import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TRAIN_DIR)
if TRAIN_DIR in sys.path:
    sys.path.remove(TRAIN_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from omegaconf import OmegaConf
from torch.optim import AdamW

from nemo_automodel.components.distillation.vlm_kd import derive_response_segments
from nemo_automodel.components.loss.topk_kd_loss import TopKKDLoss
from train.kd_lib import (
    build_components,
    build_debug_config,
    build_epoch_lr_schedule,
    build_topk_overlap_debug,
    create_attention_mask,
    get_lr_for_global_step,
    instantiate_config,
    log_rank0_topk_overlap,
    log_student_model_info,
    resolve_training_schedule,
    save_checkpoint,
    set_optimizer_lr,
    to_device,
)
from train.utils import flatten_omega_conf, get_config

logger = get_logger(__name__, log_level="INFO")


def should_trace_pipeline_step(step: int) -> bool:
    trace_until_step = int(os.environ.get("BARD_TRACE_PIPELINE_UNTIL_STEP", "128"))
    return trace_until_step > 0 and step < trace_until_step


def should_trace_detailed_step(step: int) -> bool:
    trace_start = int(os.environ.get("BARD_TRACE_DETAILED_START_STEP", "-1"))
    trace_end = int(os.environ.get("BARD_TRACE_DETAILED_END_STEP", "-1"))
    return trace_start >= 0 and trace_end >= trace_start and trace_start <= step <= trace_end


def get_tracker_backend(config) -> str:
    tracker_backend = config.experiment.get("log_with", None)
    if tracker_backend is None:
        return ""
    return str(tracker_backend).strip().lower()


def create_rank0_tensorboard_writer(project_dir: str, config):
    from torch.utils.tensorboard import SummaryWriter

    log_dir = Path(project_dir) / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(
        log_dir=str(log_dir),
        max_queue=int(config.experiment.get("tensorboard_max_queue", 100)),
        flush_secs=int(config.experiment.get("tensorboard_flush_secs", 120)),
    )


def format_token_for_log(processor, token_id: int) -> str:
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        token_text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        try:
            token_text = tokenizer.convert_ids_to_tokens(int(token_id))
        except Exception:
            token_text = str(int(token_id))
    token_text = str(token_text).replace("\n", "\\n")
    if len(token_text) > 40:
        token_text = token_text[:37] + "..."
    return f"{int(token_id)}:{token_text}"


def format_position_triplet(position_ids: torch.Tensor | None, batch_idx: int, seq_pos: int) -> str:
    if position_ids is None or batch_idx < 0 or seq_pos < 0:
        return "none"
    if position_ids.ndim == 3:
        values = position_ids[:, batch_idx, seq_pos].detach().cpu().tolist()
        return ",".join(str(int(v)) for v in values)
    return str(int(position_ids[batch_idx, seq_pos].item()))


def log_rank0_diffusion_alignment_samples(
    *,
    processor,
    student_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    target_tokens: torch.Tensor,
    sample_batch_indices: torch.Tensor | None,
    prompt_lengths: torch.Tensor | None,
    response_lengths: torch.Tensor | None,
    response_positions: torch.Tensor | None,
    student_batch: dict[str, torch.Tensor],
    teacher_batch: dict[str, torch.Tensor],
    step: int | None,
):
    max_steps = max(int(os.environ.get("BARD_LOG_DIFFUSION_ALIGN_STEPS", "4")), 0)
    if max_steps <= 0 or step is None or step >= max_steps or target_tokens.numel() == 0 or processor is None:
        return

    max_samples = max(int(os.environ.get("BARD_LOG_DIFFUSION_ALIGN_SAMPLES", "8")), 0)
    if max_samples <= 0:
        return

    topk = max(int(os.environ.get("BARD_LOG_DIFFUSION_ALIGN_TOPK", "5")), 1)
    num_rows = min(
        max_samples,
        int(target_tokens.shape[0]),
        int(student_logits.shape[0]),
        int(teacher_topk_indices.shape[0]),
    )
    if num_rows <= 0:
        return

    student_topk = min(topk, int(student_logits.shape[-1]))
    teacher_topk = min(topk, int(teacher_topk_indices.shape[-1]))
    student_topk_logits, student_topk_indices = torch.topk(student_logits[:num_rows].float(), k=student_topk, dim=-1)

    logger.info("step %s | diffusion alignment samples", step)
    for row_idx in range(num_rows):
        batch_idx = -1 if sample_batch_indices is None else int(sample_batch_indices[row_idx].item())
        prompt_len = -1 if prompt_lengths is None else int(prompt_lengths[row_idx].item())
        response_len = -1 if response_lengths is None else int(response_lengths[row_idx].item())
        response_pos = -1 if response_positions is None else int(response_positions[row_idx].item())
        target_id = int(target_tokens[row_idx].item())

        student_seq_pos = -1
        teacher_clean_seq_pos = -1
        teacher_noisy_seq_pos = -1
        if batch_idx >= 0 and prompt_len >= 0 and response_len >= 0 and response_pos >= 0:
            student_seq_pos = prompt_len + response_len + response_pos
            teacher_clean_seq_pos = prompt_len + response_pos
            teacher_noisy_seq_pos = prompt_len + response_len + response_pos

        teacher_items = ", ".join(
            f"{format_token_for_log(processor, int(teacher_topk_indices[row_idx, k].item()))}@{float(teacher_topk_logits[row_idx, k].item()):.2f}"
            for k in range(teacher_topk)
        )
        student_items = ", ".join(
            f"{format_token_for_log(processor, int(student_topk_indices[row_idx, k].item()))}@{float(student_topk_logits[row_idx, k].item()):.2f}"
            for k in range(student_topk)
        )

        logger.info(
            "step %s | align[%s] batch=%s prompt_len=%s response_pos=%s target=%s | "
            "student_pos=%s student_pos_ids=%s | teacher_clean_pos=%s teacher_clean_pos_ids=%s | "
            "teacher_noisy_pos=%s teacher_noisy_pos_ids=%s | teacher_topk=%s | student_topk=%s",
            step,
            row_idx,
            batch_idx,
            prompt_len,
            response_pos,
            format_token_for_log(processor, target_id),
            student_seq_pos,
            format_position_triplet(student_batch.get("position_ids"), batch_idx, student_seq_pos),
            teacher_clean_seq_pos,
            format_position_triplet(teacher_batch.get("position_ids"), batch_idx, teacher_clean_seq_pos),
            teacher_noisy_seq_pos,
            format_position_triplet(teacher_batch.get("position_ids"), batch_idx, teacher_noisy_seq_pos),
            teacher_items,
            student_items,
        )


def attach_all_trainable_params_to_loss(loss: torch.Tensor, model) -> torch.Tensor:
    if os.environ.get("BARD_FORCE_STATIC_GRAD_GRAPH", "1") != "1":
        return loss

    anchor = None
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.numel() == 0:
            continue
        scalar = parameter.reshape(-1)[0].to(device=loss.device, dtype=loss.dtype)
        anchor = scalar if anchor is None else anchor + scalar
    if anchor is None:
        return loss
    return loss + anchor * 0.0


def _unwrap_model(model):
    base_model = model
    while hasattr(base_model, "module"):
        base_model = base_model.module
    return base_model


def _count_parameters(model):
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total_params, trainable_params


def log_model_info(label: str, model):
    total_params, trainable_params = _count_parameters(model)
    ratio = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0
    first_param = next(model.parameters(), None)
    logger.info(
        "%s model | class=%s | dtype=%s | total_params=%d | trainable_params=%d | trainable_ratio=%.4f%%",
        label,
        model.__class__.__name__,
        str(first_param.dtype) if first_param is not None else "n/a",
        total_params,
        trainable_params,
        ratio,
    )


def log_rank_trace(accelerator: Accelerator, step: int, phase: str, **kwargs):
    if phase in {"fetch_batch_start", "loop_tail_done"}:
        return
    details = " ".join(f"{key}={value}" for key, value in kwargs.items())
    message = f"rank_trace | rank={accelerator.process_index} | local_rank={accelerator.local_process_index} | step={step} | phase={phase}"
    if details:
        message = f"{message} | {details}"
    logger.info(message, main_process_only=False)


def set_dataloader_epoch(dataloader, epoch: int) -> bool:
    """Best-effort epoch propagation for distributed shuffling."""
    candidates = [
        dataloader,
        getattr(dataloader, "sampler", None),
        getattr(dataloader, "batch_sampler", None),
        getattr(getattr(dataloader, "batch_sampler", None), "sampler", None),
        getattr(dataloader, "dataloader", None),
        getattr(getattr(dataloader, "dataloader", None), "sampler", None),
        getattr(getattr(dataloader, "dataloader", None), "batch_sampler", None),
        getattr(getattr(getattr(dataloader, "dataloader", None), "batch_sampler", None), "sampler", None),
    ]
    for candidate in candidates:
        if candidate is None or not hasattr(candidate, "set_epoch"):
            continue
        candidate.set_epoch(epoch)
        return True
    return False


def _parameter_debug_prefix(parameter_name: str) -> str:
    parts = parameter_name.split(".")
    if len(parts) >= 4 and parts[0] == "model" and parts[1] in {"language_model", "visual"} and parts[2] in {
        "layers",
        "blocks",
        "deepstack_merger_list",
    }:
        return ".".join(parts[:4])
    return ".".join(parts[: min(3, len(parts))])


def inspect_loss_graph_trainable_params(loss: torch.Tensor, model, max_names: int = 8) -> dict[str, object]:
    base_model = _unwrap_model(model)
    named_parameters = [(name, parameter) for name, parameter in base_model.named_parameters() if parameter.requires_grad]
    if loss.grad_fn is None or not named_parameters:
        return {
            "used_count": 0,
            "trainable_count": len(named_parameters),
            "unused_count": len(named_parameters),
            "unused_prefix_summary": "all_or_no_grad_fn",
            "unused_sample_names": [name for name, _ in named_parameters[:max_names]],
        }

    param_name_by_id = {id(parameter): name for name, parameter in named_parameters}
    used_param_ids: set[int] = set()
    visited: set[int] = set()
    stack = [loss.grad_fn]

    while stack:
        grad_fn = stack.pop()
        if grad_fn is None:
            continue

        grad_fn_id = id(grad_fn)
        if grad_fn_id in visited:
            continue
        visited.add(grad_fn_id)

        variable = getattr(grad_fn, "variable", None)
        if isinstance(variable, torch.Tensor):
            parameter_name = param_name_by_id.get(id(variable))
            if parameter_name is not None:
                used_param_ids.add(id(variable))

        for next_fn, _ in getattr(grad_fn, "next_functions", ()):
            if next_fn is not None:
                stack.append(next_fn)

    unused_names = [name for name, parameter in named_parameters if id(parameter) not in used_param_ids]
    unused_prefix_counts = Counter(_parameter_debug_prefix(name) for name in unused_names)
    unused_prefix_summary = ",".join(
        f"{prefix}:{count}" for prefix, count in unused_prefix_counts.most_common(8)
    ) or "none"

    return {
        "used_count": len(named_parameters) - len(unused_names),
        "trainable_count": len(named_parameters),
        "unused_count": len(unused_names),
        "unused_prefix_summary": unused_prefix_summary,
        "unused_sample_names": unused_names[:max_names],
    }


def build_teacher_model(config, accelerator: Accelerator | None = None):
    plugin = None if accelerator is None else getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is not None and accelerator.distributed_type == DistributedType.DEEPSPEED:
        with plugin.zero3_init_context_manager(enable=False):
            teacher = instantiate_config(config.teacher_model)
    else:
        teacher = instantiate_config(config.teacher_model)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def build_diffusion_kd_config(config) -> SimpleNamespace:
    diffusion_cfg = config.get("diffusion_kd", None)
    collate_block_size = int(config.dataloader.collate_fn.get("block_size", 4))
    teacher_server_cfg = config.get("teacher_server", None)
    return SimpleNamespace(
        max_response_len=int(diffusion_cfg.get("max_response_len", 0)) if diffusion_cfg is not None else 0,
        student_block_size=int(diffusion_cfg.get("student_block_size", collate_block_size))
        if diffusion_cfg is not None
        else collate_block_size,
        teacher_block_size=int(diffusion_cfg.get("teacher_block_size", collate_block_size))
        if diffusion_cfg is not None
        else collate_block_size,
        teacher_topk=int(
            diffusion_cfg.get(
                "teacher_topk",
                teacher_server_cfg.get("topk", 16) if teacher_server_cfg is not None else 16,
            )
        )
        if diffusion_cfg is not None
        else int(teacher_server_cfg.get("topk", 16) if teacher_server_cfg is not None else 16),
    )


def build_diffusion_batch(
    batch: dict[str, torch.Tensor],
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
    block_size: int,
    max_response_len: int = 0,
    use_existing_attention_mask: bool = False,
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    int,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    batch_size = batch["input_ids"].shape[0]
    device = batch["input_ids"].device
    capped_response_lens = response_lens.clone()
    if max_response_len > 0:
        capped_response_lens = torch.clamp(capped_response_lens, max=max_response_len)

    logits_to_keep = (batch["response_mask"] & batch["loss_mask"]).clone()
    target_tokens = []
    sample_batch_indices = []
    prompt_length_slices = []
    response_length_slices = []
    response_position_slices = []
    max_len = int(batch["input_ids"].shape[1])

    raw_block_size = batch.get("block_size", block_size)
    if isinstance(raw_block_size, torch.Tensor):
        raw_block_size = int(raw_block_size.item())
    else:
        raw_block_size = int(raw_block_size)

    if use_existing_attention_mask and raw_block_size == int(block_size) and torch.equal(capped_response_lens, response_lens):
        attention_mask = batch["attention_mask"]
    else:
        attention_masks = []
        for batch_idx in range(batch_size):
            attention_masks.append(
                create_attention_mask(
                    prefix_len=int(prompt_lens[batch_idx].item()),
                    response_len=int(capped_response_lens[batch_idx].item()),
                    block_size=block_size,
                    max_len=max_len,
                    device=device,
                )
            )
        attention_mask = torch.cat(attention_masks, dim=0)

    for batch_idx in range(batch_size):
        prompt_len = int(prompt_lens[batch_idx].item())
        original_response_len = int(response_lens[batch_idx].item())
        response_len = int(capped_response_lens[batch_idx].item())
        noisy_start = prompt_len + original_response_len

        if response_len < original_response_len:
            logits_to_keep[batch_idx, noisy_start + response_len : noisy_start + original_response_len] = False

        selected_positions = torch.nonzero(logits_to_keep[batch_idx], as_tuple=False).flatten()
        if selected_positions.numel() > 0:
            response_positions = selected_positions - noisy_start
            valid = (response_positions >= 0) & (response_positions < response_len)
            if valid.any():
                selected_positions = selected_positions[valid]
                response_positions = response_positions[valid]
                target_tokens.append(batch["labels"][batch_idx, selected_positions])
                sample_batch_indices.append(
                    torch.full(
                        (int(response_positions.numel()),),
                        fill_value=batch_idx,
                        dtype=torch.long,
                        device=device,
                    )
                )
                prompt_length_slices.append(
                    torch.full(
                        (int(response_positions.numel()),),
                        fill_value=prompt_len,
                        dtype=torch.long,
                        device=device,
                    )
                )
                response_length_slices.append(
                    torch.full(
                        (int(response_positions.numel()),),
                        fill_value=response_len,
                        dtype=torch.long,
                        device=device,
                    )
                )
                response_position_slices.append(response_positions)

    result = {
        "input_ids": batch["input_ids"],
        "labels": batch["labels"],
        "attention_mask": attention_mask,
        "logits_to_keep": logits_to_keep,
    }
    if "position_ids" in batch:
        result["position_ids"] = batch["position_ids"]
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if key in batch:
            result[key] = batch[key]

    return (
        result,
        torch.cat(target_tokens, dim=0) if target_tokens else batch["input_ids"].new_empty((0,)),
        int(logits_to_keep.sum().item()),
        torch.cat(sample_batch_indices, dim=0) if sample_batch_indices else None,
        torch.cat(prompt_length_slices, dim=0) if prompt_length_slices else None,
        torch.cat(response_length_slices, dim=0) if response_length_slices else None,
        torch.cat(response_position_slices, dim=0) if response_position_slices else None,
    )


def forward_diffusion_logits(model, diffusion_batch: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=diffusion_batch["input_ids"],
        attention_mask=diffusion_batch["attention_mask"],
        position_ids=diffusion_batch.get("position_ids", None),
        logits_to_keep=diffusion_batch["logits_to_keep"],
        pixel_values=diffusion_batch.get("pixel_values", None),
        pixel_values_videos=diffusion_batch.get("pixel_values_videos", None),
        image_grid_thw=diffusion_batch.get("image_grid_thw", None),
        video_grid_thw=diffusion_batch.get("video_grid_thw", None),
        use_cache=False,
    )
    if not hasattr(outputs, "logits"):
        raise AttributeError(f"`forward_diffusion_logits` expects model outputs exposing `.logits`, got {type(outputs)}")
    logits = outputs.logits
    return logits.reshape(-1, logits.shape[-1]) if logits.ndim == 3 else logits


def build_teacher_topk(
    teacher_model,
    teacher_batch: dict[str, torch.Tensor],
    teacher_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        teacher_logits = forward_diffusion_logits(teacher_model, teacher_batch)
        if teacher_logits.numel() == 0:
            empty = teacher_logits.new_empty((0, 0))
            return empty.long(), empty, teacher_logits.new_tensor(0.0)

        topk = min(max(int(teacher_topk), 1), int(teacher_logits.shape[-1]))
        teacher_topk_logits, teacher_topk_indices = torch.topk(teacher_logits, k=topk, dim=-1)
        teacher_prob = F.softmax(teacher_topk_logits.float(), dim=-1)
        teacher_entropy = -(teacher_prob * torch.log(teacher_prob.clamp_min(1e-12))).sum(dim=-1).mean()
        return teacher_topk_indices, teacher_topk_logits, teacher_entropy


def compute_diffusion_kd_losses(
    student_model,
    teacher_model,
    student_batch: dict[str, torch.Tensor],
    teacher_batch: dict[str, torch.Tensor],
    target_tokens: torch.Tensor,
    teacher_topk: int,
    ce_weight: float,
    kd_weight: float,
    topk_kd_loss: TopKKDLoss,
    processor=None,
    debug_config: SimpleNamespace | None = None,
    sample_batch_indices: torch.Tensor | None = None,
    prompt_lengths: torch.Tensor | None = None,
    response_lengths: torch.Tensor | None = None,
    response_positions: torch.Tensor | None = None,
    accelerator: Accelerator | None = None,
    step: int | None = None,
    trace_first_step: bool = False,
):
    if target_tokens.numel() == 0:
        zero = attach_all_trainable_params_to_loss(
            target_tokens.new_tensor(0.0, dtype=torch.float32),
            student_model,
        )
        return zero, zero, zero, zero, 0, None

    if trace_first_step and accelerator is not None and step is not None:
        log_rank_trace(
            accelerator,
            step,
            "teacher_forward_start",
            teacher_input_shape=tuple(teacher_batch["input_ids"].shape),
        )
    teacher_topk_indices, teacher_topk_logits, teacher_entropy = build_teacher_topk(
        teacher_model=teacher_model,
        teacher_batch=teacher_batch,
        teacher_topk=teacher_topk,
    )
    if trace_first_step and accelerator is not None and step is not None:
        log_rank_trace(
            accelerator,
            step,
            "teacher_forward_done",
            teacher_topk_shape=tuple(teacher_topk_indices.shape),
        )
    if trace_first_step and accelerator is not None and step is not None:
        log_rank_trace(
            accelerator,
            step,
            "student_forward_start",
            target_tokens=int(target_tokens.numel()),
            student_input_shape=tuple(student_batch["input_ids"].shape),
        )
    student_logits = forward_diffusion_logits(student_model, student_batch)
    if trace_first_step and accelerator is not None and step is not None:
        log_rank_trace(
            accelerator,
            step,
            "student_forward_done",
            student_logits_shape=tuple(student_logits.shape),
        )
    if student_logits.shape[0] != target_tokens.shape[0]:
        raise ValueError(
            f"Student logits/targets mismatch: logits={int(student_logits.shape[0])}, targets={int(target_tokens.shape[0])}."
        )
    if teacher_topk_indices.shape[0] != student_logits.shape[0]:
        raise ValueError(
            f"Teacher/student token mismatch: teacher={int(teacher_topk_indices.shape[0])}, student={int(student_logits.shape[0])}."
        )

    ce_loss = student_logits.new_tensor(0.0)
    if ce_weight > 0.0:
        ce_loss = F.cross_entropy(student_logits.float(), target_tokens.long())

    kd_loss = student_logits.new_tensor(0.0)
    if kd_weight > 0.0:
        kd_loss = topk_kd_loss(
            student_logits,
            teacher_topk_indices=teacher_topk_indices,
            teacher_topk_logits=teacher_topk_logits,
            num_batch_labels=student_logits.shape[0],
        )
    if accelerator is not None and accelerator.is_main_process:
        log_rank0_diffusion_alignment_samples(
            processor=processor,
            student_logits=student_logits.detach(),
            teacher_topk_indices=teacher_topk_indices.detach(),
            teacher_topk_logits=teacher_topk_logits.detach(),
            target_tokens=target_tokens.detach(),
            sample_batch_indices=sample_batch_indices.detach() if sample_batch_indices is not None else None,
            prompt_lengths=prompt_lengths.detach() if prompt_lengths is not None else None,
            response_lengths=response_lengths.detach() if response_lengths is not None else None,
            response_positions=response_positions.detach() if response_positions is not None else None,
            student_batch=student_batch,
            teacher_batch=teacher_batch,
            step=step,
        )
    if trace_first_step and accelerator is not None and step is not None:
        log_rank_trace(
            accelerator,
            step,
            "loss_compute_done",
            ce_loss=f"{float(ce_loss.detach().float().item()):.6f}",
            kd_loss=f"{float(kd_loss.detach().float().item()):.6f}",
        )

    total_loss = ce_weight * ce_loss + kd_weight * kd_loss
    total_loss = attach_all_trainable_params_to_loss(total_loss, student_model)
    if trace_first_step and accelerator is not None and step is not None:
        loss_graph_diag = inspect_loss_graph_trainable_params(total_loss, student_model)
        log_rank_trace(
            accelerator,
            step,
            "loss_graph_diag",
            trainable_params=loss_graph_diag["trainable_count"],
            used_params=loss_graph_diag["used_count"],
            unused_params=loss_graph_diag["unused_count"],
            unused_prefixes=loss_graph_diag["unused_prefix_summary"],
            unused_samples=";".join(loss_graph_diag["unused_sample_names"]),
            has_pixel_values="pixel_values" in student_batch,
            pixel_values_shape=tuple(student_batch["pixel_values"].shape)
            if "pixel_values" in student_batch
            else "none",
            image_grid_shape=tuple(student_batch["image_grid_thw"].shape)
            if "image_grid_thw" in student_batch
            else "none",
        )
    diagnostics = None
    if debug_config is not None and debug_config.enabled and debug_config.log_topk_overlap:
        diagnostics = build_topk_overlap_debug(
            processor=processor,
            selected_student_logits=student_logits.detach(),
            teacher_topk_indices=teacher_topk_indices.detach(),
            debug_config=debug_config,
            target_tokens=target_tokens.detach(),
            prompt_lengths=prompt_lengths.detach() if prompt_lengths is not None else None,
            response_positions=response_positions.detach() if response_positions is not None else None,
            teacher_positions=response_positions.detach() if response_positions is not None else None,
        )

    return total_loss, ce_loss, kd_loss, teacher_entropy, int(target_tokens.numel()), diagnostics


def train():
    config = get_config()
    project_dir = str(Path(config.experiment.project) / "logs")
    tracker_backend = get_tracker_backend(config)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config.training.gradient_accumulation_steps),
        mixed_precision=config.training.mixed_precision,
        log_with=None if tracker_backend == "tensorboard" else config.experiment.get("log_with", None),
        project_dir=project_dir,
        split_batches=False,
        cpu=bool(config.training.get("cpu", False)),
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if config.training.seed is not None:
        set_seed(int(config.training.seed))

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        OmegaConf.save(config, Path(config.experiment.project) / "config.yaml")

    use_accelerate_trackers = bool(tracker_backend) and tracker_backend != "tensorboard"
    tb_writer = None
    tb_flush_every_steps = max(0, int(config.experiment.get("tensorboard_flush_every_steps", 50)))
    if tracker_backend == "tensorboard" and accelerator.is_main_process:
        tb_writer = create_rank0_tensorboard_writer(project_dir, config)
    elif use_accelerate_trackers:
        accelerator.init_trackers(
            config.experiment.project,
            config={k: v for k, v in flatten_omega_conf(config, resolve=True)},
        )

    dataloader, student, processor = build_components(config)
    teacher = build_teacher_model(config, accelerator=accelerator)
    debug_config = build_debug_config(config)
    diffusion_kd_config = build_diffusion_kd_config(config)

    if bool(config.training.get("gradient_checkpointing_enable", False)):
        if hasattr(student, "gradient_checkpointing_enable"):
            logger.info("Enabling gradient checkpointing on student")
            student.gradient_checkpointing_enable()
        else:
            logger.warning("training.gradient_checkpointing_enable=true but student has no gradient_checkpointing_enable()")

    if accelerator.is_main_process:
        log_student_model_info(student)
        log_model_info("teacher", teacher)
        logger.info(
            "diffusion kd config | student_block_size=%d | teacher_block_size=%d | teacher_topk=%d | max_response_len=%d | kd_positions=response_mask&loss_mask",
            diffusion_kd_config.student_block_size,
            diffusion_kd_config.teacher_block_size,
            diffusion_kd_config.teacher_topk,
            diffusion_kd_config.max_response_len,
        )

    base_lr = float(config.optimizer.lr)
    optimizer = AdamW(
        student.parameters(),
        lr=base_lr,
        betas=tuple(config.optimizer.get("betas", [0.9, 0.95])),
        weight_decay=float(config.optimizer.get("weight_decay", 0.0)),
    )
    student, optimizer, dataloader = accelerator.prepare(student, optimizer, dataloader)
    teacher.to(accelerator.device)
    teacher.eval()

    training_schedule = resolve_training_schedule(config, dataloader)
    if config.training.get("max_steps", None) is not None and accelerator.is_main_process:
        logger.info(
            "training.max_steps=%s is ignored; using training.num_epochs=%d and dataloader_steps_per_epoch=%d",
            str(config.training.max_steps),
            training_schedule.num_epochs,
            training_schedule.epoch_steps,
        )
    lr_schedule = build_epoch_lr_schedule(config, training_schedule.total_train_steps)
    set_optimizer_lr(optimizer, get_lr_for_global_step(lr_schedule, 0))
    student.train()

    if accelerator.is_main_process:
        logger.info(
            "training schedule | num_epochs=%d | dataloader_steps_per_epoch=%d | optimizer_steps_per_epoch=%d | total_micro_steps=%d | total_train_steps=%d",
            training_schedule.num_epochs,
            training_schedule.epoch_steps,
            training_schedule.optimizer_steps_per_epoch,
            training_schedule.total_micro_steps,
            training_schedule.total_train_steps,
        )
        logger.info(
            "lr schedule | base_lr=%.8f | warmup_start_ratio=%.4f | warmup_steps=%d | total_train_steps=%d | min_lr_ratio=%.4f",
            lr_schedule.base_lr,
            lr_schedule.warmup_start_ratio,
            lr_schedule.warmup_steps,
            lr_schedule.total_train_steps,
            lr_schedule.min_lr_ratio,
        )

    topk_kd_loss = TopKKDLoss(
        temperature=float(config.distillation.get("temperature", 1.0)),
        fp32_upcast=bool(config.distillation.get("fp32_upcast", True)),
        direction=str(config.distillation.get("kl_direction", "forward-kl")),
    )
    ce_weight = float(config.loss.get("ce_weight", 1.0))
    kd_weight = float(config.loss.get("kd_weight", 1.0))

    max_micro_steps = int(training_schedule.total_micro_steps)
    global_step = 0
    optimizer_step = 0

    try:
        for epoch in range(training_schedule.num_epochs):
            if global_step >= max_micro_steps:
                break

            if accelerator.is_main_process:
                logger.info(
                    "starting epoch %d/%d | micro_step=%d/%d | optimizer_step=%d/%d",
                    epoch + 1,
                    training_schedule.num_epochs,
                    global_step,
                    max_micro_steps,
                    optimizer_step,
                    training_schedule.total_train_steps,
                )

            if not set_dataloader_epoch(dataloader, epoch) and accelerator.is_main_process and epoch == 0:
                logger.warning("dataloader/sampler does not expose set_epoch(); shuffle order may repeat across epochs")

            dataloader_iter = iter(dataloader)
            while True:
                if global_step >= max_micro_steps:
                    break

                trace_pipeline_step = should_trace_pipeline_step(global_step)
                trace_detailed_step = should_trace_detailed_step(global_step)
                if trace_pipeline_step:
                    log_rank_trace(accelerator, global_step, "fetch_batch_start")
                step_start_time = time.perf_counter()
                fetch_batch_start_time = time.perf_counter()
                try:
                    raw_batch = next(dataloader_iter)
                except StopIteration:
                    break
                fetch_batch_time_s = time.perf_counter() - fetch_batch_start_time
                if trace_pipeline_step or fetch_batch_time_s >= 5.0:
                    input_ids = raw_batch.get("input_ids")
                    log_rank_trace(
                        accelerator,
                        global_step,
                        "fetch_batch_done",
                        fetch_batch_s=f"{fetch_batch_time_s:.4f}",
                        input_shape=tuple(input_ids.shape) if input_ids is not None else "n/a",
                    )

                if accelerator.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(accelerator.device)
                raw_batch = to_device(raw_batch, accelerator.device)
                prompt_lens, response_lens, _ = derive_response_segments(raw_batch["response_mask"])
                build_batch_start_time = time.perf_counter()
                (
                    student_batch,
                    target_tokens,
                    kd_tokens,
                    sample_batch_indices,
                    prompt_lengths,
                    response_lengths,
                    response_positions,
                ) = build_diffusion_batch(
                    batch=raw_batch,
                    prompt_lens=prompt_lens,
                    response_lens=response_lens,
                    block_size=diffusion_kd_config.student_block_size,
                    max_response_len=diffusion_kd_config.max_response_len,
                    use_existing_attention_mask=True,
                )
                teacher_batch, _, _, _, _, _, _ = build_diffusion_batch(
                    batch=raw_batch,
                    prompt_lens=prompt_lens,
                    response_lens=response_lens,
                    block_size=diffusion_kd_config.teacher_block_size,
                    max_response_len=diffusion_kd_config.max_response_len,
                )
                build_batch_time_s = time.perf_counter() - build_batch_start_time

                forward_time_s = 0.0
                loss_backward_time_s = 0.0
                trace_first_step = global_step == 0

                if trace_first_step:
                    log_rank_trace(
                        accelerator,
                        global_step,
                        "build_batch_done",
                        student_input_shape=tuple(student_batch["input_ids"].shape),
                        teacher_input_shape=tuple(teacher_batch["input_ids"].shape),
                        kd_tokens=int(kd_tokens),
                    )

                with accelerator.accumulate(student):
                    loss_backward_start_time = time.perf_counter()
                    forward_start_time = time.perf_counter()
                    if trace_detailed_step:
                        log_rank_trace(accelerator, global_step, "compute_loss_start")
                    total_loss, ce_loss, kd_loss, teacher_entropy_topk, kd_tokens, topk_overlap_diag = compute_diffusion_kd_losses(
                        student_model=student,
                        teacher_model=teacher,
                        student_batch=student_batch,
                        teacher_batch=teacher_batch,
                        target_tokens=target_tokens,
                        teacher_topk=diffusion_kd_config.teacher_topk,
                        ce_weight=ce_weight,
                        kd_weight=kd_weight,
                        topk_kd_loss=topk_kd_loss,
                        processor=processor if accelerator.is_main_process else None,
                        debug_config=debug_config,
                        sample_batch_indices=sample_batch_indices,
                        prompt_lengths=prompt_lengths,
                        response_lengths=response_lengths,
                        response_positions=response_positions,
                        accelerator=accelerator,
                        step=global_step,
                        trace_first_step=trace_first_step,
                    )
                    if trace_detailed_step:
                        log_rank_trace(
                            accelerator,
                            global_step,
                            "compute_loss_done",
                            ce_loss=f"{float(ce_loss.detach()):.6f}",
                            kd_loss=f"{float(kd_loss.detach()):.6f}",
                            kd_tokens=int(kd_tokens),
                        )
                    forward_time_s = time.perf_counter() - forward_start_time

                    if total_loss.requires_grad:
                        if trace_first_step or trace_detailed_step:
                            log_rank_trace(accelerator, global_step, "backward_start")
                        accelerator.backward(total_loss)
                        if trace_first_step or trace_detailed_step:
                            log_rank_trace(accelerator, global_step, "backward_done")

                    if accelerator.sync_gradients:
                        optimizer_step += 1
                        current_lr = get_lr_for_global_step(lr_schedule, optimizer_step)
                        set_optimizer_lr(optimizer, current_lr)
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = torch.tensor(0.0, device=accelerator.device)
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                student.parameters(), float(config.training.max_grad_norm)
                            )
                        if trace_first_step or trace_detailed_step:
                            log_rank_trace(accelerator, global_step, "optimizer_step_start", lr=f"{current_lr:.8f}")
                        optimizer.step()
                        if trace_first_step or trace_detailed_step:
                            log_rank_trace(accelerator, global_step, "optimizer_step_done")
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            ds_engine = getattr(accelerator, "deepspeed_engine_wrapped", None)
                            ds_grad_norm = None if ds_engine is None else ds_engine.get_global_grad_norm()
                            grad_norm = torch.tensor(
                                0.0 if ds_grad_norm is None else float(ds_grad_norm),
                                device=accelerator.device,
                                dtype=torch.float32,
                            )
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        grad_norm = torch.tensor(0.0, device=accelerator.device)
                        current_lr = optimizer.param_groups[0]["lr"]
                    loss_backward_time_s = time.perf_counter() - loss_backward_start_time

                metrics = torch.tensor(
                    [
                        total_loss.detach(),
                        ce_loss.detach(),
                        kd_loss.detach(),
                        teacher_entropy_topk.detach(),
                        float(kd_tokens),
                        float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                        float(current_lr),
                        float(topk_overlap_diag["topk_overlap_mean"]) if topk_overlap_diag is not None else 0.0,
                        float(fetch_batch_time_s),
                        float(build_batch_time_s),
                        float(forward_time_s),
                        float(loss_backward_time_s),
                        float(
                            torch.cuda.max_memory_allocated(accelerator.device) / (1024.0 * 1024.0 * 1024.0)
                            if accelerator.device.type == "cuda"
                            else 0.0
                        ),
                        float(
                            torch.cuda.max_memory_reserved(accelerator.device) / (1024.0 * 1024.0 * 1024.0)
                            if accelerator.device.type == "cuda"
                            else 0.0
                        ),
                    ],
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                if trace_detailed_step:
                    log_rank_trace(accelerator, global_step, "metrics_gather_start")
                gathered = accelerator.gather(metrics.unsqueeze(0))
                if trace_detailed_step:
                    log_rank_trace(accelerator, global_step, "metrics_gather_done")
                step_time_s = time.perf_counter() - step_start_time

                if accelerator.is_main_process:
                    mean_metrics = gathered.mean(dim=0).detach().float().cpu().tolist()
                    max_metrics = gathered.max(dim=0).values.detach().float().cpu().tolist()
                    log_rank0_topk_overlap(global_step, topk_overlap_diag, debug_config)
                    log_parts = [
                        f"micro_step {global_step}",
                        f"optimizer_step {optimizer_step}",
                        f"loss {mean_metrics[0]:.6f}",
                        f"ce_loss {mean_metrics[1]:.6f}",
                        f"kd_loss {mean_metrics[2]:.6f}",
                        f"teacher_entropy_topk {mean_metrics[3]:.6f}",
                        f"kd_tokens {mean_metrics[4]:.0f}",
                        f"grad_norm {mean_metrics[5]:.6f}",
                        f"lr {mean_metrics[6]:.8f}",
                    ]
                    if debug_config.enabled and debug_config.log_topk_overlap:
                        log_parts.append(f"topk_overlap@{debug_config.topk_overlap_k} {mean_metrics[7]:.4f}")
                    log_parts.extend(
                        [
                            f"fetch_batch_s mean/max {mean_metrics[8]:.4f}/{max_metrics[8]:.4f}",
                            f"build_batch_s mean/max {mean_metrics[9]:.4f}/{max_metrics[9]:.4f}",
                            f"forward_s mean/max {mean_metrics[10]:.4f}/{max_metrics[10]:.4f}",
                            f"loss_bw_s mean/max {mean_metrics[11]:.4f}/{max_metrics[11]:.4f}",
                            f"step_s {step_time_s:.4f}",
                            f"cuda_alloc_gb mean/max {mean_metrics[12]:.2f}/{max_metrics[12]:.2f}",
                            f"cuda_reserved_gb mean/max {mean_metrics[13]:.2f}/{max_metrics[13]:.2f}",
                        ]
                    )
                    logger.info(" | ".join(log_parts))

                    if tracker_backend:
                        if trace_pipeline_step:
                            log_rank_trace(accelerator, global_step, "tracker_log_start")
                        tracker_metrics = {
                            "train/micro_step": float(global_step),
                            "train/optimizer_step": float(optimizer_step),
                            "train/loss": mean_metrics[0],
                            "train/ce_loss": mean_metrics[1],
                            "train/kd_loss": mean_metrics[2],
                            "train/teacher_entropy_topk": mean_metrics[3],
                            "train/kd_tokens": mean_metrics[4],
                            "train/grad_norm": mean_metrics[5],
                            "train/lr": mean_metrics[6],
                            "train/fetch_batch_s_mean": mean_metrics[8],
                            "train/build_batch_s_mean": mean_metrics[9],
                            "train/forward_s_mean": mean_metrics[10],
                            "train/loss_backward_s_mean": mean_metrics[11],
                            "train/cuda_alloc_gb_mean": mean_metrics[12],
                            "train/cuda_reserved_gb_mean": mean_metrics[13],
                        }
                        if debug_config.enabled and debug_config.log_topk_overlap:
                            tracker_metrics[f"train/topk_overlap_at_{debug_config.topk_overlap_k}"] = mean_metrics[7]
                        if tb_writer is not None:
                            for key, value in tracker_metrics.items():
                                tb_writer.add_scalar(key, value, global_step=global_step)
                            if tb_flush_every_steps > 0 and (global_step + 1) % tb_flush_every_steps == 0:
                                tb_writer.flush()
                        else:
                            accelerator.log(tracker_metrics, step=global_step)
                        if trace_pipeline_step:
                            log_rank_trace(accelerator, global_step, "tracker_log_done")

                global_step += 1
                if trace_pipeline_step:
                    log_rank_trace(accelerator, global_step - 1, "loop_tail_done")
                if int(config.training.get("save_every", 0)) > 0 and accelerator.sync_gradients:
                    if optimizer_step % int(config.training.save_every) == 0:
                        save_checkpoint(accelerator, student, processor, config.experiment.project, optimizer_step)

        save_checkpoint(accelerator, student, processor, config.experiment.project, optimizer_step)
    finally:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
        if use_accelerate_trackers:
            accelerator.end_training()


if __name__ == "__main__":
    train()
