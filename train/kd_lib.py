from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque
from math import ceil, cos, pi
from pathlib import Path
from types import SimpleNamespace

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TRAIN_DIR)
if TRAIN_DIR in sys.path:
    sys.path.remove(TRAIN_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from omegaconf import OmegaConf
from torch.optim import AdamW

from nemo_automodel.components.config.loader import _resolve_target
from nemo_automodel.components.distillation.teacher_rpc import AsyncTeacherClient
from nemo_automodel.components.distillation.vlm_kd import collect_generation_kd_indices
from nemo_automodel.components.distillation.vlm_on_policy import (
    build_prompt_batch,
    build_teacher_batch_from_gold_response,
    build_teacher_batch_from_on_policy_response,
    sample_on_policy_responses,
)
from nemo_automodel.components.loss.topk_kd_loss import TopKKDLoss
from train.utils import flatten_omega_conf, get_config

logger = get_logger(__name__, log_level="INFO")


def to_device(data, device):
    if isinstance(data, dict):
        return type(data)({k: to_device(v, device) for k, v in data.items()})
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    return data


def to_cpu(data):
    if isinstance(data, dict):
        return type(data)({k: to_cpu(v) for k, v in data.items()})
    if isinstance(data, (list, tuple)):
        return type(data)(to_cpu(v) for v in data)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    return data


def instantiate_config(cfg, **extra_kwargs):
    if cfg is None:
        return None
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict) or "_target_" not in container:
        raise ValueError(f"Config must be a mapping with `_target_`, got: {container}")
    target = container.pop("_target_")
    container.update(extra_kwargs)
    fn = _resolve_target(target)
    return fn(**container)


def build_components(config):
    from transformers import AutoProcessor

    processor = instantiate_config(config.get("processor", None))
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            config.model.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

    dataset_kwargs = {k: v for k, v in OmegaConf.to_container(config.dataset, resolve=True).items() if k != "_target_"}
    dataset = instantiate_config(config.dataset, **dataset_kwargs)

    collate_fn = None
    if config.dataloader.get("collate_fn", None) is not None:
        collate_target = str(config.dataloader.collate_fn.get("_target_", ""))
        collate_extra_kwargs = {}
        diffusion_cfg = config.get("diffusion_kd", None)
        if diffusion_cfg is not None and collate_target.endswith("qwen_vl_block_collate_fn"):
            collate_extra_kwargs["min_mask_rate"] = float(diffusion_cfg.get("min_mask_rate", 0.001))
            collate_extra_kwargs["max_mask_rate"] = float(diffusion_cfg.get("max_mask_rate", 1.0))
        collate_fn = instantiate_config(
            config.dataloader.collate_fn,
            processor=processor,
            max_len=config.dataset.max_len,
            **collate_extra_kwargs,
        )

    dataloader_kwargs = {
        k: v
        for k, v in OmegaConf.to_container(config.dataloader, resolve=True).items()
        if k not in ("_target_", "collate_fn")
    }
    dataloader = instantiate_config(
        config.dataloader,
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=int(config.training.per_device_train_batch_size),
        **dataloader_kwargs,
    )
    student = instantiate_config(config.model)
    return dataloader, student, processor


def save_checkpoint(accelerator, model, processor, output_dir: str, step: int):
    if not accelerator.is_main_process:
        return
    save_dir = Path(output_dir) / f"step-{step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    if hasattr(unwrapped, "save_pretrained"):
        unwrapped.save_pretrained(save_dir)
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(save_dir)
    else:
        torch.save(unwrapped.state_dict(), save_dir / "model.pt")


def build_generation_config(config) -> SimpleNamespace:
    return SimpleNamespace(**OmegaConf.to_container(config.on_policy, resolve=True))


def build_lr_schedule(config) -> SimpleNamespace:
    scheduler_cfg = config.get("lr_scheduler", None)
    warmup_steps = int(scheduler_cfg.get("warmup_steps", 0)) if scheduler_cfg is not None else 0
    min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.1)) if scheduler_cfg is not None else 0.1
    warmup_start_ratio = float(scheduler_cfg.get("warmup_start_ratio", 0.1)) if scheduler_cfg is not None else 0.1
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))
    warmup_start_ratio = max(0.0, min(1.0, warmup_start_ratio))

    grad_accum_steps = max(int(config.training.gradient_accumulation_steps), 1)
    total_train_steps = max(int(config.training.max_steps), 1)
    total_optimizer_steps = max(ceil(total_train_steps / grad_accum_steps), 1)
    warmup_steps = min(max(warmup_steps, 0), total_optimizer_steps)
    base_lr = float(config.optimizer.lr)

    return SimpleNamespace(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
        warmup_start_ratio=warmup_start_ratio,
        total_optimizer_steps=total_optimizer_steps,
    )


def build_epoch_lr_schedule(config, total_train_steps: int) -> SimpleNamespace:
    scheduler_cfg = config.get("lr_scheduler", None)
    warmup_steps = int(scheduler_cfg.get("warmup_steps", 0)) if scheduler_cfg is not None else 0
    min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.1)) if scheduler_cfg is not None else 0.1
    warmup_start_ratio = float(scheduler_cfg.get("warmup_start_ratio", 0.1)) if scheduler_cfg is not None else 0.1
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))
    warmup_start_ratio = max(0.0, min(1.0, warmup_start_ratio))

    total_train_steps = max(int(total_train_steps), 1)
    warmup_steps = min(max(warmup_steps, 0), total_train_steps)
    base_lr = float(config.optimizer.lr)

    return SimpleNamespace(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
        warmup_start_ratio=warmup_start_ratio,
        total_train_steps=total_train_steps,
    )


def resolve_training_schedule(config, dataloader) -> SimpleNamespace:
    if "num_epochs" not in config.training:
        raise ValueError("training.num_epochs is required to derive max_steps from epochs")

    num_epochs = int(config.training.num_epochs)
    if num_epochs <= 0:
        raise ValueError(f"training.num_epochs must be greater than 0, got {num_epochs}")

    try:
        epoch_steps = int(len(dataloader))
    except TypeError as exc:
        raise ValueError("Epoch-based training requires a dataloader with a defined length") from exc

    if epoch_steps <= 0:
        raise ValueError(f"Dataloader length must be greater than 0, got {epoch_steps}")

    grad_accum_steps = max(int(config.training.gradient_accumulation_steps), 1)
    optimizer_steps_per_epoch = max(ceil(epoch_steps / grad_accum_steps), 1)
    total_micro_steps = epoch_steps * num_epochs
    total_train_steps = optimizer_steps_per_epoch * num_epochs
    return SimpleNamespace(
        num_epochs=num_epochs,
        epoch_steps=epoch_steps,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        total_micro_steps=total_micro_steps,
        total_train_steps=total_train_steps,
    )


def get_lr_for_optimizer_step(schedule: SimpleNamespace, optimizer_step: int) -> float:
    if optimizer_step <= 0:
        return schedule.base_lr * schedule.warmup_start_ratio

    if schedule.warmup_steps > 0 and optimizer_step <= schedule.warmup_steps:
        warmup_progress = float(optimizer_step - 1) / float(max(schedule.warmup_steps - 1, 1))
        lr_ratio = schedule.warmup_start_ratio + (1.0 - schedule.warmup_start_ratio) * warmup_progress
        return schedule.base_lr * lr_ratio

    if schedule.total_optimizer_steps <= schedule.warmup_steps:
        return schedule.base_lr * schedule.min_lr_ratio

    progress = float(optimizer_step - schedule.warmup_steps) / float(
        schedule.total_optimizer_steps - schedule.warmup_steps
    )
    progress = max(0.0, min(1.0, progress))
    cosine_decay = 0.5 * (1.0 + cos(pi * progress))
    return schedule.base_lr * (schedule.min_lr_ratio + (1.0 - schedule.min_lr_ratio) * cosine_decay)


def get_lr_for_global_step(schedule: SimpleNamespace, global_step: int) -> float:
    if global_step <= 0:
        return schedule.base_lr * schedule.warmup_start_ratio

    if schedule.warmup_steps > 0 and global_step <= schedule.warmup_steps:
        warmup_progress = float(global_step - 1) / float(max(schedule.warmup_steps - 1, 1))
        lr_ratio = schedule.warmup_start_ratio + (1.0 - schedule.warmup_start_ratio) * warmup_progress
        return schedule.base_lr * lr_ratio

    if schedule.total_train_steps <= schedule.warmup_steps:
        return schedule.base_lr * schedule.min_lr_ratio

    progress = float(global_step - schedule.warmup_steps) / float(
        schedule.total_train_steps - schedule.warmup_steps
    )
    progress = max(0.0, min(1.0, progress))
    cosine_decay = 0.5 * (1.0 + cos(pi * progress))
    return schedule.base_lr * (schedule.min_lr_ratio + (1.0 - schedule.min_lr_ratio) * cosine_decay)


def set_optimizer_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _count_parameters(model):
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total_params, trainable_params


def log_student_model_info(model):
    total_params, trainable_params = _count_parameters(model)
    ratio = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0
    first_param = next(model.parameters(), None)
    logger.info(
        "student model | class=%s | dtype=%s | total_params=%d | trainable_params=%d | trainable_ratio=%.4f%%",
        model.__class__.__name__,
        str(first_param.dtype) if first_param is not None else "n/a",
        total_params,
        trainable_params,
        ratio,
    )


def build_debug_config(config) -> SimpleNamespace:
    debug_cfg = config.get("debug", None)
    if debug_cfg is None:
        return SimpleNamespace(
            enabled=False,
            log_generated_text=False,
            log_topk_overlap=False,
            generated_text_every_steps=1,
            generated_text_num_samples=1,
            generated_text_skip_special_tokens=True,
            log_prompt_text=True,
            topk_overlap_every_steps=1,
            topk_overlap_k=10,
            topk_overlap_num_samples=2,
        )
    return SimpleNamespace(
        enabled=bool(debug_cfg.get("enabled", False)),
        log_generated_text=bool(debug_cfg.get("log_generated_text", False)),
        log_topk_overlap=bool(debug_cfg.get("log_topk_overlap", False)),
        generated_text_every_steps=int(debug_cfg.get("generated_text_every_steps", 1)),
        generated_text_num_samples=int(debug_cfg.get("generated_text_num_samples", 1)),
        generated_text_skip_special_tokens=bool(debug_cfg.get("generated_text_skip_special_tokens", True)),
        log_prompt_text=bool(debug_cfg.get("log_prompt_text", True)),
        topk_overlap_every_steps=int(debug_cfg.get("topk_overlap_every_steps", 1)),
        topk_overlap_k=int(debug_cfg.get("topk_overlap_k", 10)),
        topk_overlap_num_samples=int(debug_cfg.get("topk_overlap_num_samples", 2)),
    )


def build_stability_config(config) -> SimpleNamespace:
    stability_cfg = config.get("stability", None)
    if stability_cfg is None:
        return SimpleNamespace(
            log_timing=True,
            log_timing_every_steps=1,
            warn_on_max_len_hit=True,
            drop_rollout_if_hits_max_len=False,
        )
    return SimpleNamespace(
        log_timing=bool(stability_cfg.get("log_timing", True)),
        log_timing_every_steps=int(stability_cfg.get("log_timing_every_steps", 1)),
        warn_on_max_len_hit=bool(stability_cfg.get("warn_on_max_len_hit", True)),
        drop_rollout_if_hits_max_len=bool(stability_cfg.get("drop_rollout_if_hits_max_len", False)),
    )


def _get_decoder(processor):
    if processor is None:
        return None
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "batch_decode"):
        return tokenizer.batch_decode
    return None


def _decode_sequences(processor, sequences: torch.Tensor, skip_special_tokens: bool) -> list[str]:
    decoder = _get_decoder(processor)
    if decoder is None:
        return []
    return decoder(
        sequences.detach().cpu(),
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )


def _decode_token_ids(processor, token_ids: torch.Tensor, skip_special_tokens: bool) -> list[str]:
    if token_ids.numel() == 0:
        return []
    sequences = token_ids.detach().cpu().view(-1, 1)
    return _decode_sequences(processor, sequences, skip_special_tokens=skip_special_tokens)


def _format_token_debug(token_id: int, token_text: str | None) -> str:
    if token_text is None:
        return str(token_id)
    sanitized = (
        token_text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f"{token_id}:{sanitized}"


def infer_effective_response_lens(
    sampled_responses: torch.Tensor,
    response_lens: torch.Tensor,
    eos_token_id: int,
    mask_token_id: int,
) -> torch.Tensor:
    if sampled_responses.ndim != 2:
        raise ValueError(
            f"`sampled_responses` must be 2D to infer effective lengths, got shape={tuple(sampled_responses.shape)}"
        )

    effective_lens = response_lens.clone()
    max_generated_width = int(sampled_responses.shape[1])
    if max_generated_width <= 0:
        return torch.zeros_like(response_lens)

    for batch_idx in range(sampled_responses.shape[0]):
        upper_bound = min(int(response_lens[batch_idx].item()), max_generated_width)
        if upper_bound <= 0:
            effective_lens[batch_idx] = 0
            continue

        current = sampled_responses[batch_idx, :upper_bound]
        eos_positions = torch.nonzero(current == eos_token_id, as_tuple=False).flatten()
        if eos_positions.numel() > 0:
            effective_lens[batch_idx] = int(eos_positions[0].item()) + 1
            continue

        mask_positions = torch.nonzero(current == mask_token_id, as_tuple=False).flatten()
        if mask_positions.numel() > 0:
            effective_lens[batch_idx] = int(mask_positions[0].item())
            continue

        effective_lens[batch_idx] = upper_bound

    return effective_lens


def estimate_tensor_payload_nbytes(data) -> int:
    if isinstance(data, dict):
        return sum(estimate_tensor_payload_nbytes(value) for value in data.values())
    if isinstance(data, (list, tuple)):
        return sum(estimate_tensor_payload_nbytes(value) for value in data)
    if isinstance(data, torch.Tensor):
        return int(data.numel() * data.element_size())
    return 0


def count_grid_tokens(grid_thw: torch.Tensor | None) -> int:
    if grid_thw is None or grid_thw.numel() == 0:
        return 0
    return int(torch.prod(grid_thw.long(), dim=-1).sum().item())


def build_rollout_request_lens(
    original_response_lens: torch.Tensor,
    configured_max_response_len: int,
) -> torch.Tensor:
    if configured_max_response_len > 0:
        return torch.clamp(original_response_lens, max=configured_max_response_len)
    return original_response_lens.clone()


def analyze_rollout_stop_reasons(
    sampled_responses: torch.Tensor,
    request_response_lens: torch.Tensor,
    effective_response_lens: torch.Tensor,
    eos_token_id: int,
    mask_token_id: int,
    configured_max_response_len: int,
) -> dict[str, int]:
    upper_bounds = torch.clamp(request_response_lens, max=int(sampled_responses.shape[1]))
    eos_stop_samples = 0
    mask_trunc_samples = 0
    max_len_stop_samples = 0
    request_cap_stop_samples = 0

    for batch_idx in range(sampled_responses.shape[0]):
        upper_bound = int(upper_bounds[batch_idx].item())
        if upper_bound <= 0:
            continue
        current = sampled_responses[batch_idx, :upper_bound]
        eos_positions = torch.nonzero(current == eos_token_id, as_tuple=False).flatten()
        if eos_positions.numel() > 0:
            eos_stop_samples += 1
            continue

        mask_positions = torch.nonzero(current == mask_token_id, as_tuple=False).flatten()
        if mask_positions.numel() > 0:
            mask_trunc_samples += 1
            continue

        effective_len = int(effective_response_lens[batch_idx].item())
        if configured_max_response_len > 0 and effective_len >= configured_max_response_len:
            max_len_stop_samples += 1
        else:
            request_cap_stop_samples += 1

    return {
        "request_response_len_max": int(request_response_lens.max().item()) if request_response_lens.numel() > 0 else 0,
        "effective_response_len_max": int(effective_response_lens.max().item()) if effective_response_lens.numel() > 0 else 0,
        "eos_stop_samples": eos_stop_samples,
        "mask_trunc_samples": mask_trunc_samples,
        "max_len_stop_samples": max_len_stop_samples,
        "request_cap_stop_samples": request_cap_stop_samples,
    }


def create_attention_mask(
    prefix_len: int,
    response_len: int,
    block_size: int,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    seq_len = min(prefix_len + 2 * response_len, max_len)
    mask = torch.zeros((max_len, max_len), dtype=torch.bool, device=device)
    if seq_len <= 0:
        return mask.unsqueeze(0).unsqueeze(0)

    block_ids = torch.empty((seq_len,), dtype=torch.long, device=device)
    token_types = torch.empty((seq_len,), dtype=torch.int8, device=device)

    clean_start = min(prefix_len, seq_len)
    clean_end = min(prefix_len + response_len, seq_len)
    noisy_start = clean_end

    block_ids[:clean_start] = torch.arange(clean_start, device=device) // block_size
    token_types[:clean_start] = 0

    if clean_end > clean_start:
        num_prefix_blocks = (prefix_len + block_size - 1) // block_size
        clean_len = clean_end - clean_start
        clean_rel_positions = torch.arange(clean_len, device=device)
        clean_blocks = num_prefix_blocks + clean_rel_positions // block_size
        block_ids[clean_start:clean_end] = clean_blocks
        token_types[clean_start:clean_end] = 1

    if noisy_start < seq_len:
        num_prefix_blocks = (prefix_len + block_size - 1) // block_size
        noisy_len = seq_len - noisy_start
        noisy_rel_positions = torch.arange(noisy_len, device=device)
        noisy_blocks = num_prefix_blocks + noisy_rel_positions // block_size
        block_ids[noisy_start:seq_len] = noisy_blocks
        token_types[noisy_start:seq_len] = 2

    q_blocks = block_ids.view(-1, 1)
    k_blocks = block_ids.view(1, -1)
    q_types = token_types.view(-1, 1)
    k_types = token_types.view(1, -1)

    base_mask = q_blocks >= k_blocks
    noisy_query_mask = q_types == 2
    noisy_visibility = (k_types == 0) | ((k_types == 1) & (k_blocks < q_blocks)) | ((k_types == 2) & (k_blocks == q_blocks))
    local_mask = torch.where(noisy_query_mask, noisy_visibility, base_mask)
    mask[:seq_len, :seq_len] = local_mask
    return mask.unsqueeze(0).unsqueeze(0)


def slice_sample_position_ids(position_ids: torch.Tensor, batch_idx: int, length: int) -> torch.Tensor:
    if position_ids.ndim == 3:
        return position_ids[:, batch_idx : batch_idx + 1, :length]
    return position_ids[batch_idx : batch_idx + 1, :length]


def build_gold_ce_batch(
    batch: dict[str, torch.Tensor],
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
    pad_token_id: int,
    mask_token_id: int | None = None,
    max_response_len: int = 0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    if mask_token_id is not None:
        return build_masked_gold_ce_batch(
            batch=batch,
            prompt_lens=prompt_lens,
            response_lens=response_lens,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            max_response_len=max_response_len,
        )

    batch_size = batch["input_ids"].shape[0]
    device = batch["input_ids"].device
    capped_response_lens = response_lens.clone()
    if max_response_len > 0:
        capped_response_lens = torch.clamp(capped_response_lens, max=max_response_len)

    max_len = int((prompt_lens + capped_response_lens).max().item()) if batch_size > 0 else 0
    packed_input_ids = torch.full(
        (batch_size, max_len),
        fill_value=pad_token_id,
        dtype=batch["input_ids"].dtype,
        device=device,
    )
    labels = torch.full((batch_size, max_len), fill_value=-100, dtype=batch["input_ids"].dtype, device=device)
    masked_indices = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    target_tokens = []

    for batch_idx in range(batch_size):
        prompt_len = int(prompt_lens[batch_idx].item())
        response_len = int(capped_response_lens[batch_idx].item())
        total_len = prompt_len + response_len
        packed_input_ids[batch_idx, :prompt_len] = batch["input_ids"][batch_idx, :prompt_len]
        if response_len > 0:
            gold_response = batch["labels"][batch_idx, prompt_len : prompt_len + response_len]
            packed_input_ids[batch_idx, prompt_len:total_len] = gold_response
            labels[batch_idx, prompt_len:total_len] = gold_response
            masked_indices[batch_idx, prompt_len:total_len] = True
            target_tokens.append(gold_response)

    result = {
        "input_ids": packed_input_ids,
        "labels": labels,
        "masked_indices": masked_indices,
    }
    if "position_ids" in batch:
        result["position_ids"] = batch["position_ids"][..., :max_len]
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if key in batch:
            result[key] = batch[key]

    return (
        result,
        torch.cat(target_tokens, dim=0) if target_tokens else packed_input_ids.new_empty((0,)),
        int(capped_response_lens.sum().item()) if capped_response_lens.numel() > 0 else 0,
    )


def build_masked_gold_ce_batch(
    batch: dict[str, torch.Tensor],
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
    pad_token_id: int,
    mask_token_id: int,
    max_response_len: int = 0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    batch_size = batch["input_ids"].shape[0]
    device = batch["input_ids"].device
    block_size = int(batch.get("block_size", 4))
    capped_response_lens = response_lens.clone()
    if max_response_len > 0:
        capped_response_lens = torch.clamp(capped_response_lens, max=max_response_len)

    max_len = int((prompt_lens + 2 * capped_response_lens).max().item()) if batch_size > 0 else 0
    packed_input_ids = torch.full(
        (batch_size, max_len),
        fill_value=pad_token_id,
        dtype=batch["input_ids"].dtype,
        device=device,
    )
    labels = torch.full((batch_size, max_len), fill_value=-100, dtype=batch["input_ids"].dtype, device=device)
    masked_indices = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    attention_masks = []
    target_tokens = []
    packed_position_ids = None

    if "position_ids" in batch:
        batch_position_ids = batch["position_ids"]
        if batch_position_ids.ndim == 3:
            packed_position_ids = torch.zeros(
                (batch_position_ids.shape[0], batch_size, max_len),
                dtype=batch_position_ids.dtype,
                device=device,
            )
        else:
            packed_position_ids = torch.zeros(
                (batch_size, max_len),
                dtype=batch_position_ids.dtype,
                device=device,
            )

    for batch_idx in range(batch_size):
        prompt_len = int(prompt_lens[batch_idx].item())
        response_len = int(capped_response_lens[batch_idx].item())
        clean_start = prompt_len
        clean_end = prompt_len + response_len
        noisy_start = clean_end
        total_len = prompt_len + 2 * response_len
        packed_input_ids[batch_idx, :prompt_len] = batch["input_ids"][batch_idx, :prompt_len]
        if response_len > 0:
            gold_response = batch["labels"][batch_idx, prompt_len : prompt_len + response_len]
            packed_input_ids[batch_idx, clean_start:clean_end] = gold_response
            packed_input_ids[batch_idx, noisy_start:total_len] = mask_token_id
            labels[batch_idx, noisy_start:total_len] = gold_response
            masked_indices[batch_idx, noisy_start:total_len] = True
            target_tokens.append(gold_response)

            if packed_position_ids is not None:
                sample_position_ids = slice_sample_position_ids(batch["position_ids"], batch_idx, prompt_len + response_len)
                prompt_position_ids = sample_position_ids[..., :prompt_len]
                clean_position_ids = sample_position_ids[..., prompt_len : prompt_len + response_len]
                if packed_position_ids.ndim == 3:
                    packed_position_ids[:, batch_idx : batch_idx + 1, :prompt_len] = prompt_position_ids
                    packed_position_ids[:, batch_idx : batch_idx + 1, clean_start:clean_end] = clean_position_ids
                    packed_position_ids[:, batch_idx : batch_idx + 1, noisy_start:total_len] = clean_position_ids
                else:
                    packed_position_ids[batch_idx : batch_idx + 1, :prompt_len] = prompt_position_ids
                    packed_position_ids[batch_idx : batch_idx + 1, clean_start:clean_end] = clean_position_ids
                    packed_position_ids[batch_idx : batch_idx + 1, noisy_start:total_len] = clean_position_ids
        elif packed_position_ids is not None and prompt_len > 0:
            sample_position_ids = slice_sample_position_ids(batch["position_ids"], batch_idx, prompt_len)
            if packed_position_ids.ndim == 3:
                packed_position_ids[:, batch_idx : batch_idx + 1, :prompt_len] = sample_position_ids
            else:
                packed_position_ids[batch_idx : batch_idx + 1, :prompt_len] = sample_position_ids

        attention_masks.append(
            create_attention_mask(
                prefix_len=prompt_len,
                response_len=response_len,
                block_size=block_size,
                max_len=max_len,
                device=device,
            )
        )

    result = {
        "input_ids": packed_input_ids,
        "labels": labels,
        "masked_indices": masked_indices,
        "attention_mask": torch.cat(attention_masks, dim=0),
        "logits_to_keep": masked_indices,
    }
    if packed_position_ids is not None:
        result["position_ids"] = packed_position_ids
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if key in batch:
            result[key] = batch[key]

    return (
        result,
        torch.cat(target_tokens, dim=0) if target_tokens else packed_input_ids.new_empty((0,)),
        int(capped_response_lens.sum().item()) if capped_response_lens.numel() > 0 else 0,
    )


def collect_gold_kd_indices(
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    batch_indices = []
    teacher_positions = []
    total_tokens = 0

    for batch_idx in range(prompt_lens.shape[0]):
        prompt_len = int(prompt_lens[batch_idx].item())
        response_len = int(response_lens[batch_idx].item())
        if response_len <= 0:
            continue

        current_positions = torch.arange(
            prompt_len - 1,
            prompt_len + response_len - 1,
            dtype=torch.long,
            device=prompt_lens.device,
        )
        valid = current_positions >= 0
        if not torch.any(valid):
            continue

        current_positions = current_positions[valid]
        batch_indices.append(
            torch.full(
                (int(current_positions.shape[0]),),
                fill_value=batch_idx,
                dtype=torch.long,
                device=prompt_lens.device,
            )
        )
        teacher_positions.append(current_positions)
        total_tokens += int(current_positions.shape[0])

    if not batch_indices:
        return None, None, 0

    return torch.cat(batch_indices, dim=0), torch.cat(teacher_positions, dim=0), total_tokens


def compute_gold_losses(
    student_model,
    gold_ce_batch: dict | None,
    gold_target_tokens: torch.Tensor | None,
    gold_teacher_reply: dict | None = None,
    topk_kd_loss: TopKKDLoss | None = None,
):
    if gold_ce_batch is None or gold_target_tokens is None or gold_target_tokens.numel() == 0:
        zero_source = gold_target_tokens
        if zero_source is None:
            first_param = next(student_model.parameters(), None)
            if first_param is not None:
                zero_source = first_param
            else:
                zero_source = torch.tensor(0.0)
        zero = zero_source.new_tensor(0.0)
        return zero, zero, 0

    if "attention_mask" not in gold_ce_batch or "logits_to_keep" not in gold_ce_batch:
        gold_ce_loss, gold_tokens = compute_gold_ce_loss(student_model, gold_ce_batch, gold_target_tokens)
        zero = gold_ce_loss.new_tensor(0.0)
        if gold_teacher_reply is not None:
            raise ValueError("Gold KD requires a masked gold CE batch exposing `attention_mask` and `logits_to_keep`.")
        return gold_ce_loss, zero, gold_tokens

    base_model = student_model
    while hasattr(base_model, "module"):
        base_model = base_model.module

    if not hasattr(base_model, "model") or not hasattr(base_model, "lm_head"):
        raise AttributeError(
            f"`compute_gold_losses` expects an unwrapped model exposing `.model` and `.lm_head`, got {type(base_model)}"
        )

    outputs = base_model.model(
        input_ids=gold_ce_batch["input_ids"],
        attention_mask=gold_ce_batch["attention_mask"],
        position_ids=gold_ce_batch.get("position_ids", None),
        pixel_values=gold_ce_batch.get("pixel_values", None),
        pixel_values_videos=gold_ce_batch.get("pixel_values_videos", None),
        image_grid_thw=gold_ce_batch.get("image_grid_thw", None),
        video_grid_thw=gold_ce_batch.get("video_grid_thw", None),
        use_cache=False,
    )
    hidden_states = outputs[0]
    logits = base_model.lm_head(hidden_states[gold_ce_batch["logits_to_keep"]].contiguous())
    target_tokens = gold_target_tokens.to(logits.device)
    gold_ce_loss = F.cross_entropy(logits.float(), target_tokens.long())

    gold_kd_loss = logits.new_tensor(0.0)
    if gold_teacher_reply is not None:
        if topk_kd_loss is None:
            raise ValueError("`topk_kd_loss` is required when `gold_teacher_reply` is provided.")
        teacher_topk_indices = gold_teacher_reply["teacher_topk_indices"].to(logits.device)
        teacher_topk_logits = gold_teacher_reply["teacher_topk_logits"].to(logits.device)
        if teacher_topk_indices.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Gold KD token mismatch: student={int(logits.shape[0])}, teacher={int(teacher_topk_indices.shape[0])}."
            )
        gold_kd_loss = topk_kd_loss(
            logits,
            teacher_topk_indices=teacher_topk_indices,
            teacher_topk_logits=teacher_topk_logits,
            num_batch_labels=logits.shape[0],
        )

    return gold_ce_loss, gold_kd_loss, int(target_tokens.numel())


def compute_gold_ce_loss(student_model, gold_ce_batch: dict | None, gold_target_tokens: torch.Tensor | None):
    if gold_ce_batch is None or gold_target_tokens is None or gold_target_tokens.numel() == 0:
        zero_source = gold_target_tokens
        if zero_source is None:
            first_param = next(student_model.parameters(), None)
            if first_param is not None:
                zero_source = first_param
            else:
                zero_source = torch.tensor(0.0)
        zero = zero_source.new_tensor(0.0)
        return zero, 0

    if "attention_mask" in gold_ce_batch and "logits_to_keep" in gold_ce_batch:
        gold_ce_loss, _, gold_tokens = compute_gold_losses(
            student_model=student_model,
            gold_ce_batch=gold_ce_batch,
            gold_target_tokens=gold_target_tokens,
        )
        return gold_ce_loss, gold_tokens

    outputs = student_model(
        input_ids=gold_ce_batch["input_ids"],
        position_ids=gold_ce_batch.get("position_ids", None),
        labels=gold_ce_batch["labels"],
        masked_indices=gold_ce_batch["masked_indices"],
        return_logits=True,
        pixel_values=gold_ce_batch.get("pixel_values", None),
        pixel_values_videos=gold_ce_batch.get("pixel_values_videos", None),
        image_grid_thw=gold_ce_batch.get("image_grid_thw", None),
        video_grid_thw=gold_ce_batch.get("video_grid_thw", None),
    )
    logits = outputs.logits
    target_tokens = gold_target_tokens.to(logits.device)
    gold_ce_loss = F.cross_entropy(logits.float(), target_tokens.long())
    return gold_ce_loss, int(target_tokens.numel())


def log_rank0_generated_text(processor, item: dict, global_step: int, debug_config: SimpleNamespace):
    if not debug_config.enabled or not debug_config.log_generated_text:
        return
    if global_step % max(debug_config.generated_text_every_steps, 1) != 0:
        return
    if "debug_prompt_input_ids" not in item or "debug_sampled_responses" not in item:
        return

    num_samples = min(
        int(debug_config.generated_text_num_samples),
        int(item["debug_sampled_responses"].shape[0]),
    )
    if num_samples <= 0:
        return

    sampled_responses = item["debug_sampled_responses"][:num_samples]
    prompt_input_ids = item["debug_prompt_input_ids"][:num_samples]
    prompt_lens = item["debug_prompt_lens"][:num_samples].tolist()
    response_lens = item.get("debug_response_lens", None)
    if response_lens is not None:
        response_lens = response_lens[:num_samples].tolist()

    trimmed_responses = []
    for idx in range(num_samples):
        current_response_len = int(sampled_responses.shape[1]) if response_lens is None else int(response_lens[idx])
        trimmed_responses.append(sampled_responses[idx, :current_response_len])
    max_response_len = max((response.shape[0] for response in trimmed_responses), default=0)
    padded_responses = torch.full(
        (num_samples, max_response_len),
        fill_value=0,
        dtype=sampled_responses.dtype,
    )
    for idx, response in enumerate(trimmed_responses):
        if response.numel() > 0:
            padded_responses[idx, : response.shape[0]] = response.cpu()

    generated_texts = _decode_sequences(
        processor,
        padded_responses,
        skip_special_tokens=debug_config.generated_text_skip_special_tokens,
    )
    if not generated_texts:
        logger.info("step %s | debug text decode skipped because processor has no batch_decode()", global_step)
        return

    prompt_texts = []
    if debug_config.log_prompt_text:
        trimmed_prompts = [prompt_input_ids[idx, : int(prompt_lens[idx])] for idx in range(num_samples)]
        max_prompt_len = max((prompt.shape[0] for prompt in trimmed_prompts), default=0)
        padded_prompts = torch.full(
            (num_samples, max_prompt_len),
            fill_value=0,
            dtype=prompt_input_ids.dtype,
        )
        for idx, prompt in enumerate(trimmed_prompts):
            if prompt.numel() > 0:
                padded_prompts[idx, : prompt.shape[0]] = prompt.cpu()
        prompt_texts = _decode_sequences(
            processor,
            padded_prompts,
            skip_special_tokens=debug_config.generated_text_skip_special_tokens,
        )

    for idx in range(num_samples):
        if prompt_texts:
            logger.info("step %s | rank0 sample %s | prompt: %s", global_step, idx, prompt_texts[idx])
        logger.info("step %s | rank0 sample %s | generated: %s", global_step, idx, generated_texts[idx])


def build_topk_overlap_debug(
    processor,
    selected_student_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    debug_config: SimpleNamespace,
    target_tokens: torch.Tensor | None = None,
    prompt_lengths: torch.Tensor | None = None,
    response_positions: torch.Tensor | None = None,
    teacher_positions: torch.Tensor | None = None,
):
    if teacher_topk_indices.numel() == 0:
        return None

    overlap_k = min(
        max(int(debug_config.topk_overlap_k), 1),
        int(teacher_topk_indices.shape[-1]),
        int(selected_student_logits.shape[-1]),
    )
    student_top_indices = torch.topk(selected_student_logits.float(), k=overlap_k, dim=-1).indices
    teacher_top_indices = teacher_topk_indices[:, :overlap_k]

    overlap_matrix = (student_top_indices.unsqueeze(-1) == teacher_top_indices.unsqueeze(-2)).any(dim=-1)
    overlap_counts = overlap_matrix.sum(dim=-1)
    overlap_ratio = overlap_counts.float() / float(overlap_k)

    diagnostics = {
        "topk_overlap_k": overlap_k,
        "topk_overlap_mean": float(overlap_ratio.mean().item()),
    }

    sample_count = min(
        max(int(debug_config.topk_overlap_num_samples), 0),
        int(student_top_indices.shape[0]),
    )
    if sample_count <= 0:
        return diagnostics

    total_positions = int(student_top_indices.shape[0])
    if sample_count >= total_positions:
        sampled_indices = torch.arange(total_positions, device=student_top_indices.device)
    else:
        sampled_indices = torch.randperm(total_positions, device=student_top_indices.device)[:sample_count]

    decode_enabled = bool(debug_config.generated_text_skip_special_tokens)
    sample_entries = []
    for sampled_idx in sampled_indices.tolist():
        current_student = student_top_indices[sampled_idx]
        current_teacher = teacher_top_indices[sampled_idx]
        current_overlap = int(overlap_counts[sampled_idx].item())

        student_decoded = _decode_token_ids(processor, current_student, skip_special_tokens=decode_enabled)
        teacher_decoded = _decode_token_ids(processor, current_teacher, skip_special_tokens=decode_enabled)

        student_items = [
            _format_token_debug(
                int(current_student[idx].item()),
                student_decoded[idx] if idx < len(student_decoded) else None,
            )
            for idx in range(current_student.shape[0])
        ]
        teacher_items = [
            _format_token_debug(
                int(current_teacher[idx].item()),
                teacher_decoded[idx] if idx < len(teacher_decoded) else None,
            )
            for idx in range(current_teacher.shape[0])
        ]
        sample_entries.append(
            {
                "sample_idx": sampled_idx,
                "overlap_count": current_overlap,
                "response_position": None
                if response_positions is None
                else int(response_positions[sampled_idx].item()),
                "prompt_length": None
                if prompt_lengths is None
                else int(prompt_lengths[sampled_idx].item()),
                "teacher_position": None
                if teacher_positions is None
                else int(teacher_positions[sampled_idx].item()),
                "target_token": None if target_tokens is None else _format_token_debug(
                    int(target_tokens[sampled_idx].item()),
                    _decode_token_ids(
                        processor,
                        target_tokens[sampled_idx : sampled_idx + 1],
                        skip_special_tokens=decode_enabled,
                    )[0]
                    if processor is not None
                    else None,
                ),
                "student_top_tokens": student_items,
                "teacher_top_tokens": teacher_items,
            }
        )

    diagnostics["samples"] = sample_entries
    return diagnostics


def log_rank0_topk_overlap(global_step: int, diagnostics: dict | None, debug_config: SimpleNamespace):
    if not debug_config.enabled or not debug_config.log_topk_overlap:
        return
    if diagnostics is None:
        return
    if global_step % max(debug_config.topk_overlap_every_steps, 1) != 0:
        return

    logger.info(
        "step %s | rank0 topk overlap@%s mean: %.4f",
        global_step,
        diagnostics["topk_overlap_k"],
        diagnostics["topk_overlap_mean"],
    )
    for sample in diagnostics.get("samples", []):
        logger.info(
            "step %s | rank0 topk sample %s | overlap=%s/%s | response_pos=%s | prompt_len=%s | teacher_pos=%s | target=%s | student=%s | teacher=%s",
            global_step,
            sample["sample_idx"],
            sample["overlap_count"],
            diagnostics["topk_overlap_k"],
            sample.get("response_position"),
            sample.get("prompt_length"),
            sample.get("teacher_position"),
            sample.get("target_token"),
            sample["student_top_tokens"],
            sample["teacher_top_tokens"],
        )


def move_prefetch_item_to_device(item: dict, device):
    if item["prompt_batch"] is not None:
        item["prompt_batch"] = to_device(item["prompt_batch"], device)
    if item["rollout_state"] is not None:
        item["rollout_state"] = to_device(item["rollout_state"], device)
    if item["gold_ce_batch"] is not None:
        item["gold_ce_batch"] = to_device(item["gold_ce_batch"], device)
    if item["target_tokens"].device != device:
        item["target_tokens"] = item["target_tokens"].to(device, non_blocking=True)
    if item["gold_target_tokens"] is not None and item["gold_target_tokens"].device != device:
        item["gold_target_tokens"] = item["gold_target_tokens"].to(device, non_blocking=True)
    if item["generation_entry_indices"] is not None and item["generation_entry_indices"].device != device:
        item["generation_entry_indices"] = item["generation_entry_indices"].to(device, non_blocking=True)
    if item["batch_indices"] is not None and item["batch_indices"].device != device:
        item["batch_indices"] = item["batch_indices"].to(device, non_blocking=True)
    if item["prompt_lengths"] is not None and item["prompt_lengths"].device != device:
        item["prompt_lengths"] = item["prompt_lengths"].to(device, non_blocking=True)
    if item["replay_keep_mask"] is not None and item["replay_keep_mask"].device != device:
        item["replay_keep_mask"] = item["replay_keep_mask"].to(device, non_blocking=True)
    if item["response_positions"] is not None and item["response_positions"].device != device:
        item["response_positions"] = item["response_positions"].to(device, non_blocking=True)
    if item["teacher_positions"] is not None and item["teacher_positions"].device != device:
        item["teacher_positions"] = item["teacher_positions"].to(device, non_blocking=True)
    return item


def compute_losses(
    student_model,
    prompt_batch,
    rollout_state,
    student_generation_logits,
    target_tokens,
    generation_entry_indices,
    batch_indices,
    prompt_lengths,
    replay_keep_mask,
    response_positions,
    teacher_positions,
    teacher_reply,
    ce_weight: float,
    kd_weight: float,
    topk_kd_loss,
    processor=None,
    debug_config: SimpleNamespace | None = None,
):
    if generation_entry_indices is None or batch_indices is None or target_tokens.numel() == 0:
        zero_source = student_generation_logits
        if zero_source is None:
            zero_source = target_tokens
        zero = zero_source.new_tensor(0.0)
        return zero, zero, zero, zero, 0, None

    if rollout_state is not None:
        replay_model = student_model
        if not hasattr(replay_model, "iter_kd_replay_logits") and hasattr(replay_model, "module"):
            replay_model = replay_model.module
        if not hasattr(replay_model, "iter_kd_replay_logits"):
            raise AttributeError("Student model does not expose `iter_kd_replay_logits()` required for rollout replay.")
        teacher_topk_indices = teacher_reply["teacher_topk_indices"].to(target_tokens.device)
        teacher_topk_logits = teacher_reply["teacher_topk_logits"].to(target_tokens.device)
        teacher_prob = F.softmax(teacher_topk_logits.float(), dim=-1)
        teacher_entropy = -(teacher_prob * torch.log(teacher_prob.clamp_min(1e-12))).sum(dim=-1).mean()
        total_tokens = int(target_tokens.numel())
        ce_sum = target_tokens.new_tensor(0.0, dtype=torch.float32)
        kd_sum = target_tokens.new_tensor(0.0, dtype=torch.float32)
        debug_student_logits = []
        token_offset = 0
        raw_token_offset = 0

        for selected_student_logits in replay_model.iter_kd_replay_logits(prompt_batch, rollout_state):
            raw_chunk_tokens = int(selected_student_logits.shape[0])
            if raw_chunk_tokens <= 0:
                continue
            if replay_keep_mask is None:
                raise ValueError("Replay mode requires `replay_keep_mask` to align filtered logits with targets.")

            keep_mask_chunk = replay_keep_mask[raw_token_offset : raw_token_offset + raw_chunk_tokens].to(
                device=selected_student_logits.device,
                dtype=torch.bool,
            )
            raw_token_offset += raw_chunk_tokens
            if not torch.any(keep_mask_chunk):
                continue

            selected_student_logits = selected_student_logits[keep_mask_chunk]
            chunk_tokens = int(selected_student_logits.shape[0])

            target_chunk = target_tokens[token_offset : token_offset + chunk_tokens].to(selected_student_logits.device)
            teacher_topk_indices_chunk = teacher_topk_indices[token_offset : token_offset + chunk_tokens].to(
                selected_student_logits.device
            )
            teacher_topk_logits_chunk = teacher_topk_logits[token_offset : token_offset + chunk_tokens].to(
                selected_student_logits.device
            )

            ce_sum = ce_sum + F.cross_entropy(selected_student_logits.float(), target_chunk.long(), reduction="sum")
            kd_sum = kd_sum + (
                topk_kd_loss(
                    selected_student_logits,
                    teacher_topk_indices=teacher_topk_indices_chunk,
                    teacher_topk_logits=teacher_topk_logits_chunk,
                    num_batch_labels=chunk_tokens,
                )
                * chunk_tokens
            )
            token_offset += chunk_tokens

            if debug_config is not None:
                debug_student_logits.append(selected_student_logits.detach())

        if token_offset != total_tokens:
            raise ValueError(
                f"Replay logits/token count mismatch: replayed={token_offset}, targets={total_tokens}."
            )
        if replay_keep_mask is not None and raw_token_offset != int(replay_keep_mask.shape[0]):
            raise ValueError(
                f"Replay raw-token count mismatch: replayed={raw_token_offset}, recorded={int(replay_keep_mask.shape[0])}."
            )

        normalizer = max(total_tokens, 1)
        ce_loss = ce_sum / normalizer
        kd_loss = kd_sum / normalizer
        total_loss = ce_weight * ce_loss + kd_weight * kd_loss
        diagnostics = None
        if debug_config is not None and debug_student_logits:
            debug_device = debug_student_logits[0].device
            diagnostics = build_topk_overlap_debug(
                processor=processor,
                selected_student_logits=torch.cat(debug_student_logits, dim=0),
                teacher_topk_indices=teacher_topk_indices.to(debug_device),
                debug_config=debug_config,
                target_tokens=target_tokens.to(debug_device),
                prompt_lengths=prompt_lengths.to(debug_device) if prompt_lengths is not None else None,
                response_positions=response_positions.to(debug_device) if response_positions is not None else None,
                teacher_positions=teacher_positions.to(debug_device) if teacher_positions is not None else None,
            )
        return total_loss, ce_loss, kd_loss, teacher_entropy, total_tokens, diagnostics

    selected_student_logits = student_generation_logits[generation_entry_indices]
    target_tokens = target_tokens.to(selected_student_logits.device)
    ce_loss = F.cross_entropy(selected_student_logits.float(), target_tokens.long())

    teacher_topk_indices = teacher_reply["teacher_topk_indices"].to(selected_student_logits.device)
    teacher_topk_logits = teacher_reply["teacher_topk_logits"].to(selected_student_logits.device)
    teacher_prob = F.softmax(teacher_topk_logits.float(), dim=-1)
    teacher_entropy = -(teacher_prob * torch.log(teacher_prob.clamp_min(1e-12))).sum(dim=-1).mean()
    kd_loss = topk_kd_loss(
        selected_student_logits,
        teacher_topk_indices=teacher_topk_indices,
        teacher_topk_logits=teacher_topk_logits,
        num_batch_labels=selected_student_logits.shape[0],
    )
    total_loss = ce_weight * ce_loss + kd_weight * kd_loss
    diagnostics = None
    if debug_config is not None:
        diagnostics = build_topk_overlap_debug(
            processor=processor,
            selected_student_logits=selected_student_logits,
            teacher_topk_indices=teacher_topk_indices,
            debug_config=debug_config,
            target_tokens=target_tokens,
            prompt_lengths=prompt_lengths,
            response_positions=response_positions,
            teacher_positions=teacher_positions,
        )
    return total_loss, ce_loss, kd_loss, teacher_entropy, int(selected_student_logits.shape[0]), diagnostics


def prepare_on_policy_prefetch_item(
    student_model,
    raw_batch,
    generation_config,
    pad_token_id: int,
    mask_token_id: int,
    mask_rate: float,
    teacher_client: AsyncTeacherClient,
    enable_gold_ce: bool = False,
    enable_gold_kd: bool = False,
    include_debug_payload: bool = False,
    drop_rollout_if_hits_max_len: bool = False,
):
    del mask_rate
    prep_start_time = time.perf_counter()
    prompt_batch, prompt_lens, response_lens = build_prompt_batch(raw_batch, pad_token_id=pad_token_id)
    original_response_lens = response_lens.clone()
    configured_max_response_len = int(getattr(generation_config, "max_response_len", 0))
    rollout_request_lens = build_rollout_request_lens(
        original_response_lens=original_response_lens,
        configured_max_response_len=configured_max_response_len,
    )
    use_rollout_replay = hasattr(student_model, "iter_kd_replay_logits")
    rollout_start_time = time.perf_counter()
    (
        sampled_responses,
        generation_payload,
        student_generation_batch_indices,
        student_generation_response_positions,
        student_generation_block_step_counts,
    ) = sample_on_policy_responses(
        student_model,
        prompt_batch=prompt_batch,
        response_lens=rollout_request_lens,
        generation_config=generation_config,
        pad_token_id=pad_token_id,
        return_generation_logits=not use_rollout_replay,
        return_rollout_state=use_rollout_replay,
    )
    rollout_time_s = time.perf_counter() - rollout_start_time
    if sampled_responses.shape[1] > 0:
        response_lens = infer_effective_response_lens(
            sampled_responses=sampled_responses,
            response_lens=torch.clamp(rollout_request_lens, max=int(sampled_responses.shape[1])),
            eos_token_id=int(getattr(generation_config, "eos_token_id", pad_token_id)),
            mask_token_id=mask_token_id,
        )
    rollout_stop_diag = analyze_rollout_stop_reasons(
        sampled_responses=sampled_responses,
        request_response_lens=rollout_request_lens,
        effective_response_lens=response_lens,
        eos_token_id=int(getattr(generation_config, "eos_token_id", pad_token_id)),
        mask_token_id=mask_token_id,
        configured_max_response_len=configured_max_response_len,
    )
    hit_max_len_mask = torch.zeros_like(response_lens, dtype=torch.bool)
    if configured_max_response_len > 0 and response_lens.numel() > 0:
        hit_max_len_mask = response_lens >= configured_max_response_len
    raw_rollout_generated_tokens = int(response_lens.sum().item()) if response_lens.numel() > 0 else 0
    if drop_rollout_if_hits_max_len and torch.any(hit_max_len_mask):
        response_lens = response_lens.masked_fill(hit_max_len_mask, 0)
    if sampled_responses.shape[1] == 0:
        generation_payload = None
        student_generation_batch_indices = torch.zeros(
            (0,),
            dtype=torch.long,
            device=sampled_responses.device,
        )
        student_generation_response_positions = torch.zeros(
            (0,),
            dtype=torch.long,
            device=sampled_responses.device,
        )
    if student_generation_batch_indices is None or student_generation_response_positions is None:
        raise ValueError("Student generation did not return unmasked-token logits required for diffusion KD.")

    generation_entry_indices, batch_indices, teacher_positions, kd_tokens = collect_generation_kd_indices(
        prompt_lens=prompt_lens,
        response_lens=response_lens,
        student_batch_indices=student_generation_batch_indices,
        student_response_positions=student_generation_response_positions,
    )
    if generation_entry_indices is None or batch_indices is None:
        target_tokens = sampled_responses.new_empty((0,))
    else:
        target_tokens = sampled_responses[
            student_generation_batch_indices[generation_entry_indices],
            student_generation_response_positions[generation_entry_indices],
        ]
    result = {
        "student_generation_logits": generation_payload if not use_rollout_replay else None,
        "prompt_batch": prompt_batch if use_rollout_replay else None,
        "rollout_state": generation_payload if use_rollout_replay else None,
        "target_tokens": target_tokens,
        "generation_entry_indices": generation_entry_indices,
        "batch_indices": batch_indices,
        "prompt_lengths": None,
        "replay_keep_mask": None,
        "response_positions": None,
        "teacher_positions": teacher_positions,
        "future": None,
        "kd_tokens": kd_tokens,
        "rollout_generated_tokens": raw_rollout_generated_tokens,
        "kd_response_tokens": int(response_lens.sum().item()) if response_lens.numel() > 0 else 0,
        "rollout_forward_steps": 0,
        "rollout_time_s": rollout_time_s,
        "prepare_time_s": time.perf_counter() - prep_start_time,
        "teacher_submit_time_s": 0.0,
        "rollout_hit_max_len_samples": int(hit_max_len_mask.sum().item()) if hit_max_len_mask.numel() > 0 else 0,
        "rollout_dropped_max_len_samples": int(hit_max_len_mask.sum().item())
        if drop_rollout_if_hits_max_len and hit_max_len_mask.numel() > 0
        else 0,
        "dataset_response_len_max": int(original_response_lens.max().item()) if original_response_lens.numel() > 0 else 0,
        "request_response_len_max": int(rollout_stop_diag["request_response_len_max"]),
        "effective_response_len_max": int(rollout_stop_diag["effective_response_len_max"]),
        "eos_stop_samples": int(rollout_stop_diag["eos_stop_samples"]),
        "mask_trunc_samples": int(rollout_stop_diag["mask_trunc_samples"]),
        "max_len_stop_samples": int(rollout_stop_diag["max_len_stop_samples"]),
        "request_cap_stop_samples": int(rollout_stop_diag["request_cap_stop_samples"]),
        "image_tokens": count_grid_tokens(raw_batch.get("image_grid_thw", None)),
        "video_tokens": count_grid_tokens(raw_batch.get("video_grid_thw", None)),
        "gold_ce_batch": None,
        "gold_target_tokens": None,
        "gold_ce_tokens": 0,
        "gold_kd_future": None,
    }
    if student_generation_block_step_counts is not None:
        result["rollout_forward_steps"] = int(student_generation_block_step_counts.sum().item())
    if use_rollout_replay:
        replay_keep_mask = torch.zeros(
            (student_generation_batch_indices.shape[0],),
            dtype=torch.bool,
            device=student_generation_batch_indices.device,
        )
        if generation_entry_indices is not None:
            replay_keep_mask[generation_entry_indices] = True
        result["replay_keep_mask"] = replay_keep_mask
    if generation_entry_indices is not None:
        result["prompt_lengths"] = prompt_lens[batch_indices]
        result["response_positions"] = student_generation_response_positions[generation_entry_indices]
    if include_debug_payload:
        result["debug_prompt_input_ids"] = prompt_batch["input_ids"].detach().cpu()
        result["debug_prompt_lens"] = prompt_lens.detach().cpu()
        result["debug_response_lens"] = response_lens.detach().cpu()
        result["debug_sampled_responses"] = sampled_responses.detach().cpu()

    capped_gold_response_lens = None
    if enable_gold_ce or enable_gold_kd:
        if not hasattr(student_model, "prepare_for_rl_training"):
            raise ValueError("Gold CE is only supported for models exposing `prepare_for_rl_training()`.")
        capped_gold_response_lens = original_response_lens.clone()
        if configured_max_response_len > 0:
            capped_gold_response_lens = torch.clamp(capped_gold_response_lens, max=configured_max_response_len)
        gold_ce_batch, gold_target_tokens, gold_ce_tokens = build_gold_ce_batch(
            batch=raw_batch,
            prompt_lens=prompt_lens,
            response_lens=original_response_lens,
            pad_token_id=pad_token_id,
            mask_token_id=int(getattr(generation_config, "mask_token_id", 151671)),
            max_response_len=configured_max_response_len,
        )
        result["gold_ce_batch"] = to_cpu(gold_ce_batch)
        result["gold_target_tokens"] = to_cpu(gold_target_tokens)
        result["gold_ce_tokens"] = gold_ce_tokens

        if enable_gold_kd:
            gold_batch_indices, gold_teacher_positions, gold_kd_tokens = collect_gold_kd_indices(
                prompt_lens=prompt_lens,
                response_lens=capped_gold_response_lens,
            )
            if gold_kd_tokens > 0 and gold_batch_indices is not None and gold_teacher_positions is not None:
                gold_teacher_submit_start_time = time.perf_counter()
                gold_teacher_batch = build_teacher_batch_from_gold_response(
                    batch=raw_batch,
                    prompt_lens=prompt_lens,
                    response_lens=capped_gold_response_lens,
                    pad_token_id=pad_token_id,
                )
                result["teacher_request_nbytes"] = result.get("teacher_request_nbytes", 0) + estimate_tensor_payload_nbytes(
                    gold_teacher_batch
                )
                result["gold_kd_future"] = teacher_client.submit(
                    {
                        "teacher_batch": gold_teacher_batch,
                        "batch_indices": gold_batch_indices,
                        "teacher_positions": gold_teacher_positions,
                    }
                )
                result["teacher_submit_time_s"] += time.perf_counter() - gold_teacher_submit_start_time

    if use_rollout_replay:
        result["prompt_batch"] = to_cpu(result["prompt_batch"])
        result["rollout_state"] = to_cpu(result["rollout_state"])
        result["target_tokens"] = to_cpu(result["target_tokens"])
        result["generation_entry_indices"] = to_cpu(result["generation_entry_indices"])
        result["batch_indices"] = to_cpu(result["batch_indices"])
        result["prompt_lengths"] = to_cpu(result["prompt_lengths"])
        result["replay_keep_mask"] = to_cpu(result["replay_keep_mask"])
        result["response_positions"] = to_cpu(result["response_positions"])
        result["teacher_positions"] = to_cpu(result["teacher_positions"])

    if kd_tokens == 0:
        result["generation_entry_indices"] = None
        result["batch_indices"] = None
        return result

    teacher_submit_start_time = time.perf_counter()
    teacher_batch = build_teacher_batch_from_on_policy_response(
        batch=raw_batch,
        prompt_lens=prompt_lens,
        response_lens=response_lens,
        sampled_responses=sampled_responses,
        pad_token_id=pad_token_id,
    )
    result["teacher_request_nbytes"] = estimate_tensor_payload_nbytes(teacher_batch)
    result["future"] = teacher_client.submit(
        {
            "teacher_batch": teacher_batch,
            "batch_indices": batch_indices,
            "teacher_positions": teacher_positions,
        }
    )
    result["teacher_submit_time_s"] = time.perf_counter() - teacher_submit_start_time
    return result


def compute_on_policy_losses(
    student_model,
    prompt_batch,
    rollout_state,
    student_generation_logits,
    target_tokens,
    generation_entry_indices,
    batch_indices,
    prompt_lengths,
    replay_keep_mask,
    response_positions,
    teacher_positions,
    teacher_reply,
    ce_weight: float,
    kd_weight: float,
    topk_kd_loss: TopKKDLoss,
    processor=None,
    debug_config: SimpleNamespace | None = None,
    backward_fn=None,
):
    if generation_entry_indices is None or batch_indices is None or target_tokens.numel() == 0:
        zero_source = student_generation_logits
        if zero_source is None:
            zero_source = target_tokens
        zero = zero_source.new_tensor(0.0)
        return zero, zero, zero, zero, 0, None

    if rollout_state is not None:
        replay_model = student_model
        if not hasattr(replay_model, "iter_kd_replay_logits") and hasattr(replay_model, "module"):
            replay_model = replay_model.module
        if not hasattr(replay_model, "iter_kd_replay_logits"):
            raise AttributeError("Student model does not expose `iter_kd_replay_logits()` required for rollout replay.")
        teacher_topk_indices = teacher_reply["teacher_topk_indices"].to(target_tokens.device)
        teacher_topk_logits = teacher_reply["teacher_topk_logits"].to(target_tokens.device)
        teacher_prob = F.softmax(teacher_topk_logits.float(), dim=-1)
        teacher_entropy = -(teacher_prob * torch.log(teacher_prob.clamp_min(1e-12))).sum(dim=-1).mean()
        total_tokens = int(target_tokens.numel())
        ce_sum = target_tokens.new_tensor(0.0, dtype=torch.float32)
        kd_sum = target_tokens.new_tensor(0.0, dtype=torch.float32)
        debug_student_logits = []
        token_offset = 0
        raw_token_offset = 0

        for selected_student_logits in replay_model.iter_kd_replay_logits(prompt_batch, rollout_state):
            raw_chunk_tokens = int(selected_student_logits.shape[0])
            if raw_chunk_tokens <= 0:
                continue
            if replay_keep_mask is None:
                raise ValueError("Replay mode requires `replay_keep_mask` to align filtered logits with targets.")

            keep_mask_chunk = replay_keep_mask[raw_token_offset : raw_token_offset + raw_chunk_tokens].to(
                device=selected_student_logits.device,
                dtype=torch.bool,
            )
            raw_token_offset += raw_chunk_tokens
            if not torch.any(keep_mask_chunk):
                continue

            selected_student_logits = selected_student_logits[keep_mask_chunk]
            chunk_tokens = int(selected_student_logits.shape[0])

            target_chunk = target_tokens[token_offset : token_offset + chunk_tokens].to(selected_student_logits.device)
            teacher_topk_indices_chunk = teacher_topk_indices[token_offset : token_offset + chunk_tokens].to(
                selected_student_logits.device
            )
            teacher_topk_logits_chunk = teacher_topk_logits[token_offset : token_offset + chunk_tokens].to(
                selected_student_logits.device
            )

            ce_chunk_sum = F.cross_entropy(selected_student_logits.float(), target_chunk.long(), reduction="sum")
            kd_chunk_sum = (
                topk_kd_loss(
                    selected_student_logits,
                    teacher_topk_indices=teacher_topk_indices_chunk,
                    teacher_topk_logits=teacher_topk_logits_chunk,
                    num_batch_labels=chunk_tokens,
                )
                * chunk_tokens
            )

            if backward_fn is None:
                ce_sum = ce_sum + ce_chunk_sum
                kd_sum = kd_sum + kd_chunk_sum
            else:
                ce_sum = ce_sum + ce_chunk_sum.detach()
                kd_sum = kd_sum + kd_chunk_sum.detach()
                if ce_weight != 0.0 or kd_weight != 0.0:
                    chunk_total_loss = (ce_weight * ce_chunk_sum + kd_weight * kd_chunk_sum) / max(total_tokens, 1)
                    backward_fn(chunk_total_loss)

            token_offset += chunk_tokens

            if debug_config is not None:
                debug_student_logits.append(selected_student_logits.detach())

        if token_offset != total_tokens:
            raise ValueError(
                f"Replay logits/token count mismatch: replayed={token_offset}, targets={total_tokens}."
            )
        if replay_keep_mask is not None and raw_token_offset != int(replay_keep_mask.shape[0]):
            raise ValueError(
                f"Replay raw-token count mismatch: replayed={raw_token_offset}, recorded={int(replay_keep_mask.shape[0])}."
            )

        normalizer = max(total_tokens, 1)
        ce_loss = ce_sum / normalizer
        kd_loss = kd_sum / normalizer
        total_loss = ce_weight * ce_loss + kd_weight * kd_loss
        if backward_fn is not None:
            total_loss = total_loss.detach()
        diagnostics = None
        if debug_config is not None and debug_student_logits:
            debug_device = debug_student_logits[0].device
            diagnostics = build_topk_overlap_debug(
                processor=processor,
                selected_student_logits=torch.cat(debug_student_logits, dim=0),
                teacher_topk_indices=teacher_topk_indices.to(debug_device),
                debug_config=debug_config,
                target_tokens=target_tokens.to(debug_device),
                prompt_lengths=prompt_lengths.to(debug_device) if prompt_lengths is not None else None,
                response_positions=response_positions.to(debug_device) if response_positions is not None else None,
                teacher_positions=teacher_positions.to(debug_device) if teacher_positions is not None else None,
            )
        return total_loss, ce_loss, kd_loss, teacher_entropy, total_tokens, diagnostics

    selected_student_logits = student_generation_logits[generation_entry_indices]
    target_tokens = target_tokens.to(selected_student_logits.device)
    ce_loss = F.cross_entropy(selected_student_logits.float(), target_tokens.long())

    teacher_topk_indices = teacher_reply["teacher_topk_indices"].to(selected_student_logits.device)
    teacher_topk_logits = teacher_reply["teacher_topk_logits"].to(selected_student_logits.device)
    teacher_prob = F.softmax(teacher_topk_logits.float(), dim=-1)
    teacher_entropy = -(teacher_prob * torch.log(teacher_prob.clamp_min(1e-12))).sum(dim=-1).mean()
    kd_loss = topk_kd_loss(
        selected_student_logits,
        teacher_topk_indices=teacher_topk_indices,
        teacher_topk_logits=teacher_topk_logits,
        num_batch_labels=selected_student_logits.shape[0],
    )
    total_loss = ce_weight * ce_loss + kd_weight * kd_loss
    diagnostics = None
    if debug_config is not None:
        diagnostics = build_topk_overlap_debug(
            processor=processor,
            selected_student_logits=selected_student_logits,
            teacher_topk_indices=teacher_topk_indices,
            debug_config=debug_config,
            target_tokens=target_tokens,
            prompt_lengths=prompt_lengths,
            response_positions=response_positions,
            teacher_positions=teacher_positions,
        )
    return total_loss, ce_loss, kd_loss, teacher_entropy, int(selected_student_logits.shape[0]), diagnostics


def train():
    config = get_config()
    project_dir = str(Path(config.experiment.project) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config.training.gradient_accumulation_steps),
        mixed_precision=config.training.mixed_precision,
        log_with=config.experiment.get("log_with", None),
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

    if config.experiment.get("log_with", None):
        accelerator.init_trackers(
            config.experiment.project,
            config={k: v for k, v in flatten_omega_conf(config, resolve=True)},
        )

    dataloader, student, processor = build_components(config)
    debug_config = build_debug_config(config)
    stability_config = build_stability_config(config)
    if accelerator.is_main_process:
        log_student_model_info(student)

    base_lr = float(config.optimizer.lr)
    optimizer = AdamW(
        student.parameters(),
        lr=base_lr,
        betas=tuple(config.optimizer.get("betas", [0.9, 0.95])),
        weight_decay=float(config.optimizer.get("weight_decay", 0.0)),
    )
    student, optimizer, dataloader = accelerator.prepare(student, optimizer, dataloader)
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

    generation_config = build_generation_config(config)
    topk_kd_loss = TopKKDLoss(
        temperature=float(config.distillation.get("temperature", 1.0)),
        fp32_upcast=bool(config.distillation.get("fp32_upcast", True)),
        direction=str(config.distillation.get("kl_direction", "forward-kl")),
    )
    teacher_client = AsyncTeacherClient(
        host=str(config.teacher_server.host),
        port=int(config.teacher_server.port),
        timeout_s=float(config.teacher_server.get("timeout_s", 600.0)),
        max_pending=int(config.teacher_server.get("prefetch_depth", 2)),
        compress_fp32_to_bf16=bool(config.teacher_server.get("compress_fp32_to_bf16", True)),
    )

    pad_token_id = int(config.on_policy.pad_token_id)
    mask_token_id = int(config.on_policy.mask_token_id)
    mask_rate = float(config.on_policy.mask_rate)
    ce_weight = float(config.loss.ce_weight)
    kd_weight = float(config.loss.kd_weight)
    enable_gold_ce = bool(config.loss.get("enable_gold_ce", False))
    gold_ce_weight = float(config.loss.get("gold_ce_weight", 1.0))
    enable_gold_kd = bool(config.loss.get("enable_gold_kd", False))
    gold_kd_weight = float(config.loss.get("gold_kd_weight", 0.0))

    max_micro_steps = int(training_schedule.total_micro_steps)
    requested_prefetch_depth = max(int(config.teacher_server.get("prefetch_depth", 2)), 1)
    supports_rollout_replay = hasattr(accelerator.unwrap_model(student), "iter_kd_replay_logits")
    prefetch_depth = requested_prefetch_depth if supports_rollout_replay else 1
    if accelerator.is_main_process and requested_prefetch_depth != prefetch_depth:
        logger.warning(
            "teacher_server.prefetch_depth=%d is overridden to %d because this model still keeps generation-time KD logits alive.",
            requested_prefetch_depth,
            prefetch_depth,
        )
    global_step = 0
    optimizer_step = 0
    pending = deque()

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

            data_iter = iter(dataloader)
            exhausted = False

            while global_step < max_micro_steps:
                while len(pending) < prefetch_depth and not exhausted:
                    try:
                        raw_batch = next(data_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    raw_batch = to_device(raw_batch, accelerator.device)
                    with torch.no_grad():
                        pending.append(
                            prepare_on_policy_prefetch_item(
                                student_model=accelerator.unwrap_model(student),
                                raw_batch=raw_batch,
                                generation_config=generation_config,
                                pad_token_id=pad_token_id,
                                mask_token_id=mask_token_id,
                                mask_rate=mask_rate,
                                teacher_client=teacher_client,
                                enable_gold_ce=enable_gold_ce,
                                enable_gold_kd=enable_gold_kd and gold_kd_weight > 0.0,
                                include_debug_payload=accelerator.is_main_process and debug_config.enabled,
                                drop_rollout_if_hits_max_len=stability_config.drop_rollout_if_hits_max_len,
                            )
                        )

                if not pending:
                    break

                step_start_time = time.perf_counter()
                if accelerator.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(accelerator.device)
                item = pending.popleft()
                if accelerator.is_main_process:
                    log_rank0_generated_text(processor, item, global_step, debug_config)
                    if stability_config.warn_on_max_len_hit and int(item["rollout_hit_max_len_samples"]) > 0:
                        logger.warning(
                            "step %s | rank0 rollout hit max_response_len on %s sample(s) | raw_rollout_tokens=%s | kd_response_tokens=%s | dropped=%s",
                            global_step,
                            int(item["rollout_hit_max_len_samples"]),
                            int(item["rollout_generated_tokens"]),
                            int(item["kd_response_tokens"]),
                            int(item["rollout_dropped_max_len_samples"]),
                        )
                    if stability_config.log_timing and global_step % max(stability_config.log_timing_every_steps, 1) == 0:
                        logger.info(
                            "step %s | rank0 teacher payload %.2f MiB | kd_tokens=%s | raw_rollout_tokens=%s | kd_response_tokens=%s",
                            global_step,
                            float(item.get("teacher_request_nbytes", 0)) / (1024.0 * 1024.0),
                            int(item["kd_tokens"]),
                            int(item["rollout_generated_tokens"]),
                            int(item["kd_response_tokens"]),
                        )
                        logger.info(
                            "step %s | rank0 rollout stop diag | dataset_resp_max=%s | request_resp_max=%s | effective_resp_max=%s | eos_stop=%s | mask_trunc=%s | request_cap=%s | max_len_stop=%s | configured_max=%s",
                            global_step,
                            int(item.get("dataset_response_len_max", 0)),
                            int(item.get("request_response_len_max", 0)),
                            int(item.get("effective_response_len_max", 0)),
                            int(item.get("eos_stop_samples", 0)),
                            int(item.get("mask_trunc_samples", 0)),
                            int(item.get("request_cap_stop_samples", 0)),
                            int(item.get("max_len_stop_samples", 0)),
                            int(getattr(generation_config, "max_response_len", 0)),
                        )
                teacher_wait_start_time = time.perf_counter()
                teacher_reply = None if item["future"] is None else item["future"].result()
                gold_teacher_reply = None if item["gold_kd_future"] is None else item["gold_kd_future"].result()
                teacher_wait_time_s = time.perf_counter() - teacher_wait_start_time
                device_move_start_time = time.perf_counter()
                item = move_prefetch_item_to_device(item, accelerator.device)
                device_move_time_s = time.perf_counter() - device_move_start_time
                loss_backward_start_time = time.perf_counter()
                with accelerator.accumulate(student):
                    total_loss, ce_loss, kd_loss, teacher_entropy_topk, kd_tokens, topk_overlap_diag = compute_on_policy_losses(
                        student_model=student,
                        prompt_batch=item["prompt_batch"],
                        rollout_state=item["rollout_state"],
                        student_generation_logits=item["student_generation_logits"],
                        target_tokens=item["target_tokens"],
                        generation_entry_indices=item["generation_entry_indices"],
                        batch_indices=item["batch_indices"],
                        prompt_lengths=item["prompt_lengths"],
                        replay_keep_mask=item["replay_keep_mask"],
                        response_positions=item["response_positions"],
                        teacher_positions=item["teacher_positions"],
                        teacher_reply=teacher_reply,
                        ce_weight=ce_weight,
                        kd_weight=kd_weight,
                        topk_kd_loss=topk_kd_loss,
                        processor=processor if accelerator.is_main_process else None,
                        debug_config=debug_config,
                        backward_fn=accelerator.backward if item["rollout_state"] is not None else None,
                    )
                    gold_ce_loss, gold_kd_loss, gold_ce_tokens = compute_gold_losses(
                        student_model=student,
                        gold_ce_batch=item["gold_ce_batch"] if (enable_gold_ce or enable_gold_kd) else None,
                        gold_target_tokens=item["gold_target_tokens"] if (enable_gold_ce or enable_gold_kd) else None,
                        gold_teacher_reply=gold_teacher_reply if enable_gold_kd and gold_kd_weight > 0.0 else None,
                        topk_kd_loss=topk_kd_loss if enable_gold_kd and gold_kd_weight > 0.0 else None,
                    )
                    if enable_gold_ce:
                        total_loss = total_loss + gold_ce_weight * gold_ce_loss
                    if enable_gold_kd and gold_kd_weight > 0.0:
                        total_loss = total_loss + gold_kd_weight * gold_kd_loss
                    if total_loss.requires_grad:
                        accelerator.backward(total_loss)

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
                        optimizer.step()
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
                        gold_ce_loss.detach(),
                        gold_kd_loss.detach(),
                        teacher_entropy_topk.detach(),
                        float(kd_tokens),
                        float(gold_ce_tokens),
                        float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                        float(current_lr),
                        float(topk_overlap_diag["topk_overlap_mean"]) if topk_overlap_diag is not None else 0.0,
                        float(item["rollout_generated_tokens"]),
                        float(item["rollout_forward_steps"]),
                        float(item["kd_response_tokens"]),
                        float(item["prepare_time_s"]),
                        float(item["rollout_time_s"]),
                        float(item.get("teacher_submit_time_s", 0.0)),
                        float(teacher_wait_time_s),
                        float(device_move_time_s),
                        float(loss_backward_time_s),
                        float(item["rollout_hit_max_len_samples"]),
                        float(item["rollout_dropped_max_len_samples"]),
                        float(item.get("teacher_request_nbytes", 0)),
                        float(item.get("image_tokens", 0)),
                        float(item.get("video_tokens", 0)),
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
                gathered = accelerator.gather(metrics.unsqueeze(0))
                step_time_s = time.perf_counter() - step_start_time
                mean_metrics = gathered.mean(dim=0)
                max_metrics = gathered.max(dim=0).values
                rollout_tokens_sum = float(gathered[:, 11].sum().item())
                rollout_forwards_sum = float(gathered[:, 12].sum().item())
                rollout_tpf = rollout_tokens_sum / rollout_forwards_sum if rollout_forwards_sum > 0.0 else 0.0

                if accelerator.is_main_process:
                    log_rank0_topk_overlap(global_step, topk_overlap_diag, debug_config)
                    timing_suffix = ""
                    if stability_config.log_timing and global_step % max(stability_config.log_timing_every_steps, 1) == 0:
                        timing_parts = [
                            "prep_s mean/max %.4f/%.4f" % (mean_metrics[14].item(), max_metrics[14].item()),
                            "rollout_s mean/max %.4f/%.4f" % (mean_metrics[15].item(), max_metrics[15].item()),
                            "teacher_wait_s mean/max %.4f/%.4f" % (mean_metrics[17].item(), max_metrics[17].item()),
                            "loss_bw_s mean/max %.4f/%.4f" % (mean_metrics[19].item(), max_metrics[19].item()),
                            "step_s %.4f" % step_time_s,
                            "teacher_payload_mb %.2f" % (mean_metrics[22].item() / (1024.0 * 1024.0)),
                            "image_tokens %.0f" % mean_metrics[23].item(),
                            "cuda_alloc_gb mean/max %.2f/%.2f" % (mean_metrics[25].item(), max_metrics[25].item()),
                            "cuda_reserved_gb mean/max %.2f/%.2f" % (mean_metrics[26].item(), max_metrics[26].item()),
                        ]
                        raw_rollout_tokens = mean_metrics[11].item()
                        kd_response_tokens = mean_metrics[13].item()
                        if abs(raw_rollout_tokens - kd_response_tokens) > 0.5:
                            timing_parts.append("raw_rollout_tokens %.0f" % raw_rollout_tokens)
                            timing_parts.append("kd_response_tokens %.0f" % kd_response_tokens)
                        max_len_hits = mean_metrics[20].item()
                        dropped_max_len = mean_metrics[21].item()
                        if max_len_hits > 0.0:
                            timing_parts.append("max_len_hits %.0f" % max_len_hits)
                        if dropped_max_len > 0.0:
                            timing_parts.append("dropped_max_len %.0f" % dropped_max_len)
                        video_tokens = mean_metrics[24].item()
                        if video_tokens > 0.0:
                            timing_parts.append("video_tokens %.0f" % video_tokens)
                        timing_suffix = " | " + " | ".join(timing_parts)

                    log_parts = [
                        f"micro_step {global_step}",
                        f"optimizer_step {optimizer_step}",
                        f"loss {mean_metrics[0].item():.6f}",
                        f"ce_loss {mean_metrics[1].item():.6f}",
                        f"kd_loss {mean_metrics[2].item():.6f}",
                    ]
                    if bool(config.loss.get("enable_gold_ce", False)) or mean_metrics[3].item() > 0.0 or mean_metrics[7].item() > 0.0:
                        log_parts.append(f"gold_ce_loss {mean_metrics[3].item():.6f}")
                    if bool(config.loss.get("enable_gold_kd", False)) or mean_metrics[4].item() > 0.0:
                        log_parts.append(f"gold_kd_loss {mean_metrics[4].item():.6f}")
                    log_parts.extend(
                        [
                            f"teacher_entropy_topk {mean_metrics[5].item():.6f}",
                            f"kd_tokens {mean_metrics[6].item():.0f}",
                        ]
                    )
                    if bool(config.loss.get("enable_gold_ce", False)) or mean_metrics[7].item() > 0.0:
                        log_parts.append(f"gold_ce_tokens {mean_metrics[7].item():.0f}")
                    log_parts.extend(
                        [
                            f"grad_norm {mean_metrics[8].item():.6f}",
                            f"lr {mean_metrics[9].item():.8f}",
                            f"rollout_tpf {rollout_tpf:.4f}",
                        ]
                    )
                    if debug_config.enabled and debug_config.log_topk_overlap:
                        log_parts.append(f"topk_overlap@{debug_config.topk_overlap_k} {mean_metrics[10].item():.4f}")
                    if timing_suffix:
                        log_parts.append(timing_suffix[3:])
                    logger.info(" | ".join(log_parts))

                if config.experiment.get("log_with", None):
                    tracker_metrics = {
                        "train/micro_step": float(global_step),
                        "train/optimizer_step": float(optimizer_step),
                        "train/loss": mean_metrics[0].item(),
                        "train/ce_loss": mean_metrics[1].item(),
                        "train/kd_loss": mean_metrics[2].item(),
                        "train/gold_ce_loss": mean_metrics[3].item(),
                        "train/gold_kd_loss": mean_metrics[4].item(),
                        "train/teacher_entropy_topk": mean_metrics[5].item(),
                        "train/kd_tokens": mean_metrics[6].item(),
                        "train/gold_ce_tokens": mean_metrics[7].item(),
                        "train/grad_norm": mean_metrics[8].item(),
                        "train/lr": mean_metrics[9].item(),
                        "train/rollout_tpf": rollout_tpf,
                        "train/kd_response_tokens": mean_metrics[13].item(),
                        "train/raw_rollout_tokens": mean_metrics[11].item(),
                        "train/rollout_prep_s_mean": mean_metrics[14].item(),
                        "train/rollout_s_mean": mean_metrics[15].item(),
                        "train/teacher_submit_s_mean": mean_metrics[16].item(),
                        "train/teacher_wait_s_mean": mean_metrics[17].item(),
                        "train/device_move_s_mean": mean_metrics[18].item(),
                        "train/loss_backward_s_mean": mean_metrics[19].item(),
                        "train/max_len_hit_samples": mean_metrics[20].item(),
                        "train/dropped_max_len_samples": mean_metrics[21].item(),
                        "train/teacher_payload_mb": mean_metrics[22].item() / (1024.0 * 1024.0),
                        "train/image_tokens": mean_metrics[23].item(),
                        "train/video_tokens": mean_metrics[24].item(),
                        "train/cuda_alloc_gb_mean": mean_metrics[25].item(),
                        "train/cuda_reserved_gb_mean": mean_metrics[26].item(),
                    }
                    if debug_config.enabled and debug_config.log_topk_overlap:
                        tracker_metrics[f"train/topk_overlap_at_{debug_config.topk_overlap_k}"] = mean_metrics[10].item()
                    accelerator.log(tracker_metrics, step=global_step)

                global_step += 1
                if int(config.training.get("save_every", 0)) > 0:
                    if accelerator.sync_gradients and optimizer_step % int(config.training.save_every) == 0:
                        save_checkpoint(accelerator, student, processor, config.experiment.project, optimizer_step)

        save_checkpoint(accelerator, student, processor, config.experiment.project, optimizer_step)
    finally:
        teacher_client.close()
        if config.experiment.get("log_with", None):
            accelerator.end_training()


if __name__ == "__main__":
    train()
