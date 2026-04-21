import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TRAIN_DIR)
if TRAIN_DIR in sys.path:
    sys.path.remove(TRAIN_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from omegaconf import OmegaConf
from torch.optim import AdamW

from nemo_automodel.components.distillation.trajectory_store import DiskTrajectoryStore, VersionedModelRegistry
from nemo_automodel.components.loss.topk_kd_loss import TopKKDLoss
from train.kd_lib import (
    build_debug_config,
    build_lr_schedule,
    compute_gold_ce_loss,
    compute_losses,
    get_lr_for_optimizer_step,
    instantiate_config,
    log_rank0_topk_overlap,
    log_student_model_info,
    move_prefetch_item_to_device,
    save_checkpoint,
    set_optimizer_lr,
)
from train.utils import flatten_omega_conf, get_config

logger = get_logger(__name__, log_level="INFO")


def build_actor_learner_config(config) -> SimpleNamespace:
    actor_cfg = config.get("actor_learner", None)
    root_dir = Path(config.experiment.project) / "actor_learner"
    publish_every_optimizer_steps = (
        int(actor_cfg.get("publish_every_optimizer_steps", actor_cfg.get("publish_every_steps", 0)))
        if actor_cfg is not None
        else 0
    )
    return SimpleNamespace(
        spool_dir=str(Path(actor_cfg.get("spool_dir", root_dir / "spool"))) if actor_cfg is not None else str(root_dir / "spool"),
        model_registry_dir=str(
            Path(actor_cfg.get("model_registry_dir", root_dir / "models")) if actor_cfg is not None else root_dir / "models"
        ),
        expected_actors=int(actor_cfg.get("expected_actors", 1)) if actor_cfg is not None else 1,
        claim_timeout_s=float(actor_cfg.get("claim_timeout_s", 30.0)) if actor_cfg is not None else 30.0,
        poll_interval_s=float(actor_cfg.get("poll_interval_s", 1.0)) if actor_cfg is not None else 1.0,
        publish_every_optimizer_steps=publish_every_optimizer_steps,
        publish_initial_model=bool(actor_cfg.get("publish_initial_model", True)) if actor_cfg is not None else True,
        model_version=str(actor_cfg.get("model_version", "bootstrap")) if actor_cfg is not None else "bootstrap",
        strict_version_match=bool(actor_cfg.get("strict_version_match", False)) if actor_cfg is not None else False,
        drain_threshold=int(actor_cfg.get("drain_threshold", 0)) if actor_cfg is not None else 0,
    )


def build_student_and_processor(config):
    from transformers import AutoProcessor

    processor = instantiate_config(config.get("processor", None))
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            config.model.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
    student = instantiate_config(config.model)
    return student, processor

def claim_paths_for_version(
    accelerator: Accelerator,
    store: DiskTrajectoryStore,
    cfg: SimpleNamespace,
    version: str | None,
):
    claimed_paths = None
    world_size = int(accelerator.num_processes)
    if accelerator.is_main_process:
        claimed_paths = []
        while len(claimed_paths) < world_size:
            claimed_path, _ = store.claim_next(
                timeout_s=cfg.claim_timeout_s,
                poll_interval_s=cfg.poll_interval_s,
                consumer_id=f"learner-main-{os.getpid()}",
                version=version,
            )
            if claimed_path is not None:
                claimed_paths.append(str(claimed_path))
                continue
            target_ready = store.count_ready(version)
            if store.all_expected_producers_done(cfg.expected_actors) and target_ready == 0:
                for partial_path in claimed_paths:
                    store.requeue(partial_path)
                claimed_paths = None
                break
        if claimed_paths is not None and len(claimed_paths) != world_size:
            raise RuntimeError(f"Expected {world_size} claimed paths, got {len(claimed_paths)}.")

    if accelerator.num_processes > 1:
        object_list = [claimed_paths]
        torch.distributed.broadcast_object_list(object_list, src=0)
        claimed_paths = object_list[0]
    return claimed_paths


def load_record_for_rank(claimed_paths: list[str]) -> tuple[str, dict]:
    local_rank = int(os.environ.get("RANK", "0"))
    claimed_path = claimed_paths[local_rank]
    record = torch.load(claimed_path, map_location="cpu")
    return claimed_path, record


def publish_model_version(accelerator: Accelerator, registry: VersionedModelRegistry, model, version: str, step: int):
    if not accelerator.is_main_process:
        return None
    state_dict = accelerator.unwrap_model(model).state_dict()
    payload = registry.publish_state_dict(
        state_dict=state_dict,
        version=version,
        step=step,
        metadata={"published_by": "learner"},
    )
    logger.info("learner publish | version=%s | step=%s", payload["version"], payload["step"])
    return payload


def select_claim_version(
    accelerator: Accelerator,
    store: DiskTrajectoryStore,
    cfg: SimpleNamespace,
    train_version: str,
    published_version: str,
) -> str | None:
    claim_version = None
    if accelerator.is_main_process:
        world_size = int(accelerator.num_processes)
        version_counts = {version: store.count_ready(version) for version in store.list_ready_versions()}
        if cfg.strict_version_match:
            claim_version = train_version
        else:
            preferred_versions = []
            for version in (train_version, published_version):
                if version not in preferred_versions:
                    preferred_versions.append(version)
            for version in preferred_versions:
                if version_counts.get(version, 0) >= world_size:
                    claim_version = version
                    break
            if claim_version is None and version_counts:
                claim_version = max(
                    version_counts.items(),
                    key=lambda item: (item[1], item[0]),
                )[0]

    if accelerator.num_processes > 1:
        object_list = [claim_version]
        torch.distributed.broadcast_object_list(object_list, src=0)
        claim_version = object_list[0]
    return claim_version


def decide_version_acceptance(
    accelerator: Accelerator,
    store: DiskTrajectoryStore,
    cfg: SimpleNamespace,
    train_version: str,
    published_version: str,
    trajectory_version: str,
) -> tuple[str, bool]:
    decision = {"train_version": str(train_version), "accept": True}
    if accelerator.is_main_process:
        if not cfg.strict_version_match or trajectory_version == train_version:
            decision = {"train_version": str(train_version), "accept": True}
        else:
            old_ready_count = store.count_ready(train_version)
            new_ready_count = store.count_ready(published_version)
            if (
                old_ready_count <= cfg.drain_threshold
                and new_ready_count >= int(accelerator.num_processes)
                and trajectory_version == published_version
            ):
                logger.info(
                    "learner version switch | old=%s | new=%s | old_ready=%s | new_ready=%s | drain_threshold=%s",
                    train_version,
                    trajectory_version,
                    old_ready_count,
                    new_ready_count,
                    cfg.drain_threshold,
                )
                decision = {"train_version": str(trajectory_version), "accept": True}
            else:
                decision = {"train_version": str(train_version), "accept": False}

    if accelerator.num_processes > 1:
        object_list = [decision]
        torch.distributed.broadcast_object_list(object_list, src=0)
        decision = object_list[0]
    return str(decision["train_version"]), bool(decision["accept"])


def main():
    config = get_config()
    actor_learner_config = build_actor_learner_config(config)
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
        OmegaConf.save(config, Path(config.experiment.project) / "learner_config.yaml")

    if config.experiment.get("log_with", None):
        accelerator.init_trackers(
            config.experiment.project,
            config={k: v for k, v in flatten_omega_conf(config, resolve=True)},
        )

    student, processor = build_student_and_processor(config)
    debug_config = build_debug_config(config)
    if accelerator.is_main_process:
        log_student_model_info(student)

    base_lr = float(config.optimizer.lr)
    optimizer = AdamW(
        student.parameters(),
        lr=base_lr,
        betas=tuple(config.optimizer.get("betas", [0.9, 0.95])),
        weight_decay=float(config.optimizer.get("weight_decay", 0.0)),
    )
    lr_schedule = build_lr_schedule(config)
    set_optimizer_lr(optimizer, get_lr_for_optimizer_step(lr_schedule, 0))
    student, optimizer = accelerator.prepare(student, optimizer)
    student.train()
    if accelerator.is_main_process:
        logger.info(
            "lr schedule | base_lr=%.8f | warmup_start_ratio=%.4f | warmup_steps=%d | total_optimizer_steps=%d | min_lr_ratio=%.4f",
            lr_schedule.base_lr,
            lr_schedule.warmup_start_ratio,
            lr_schedule.warmup_steps,
            lr_schedule.total_optimizer_steps,
            lr_schedule.min_lr_ratio,
        )

    topk_kd_loss = TopKKDLoss(
        temperature=float(config.distillation.get("temperature", 1.0)),
        fp32_upcast=bool(config.distillation.get("fp32_upcast", True)),
        direction=str(config.distillation.get("kl_direction", "forward-kl")),
    )
    store = DiskTrajectoryStore(actor_learner_config.spool_dir)
    registry = VersionedModelRegistry(actor_learner_config.model_registry_dir)

    ce_weight = float(config.loss.ce_weight)
    kd_weight = float(config.loss.kd_weight)
    enable_gold_ce = bool(config.loss.get("enable_gold_ce", False))
    gold_ce_weight = float(config.loss.get("gold_ce_weight", 1.0))
    max_steps = int(config.training.max_steps)

    train_version = actor_learner_config.model_version
    published_version = actor_learner_config.model_version
    if actor_learner_config.publish_initial_model:
        publish_model_version(
            accelerator=accelerator,
            registry=registry,
            model=student,
            version=published_version,
            step=0,
        )
    accelerator.wait_for_everyone()

    global_step = 0
    optimizer_step = 0

    try:
        while global_step < max_steps:
            claim_version = select_claim_version(
                accelerator=accelerator,
                store=store,
                cfg=actor_learner_config,
                train_version=train_version,
                published_version=published_version,
            )
            claimed_paths = claim_paths_for_version(
                accelerator=accelerator,
                store=store,
                cfg=actor_learner_config,
                version=claim_version,
            )
            if claimed_paths is None:
                break

            claimed_path, record = load_record_for_rank(claimed_paths)
            trajectory_version = str(record.get("model_version", "unknown"))
            train_version, accept_record = decide_version_acceptance(
                accelerator=accelerator,
                store=store,
                cfg=actor_learner_config,
                train_version=train_version,
                published_version=published_version,
                trajectory_version=trajectory_version,
            )
            if not accept_record:
                if accelerator.is_main_process:
                    for path in claimed_paths:
                        store.requeue(path)
                time.sleep(actor_learner_config.poll_interval_s)
                continue

            item = record["item"]
            teacher_reply = record["teacher_reply"]
            item = move_prefetch_item_to_device(item, accelerator.device)

            with accelerator.accumulate(student):
                total_loss, ce_loss, kd_loss, teacher_entropy_topk, kd_tokens, topk_overlap_diag = compute_losses(
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
                )
                gold_ce_loss, gold_ce_tokens = compute_gold_ce_loss(
                    student_model=student,
                    gold_ce_batch=item["gold_ce_batch"] if enable_gold_ce else None,
                    gold_target_tokens=item["gold_target_tokens"] if enable_gold_ce else None,
                )
                total_loss = total_loss + gold_ce_weight * gold_ce_loss
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    optimizer_step += 1
                    current_lr = get_lr_for_optimizer_step(lr_schedule, optimizer_step)
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

            metrics = torch.tensor(
                [
                    total_loss.detach(),
                    ce_loss.detach(),
                    kd_loss.detach(),
                    gold_ce_loss.detach(),
                    teacher_entropy_topk.detach(),
                    float(kd_tokens),
                    float(gold_ce_tokens),
                    float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                    float(current_lr),
                    float(topk_overlap_diag["topk_overlap_mean"]) if topk_overlap_diag is not None else 0.0,
                    float(item["rollout_generated_tokens"]),
                    float(item["rollout_forward_steps"]),
                ],
                device=accelerator.device,
                dtype=torch.float32,
            )
            gathered = accelerator.gather(metrics.unsqueeze(0))
            mean_metrics = gathered.mean(dim=0)
            rollout_tokens_sum = float(gathered[:, 10].sum().item())
            rollout_forwards_sum = float(gathered[:, 11].sum().item())
            rollout_tpf = rollout_tokens_sum / rollout_forwards_sum if rollout_forwards_sum > 0.0 else 0.0

            if accelerator.is_main_process:
                for path in claimed_paths:
                    store.ack(path)
                log_rank0_topk_overlap(global_step, topk_overlap_diag, debug_config)
                if debug_config.enabled and debug_config.log_topk_overlap:
                    logger.info(
                        "learner step %s | claim_version=%s | traj_version=%s | train_version=%s | published_version=%s | loss %.6f | ce_loss %.6f | kd_loss %.6f | gold_ce_loss %.6f | teacher_entropy_topk %.6f | kd_tokens %.0f | gold_ce_tokens %.0f | grad_norm %.6f | lr %.8f | rollout_tpf %.4f | topk_overlap@%s %.4f",
                        global_step,
                        claim_version,
                        trajectory_version,
                        train_version,
                        published_version,
                        mean_metrics[0].item(),
                        mean_metrics[1].item(),
                        mean_metrics[2].item(),
                        mean_metrics[3].item(),
                        mean_metrics[4].item(),
                        mean_metrics[5].item(),
                        mean_metrics[6].item(),
                        mean_metrics[7].item(),
                        mean_metrics[8].item(),
                        rollout_tpf,
                        debug_config.topk_overlap_k,
                        mean_metrics[9].item(),
                    )
                else:
                    logger.info(
                        "learner step %s | claim_version=%s | traj_version=%s | train_version=%s | published_version=%s | loss %.6f | ce_loss %.6f | kd_loss %.6f | gold_ce_loss %.6f | teacher_entropy_topk %.6f | kd_tokens %.0f | gold_ce_tokens %.0f | grad_norm %.6f | lr %.8f | rollout_tpf %.4f",
                        global_step,
                        claim_version,
                        trajectory_version,
                        train_version,
                        published_version,
                        mean_metrics[0].item(),
                        mean_metrics[1].item(),
                        mean_metrics[2].item(),
                        mean_metrics[3].item(),
                        mean_metrics[4].item(),
                        mean_metrics[5].item(),
                        mean_metrics[6].item(),
                        mean_metrics[7].item(),
                        mean_metrics[8].item(),
                        rollout_tpf,
                    )

            if config.experiment.get("log_with", None):
                tracker_metrics = {
                    "train/loss": mean_metrics[0].item(),
                    "train/ce_loss": mean_metrics[1].item(),
                    "train/kd_loss": mean_metrics[2].item(),
                    "train/gold_ce_loss": mean_metrics[3].item(),
                    "train/teacher_entropy_topk": mean_metrics[4].item(),
                    "train/kd_tokens": mean_metrics[5].item(),
                    "train/gold_ce_tokens": mean_metrics[6].item(),
                    "train/grad_norm": mean_metrics[7].item(),
                    "train/lr": mean_metrics[8].item(),
                    "train/rollout_tpf": rollout_tpf,
                }
                if debug_config.enabled and debug_config.log_topk_overlap:
                    tracker_metrics[f"train/topk_overlap_at_{debug_config.topk_overlap_k}"] = mean_metrics[9].item()
                accelerator.log(tracker_metrics, step=global_step)

            global_step += 1
            if accelerator.sync_gradients and int(config.training.get("save_every", 0)) > 0:
                if global_step % int(config.training.save_every) == 0:
                    save_checkpoint(accelerator, student, processor, config.experiment.project, global_step)
            if accelerator.sync_gradients and actor_learner_config.publish_every_optimizer_steps > 0:
                if optimizer_step % actor_learner_config.publish_every_optimizer_steps == 0:
                    published_version = f"optstep-{optimizer_step}"
                    publish_model_version(
                        accelerator=accelerator,
                        registry=registry,
                        model=student,
                        version=published_version,
                        step=optimizer_step,
                    )
                    accelerator.wait_for_everyone()

        save_checkpoint(accelerator, student, processor, config.experiment.project, global_step)
    finally:
        if config.experiment.get("log_with", None):
            accelerator.end_training()


if __name__ == "__main__":
    main()
