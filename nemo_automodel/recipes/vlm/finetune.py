# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import pathlib
import time
from omegaconf import OmegaConf
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from transformers import AutoProcessor
from transformers.modeling_utils import no_init_weights
from transformers.processing_utils import ProcessorMixin
from transformers.utils import TRANSFORMERS_CACHE, ContextManagers
from wandb import Settings

from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.vlm.collate_fns import COLLATE_FNS
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.ddp import DDPManager
from nemo_automodel.components.distributed.init_utils import (
    get_world_size_safe,
    initialize_distributed,
)
from nemo_automodel.components.distributed.megatron_fsdp import MegatronFSDPManager
from nemo_automodel.components.distributed.utils import FirstRankPerNode, get_sync_ctx
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import MetricsSample, build_metric_logger
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.loss.mpg_kl import MixturePathGeneralizeKL
from nemo_automodel.components.loss.weighted_ce import WeightedCrossEntropy
from nemo_automodel.components.datasets.vlm.sampler import BalanceSampler
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.quantization.fp8 import apply_fp8_to_model, build_fp8_config
from nemo_automodel.components.training.rng import ScopedRNG, StatefulRNG
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.components.training.utils import (
    count_tail_padding,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.components.utils.compile_utils import (
    build_compile_config,
    compile_model,
)
from nemo_automodel.components.utils.model_utils import (
    _supports_logits_to_keep,
    apply_parameter_freezing,
    init_empty_weights,
    print_trainable_parameters,
)
from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.recipes.vlm.selective_logits import (
    _check_no_cp,
    _num_samples,
    _select_response,
)

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from nemo_automodel.components.distributed.init_utils import DistInfo

logger = logging.getLogger(__name__)

# ---------------------------
#  Stateless helper functions
# ---------------------------

def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    else:
        return data

def _get_model_name(cfg_model):
    if cfg_model.get("pretrained_model_name_or_path", None) is not None:
        return cfg_model.pretrained_model_name_or_path
    elif cfg_model.get("config", None) is not None:
        return cfg_model.config.get("pretrained_model_name_or_path", None)
    else:
        return None


def _freeze_model(model: nn.Module, cfg_freeze: Optional[Dict[str, Any]] = None, freeze_embeddings: bool = True):
    """
    Freeze the model.

    Args:
        model: The model to freeze.
        cfg_freeze: The configuration for freezing the model.
        freeze_embeddings: Whether to freeze embeddings.

    Returns:
        nn.Module: The frozen model.
    """
    if cfg_freeze is not None:
        apply_parameter_freezing(model, cfg_freeze)
    elif freeze_embeddings:
        logging.info("Freezing embeddings")
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.weight.requires_grad = False
    return model


def _get_optimizer_group_family(group_name):
    if not group_name:
        return None
    for family in ("visual", "language", "merger"):
        if group_name.startswith(f"{family}_"):
            return family
    return None


def _build_optimizer_group_index_map(param_groups):
    index_map = {}
    for idx, group in enumerate(param_groups):
        family = _get_optimizer_group_family(group.get("group_name"))
        if family is None:
            continue
        index_map.setdefault(family, []).append(idx)
    return index_map


def _get_optimizer_group_lr(optimizer, family):
    group_indices = getattr(optimizer, "_vlm_group_indices", None) or {}
    for idx in group_indices.get(family, []):
        if idx < len(optimizer.param_groups):
            return optimizer.param_groups[idx]["lr"]

    for group in optimizer.param_groups:
        if _get_optimizer_group_family(group.get("group_name")) == family:
            return group["lr"]

    return None

# Modified: to support multiple groups
def get_parameter_groups(model, cfg_opt):
    # be careful here, now is only designed for qwen-vl series model.
    base_lr = cfg_opt.get("lr", 1e-5)
    visual_lr = cfg_opt.get("visual_lr", base_lr)
    language_lr = cfg_opt.get("language_lr", base_lr)
    merger_lr = cfg_opt.get("merger_lr", base_lr)
    weight_decay = cfg_opt.get("weight_decay", 0.0)

    groups = {
        "visual_decay": {
            "group_name": "visual_decay",
            "params": [],
            "lr": visual_lr,
            "max_lr": visual_lr,
            "min_lr": visual_lr * 0.1,
            "init_lr": visual_lr * 0.1,
            "weight_decay": weight_decay,
            "wd_mult": weight_decay,
        },
        "visual_no_decay": {
            "group_name": "visual_no_decay",
            "params": [],
            "lr": visual_lr,
            "max_lr": visual_lr,
            "min_lr": visual_lr * 0.1,
            "init_lr": visual_lr * 0.1,
            "weight_decay": 0.0,
            "wd_mult": 0.0,
        },
        "language_decay": {
            "group_name": "language_decay",
            "params": [],
            "lr": language_lr,
            "max_lr": language_lr,
            "min_lr": language_lr * 0.05,
            "init_lr": language_lr * 0.1,
            "weight_decay": weight_decay,
            "wd_mult": weight_decay,
        },
        "language_no_decay": {
            "group_name": "language_no_decay",
            "params": [],
            "lr": language_lr,
            "max_lr": language_lr,
            "min_lr": language_lr * 0.05,
            "init_lr": language_lr * 0.1,
            "weight_decay": 0.0,
            "wd_mult": 0.0,
        },
        "merger_decay": {
            "group_name": "merger_decay",
            "params": [],
            "lr": merger_lr,
            "max_lr": merger_lr,
            "min_lr": merger_lr * 0.05,
            "init_lr": merger_lr * 0.1,
            "weight_decay": weight_decay,
            "wd_mult": weight_decay,
        },
        "merger_no_decay": {
            "group_name": "merger_no_decay",
            "params": [],
            "lr": merger_lr,
            "max_lr": merger_lr,
            "min_lr": merger_lr * 0.05,
            "init_lr": merger_lr * 0.1,
            "weight_decay": 0.0,
            "wd_mult": 0.0,
        },
        "other_decay": {
            "group_name": "other_decay",
            "params": [],
            "lr": base_lr,
            "max_lr": base_lr,
            "min_lr": base_lr * 0.05,
            "init_lr": base_lr * 0.1,
            "weight_decay": weight_decay,
            "wd_mult": weight_decay,
        },
        "other_no_decay": {
            "group_name": "other_no_decay",
            "params": [],
            "lr": base_lr,
            "max_lr": base_lr,
            "min_lr": base_lr * 0.05,
            "init_lr": base_lr * 0.1,
            "weight_decay": 0.0,
            "wd_mult": 0.0,
        },
    }

    no_decay_keywords = ["norm", "bias", "embed_tokens", "pos_embed"]
    seen_param_ids = set()

    visual_decay_names = []
    visual_no_decay_names = []
    language_decay_names = []
    language_no_decay_names = []
    merger_decay_names = []
    merger_no_decay_names = []
    other_decay_names = []
    other_no_decay_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 权重共享检测：如果这个物理参数已经分过组了，直接跳过
        if id(param) in seen_param_ids:
            logger.info(f"Skipping tied parameter: {name}")
            continue
        seen_param_ids.add(id(param))

        is_no_decay = any(k in name.lower() for k in no_decay_keywords) or (param.ndim <= 1)

        if "visual.merger" in name or "deepstack_merger_list" in name or "multi_modal_projector" in name:
            if is_no_decay:
                groups["merger_no_decay"]["params"].append(param)
                merger_no_decay_names.append(name)
            else:
                groups["merger_decay"]["params"].append(param)
                merger_decay_names.append(name)
        elif "visual" in name or "vision_tower" in name:
            if is_no_decay:
                groups["visual_no_decay"]["params"].append(param)
                visual_no_decay_names.append(name)
            else:
                groups["visual_decay"]["params"].append(param)
                visual_decay_names.append(name)    
        elif name.startswith("model.") or "language_model" in name or "lm_head" in name:
            if is_no_decay:
                groups["language_no_decay"]["params"].append(param)
                language_no_decay_names.append(name)    
            else:
                groups["language_decay"]["params"].append(param)
                language_decay_names.append(name)    
        else:
            if is_no_decay:
                groups["other_no_decay"]["params"].append(param)
                other_no_decay_names.append(name)
            else:
                groups["other_decay"]["params"].append(param)
                other_decay_names.append(name)

    param_groups = [v for k, v in groups.items() if len(v["params"]) > 0]

    for k, v in groups.items():
        if len(v["params"]) > 0:
            logger.info(f"Group {k}: {len(v['params'])} params, max_lr={v['max_lr']:.1e}, min_lr={v['min_lr']:.1e}")

    return param_groups


def build_model_and_optimizer(
    device,
    cfg_model,
    cfg_opt,
    cfg_freeze,
    cfg_peft,
    model_wrapper,
    seed,
    checkpointer: Checkpointer,
    tp_size=1,
    cp_size=1,
    freeze_embeddings=True,
    cfg_fp8=None,
    cfg_compile=None,
    loss_fn=None,
    parallelize_fn=None,
    load_base_model=True,
) -> tuple[nn.Module, list[str], "Optimizer"]:  # noqa: F821
    """
    Build and initialize a model for VLM.

    Args:
        device: The target device.
        cfg_model: Configuration for model instantiation.
        cfg_opt: Configuration for optimizer instantiation.
        cfg_freeze: Configuration for freezing parameters.
        cfg_peft: Configuration for PEFT.
        model_wrapper: Optional parallelism wrapper.
        seed: Random seed.
        tp_size: Tensor parallel size.
        freeze_embeddings: Whether to freeze embeddings.
        cfg_fp8: Configuration for FP8.
        cfg_compile: Configuration for torch.compile.
        parallelize_fn: Optional parallelization function.
        load_base_model: Whether to load the base model.

    Returns:
        The instantiated model on the specified device, the state dict keys before any parallelization, and the optimizer.
    """
    is_meta_device = not isinstance(model_wrapper, (MegatronFSDPManager, DDPManager))

    init_ctx = ContextManagers([no_init_weights(), init_empty_weights()]) if is_meta_device else nullcontext()
    with ScopedRNG(seed=seed, ranked=True):
        kwargs = {"tp_size": tp_size, "cp_size": cp_size}

        # Instantiate the model in meta device to avoid OOM
        with init_ctx:
            model = cfg_model.instantiate(**kwargs)
            model = _freeze_model(model, cfg_freeze, freeze_embeddings)
            # Optionally apply PEFT (e.g., LoRA/DoRA, etc)
            if cfg_peft is not None:
                if tp_size > 1:
                    logger.info("Disabling Triton with TP ({})".format(tp_size))
                    cfg_peft.use_triton = False
                apply_lora_to_linear_modules(model, cfg_peft)

            if cfg_fp8 is not None:
                fp8_config = build_fp8_config(cfg_fp8)
                model = apply_fp8_to_model(model, config=fp8_config)

        # hold a copy of the model state dict keys before any parallelization
        state_dict_keys = model.state_dict().keys()

        if not _supports_logits_to_keep(model) and not isinstance(loss_fn, MaskedCrossEntropy):
            logger.warning("logits_to_keep not found in model.forward. Using MaskedCrossEntropy instead.")
            loss_fn = MaskedCrossEntropy()

        load_weights = False
        if parallelize_fn is not None and get_world_size_safe() > 1:
            moe_mesh = getattr(model_wrapper, "moe_mesh", None)
            ep_axis_name = "ep" if moe_mesh is not None and "ep" in moe_mesh.mesh_dim_names else None
            ep_shard_axis_names = (
                ("ep_shard",) if moe_mesh is not None and "ep_shard" in moe_mesh.mesh_dim_names else None
            )
            parallelize_fn(
                model,
                world_mesh=model_wrapper.device_mesh,
                moe_mesh=moe_mesh,
                pp_enabled=False,
                dp_axis_names=(
                    ("dp_replicate", "dp_shard_cp")
                    if "dp_replicate" in model_wrapper.device_mesh.mesh_dim_names
                    and "dp_shard_cp" in model_wrapper.device_mesh.mesh_dim_names
                    else ("dp_shard_cp",)
                ),
                cp_axis_name="cp",
                tp_axis_name="tp",
                ep_axis_name=ep_axis_name,
                ep_shard_axis_names=ep_shard_axis_names,
            )
            load_weights = True
        elif callable(getattr(model_wrapper, "parallelize", None)):
            if isinstance(model_wrapper, MegatronFSDPManager):
                trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
                assert len(trainable_params) > 0, "trainable_params cannot be empty"
                if tp_size > 1:
                    cfg_opt.foreach = False
                optimizer = cfg_opt.instantiate(params=trainable_params)
                model, optimizer = model_wrapper.parallelize(model, optimizer)
                return model, state_dict_keys, optimizer
            else:
                load_weights = True
                model = model_wrapper.parallelize(model)

        # Load the weights into the model in parallel.
        if is_meta_device and load_weights:
            checkpointer.load_base_model(
                model,
                device,
                cfg_model.get("cache_dir", TRANSFORMERS_CACHE),
                _get_model_name(cfg_model),
                getattr(cfg_peft, "lora_A_init", None),
                load_base_model=load_base_model,
            )

        print_trainable_parameters(model)
        model = model.to(device)

        # Apply torch.compile if configured
        if cfg_compile is not None:
            compile_config = build_compile_config(cfg_compile)
            model = compile_model(model, compile_config)

        if tp_size > 1:
            # TP does not support foreach
            cfg_opt.foreach = False

        # modified
        # trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
        # assert len(trainable_params) > 0, "trainable_params cannot be empty"
        # optimizer = cfg_opt.instantiate(params=trainable_params)

        param_groups = get_parameter_groups(model, cfg_opt)
        assert len(param_groups) > 0, "No trainable parameters found!"
        optimizer_group_indices = _build_optimizer_group_index_map(param_groups)

        # remove useless field
        if hasattr(cfg_opt, 'visual_lr'):
            delattr(cfg_opt, 'visual_lr')
        if hasattr(cfg_opt, 'merger_lr'):
            delattr(cfg_opt, 'merger_lr')

        optimizer = cfg_opt.instantiate(params=param_groups)
        optimizer._vlm_group_indices = optimizer_group_indices

        return model, state_dict_keys, optimizer


def build_checkpoint_config(cfg_ckpt, cache_dir, model_repo_id, is_peft) -> CheckpointingConfig:
    """Build a checkpoint configuration.

    Args:
        cfg_ckpt: Configuration for checkpointing.
        cache_dir: Cache directory for the model.
        model_repo_id: Model repository ID.
        is_peft: Whether the model is PEFT.

    Returns:
        The instantiated checkpoint configuration.
    """
    ckpt_kwargs = dict(
        enabled=True,
        checkpoint_dir="checkpoints/",
        model_save_format="safetensors",
        model_repo_id=model_repo_id,
        model_cache_dir=cache_dir if cache_dir is not None else TRANSFORMERS_CACHE,
        save_consolidated=True,
        is_peft=is_peft,
    )
    if cfg_ckpt is not None:
        cfg_ckpt = cfg_ckpt.to_dict()
        cfg_ckpt.pop("restore_from", None)
        cfg_ckpt.pop("load_base_model", None)
        ckpt_kwargs |= cfg_ckpt
    if ckpt_kwargs.get("is_peft", False) and ckpt_kwargs.get("model_save_format") == "torch_save":
        raise ValueError(
            "PEFT checkpointing is not supported for torch_save format. Save using `safetensors` format instead."
        )
    checkpoint_config = CheckpointingConfig(**ckpt_kwargs)
    return checkpoint_config


def build_loss_fn(cfg_loss):
    """Build a loss function.

    Args:
        cfg_loss: Loss function configuration.

    Returns:
        The instantiated loss function.
    """
    return cfg_loss.instantiate()


def build_dataloader(
    cfg_ds, cfg_dl, pretrained_model_name_or_path, cfg_processor, device_mesh, seed, local_batch_size
) -> tuple[DataLoader, ProcessorMixin]:
    """Build a DataLoader for the VLM dataset.

    Args:
        cfg_ds: Dataset configuration.
        cfg_dl: DataLoader configuration.
        pretrained_model_name_or_path: Pretrained model name or path for processor loading.
        cfg_processor: Processor configuration or None.
        device_mesh: Device mesh for distributed training.
        seed: Random seed.
        local_batch_size: Local batch size.

    Returns:
        The instantiated DataLoader and processor.
    """
    dist_sampler_kwargs = {
        "shuffle": cfg_dl.get("shuffle", True),
    }
    if device_mesh is not None:
        dist_sampler_kwargs |= {
            "num_replicas": device_mesh["dp"].size(),
            "rank": device_mesh["dp"].get_local_rank(),
        }

    with ScopedRNG(seed=seed, ranked=True):
        processor = None
        processor_kwargs = {}
        if cfg_processor is not None and hasattr(cfg_processor, "instantiate"):
            processor = cfg_processor.instantiate()
        elif cfg_processor is not None:
            processor_kwargs = cfg_processor.to_dict()

        # If no processor was instantiated, try AutoProcessor
        if processor is None:
            try:
                processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **processor_kwargs)
            except Exception as e:
                # Some models do not provide an AutoProcessor
                processor = None
                logging.warning(f"AutoProcessor not available for {pretrained_model_name_or_path} ({e}). ")

        with FirstRankPerNode():
            # ds = cfg_ds.instantiate(path_or_dataset=cfg_ds.path_or_dataset)
            ds_dict = {k: v for k, v in cfg_ds.__dict__.items() if not k.startswith('_')}
            ds = cfg_ds.instantiate(**ds_dict)

        collate_cfg = cfg_dl.get("collate_fn", None)
        if collate_cfg:
            collate_fn = collate_cfg.instantiate(processor=processor, max_len=cfg_ds.max_len)
        else:
            processor_type = type(processor).__name__
            if processor_type not in COLLATE_FNS:
                processor_type = "default"
                logging.warning(f"You are using {processor_type} with default collate function.")
            collate_fn = lambda examples: COLLATE_FNS[processor_type](examples, processor)

        dataset_meta = ds.get_metadata()
        if dataset_meta is not None:
            logging.info("Using BalanceSampler.")
            lengths, v_tokens = dataset_meta[0], dataset_meta[1]
            sampler = BalanceSampler(
                lengths=lengths,
                v_tokens=v_tokens,
                local_batch_size=local_batch_size,
                seed=seed,
                drop_last=cfg_ds.drop_last,
                **dist_sampler_kwargs,
            )
            dl_kwargs = dict(
                dataset=ds,
                batch_sampler=sampler,
                collate_fn=collate_fn,
            )
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds,
                **dist_sampler_kwargs,
            )
            dl_kwargs = dict(
                dataset=ds,
                sampler=sampler,
                collate_fn=collate_fn,
                batch_size=local_batch_size,
            )
        # DataLoader workers should use spawn to avoid CUDA fork-safety issues.
        if cfg_dl.get("num_workers", 0) > 0 and cfg_dl.get("multiprocessing_context", None) is None:
            dl_kwargs["multiprocessing_context"] = "spawn"

        return cfg_dl.instantiate(**dl_kwargs), processor


def build_multi_task_dataloader(
    cfg, pretrained_model_name_or_path, cfg_processor, device_mesh, seed, local_batch_size
) -> tuple:
    """Build a MultiTaskDataLoader when task_weights is configured.

    Constructs independent DataLoaders for each task (understanding, generation, editing)
    and wraps them in a MultiTaskDataLoader that yields batches interleaved by weight.

    Returns:
        (MultiTaskDataLoader, processor)
    """
    from nemo_automodel.components.datasets.vlm.multi_task_dataloader import MultiTaskDataLoader

    task_weights = cfg.task_weights.to_dict() if hasattr(cfg.task_weights, "to_dict") else dict(cfg.task_weights)

    # Build the primary (understanding) dataloader
    primary_dl, processor = build_dataloader(
        cfg.dataset, cfg.dataloader, pretrained_model_name_or_path,
        cfg_processor, device_mesh, seed, local_batch_size,
    )
    dataloaders = {"understanding": primary_dl}

    # Build generation dataloader if configured
    if "generation_dataset" in cfg and "generation_dataloader" in cfg:
        gen_dl, _ = build_dataloader(
            cfg.generation_dataset, cfg.generation_dataloader, pretrained_model_name_or_path,
            cfg_processor, device_mesh, seed, local_batch_size,
        )
        dataloaders["generation"] = gen_dl

    # Build editing dataloader if configured
    if "editing_dataset" in cfg and "editing_dataloader" in cfg:
        edit_dl, _ = build_dataloader(
            cfg.editing_dataset, cfg.editing_dataloader, pretrained_model_name_or_path,
            cfg_processor, device_mesh, seed, local_batch_size,
        )
        dataloaders["editing"] = edit_dl

    multi_dl = MultiTaskDataLoader(dataloaders, task_weights, seed=seed)
    return multi_dl, processor


def build_distributed(cfg_dist: Dict[str, Any]) -> "DistInfo":  # noqa: F821
    """Build and initialize distributed training resources.

    Args:
        cfg_dist: Configuration for distributed training.

    Returns:
        Distributed training information from initialize_distributed.
    """
    backend = cfg_dist.get("backend", "nccl")
    timeout = cfg_dist.get("timeout_minutes", 1)
    return initialize_distributed(backend=backend, timeout_minutes=timeout)


def build_step_scheduler(cfg, dataloader, dp_group_size, local_batch_size):
    """Build the step scheduler.

    Args:
        cfg: configuration for the StepScheduler class.
        dataloader: the training dataloader, used for extracting the epoch_len (in batches).
        dp_group_size: the size of the data parallel group.
        micro_batch_size: the size of the micro batch.

    Returns:
        StepScheduler: the configured StepScheduler.
    """
    assert "_target_" not in cfg, "_target_ not permitted in step scheduler"
    default_kwargs = dict(
        num_epochs=10,
        global_batch_size=32,
        local_batch_size=local_batch_size,
        dp_size=dp_group_size,
        ckpt_every_steps=100,
        dataloader=dataloader,
    )
    if cfg is not None:
        default_kwargs |= cfg.to_dict()
    return StepScheduler(**default_kwargs)


def build_lr_scheduler(cfg, optimizer, step_scheduler) -> OptimizerParamScheduler | None:  # noqa: F821
    """Build the learning rate scheduler.

    Args:
        cfg: Configuration for the OptimizerParamScheduler.
        optimizer: The optimizer to be scheduled.
        step_scheduler: The step scheduler to extract training parameters.

    Returns:
        OptimizerParamScheduler: The configured learning rate scheduler, or None if not configured.
    """
    if cfg is None:
        return None

    # Calculate total steps for the training run
    total_epochs = step_scheduler.num_epochs
    epoch_len = len(step_scheduler.dataloader)
    grad_acc_steps = step_scheduler.grad_acc_steps

    # Total optimizer steps (accounting for gradient accumulation)
    total_steps = (total_epochs * epoch_len) // grad_acc_steps
    if step_scheduler.max_steps is not None:
        total_steps = min(total_steps, step_scheduler.max_steps)

    # Extract learning rate from optimizer
    # 取第一组的 lr 作为 scheduler 的名义全局 lr, 用于过 init 校验.
    base_lr = optimizer.param_groups[0].get("max_lr", optimizer.param_groups[0]["lr"])

    # Set defaults for scheduler parameters
    default_kwargs = dict(
        optimizer=optimizer,
        init_lr=base_lr * 0.1,  # Start warmup at 10% of base LR
        max_lr=base_lr,
        min_lr=base_lr * 0.01,  # End at 1% of base LR
        lr_warmup_steps=min(1000, total_steps // 10),  # 10% warmup or max 1000 steps
        lr_decay_steps=total_steps,
        lr_decay_style="cosine",
        # start_wd=optimizer.param_groups[0].get("weight_decay", 0.0),
        # end_wd=optimizer.param_groups[0].get("weight_decay", 0.0),
        start_wd=1.0,
        end_wd=1.0,
        wd_incr_steps=total_steps,
        wd_incr_style="constant",
    )

    # Override with user-provided config
    if cfg is not None:
        user_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
        default_kwargs.update(user_cfg)

    # Clamp warmup to be strictly less than total decay steps
    if default_kwargs["lr_warmup_steps"] >= default_kwargs["lr_decay_steps"]:
        clamped = max(default_kwargs["lr_decay_steps"] - 1, 0)
        logger.warning(
            f"lr_warmup_steps ({default_kwargs['lr_warmup_steps']}) >= lr_decay_steps ({default_kwargs['lr_decay_steps']}), "
            f"clamping to {clamped}"
        )
        default_kwargs["lr_warmup_steps"] = clamped

    logger.info(
        f"Building LR scheduler with total_steps={total_steps}, "
        f"warmup_steps={default_kwargs['lr_warmup_steps']}, "
        f"decay_style={default_kwargs['lr_decay_style']}"
    )

    return OptimizerParamScheduler(**default_kwargs)


def build_wandb(cfg) -> wandb.Run:
    """Instantiates wandb and returns the instance. If no name is given, it will use the model name.

    Args:
        cfg: Configuration for wandb.

    Returns:
        The wandb instance.
    """
    assert cfg.get("wandb", None) is not None
    kwargs = cfg.wandb.to_dict()
    if kwargs.get("name", "") == "":
        kwargs["name"] = "_".join(_get_model_name(cfg.model).split("/")[-2:])
    run = wandb.init(
        **kwargs,
        config=cfg.to_dict(),
        settings=Settings(silent=True),
    )
    return run


def calculate_loss(loss_fn, **kwargs) -> torch.Tensor:
    """Calculate the loss.

    Args:
        loss_fn: Loss function.
        **kwargs: Keyword arguments for the loss function.

    Returns:
        The loss.
    """
    loss_fn_kwargs = {}
    if isinstance(loss_fn, FusedLinearCrossEntropy):
        model = kwargs.pop("model")
        labels = kwargs.pop("labels")

        # find the lm_head in the model
        lm_head = None
        if hasattr(model, "get_output_embeddings"):
            lm_head = model.get_output_embeddings().weight
        else:
            for n, p in model.named_parameters(remove_duplicate=False):
                if "lm_head" in n and n.endswith(".weight"):
                    lm_head = p
                    break
        if lm_head is None:
            raise ValueError("lm_head.weight not found in model")

        # unshard the possibly sharded lm_head
        lm_head = lm_head.full_tensor() if hasattr(lm_head, "full_tensor") else lm_head
        loss_fn_kwargs.update(
            {
                "hidden_states": kwargs.pop("hidden_states"),
                "labels": labels,
                "lm_weight": lm_head,
                "num_label_tokens": kwargs.pop("num_label_tokens", None),
            }
        )
    elif isinstance(loss_fn, MixturePathGeneralizeKL):
        loss_fn_kwargs.update(
            {
                "logits": kwargs.pop("logits"),
                "labels": kwargs.pop("labels"),
                "x_t": kwargs.pop("x_t"),
                "t": kwargs.pop("t"),
                "response_mask": kwargs.pop("response_mask"),
                "loss_mask": kwargs.pop("loss_mask"),
                "num_label_tokens": kwargs.pop("num_label_tokens", None),
                "num_samples": kwargs.pop("num_samples", None),
                "block_size": kwargs.pop("block_size", 0),
            }
        )
    elif isinstance(loss_fn, WeightedCrossEntropy):
        loss_fn_kwargs.update(
            {
                "logits": kwargs.pop("logits"),
                "labels": kwargs.pop("labels"),
                "x_t": kwargs.pop("x_t"),
                "t": kwargs.pop("t"),
                "response_mask": kwargs.pop("response_mask"),
                "loss_mask": kwargs.pop("loss_mask"),
                "num_label_tokens": kwargs.pop("num_label_tokens", None),
                "num_samples": kwargs.pop("num_samples", None),
                "block_size": kwargs.pop("block_size", 0),
            }
        )
    else:
        loss_fn_kwargs.update(
            {
                "logits": kwargs.pop("logits"),
                "labels": kwargs.pop("labels"),
                "num_label_tokens": kwargs.pop("num_label_tokens", None),
            }
        )

    return loss_fn(**loss_fn_kwargs)


def _compute_dual_head_loss(
    loss_fn, out, labels, generation_mask, *,
    x_t, t, loss_mask, response_mask, num_label_tokens, num_samples, block_size,
):
    """Dual-head loss for Bard-Uni: text region uses lm_head, VQ region uses image_head."""
    logits = out.logits
    image_logits = getattr(out, "image_logits", None)
    device = logits.device

    text_mask = loss_mask & ~generation_mask
    text_tokens = int(text_mask.sum().item())
    text_loss = torch.tensor(0.0, device=device)

    if text_tokens > 0:
        text_loss = loss_fn(
            logits=logits, labels=labels, x_t=x_t, t=t,
            response_mask=response_mask, loss_mask=text_mask,
            num_label_tokens=max(text_tokens, 1),
            num_samples=num_samples, block_size=block_size,
        )

    vq_mask = loss_mask & generation_mask
    vq_tokens = int(vq_mask.sum().item())
    vq_loss = torch.tensor(0.0, device=device)

    if vq_tokens > 0 and image_logits is not None:
        # Clamp labels to valid range for image_head (codebook_size)
        vq_labels = labels.clamp(0, image_logits.size(-1) - 1)
        vq_loss = loss_fn(
            logits=image_logits, labels=vq_labels, x_t=x_t, t=t,
            response_mask=response_mask, loss_mask=vq_mask,
            num_label_tokens=max(vq_tokens, 1),
            num_samples=num_samples, block_size=block_size,
        )

    total_tokens = max(text_tokens + vq_tokens, 1)
    total_loss = (text_loss * text_tokens + vq_loss * vq_tokens) / total_tokens

    # Debug: print grad info on first call
    if not hasattr(_compute_dual_head_loss, "_debug_printed"):
        _compute_dual_head_loss._debug_printed = True
        import logging
        _logger = logging.getLogger(__name__)
        _logger.info(
            f"[dual_head_loss] text_tokens={text_tokens}, vq_tokens={vq_tokens}, "
            f"image_logits is None={image_logits is None}, "
            f"total_loss.requires_grad={total_loss.requires_grad}, "
            f"total_loss.grad_fn={total_loss.grad_fn}"
        )

    return total_loss


# ---------------------------------------------------------------------------
#  Trainer class – orchestration only
# ---------------------------------------------------------------------------

class FinetuneRecipeForVLM(BaseRecipe):
    """Recipe for fine-tuning a VLM model."""

    def __init__(self, cfg):
        """Initialize the recipe with configuration.

        Args:
            cfg: Configuration dictionary/object for training.
        """
        self.cfg = cfg

    # ------------------ build phase ------------------
    def setup(self):
        """Builds all components needed for training/validation/logging/checkpointing/etc.

        This is the last place where self.cfg should be referenced.

        Raises:
            NotImplemented: Raises if it tries to restore a checkpoint; will be removed.
        """
        torch.cuda.reset_peak_memory_stats()
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        setup_logging()

        apply_cache_compatibility_patches()

        # Set up the stateful random number generator
        self.rng = StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True)

        self.device_mesh = None
        self.moe_mesh = None
        self.model_wrapper = None
        if "distributed" in self.cfg:
            self.model_wrapper = self.cfg.distributed.instantiate(world_size=self.dist_env.world_size)
            self.device_mesh = getattr(self.model_wrapper, "device_mesh", None)
            self.moe_mesh = getattr(self.model_wrapper, "moe_mesh", None)

        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = build_wandb(self.cfg)
            logging.info("🚀 View run at {}".format(run.url))

        # Log experiment details on main rank
        self._log_experiment_details()
        self._log_library_versions()

        # Build components with VLM-specific functions
        self.peft_config = None
        if self.cfg.get("peft", None) is not None:
            self.peft_config = self.cfg.peft.instantiate()
        self.loss_fn = build_loss_fn(self.cfg.loss_fn)
        parallelize_fn = getattr(self.cfg.get("parallelizer", None), "instantiate", None)

        # Build checkpoint config
        checkpoint_config = build_checkpoint_config(
            self.cfg.get("checkpoint", None),
            self.cfg.get("model.cache_dir", None),
            _get_model_name(self.cfg.model),
            True if self.cfg.get("peft", None) else False,
        )

        if self.cfg.get("clip_grad_norm.max_norm", None) is not None:
            self.max_grad_norm = float(self.cfg.clip_grad_norm.max_norm)
        else:
            logging.info("No clip_grad_norm.max_norm specified in config, using default value of 1.0")
            self.max_grad_norm = 1.0

        # Create Checkpointer instance
        self.checkpointer = Checkpointer(
            config=checkpoint_config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=self.moe_mesh,
        )

        self.model, model_state_dict_keys, self.optimizer = build_model_and_optimizer(
            self.dist_env.device,
            self.cfg.model,
            self.cfg.optimizer,
            self.cfg.get("freeze_config", None),
            self.peft_config,
            self.model_wrapper,
            seed=self.cfg.get("seed", 42),
            tp_size=self.cfg.get("distributed.tp_size", 1),
            cp_size=self.cfg.get("distributed.cp_size", 1),
            cfg_fp8=self.cfg.get("fp8", None),
            cfg_compile=self.cfg.get("compile", None),
            loss_fn=self.loss_fn,
            parallelize_fn=parallelize_fn,
            load_base_model=self.cfg.get("checkpoint.load_base_model", True),
            checkpointer=self.checkpointer,
        )
        self.checkpointer.config.model_state_dict_keys = model_state_dict_keys

        if "task_weights" in self.cfg:
            self.dataloader, self.processor = build_multi_task_dataloader(
                self.cfg,
                _get_model_name(self.cfg.model),
                self.cfg.get("processor", None),
                device_mesh=self.device_mesh,
                seed=self.cfg.get("seed", 42),
                local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
            )
        else:
            self.dataloader, self.processor = build_dataloader(
                self.cfg.dataset,
                self.cfg.dataloader,
                _get_model_name(self.cfg.model),
                self.cfg.get("processor", None),
                device_mesh=self.device_mesh,
                seed=self.cfg.get("seed", 42),
                local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
            )

        # Build validation dataloader if the config provides it
        self.val_dataloader = None
        if "validation_dataset" in self.cfg:
            self.val_dataloader, _ = build_dataloader(
                self.cfg.validation_dataset,
                self.cfg.validation_dataloader,
                _get_model_name(self.cfg.model),
                self.cfg.get("processor", None),
                device_mesh=self.device_mesh,
                seed=self.cfg.get("seed", 42),
                local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
            )

        self.best_metric_key = self.cfg.get("checkpoint.best_metric_key", "default")
        # Scheduler
        self.step_scheduler = build_step_scheduler(
            self.cfg.get("step_scheduler", None),
            self.dataloader,
            self._get_dp_group_size(),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
        )

        # Build learning rate scheduler
        self.lr_scheduler = build_lr_scheduler(self.cfg.get("lr_scheduler", None), self.optimizer, self.step_scheduler)

        # Log model, parameter counts, norms, optimizer and scheduler
        self._log_model_and_optimizer_details(self.model, self.optimizer, self.lr_scheduler)

        restore_from = self.cfg.get("checkpoint.restore_from", None)

        # Initialize JSONL loggers
        self.metric_logger_train = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "training.jsonl"
        )
        self.metric_logger_valid = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "validation.jsonl"
        )

        # Optionally resume
        self.load_checkpoint(restore_from)

        # Log step scheduler details
        self._log_step_scheduler_details(self.step_scheduler)

    # ------------------ main loop ------------------
    def run_train_validation_loop(self):
        """Run the training loop over all epochs and batches.

        For each batch, perform a forward pass, compute loss, backpropagate,
        and update model parameters when necessary. Also prints loss every gradient step.
        """
        total_steps = self.step_scheduler.max_steps
        prior_dist_2 = getattr(self.dataloader.dataset, "prior_dist_2", None)
        switch_prior_thresh = self.cfg.dataset.get("switch_prior_thresh", 1.0)
        switch_step = max(0, int(total_steps * switch_prior_thresh))

        self.model.train()
        self.timestamp = time.perf_counter()
        for epoch in self.step_scheduler.epochs:
            self.step_scheduler.set_epoch(epoch)
            for batch_idx, batches in enumerate(self.step_scheduler): # batches是个list，长度为acc_steps, 每个item维度是local_batch_size
                cur_step = self.step_scheduler.step
                # switch noisy prior dist for curriculum learning
                if prior_dist_2 is not None and cur_step > switch_step:
                    self.dataloader.dataset.prior_dist = prior_dist_2
                    prior_dist_2 = None

                log_data = self._run_train_optim_step(batches, self.max_grad_norm)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(1)

                # log
                self.log_train_metrics(log_data)

                val_loss = {}
                if self.step_scheduler.is_val_step and self.val_dataloader is not None:
                    val_log_data = self._run_validation_epoch(self.val_dataloader)
                    val_loss["val_loss"] = val_log_data.metrics["val_loss"]
                    self.log_val_metrics(val_log_data)
                    self.model.train()

                if self.step_scheduler.is_ckpt_step:
                    self.save_checkpoint(
                        epoch,
                        self.step_scheduler.step,
                        log_data.metrics["loss"],
                        val_loss,
                        best_metric_key=self.best_metric_key,
                    )

        # Close JSONL loggers after training loop completes
        self.metric_logger_train.close()
        self.metric_logger_valid.close()

        self.checkpointer.close()

    def _run_train_optim_step(self, batches, max_grad_norm: Optional[float] = None):
        """Execute a single training step.

        Args:
            batches: List of batches of training data.
            max_grad_norm: Gradient clipping norm. Optional, if None will not clip gradients.
        """
        if 'loss_mask' in batches[0]:
            num_label_tokens = torch.tensor(
                sum((batch["loss_mask"]).sum().item() for batch in batches), dtype=torch.long
            )
        else:
            num_label_tokens = torch.tensor(
                sum((batch["labels"] != -100).sum().item() for batch in batches), dtype=torch.long
            )

        global_batch_size = torch.tensor(len(batches))
        global_batch_size = self._dp_allreduce(global_batch_size).item()

        # 单节点一次iteration的样本数
        num_samples = torch.tensor(
            sum((batch["num_samples"]).sum().item() for batch in batches), dtype=torch.long
        )
        # 全部节点一次iteration的样本总数
        num_total_samples = self._dp_allreduce(num_samples).item()
        num_processes = dist.get_world_size()

        num_label_tokens = self._dp_allreduce(num_label_tokens).item()
        loss_buffer = []

        # number of tokens in the batch, excluding any tail padding.
        num_tokens_in_batch = torch.tensor(
            sum(batch["labels"].numel() - count_tail_padding(batch["labels"]) for batch in batches),
            dtype=torch.long,
        )
        num_tokens_in_batch = self._dp_allreduce(num_tokens_in_batch).item()

        num_batches = len(batches)
        for i, batch in enumerate(batches): # accumulation_steps维度迭代
            batch = to_device(batch, self.dist_env.device)
            labels = batch.pop("labels")
            generation_mask = batch.pop("generation_mask", None)

            train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch, labels) # local_batch_size维度迭代
            with (
                train_ctx(),
                get_sync_ctx(
                    self.model,
                    i == num_batches - 1,
                    defer_fsdp_grad_sync=getattr(self.model_wrapper, "defer_fsdp_grad_sync", True),
                ),
            ):
                if generation_mask is not None and generation_mask.any():
                    # Bard-Uni dual-head path: full-sequence forward, split loss by generation_mask
                    batch["generation_mask"] = generation_mask
                    if "vq_code_mask" in batch:
                        batch["vq_codes"] = batch["input_ids"].clone()
                    out = self.model(labels=labels, **batch)

                    # Debug: check model output type on first call
                    if not hasattr(self, "_out_debug_done"):
                        self._out_debug_done = True
                        _img_logits = getattr(out, "image_logits", "MISSING")
                        logging.info(
                            f"[train debug] out type={type(out).__name__}, "
                            f"has image_logits={hasattr(out, 'image_logits')}, "
                            f"image_logits is None={_img_logits is None}, "
                            f"model class={type(self.model).__name__}, "
                            f"batch keys={list(batch.keys())}"
                        )

                    local_loss = _compute_dual_head_loss(
                        self.loss_fn, out, labels, generation_mask,
                        x_t=batch.get("input_ids"),
                        t=batch.get("t"),
                        loss_mask=batch.get("loss_mask"),
                        response_mask=batch.get("response_mask"),
                        num_label_tokens=num_label_tokens,
                        num_samples=num_total_samples,
                        block_size=batch.get("block_size", 0),
                    )
                else:
                    # Standard single-head path (Bard-VL / understanding-only)
                    use_selected_response_logits = isinstance(
                        self.loss_fn,
                        (MixturePathGeneralizeKL, WeightedCrossEntropy),
                    )
                    if use_selected_response_logits:
                        _check_no_cp(self.device_mesh, "Selective noisy-response logits")

                    selected_response = None
                    if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                        out = self.model(logits_to_keep=1, labels=labels, **batch)
                        if "hidden_states" not in out:
                            raise ValueError(
                                "FusedLinearCrossEntropy requires the model to output hidden states. Set `model.output_hidden_states=True` in the config."
                            )
                    else:
                        if use_selected_response_logits:
                            selected_response = _select_response(
                                batch["response_mask"],
                                {
                                    "labels": (labels, 0),
                                    "input_ids": (batch["input_ids"], 0),
                                    "t": (batch["t"], 0.0),
                                    "loss_mask": (batch["loss_mask"], False),
                                    "response_mask": (batch["response_mask"], False),
                                },
                                empty_error="No noisy response tokens available for selective logits.",
                            )
                            out = self.model(logits_to_keep=selected_response["indices"], labels=labels, **batch)
                        else:
                            out = self.model(labels=labels, **batch)

                    local_loss = calculate_loss(
                        self.loss_fn,
                        logits=getattr(out, "logits", out),
                        labels=selected_response["labels"] if selected_response is not None else labels,
                        model=self.model,
                        hidden_states=out.hidden_states[-1] if getattr(out, "hidden_states", None) is not None else None,
                        num_label_tokens=num_label_tokens,
                        x_t=selected_response["input_ids"] if selected_response is not None else batch.get("input_ids", None),
                        t=selected_response["t"] if selected_response is not None else batch.get("t", None),
                        loss_mask=selected_response["loss_mask"] if selected_response is not None else batch.get("loss_mask", None),
                        response_mask=selected_response["response_mask"] if selected_response is not None else batch.get("response_mask", None),
                        block_size=batch.get("block_size") if "block_size" in batch else 0,
                        num_samples=num_total_samples,
                    )

                loss_buffer.append(local_loss.clone().detach())
                local_loss.backward()

        grad_norm = scale_grads_and_clip_grad_norm(
            max_grad_norm=max_grad_norm,
            model_parts=[self.model],
            norm_type=2.0,
            pp_enabled=False,
            device_mesh=self.device_mesh,
            moe_mesh=self.moe_mesh,
            ep_axis_name="ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None,
            pp_axis_name=None,
            foreach=True,
            num_label_tokens=num_label_tokens,
            dp_group_size=self._get_dp_group_size(include_cp=True),
        )

        # Note(MegatronFSDP): Need to call these functions for MegatronFSDP if not using latest api
        # self.model.finish_grad_sync()

        self.checkpointer.maybe_wait_for_staging()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if hasattr(self.model, "update_moe_gate_bias"):
            self.model.update_moe_gate_bias()

        # Precompute FP8 scales
        fp8_config = self.cfg.get("fp8", None)
        if (
            fp8_config is not None
            and fp8_config.get("enabled", False)
            and fp8_config.get("precompute_float8_dynamic_scale_for_fsdp", False)
            and self.device_mesh is not None
            and self.device_mesh["dp_shard"].size() > 1
        ):
            precompute_float8_dynamic_scale_for_fsdp(self.model)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(1)

        # Note(MegatronFSDP): Need to call these functions for MegatronFSDP if not using latest api
        # self.model.install_optimized_model_weights()
        # self.model.zero_grad_buffer()

        # TPS is calculated as follows (assuming grad-accumulation-steps=2):
        # fwd 0 | bwd 0 | fwd 1 | bwd 1 | opt 0 | fwd 2 | bwd 2 | ...
        # ^                                     ^
        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta
        reporting_loss = torch.sum(torch.stack(loss_buffer))
        reporting_loss = self._dp_allreduce(reporting_loss, include_cp=True).item()
        # fix reporting_loss, tps across ranks

        metrics = {
            "loss": reporting_loss,
            "grad_norm": grad_norm,
            "single_samples": num_samples,
            "total_samples": num_total_samples, # total train samples in each iteration
            "mem": torch.cuda.max_memory_allocated() / 1024**3,
            "tps": tps,
            "tps_per_gpu": tps / max(self._get_dp_group_size(), 1),
            "num_tokens_per_step": num_tokens_in_batch,
            "num_label_tokens": num_label_tokens,
        }
        for family in ("visual", "merger", "language"):
            lr = _get_optimizer_group_lr(self.optimizer, family)
            if lr is not None:
                metrics[f"{family}_lr"] = lr

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics=metrics,
        )

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        """Run one pass over `self.val_dataloader`."""
        with ScopedRNG(seed=1, ranked=True):
            self.model.eval()

            total_loss = 0.0
            total_tokens = 0
            total_num_label_tokens = 0
            for batch in val_dataloader:
                batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
                labels = batch.pop("labels")
                generation_mask = batch.pop("generation_mask", None)
                num_label_tokens = int(batch["loss_mask"].sum().item()) if "loss_mask" in batch else int((labels != -100).sum().item())

                if (
                    self.device_mesh
                    and "position_ids" not in batch
                    and (self.device_mesh["cp"].size() > 1 or self.device_mesh["tp"].size() > 1)
                ):
                    batch["position_ids"] = (
                        torch.arange(0, batch["input_ids"].shape[1]).unsqueeze(0).to(self.model.device)
                    )

                train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch, labels)
                with train_ctx():
                    if generation_mask is not None and generation_mask.any():
                        batch["generation_mask"] = generation_mask
                        if "vq_code_mask" in batch:
                            batch["vq_codes"] = batch["input_ids"].clone()
                        out = self.model(labels=labels, **batch)
                        local_loss = _compute_dual_head_loss(
                            self.loss_fn, out, labels, generation_mask,
                            x_t=batch.get("input_ids"),
                            t=batch.get("t"),
                            loss_mask=batch.get("loss_mask"),
                            response_mask=batch.get("response_mask"),
                            num_label_tokens=num_label_tokens,
                            num_samples=_num_samples(batch),
                            block_size=batch.get("block_size", 0),
                        )
                    else:
                        use_selected_response_logits = isinstance(
                            self.loss_fn,
                            (MixturePathGeneralizeKL, WeightedCrossEntropy),
                        )
                        if use_selected_response_logits:
                            _check_no_cp(self.device_mesh, "Selective noisy-response logits")

                        selected_response = None
                        if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                            out = self.model(logits_to_keep=1, labels=labels, **batch)
                        else:
                            if use_selected_response_logits:
                                selected_response = _select_response(
                                    batch["response_mask"],
                                    {
                                        "labels": (labels, 0),
                                        "input_ids": (batch["input_ids"], 0),
                                        "t": (batch["t"], 0.0),
                                        "loss_mask": (batch["loss_mask"], False),
                                        "response_mask": (batch["response_mask"], False),
                                    },
                                    empty_error="No noisy response tokens available for selective logits during validation.",
                                )
                                out = self.model(logits_to_keep=selected_response["indices"], labels=labels, **batch)
                            else:
                                out = self.model(labels=labels, **batch)
                        local_loss = calculate_loss(
                            self.loss_fn,
                            logits=getattr(out, "logits", out),
                            labels=selected_response["labels"] if selected_response is not None else labels,
                            model=self.model,
                            hidden_states=out.hidden_states[-1]
                            if getattr(out, "hidden_states", None) is not None
                            else None,
                            num_label_tokens=num_label_tokens,
                            x_t=selected_response["input_ids"] if selected_response is not None else batch.get("input_ids", None),
                            t=selected_response["t"] if selected_response is not None else batch.get("t", None),
                            loss_mask=selected_response["loss_mask"] if selected_response is not None else batch.get("loss_mask", None),
                            response_mask=selected_response["response_mask"] if selected_response is not None else batch.get("response_mask", None),
                            block_size=batch.get("block_size") if "block_size" in batch else 0,
                            num_samples=_num_samples(batch),
                        )
                    total_num_label_tokens += num_label_tokens

                total_loss += local_loss.item() * num_label_tokens
                total_tokens += num_label_tokens

        # Aggregate across ranks if distributed is initialized
        total_loss = self._dp_allreduce(torch.FloatTensor([total_loss]), include_cp=True).item()
        total_tokens = self._dp_allreduce(torch.LongTensor([total_tokens]), include_cp=True).item()
        total_num_label_tokens = self._dp_allreduce(torch.LongTensor([total_num_label_tokens])).item()

        val_loss = total_loss / max(total_tokens, 1e-8)

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                "num_label_tokens": total_num_label_tokens,
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
            },
        )

    def log_val_metrics(self, log_data):
        """Log metrics to wandb and other loggers
        Args:
            log_data: MetricsSample object, containing:
                step: int, the current step.
                epoch: int, the current epoch.
                metrics: Dict[str, float], containing:
                    "val_loss": Validation loss.
                    "lr": Learning rate.
                    "num_label_tokens": Number of label tokens.
                    "mem": Memory allocated.
        """

        if not self.dist_env.is_main or log_data is None:
            return

        if wandb.run is not None:
            wandb.log(log_data.to_dict(), step=log_data.step)

        # JSONL validation log
        self.metric_logger_valid.log(log_data)

        logging.info(
            "[val] step {} | epoch {} | loss {:.4f} | lr {:.2e} | num_label_tokens {}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["val_loss"],
                log_data.metrics["lr"],
                log_data.metrics["num_label_tokens"],
            )
        )

    def log_train_metrics(self, log_data) -> float:
        """Log metrics to wandb.

        Args:
            train_loss: Training loss.
            grad_norm: Grad norm from the training step.
            num_tokens_in_batch: Total number of loss tokens.
            tps: Tokens per second.
        """
        if not self.dist_env.is_main:
            return

        if wandb.run is not None:
            wandb.log(log_data.to_dict(), step=self.step_scheduler.step)
        # JSONL training log
        self.metric_logger_train.log(log_data)
        # logging.info(
        #     "step {} | epoch {} | loss {:.4f} | grad_norm {:.4f} | lr {:.2e} | mem {:.2f} GiB | tps {:.2f}({:.2f}/gpu) | num_label_tokens {}".format(
        #         log_data.step,
        #         log_data.epoch,
        #         log_data.metrics["loss"],
        #         log_data.metrics["grad_norm"],
        #         log_data.metrics["lr"],
        #         log_data.metrics["mem"],
        #         log_data.metrics["tps"],
        #         log_data.metrics["tps_per_gpu"],
        #         log_data.metrics["num_label_tokens"],
        #     )
        # )

        parts = [
            f"step {log_data.step}",
            f"epoch {log_data.epoch}",
            f"loss {log_data.metrics['loss']:.4f}",
            f"grad_norm {log_data.metrics['grad_norm']:.4f}",
        ]
        for family in ("visual", "merger", "language"):
            key = f"{family}_lr"
            if key in log_data.metrics:
                parts.append(f"{key} {log_data.metrics[key]:.2e}")
        parts.extend(
            [
                f"single_samples {log_data.metrics['single_samples']:d}",
                f"total_samples {log_data.metrics['total_samples']:d}",
                f"mem {log_data.metrics['mem']:.2f} GiB",
                f"tps {log_data.metrics['tps']:.2f}({log_data.metrics['tps_per_gpu']:.2f}/gpu)",
                f"num_label_tokens {log_data.metrics['num_label_tokens']}",
            ]
        )
        logging.info(" | ".join(parts))
                
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config_path=None):
    """Main entry point for the fine-tuning recipe.

    Loads the configuration, sets up the trainer, and initiates the training loop.
    """
    if config_path is None:
        config_path = pathlib.Path(__file__).parent.resolve() / "gemma3" / "gemma3_vl_4b_cord_v2.yaml"
    cfg = parse_args_and_load_config(config_path)
    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
