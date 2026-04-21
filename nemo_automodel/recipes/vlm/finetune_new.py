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
import math
import pathlib
import time
from omegaconf import OmegaConf
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from nemo_automodel.components.loss.revision_tristate_loss import RevisionTriStateLoss
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

# Modified: to support multiple groups
def get_parameter_groups(model, cfg_opt):
    base_lr = cfg_opt.get("lr", 1e-5)
    visual_lr = cfg_opt.get("visual_lr", base_lr)
    language_lr = cfg_opt.get("language_lr", base_lr)
    merger_lr = cfg_opt.get("merger_lr", base_lr)
    weight_decay = cfg_opt.get("weight_decay", 0.0)

    groups = {
        "visual_decay": {
            "params": [],
            "lr": visual_lr,
            "max_lr": visual_lr,
            "min_lr": visual_lr * 0.1,
            "init_lr": visual_lr * 0.1,
            "weight_decay": weight_decay,
            "wd_mult": weight_decay,
        },
        "visual_no_decay": {
            "params": [],
            "lr": visual_lr,
            "max_lr": visual_lr,
            "min_lr": visual_lr * 0.1,
            "init_lr": visual_lr * 0.1,
            "weight_decay": 0.0,
            "wd_mult": 0.0,
        },
        "language_decay": {
            "params": [],
            "lr": language_lr,
            "max_lr": language_lr,
            "min_lr": language_lr * 0.05,
            "init_lr": language_lr * 0.1,
            "weight_decay": weight_decay,
            "wd_mult": weight_decay,
        },
        "language_no_decay": {
            "params": [],
            "lr": language_lr,
            "max_lr": language_lr,
            "min_lr": language_lr * 0.05,
            "init_lr": language_lr * 0.1,
            "weight_decay": 0.0,
            "wd_mult": 0.0,
        },
        "merger_decay": {
            "params": [],
            "lr": merger_lr,
            "max_lr": merger_lr,
            "min_lr": merger_lr * 0.05,
            "init_lr": merger_lr * 0.1,
            "weight_decay": weight_decay,
            "wd_mult": weight_decay,
        },
        "merger_no_decay": {
            "params": [],
            "lr": merger_lr,
            "max_lr": merger_lr,
            "min_lr": merger_lr * 0.05,
            "init_lr": merger_lr * 0.1,
            "weight_decay": 0.0,
            "wd_mult": 0.0,
        },
        "other": {
            "params": [],
            "lr": base_lr,
            "max_lr": base_lr,
            "min_lr": base_lr * 0.05,
            "init_lr": base_lr * 0.1,
            "weight_decay": weight_decay,
            "wd_mult": weight_decay,
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
    other_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 权重共享检测：如果这个物理参数已经分过组了，直接跳过
        if id(param) in seen_param_ids:
            logger.info(f"Skipping tied parameter: {name}")
            continue
        seen_param_ids.add(id(param))

        is_no_decay = any(k in name.lower() for k in no_decay_keywords) or (param.ndim <= 1)

        if "visual.merger" in name or "deepstack_merger_list" in name:
            if is_no_decay:
                groups["merger_no_decay"]["params"].append(param)
                merger_no_decay_names.append(name)
            else:
                groups["merger_decay"]["params"].append(param)
                merger_decay_names.append(name)    
        elif "visual" in name:
            if is_no_decay:
                groups["visual_no_decay"]["params"].append(param)
                visual_no_decay_names.append(name)
            else:
                groups["visual_decay"]["params"].append(param)
                visual_decay_names.append(name)    
        elif "language_model" in name or "lm_head" in name:
            if is_no_decay:
                groups["language_no_decay"]["params"].append(param)
                language_no_decay_names.append(name)    
            else:
                groups["language_decay"]["params"].append(param)
                language_decay_names.append(name)    
        else:
            groups["other"]["params"].append(param)
            other_names.append(name)

    # print("visual_decay_names:", visual_decay_names)
    # print("visual_no_decay_names:", visual_no_decay_names)
    # print("language_decay_names:", language_decay_names)
    # print("language_no_decay_names:", language_no_decay_names)
    # print("merger_decay_names:", merger_decay_names)
    # print("merger_no_decay_names:", merger_no_decay_names)
    # print("other_names:", other_names)

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

        # remove useless field
        if hasattr(cfg_opt, 'visual_lr'):
            delattr(cfg_opt, 'visual_lr')
        if hasattr(cfg_opt, 'merger_lr'):
            delattr(cfg_opt, 'merger_lr')

        optimizer = cfg_opt.instantiate(params=param_groups)

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
            processor_target = str(cfg_processor.get("_target_", ""))
            processor_name = cfg_processor.get("pretrained_model_name_or_path", None)
            if processor_target == "transformers.AutoProcessor.from_pretrained" and processor_name is None:
                processor = cfg_processor.instantiate(pretrained_model_name_or_path=pretrained_model_name_or_path)
            else:
                processor = cfg_processor.instantiate()
        elif cfg_processor is not None:
            processor_kwargs = cfg_processor.to_dict()
            processor_kwargs.setdefault("pretrained_model_name_or_path", pretrained_model_name_or_path)

        # If no processor was instantiated, try AutoProcessor
        if processor is None:
            try:
                resolved_processor_name = processor_kwargs.pop("pretrained_model_name_or_path", pretrained_model_name_or_path)
                processor = AutoProcessor.from_pretrained(resolved_processor_name, **processor_kwargs)
            except Exception as e:
                # Some models do not provide an AutoProcessor
                processor = None
                logging.warning(f"AutoProcessor not available for {resolved_processor_name} ({e}). ")

        with FirstRankPerNode():
            # ds = cfg_ds.instantiate(path_or_dataset=cfg_ds.path_or_dataset)
            ds_dict = {k: v for k, v in cfg_ds.__dict__.items() if not k.startswith('_')}
            ds = cfg_ds.instantiate(**ds_dict)

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
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds,
                **dist_sampler_kwargs,
            )

        collate_cfg = cfg_dl.get("collate_fn", None)
        if collate_cfg:
            collate_fn = collate_cfg.instantiate(processor=processor, max_len=cfg_ds.max_len)
        else:
            processor_type = type(processor).__name__
            if processor_type not in COLLATE_FNS:
                processor_type = "default"
                logging.warning(f"You are using {processor_type} with default collate function.")
            collate_fn = lambda examples: COLLATE_FNS[processor_type](examples, processor)

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


def _unwrap_model(model):
    base_model = model
    while True:
        if hasattr(base_model, "module"):
            base_model = base_model.module
            continue
        if hasattr(base_model, "_orig_mod"):
            base_model = base_model._orig_mod
            continue
        break
    return base_model


def _build_revision_attention_mask(
    prompt_lengths: torch.Tensor,
    response_padded_lengths: torch.Tensor,
    block_size: int,
    max_total_len: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size = prompt_lengths.shape[0]
    attention_mask = torch.zeros((batch_size, 1, max_total_len, max_total_len), dtype=torch.bool, device=device)
    for idx in range(batch_size):
        prompt_len = int(prompt_lengths[idx].item())
        response_len = int(response_padded_lengths[idx].item())
        seq_len = prompt_len + response_len
        if seq_len <= 0:
            continue

        block_ids = torch.empty((seq_len,), dtype=torch.long, device=device)
        if prompt_len > 0:
            block_ids[:prompt_len] = torch.arange(prompt_len, device=device) // block_size
        if response_len > 0:
            num_prompt_blocks = (prompt_len + block_size - 1) // block_size
            response_blocks = num_prompt_blocks + torch.arange(response_len, device=device) // block_size
            block_ids[prompt_len:seq_len] = response_blocks

        local_mask = block_ids.view(-1, 1) >= block_ids.view(1, -1)
        attention_mask[idx, 0, :seq_len, :seq_len] = local_mask
    return attention_mask


def _select_quantile_remask_mask(
    confidence_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    if quantile <= 0:
        return torch.zeros_like(candidate_mask, dtype=torch.bool)

    remask_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    for batch_idx in range(candidate_mask.shape[0]):
        indices = candidate_mask[batch_idx].nonzero(as_tuple=False).flatten()
        count = int(indices.numel())
        if count <= 1:
            continue
        # Keep at least one visible editable token when a sample has multiple candidates.
        k = min(max(int(math.ceil(count * quantile)), 1), count - 1)
        values = confidence_scores[batch_idx, indices]
        _, local_indices = torch.topk(values, k=k, largest=False)
        remask_mask[batch_idx, indices[local_indices]] = True
    return remask_mask


def _build_padded_select_indices(select_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if select_mask.ndim != 2:
        raise ValueError(f"`select_mask` must have shape [B, L], got {tuple(select_mask.shape)}")

    select_mask = select_mask.to(torch.bool)
    batch_size, seq_len = select_mask.shape
    counts = select_mask.sum(dim=1)
    max_selected = int(counts.max().item()) if counts.numel() > 0 else 0

    if max_selected == 0:
        return (
            torch.zeros((batch_size, 0), dtype=torch.long, device=select_mask.device),
            torch.zeros((batch_size, 0), dtype=torch.bool, device=select_mask.device),
        )

    positions = torch.arange(seq_len, device=select_mask.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    sentinel = torch.full_like(positions, fill_value=seq_len)
    sorted_positions, _ = torch.sort(torch.where(select_mask, positions, sentinel), dim=1)
    indices = sorted_positions[:, :max_selected].clamp_max(max(seq_len - 1, 0))
    valid_mask = torch.arange(max_selected, device=select_mask.device).unsqueeze(0) < counts.unsqueeze(1)
    return indices, valid_mask


def _gather_selected_tokens(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    valid_mask: torch.Tensor,
    fill_value: Any,
) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"`tensor` must have shape [B, L], got {tuple(tensor.shape)}")
    if indices.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError("`indices` and `valid_mask` must both have shape [B, K]")

    if indices.numel() == 0:
        return tensor[:, :0]

    gathered = torch.gather(tensor, dim=1, index=indices.to(device=tensor.device, dtype=torch.long))
    return gathered.masked_fill(~valid_mask.to(gathered.device), fill_value)


def _select_tokens_by_mask(
    select_mask: torch.Tensor,
    tensors_with_fill: dict[str, tuple[torch.Tensor, Any]],
    *,
    empty_error: str,
) -> dict[str, torch.Tensor]:
    indices, valid_mask = _build_padded_select_indices(select_mask)
    if indices.shape[1] == 0:
        raise ValueError(empty_error)

    selected_tensors = {"indices": indices, "valid_mask": valid_mask}
    for name, (tensor, fill_value) in tensors_with_fill.items():
        selected_tensors[name] = _gather_selected_tokens(tensor, indices, valid_mask, fill_value)
    return selected_tensors


def _ensure_context_parallel_disabled(device_mesh, feature_name: str) -> None:
    if (
        device_mesh is not None
        and "cp" in getattr(device_mesh, "mesh_dim_names", ())
        and device_mesh["cp"].size() > 1
    ):
        raise NotImplementedError(f"{feature_name} does not support context parallel yet")


def _resolve_num_samples(batch: dict[str, Any]) -> int:
    num_samples = batch.get("num_samples", None)
    if num_samples is None:
        return int(batch["input_ids"].shape[0])
    if isinstance(num_samples, torch.Tensor):
        return int(num_samples.sum().item())
    return int(num_samples)


def _compute_revision_ce_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_loss_mask: torch.Tensor,
    edit_loss_mask: torch.Tensor,
    keep_loss_mask: torch.Tensor,
    fp32_upcast: bool = True,
) -> dict[str, torch.Tensor]:
    if fp32_upcast:
        logits = logits.float()

    vocab_size = logits.shape[-1]
    per_token_ce = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        reduction="none",
    ).view_as(labels)

    def _sum(mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=per_token_ce.dtype)
        return (per_token_ce * mask).sum()

    return {
        "mask_ce_sum": _sum(mask_loss_mask),
        "edit_ce_sum": _sum(edit_loss_mask),
        "keep_ce_sum": _sum(keep_loss_mask),
    }


def _compute_revision_accuracy_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_loss_mask: torch.Tensor,
    edit_loss_mask: torch.Tensor,
    keep_loss_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(labels)

    def _sum(mask: torch.Tensor) -> torch.Tensor:
        return (correct & mask.to(torch.bool)).sum(dtype=torch.long)

    return {
        "mask_correct": _sum(mask_loss_mask),
        "edit_correct": _sum(edit_loss_mask),
        "keep_correct": _sum(keep_loss_mask),
    }


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
    elif isinstance(loss_fn, RevisionTriStateLoss):
        loss_fn_kwargs.update(
            {
                "logits": kwargs.pop("logits"),
                "labels": kwargs.pop("labels"),
                "draft_input_ids": kwargs.pop("draft_input_ids"),
                "mask_loss_mask": kwargs.pop("mask_loss_mask"),
                "edit_loss_mask": kwargs.pop("edit_loss_mask"),
                "keep_loss_mask": kwargs.pop("keep_loss_mask"),
                "num_label_tokens": kwargs.pop("num_label_tokens", None),
                "num_samples": kwargs.pop("num_samples", None),
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
        self.revision_training_cfg = self.cfg.get("revision_training", None)
        self.revision_enabled = bool(
            self.revision_training_cfg is not None and self.revision_training_cfg.get("enabled", False)
        )
        self.revision_loss_fn = None
        if self.revision_enabled:
            loss_cfg = self.revision_training_cfg.get("loss", {})
            self.revision_loss_fn = RevisionTriStateLoss(
                mask_weight=float(loss_cfg.get("mask_weight", 1.0)),
                edit_weight=float(loss_cfg.get("edit_weight", 1.0)),
                keep_weight=float(loss_cfg.get("keep_weight", 0.05)),
            )
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

        self.dataloader, self.processor = build_dataloader(
            self.cfg.dataset,
            self.cfg.dataloader,
            _get_model_name(self.cfg.model),
            self.cfg.get("processor", None),
            device_mesh=self.device_mesh,
            seed=self.cfg.get("seed", 42),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
        )
        collate_cfg = self.cfg.dataloader.get("collate_fn", None)
        self.mask_token_id = int(getattr(collate_cfg, "mask_token_id", 151671)) if collate_cfg is not None else 151671
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if getattr(tokenizer, "pad_token_id", None) is not None:
            self.pad_token_id = int(tokenizer.pad_token_id)
        else:
            self.pad_token_id = int(tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0])

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

    def _resolve_revision_task(self, step: int, total_steps: int) -> str:
        if not self.revision_enabled:
            return "mask"

        warmup_ratio = float(self.revision_training_cfg.get("warmup_ratio", 1.0))
        warmup_steps = max(0, int(total_steps * warmup_ratio))
        if step < warmup_steps:
            return "mask"

        task_pattern = list(self.revision_training_cfg.get("task_pattern", ["revision", "revision", "revision", "mask"]))
        if len(task_pattern) == 0:
            return "revision"
        task = str(task_pattern[(step - warmup_steps) % len(task_pattern)]).strip().lower()
        return task if task in {"mask", "revision"} else "mask"

    def _get_seed_mask_ratio_range(self) -> tuple[float, float]:
        if not self.revision_enabled:
            return 0.15, 0.50
        seed_cfg = self.revision_training_cfg.get("seed", {})
        low = float(seed_cfg.get("mask_ratio_min", self.revision_training_cfg.get("seed_mask_ratio_min", 0.15)))
        high = float(seed_cfg.get("mask_ratio_max", self.revision_training_cfg.get("seed_mask_ratio_max", 0.50)))
        if low > high:
            raise ValueError(f"seed mask ratio range is invalid: min={low} max={high}")
        return low, high

    def _get_remask_quantile(self) -> float:
        if not self.revision_enabled:
            return 0.0
        remask_cfg = self.revision_training_cfg.get("remask", {})
        quantile = float(remask_cfg.get("quantile", 0.25))
        return min(max(quantile, 0.0), 1.0)

    def _sample_global_weighted_int(self, values: list[int], probs: list[float]) -> int:
        if len(values) == 0:
            raise ValueError("`values` must not be empty for weighted sampling.")
        if len(values) != len(probs):
            raise ValueError(f"`values` and `probs` must have the same length, got {len(values)} and {len(probs)}.")

        prob_tensor = torch.tensor(probs, dtype=torch.float32)
        if torch.any(prob_tensor < 0):
            raise ValueError(f"`probs` must be non-negative, got {probs}.")
        prob_sum = float(prob_tensor.sum().item())
        if prob_sum <= 0:
            raise ValueError(f"`probs` must sum to a positive value, got {probs}.")
        prob_tensor = prob_tensor / prob_sum

        broadcast_device = torch.device("cpu")
        if dist.is_available() and dist.is_initialized() and dist.get_backend() == "nccl":
            broadcast_device = self.dist_env.device

        sampled_value = torch.empty((), dtype=torch.long, device=broadcast_device)
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            sampled_idx = int(torch.multinomial(prob_tensor, num_samples=1).item())
            sampled_value.fill_(int(values[sampled_idx]))

        if dist.is_available() and dist.is_initialized():
            dist.broadcast(sampled_value, src=0)
        return int(sampled_value.item())

    def _resolve_revision_draft_steps(self, step: int) -> int:
        if not self.revision_enabled:
            return 1

        rollout_cfg = self.revision_training_cfg.get("rollout", {})
        default_steps = max(int(rollout_cfg.get("draft_steps", 1)), 1)
        schedule_cfg = rollout_cfg.get("draft_steps_schedule", None)
        if schedule_cfg is None:
            return default_steps

        start_step = int(schedule_cfg.get("start_step", 0))
        if step < start_step:
            return default_steps

        values = [max(int(v), 1) for v in schedule_cfg.get("values", [default_steps])]
        probs = schedule_cfg.get("probs", None)
        if probs is None:
            probs = [1.0 / len(values)] * len(values)
        probs = [float(p) for p in probs]
        return self._sample_global_weighted_int(values, probs)

    def _build_revision_seed_batch(self, batch, labels):
        if (
            self.device_mesh is not None
            and "cp" in getattr(self.device_mesh, "mesh_dim_names", ())
            and self.device_mesh["cp"].size() > 1
        ):
            raise NotImplementedError("revision training does not support context parallel yet")

        prefix_lengths = batch["prefix_lengths"]
        response_lengths = batch["response_lengths"]
        response_padded_lengths = batch["response_padded_lengths"]
        block_size = int(batch.get("block_size", 4))
        batch_size = labels.shape[0]
        max_prompt_len = int(prefix_lengths.max().item())
        max_response_len = int(response_padded_lengths.max().item())
        max_total_len = max_prompt_len + max_response_len
        low, high = self._get_seed_mask_ratio_range()

        seed_input_ids = torch.full(
            (batch_size, max_total_len),
            fill_value=self.pad_token_id,
            dtype=batch["input_ids"].dtype,
            device=batch["input_ids"].device,
        )
        seed_labels = torch.full_like(seed_input_ids, fill_value=self.pad_token_id)
        seed_position_ids = torch.zeros(
            (batch["position_ids"].shape[0], batch_size, max_total_len),
            dtype=batch["position_ids"].dtype,
            device=batch["position_ids"].device,
        )
        response_mask = torch.zeros((batch_size, max_total_len), dtype=torch.bool, device=batch["input_ids"].device)
        editable_mask = torch.zeros_like(response_mask)
        sequence_lengths = torch.zeros((batch_size,), dtype=torch.long, device=batch["input_ids"].device)

        for sample_idx in range(batch_size):
            prefix_len = int(prefix_lengths[sample_idx].item())
            response_len = int(response_lengths[sample_idx].item())
            response_padded_len = int(response_padded_lengths[sample_idx].item())
            total_len = prefix_len + response_padded_len
            sequence_lengths[sample_idx] = total_len

            prompt_ids = batch["input_ids"][sample_idx, :prefix_len]
            clean_response_ids = labels[sample_idx, prefix_len: prefix_len + response_padded_len].clone()
            seed_response_ids = clean_response_ids.clone()

            for block_start in range(0, response_len, block_size):
                valid_tokens = min(block_size, response_len - block_start)
                if valid_tokens <= 0:
                    continue
                ratio = float((low + torch.rand((), device=batch["input_ids"].device) * (high - low)).item())
                num_masked = int(round(ratio * valid_tokens))
                if valid_tokens > 1:
                    num_masked = min(max(num_masked, 1), valid_tokens - 1)
                else:
                    num_masked = 1
                local_indices = torch.randperm(valid_tokens, device=batch["input_ids"].device)[:num_masked]
                mask_positions = block_start + local_indices
                seed_response_ids[mask_positions] = self.mask_token_id
                editable_mask[sample_idx, prefix_len + mask_positions] = True

            seed_input_ids[sample_idx, :prefix_len] = prompt_ids
            seed_input_ids[sample_idx, prefix_len:total_len] = seed_response_ids
            seed_labels[sample_idx, :prefix_len] = prompt_ids
            seed_labels[sample_idx, prefix_len:total_len] = clean_response_ids
            seed_position_ids[:, sample_idx, :total_len] = batch["position_ids"][:, sample_idx, :total_len]
            response_mask[sample_idx, prefix_len: prefix_len + response_len] = True

        seed_attention_mask = _build_revision_attention_mask(
            prompt_lengths=prefix_lengths,
            response_padded_lengths=response_padded_lengths,
            block_size=block_size,
            max_total_len=max_total_len,
            device=batch["input_ids"].device,
        )

        model_inputs = {
            "input_ids": seed_input_ids,
            "attention_mask": seed_attention_mask,
            "position_ids": seed_position_ids.to(torch.int64),
        }
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if key in batch:
                model_inputs[key] = batch[key]

        return {
            "model_inputs": model_inputs,
            "labels": seed_labels,
            "response_mask": response_mask,
            "editable_mask": editable_mask,
            "sequence_lengths": sequence_lengths,
            "prompt_lengths": prefix_lengths,
            "response_lengths": response_lengths,
            "response_padded_lengths": response_padded_lengths,
            "block_size": block_size,
        }

    def _build_revision_second_pass_batch(self, seed_batch, draft_result):
        quantile = self._get_remask_quantile()
        draft_input_ids = draft_result["input_ids"].clone()
        confidence_scores = draft_result["confidence_scores"]
        candidate_mask = seed_batch["editable_mask"] & seed_batch["response_mask"]
        remask_mask = _select_quantile_remask_mask(confidence_scores, candidate_mask, quantile)
        if remask_mask.any():
            draft_input_ids = torch.where(
                remask_mask,
                torch.full_like(draft_input_ids, fill_value=self.mask_token_id),
                draft_input_ids,
            )

        labels = seed_batch["labels"]
        visible_editable_mask = candidate_mask & (~remask_mask)
        mask_loss_mask = remask_mask & seed_batch["response_mask"]
        edit_loss_mask = visible_editable_mask & draft_input_ids.ne(labels)
        keep_loss_mask = visible_editable_mask & draft_input_ids.eq(labels)

        model_inputs = dict(seed_batch["model_inputs"])
        model_inputs["input_ids"] = draft_input_ids

        counts = {
            "response_tokens": seed_batch["response_mask"].sum(),
            "mask_tokens": mask_loss_mask.sum(),
            "edit_tokens": edit_loss_mask.sum(),
            "keep_tokens": keep_loss_mask.sum(),
        }
        return {
            "model_inputs": model_inputs,
            "labels": labels,
            "draft_input_ids": draft_input_ids,
            "mask_loss_mask": mask_loss_mask,
            "edit_loss_mask": edit_loss_mask,
            "keep_loss_mask": keep_loss_mask,
            "counts": counts,
            "sequence_lengths": seed_batch["sequence_lengths"],
        }

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

                task = self._resolve_revision_task(cur_step, total_steps)
                log_data = self._run_train_optim_step(batches, self.max_grad_norm, task=task)
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

    def _run_train_optim_step(self, batches, max_grad_norm: Optional[float] = None, task: str = "mask"):
        """Execute a single training step.

        Args:
            batches: List of batches of training data.
            max_grad_norm: Gradient clipping norm. Optional, if None will not clip gradients.
        """
        global_batch_size = torch.tensor(len(batches))
        global_batch_size = self._dp_allreduce(global_batch_size).item()

        # 单节点一次iteration的样本数
        num_samples = torch.tensor(
            sum((batch["num_samples"]).sum().item() for batch in batches), dtype=torch.long
        )
        # 全部节点一次iteration时的样本总数
        num_total_samples = self._dp_allreduce(num_samples).item()
        num_processes = dist.get_world_size()

        loss_buffer = []
        draft_candidate_tokens = torch.tensor(0, dtype=torch.long, device=self.dist_env.device)
        draft_correct_tokens = torch.tensor(0, dtype=torch.long, device=self.dist_env.device)
        revision_response_tokens = torch.tensor(0, dtype=torch.long, device=self.dist_env.device)
        revision_mask_tokens = torch.tensor(0, dtype=torch.long, device=self.dist_env.device)
        revision_edit_tokens = torch.tensor(0, dtype=torch.long, device=self.dist_env.device)
        revision_keep_tokens = torch.tensor(0, dtype=torch.long, device=self.dist_env.device)
        revision_mask_correct = torch.tensor(0, dtype=torch.long, device=self.dist_env.device)
        revision_edit_correct = torch.tensor(0, dtype=torch.long, device=self.dist_env.device)
        revision_keep_correct = torch.tensor(0, dtype=torch.long, device=self.dist_env.device)
        revision_mask_ce_sum = torch.tensor(0.0, dtype=torch.float32, device=self.dist_env.device)
        revision_edit_ce_sum = torch.tensor(0.0, dtype=torch.float32, device=self.dist_env.device)
        revision_keep_ce_sum = torch.tensor(0.0, dtype=torch.float32, device=self.dist_env.device)
        num_label_tokens_local = 0

        # number of tokens in the batch, excluding any tail padding.
        num_tokens_in_batch_local = 0
        revision_draft_steps = self._resolve_revision_draft_steps(self.step_scheduler.step) if task == "revision" else 0

        num_batches = len(batches)
        for i, batch in enumerate(batches): # accumulation_steps维度迭代
            # batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
            batch = to_device(batch, self.dist_env.device)
            labels = batch.pop("labels")

            with (
                get_sync_ctx(
                    self.model,
                    i == num_batches - 1,
                    defer_fsdp_grad_sync=getattr(self.model_wrapper, "defer_fsdp_grad_sync", True),
                ),
            ):
                if task == "revision":
                    if self.revision_loss_fn is None:
                        raise ValueError("`revision_training.enabled=true` requires a revision loss function.")

                    seed_batch = self._build_revision_seed_batch(batch, labels)
                    rollout_model = _unwrap_model(self.model)
                    with torch.no_grad():
                        draft_result = rollout_model.rollout_revision_draft(
                            response_mask=seed_batch["response_mask"],
                            editable_mask=seed_batch["editable_mask"],
                            block_size=seed_batch["block_size"],
                            draft_steps=revision_draft_steps,
                            temperature=float(self.revision_training_cfg.get("rollout", {}).get("temperature", 0.0)),
                            top_k=int(self.revision_training_cfg.get("rollout", {}).get("top_k", 0)),
                            top_p=float(self.revision_training_cfg.get("rollout", {}).get("top_p", 1.0)),
                            confidence_metric=str(
                                self.revision_training_cfg.get("rollout", {}).get("confidence_metric", "margin")
                            ),
                            forward_model=self.model,
                            **seed_batch["model_inputs"],
                        )
                    draft_candidate_mask = seed_batch["editable_mask"] & seed_batch["response_mask"]
                    draft_candidate_tokens += draft_candidate_mask.sum()
                    draft_correct_tokens += (
                        draft_result["input_ids"].eq(seed_batch["labels"]) & draft_candidate_mask
                    ).sum(dtype=torch.long)
                    revision_batch = self._build_revision_second_pass_batch(seed_batch, draft_result)
                    forward_batch = revision_batch["model_inputs"]
                    num_tokens_in_batch_local += int(revision_batch["sequence_lengths"].sum().item())
                    num_label_tokens_local += int(
                        (
                            revision_batch["mask_loss_mask"].sum()
                            + revision_batch["edit_loss_mask"].sum()
                            + revision_batch["keep_loss_mask"].sum()
                        ).item()
                    )
                    revision_response_tokens += revision_batch["counts"]["response_tokens"]
                    revision_mask_tokens += revision_batch["counts"]["mask_tokens"]
                    revision_edit_tokens += revision_batch["counts"]["edit_tokens"]
                    revision_keep_tokens += revision_batch["counts"]["keep_tokens"]

                    selected_revision = _select_tokens_by_mask(
                        revision_batch["mask_loss_mask"]
                        | revision_batch["edit_loss_mask"]
                        | revision_batch["keep_loss_mask"],
                        {
                            "labels": (revision_batch["labels"], 0),
                            "draft_input_ids": (revision_batch["draft_input_ids"], 0),
                            "mask_loss_mask": (revision_batch["mask_loss_mask"], False),
                            "edit_loss_mask": (revision_batch["edit_loss_mask"], False),
                            "keep_loss_mask": (revision_batch["keep_loss_mask"], False),
                        },
                        empty_error="No revision tokens available for selective logits.",
                    )

                    out = self.model(logits_to_keep=selected_revision["indices"], **forward_batch)
                    logits = getattr(out, "logits", out)
                    local_loss = calculate_loss(
                        self.revision_loss_fn,
                        logits=logits,
                        labels=selected_revision["labels"],
                        draft_input_ids=selected_revision["draft_input_ids"],
                        mask_loss_mask=selected_revision["mask_loss_mask"],
                        edit_loss_mask=selected_revision["edit_loss_mask"],
                        keep_loss_mask=selected_revision["keep_loss_mask"],
                        num_label_tokens=None,
                        num_samples=num_total_samples,
                    )
                    revision_ce_stats = _compute_revision_ce_stats(
                        logits=logits.detach(),
                        labels=selected_revision["labels"],
                        mask_loss_mask=selected_revision["mask_loss_mask"],
                        edit_loss_mask=selected_revision["edit_loss_mask"],
                        keep_loss_mask=selected_revision["keep_loss_mask"],
                        fp32_upcast=getattr(self.revision_loss_fn, "fp32_upcast", True),
                    )
                    revision_accuracy_stats = _compute_revision_accuracy_stats(
                        logits=logits.detach(),
                        labels=selected_revision["labels"],
                        mask_loss_mask=selected_revision["mask_loss_mask"],
                        edit_loss_mask=selected_revision["edit_loss_mask"],
                        keep_loss_mask=selected_revision["keep_loss_mask"],
                    )
                    revision_mask_correct += revision_accuracy_stats["mask_correct"]
                    revision_edit_correct += revision_accuracy_stats["edit_correct"]
                    revision_keep_correct += revision_accuracy_stats["keep_correct"]
                    revision_mask_ce_sum += revision_ce_stats["mask_ce_sum"]
                    revision_edit_ce_sum += revision_ce_stats["edit_ce_sum"]
                    revision_keep_ce_sum += revision_ce_stats["keep_ce_sum"]
                else:
                    if 'loss_mask' in batch:
                        num_label_tokens_local += int(batch["loss_mask"].sum().item())
                    else:
                        num_label_tokens_local += int((labels != -100).sum().item())
                    num_tokens_in_batch_local += int(labels.numel() - count_tail_padding(labels))

                    use_selected_response_logits = isinstance(
                        self.loss_fn,
                        (MixturePathGeneralizeKL, WeightedCrossEntropy),
                    )
                    if use_selected_response_logits:
                        _ensure_context_parallel_disabled(self.device_mesh, "Selective noisy-response logits")

                    train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch, labels) # local_batch_size维度迭代
                    with train_ctx():
                        selected_response = None
                        if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                            # use num_logits_to_keep to avoid full logits matrix in memory
                            out = self.model(logits_to_keep=1, **batch)
                            if "hidden_states" not in out:
                                raise ValueError(
                                    "FusedLinearCrossEntropy requires the model to output hidden states. Set `model.output_hidden_states=True` in the config."
                                )
                        else:
                            if use_selected_response_logits:
                                selected_response = _select_tokens_by_mask(
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
                                out = self.model(logits_to_keep=selected_response["indices"], **batch)
                            else:
                                out = self.model(**batch)

                        local_loss = calculate_loss(
                            self.loss_fn,
                            logits=getattr(out, "logits", out),
                            labels=selected_response["labels"] if selected_response is not None else labels,
                            model=self.model,
                            hidden_states=out.hidden_states[-1] if getattr(out, "hidden_states", None) is not None else None,
                            num_label_tokens=None,
                            x_t=selected_response["input_ids"] if selected_response is not None else batch.get("input_ids", None),
                            t=selected_response["t"] if selected_response is not None else batch.get("t", None),
                            loss_mask=selected_response["loss_mask"] if selected_response is not None else batch.get("loss_mask", None),
                            response_mask=selected_response["response_mask"] if selected_response is not None else batch.get("response_mask", None),
                            block_size=batch.get("block_size") if "block_size" in batch else 0, # for block diffusion.
                            num_samples=num_total_samples,
                        )
                loss_buffer.append(local_loss.clone().detach())
                local_loss.backward()

        num_label_tokens = self._dp_allreduce(torch.tensor(num_label_tokens_local, dtype=torch.long, device=self.dist_env.device)).item()
        num_tokens_in_batch = self._dp_allreduce(
            torch.tensor(num_tokens_in_batch_local, dtype=torch.long, device=self.dist_env.device)
        ).item()
        draft_candidate_tokens = self._dp_allreduce(draft_candidate_tokens).item()
        draft_correct_tokens = self._dp_allreduce(draft_correct_tokens).item()
        revision_response_tokens = self._dp_allreduce(revision_response_tokens).item()
        revision_mask_tokens = self._dp_allreduce(revision_mask_tokens).item()
        revision_edit_tokens = self._dp_allreduce(revision_edit_tokens).item()
        revision_keep_tokens = self._dp_allreduce(revision_keep_tokens).item()
        revision_mask_correct = self._dp_allreduce(revision_mask_correct).item()
        revision_edit_correct = self._dp_allreduce(revision_edit_correct).item()
        revision_keep_correct = self._dp_allreduce(revision_keep_correct).item()
        revision_mask_ce_sum = self._dp_allreduce(revision_mask_ce_sum, include_cp=True).item()
        revision_edit_ce_sum = self._dp_allreduce(revision_edit_ce_sum, include_cp=True).item()
        revision_keep_ce_sum = self._dp_allreduce(revision_keep_ce_sum, include_cp=True).item()
        draft_acc = draft_correct_tokens / max(draft_candidate_tokens, 1) if task == "revision" else 0.0
        revision_mask_acc = revision_mask_correct / max(revision_mask_tokens, 1) if task == "revision" else 0.0
        revision_edit_acc = revision_edit_correct / max(revision_edit_tokens, 1) if task == "revision" else 0.0
        revision_keep_acc = revision_keep_correct / max(revision_keep_tokens, 1) if task == "revision" else 0.0
        revision_mask_ce = revision_mask_ce_sum / max(revision_mask_tokens, 1) if task == "revision" else 0.0
        revision_edit_ce = revision_edit_ce_sum / max(revision_edit_tokens, 1) if task == "revision" else 0.0
        revision_keep_ce = revision_keep_ce_sum / max(revision_keep_tokens, 1) if task == "revision" else 0.0

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

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "loss": reporting_loss,
                "grad_norm": grad_norm,
                # "lr": self.optimizer.param_groups[0]["lr"],
                "visual_lr": self.optimizer.param_groups[0]["lr"],      # HARDCODE HERE, be careful!
                "merger_lr": self.optimizer.param_groups[4]["lr"],      # HARDCODE HERE, be careful!
                "language_lr": self.optimizer.param_groups[2]["lr"],    # HARDCODE HERE, be careful!
                "single_samples": num_samples,
                "total_samples": num_total_samples, # total train samples in each iteration
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
                "tps": tps,
                "tps_per_gpu": tps / max(self._get_dp_group_size(), 1),
                "num_tokens_per_step": num_tokens_in_batch,
                "num_label_tokens": num_label_tokens,
                "is_revision_step": 1 if task == "revision" else 0,
                "revision_draft_steps": revision_draft_steps if task == "revision" else 0,
                "draft_acc": draft_acc,
                "revision_mask_ratio": (
                    revision_mask_tokens / max(revision_response_tokens, 1) if task == "revision" else 0.0
                ),
                "revision_edit_ratio": (
                    revision_edit_tokens / max(revision_response_tokens, 1) if task == "revision" else 0.0
                ),
                "revision_keep_ratio": (
                    revision_keep_tokens / max(revision_response_tokens, 1) if task == "revision" else 0.0
                ),
                "revision_mask_acc": revision_mask_acc,
                "revision_edit_acc": revision_edit_acc,
                "revision_keep_acc": revision_keep_acc,
                "revision_mask_ce": revision_mask_ce,
                "revision_edit_ce": revision_edit_ce,
                "revision_keep_ce": revision_keep_ce,
            },
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
                num_label_tokens = int(batch["loss_mask"].sum().item()) if "loss_mask" in batch else int((labels != -100).sum().item())

                use_selected_response_logits = isinstance(
                    self.loss_fn,
                    (MixturePathGeneralizeKL, WeightedCrossEntropy),
                )
                if use_selected_response_logits:
                    _ensure_context_parallel_disabled(self.device_mesh, "Selective noisy-response logits")

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
                    selected_response = None
                    if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                        out = self.model(logits_to_keep=1, **batch)
                    else:
                        if use_selected_response_logits:
                            selected_response = _select_tokens_by_mask(
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
                            out = self.model(logits_to_keep=selected_response["indices"], **batch)
                        else:
                            out = self.model(**batch)
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
                        num_samples=_resolve_num_samples(batch),
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

        log_parts = [
            f"step {log_data.step}",
            f"epoch {log_data.epoch}",
            f"loss {log_data.metrics['loss']:.4f}",
            f"grad_norm {log_data.metrics['grad_norm']:.4f}",
            f"visual_lr {log_data.metrics['visual_lr']:.2e}",
            f"merger_lr {log_data.metrics['merger_lr']:.2e}",
            f"language_lr {log_data.metrics['language_lr']:.2e}",
            f"single_samples {log_data.metrics['single_samples']:d}",
            f"total_samples {log_data.metrics['total_samples']:d}",
            f"mem {log_data.metrics['mem']:.2f} GiB",
            f"tps {log_data.metrics['tps']:.2f}({log_data.metrics['tps_per_gpu']:.2f}/gpu)",
            f"num_label_tokens {log_data.metrics['num_label_tokens']}",
        ]
        if log_data.metrics.get("is_revision_step", 0):
            log_parts.extend(
                [
                    f"draft_steps {int(log_data.metrics['revision_draft_steps'])}",
                    f"draft_acc {log_data.metrics['draft_acc']:.3f}",
                    "revision_ratio(m/e/k) "
                    f"{log_data.metrics['revision_mask_ratio']:.3f}/"
                    f"{log_data.metrics['revision_edit_ratio']:.3f}/"
                    f"{log_data.metrics['revision_keep_ratio']:.3f}",
                    "revision_acc(m/e/k) "
                    f"{log_data.metrics['revision_mask_acc']:.3f}/"
                    f"{log_data.metrics['revision_edit_acc']:.3f}/"
                    f"{log_data.metrics['revision_keep_acc']:.3f}",
                    "revision_ce(m/e/k) "
                    f"{log_data.metrics['revision_mask_ce']:.4f}/"
                    f"{log_data.metrics['revision_edit_ce']:.4f}/"
                    f"{log_data.metrics['revision_keep_ce']:.4f}",
                ]
            )
        logging.info(" | ".join(log_parts))
                
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
