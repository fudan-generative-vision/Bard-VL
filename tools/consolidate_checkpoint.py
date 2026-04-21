from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from collections import OrderedDict
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch


DTYPE_CHOICES = {
    "float32": "float32",
    "float": "float32",
    "fp32": "float32",
    "float16": "float16",
    "half": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}

def load_dcp(ckpt_dir: Path | str) -> tuple[dict, dict]:
    """Loads a DCP checkpoint in a state dictionary from a directory."""
    import torch
    import torch.distributed.checkpoint as dcp

    if not isinstance(ckpt_dir, Path):
        ckpt_dir = Path(ckpt_dir)
    fs_reader = dcp.FileSystemReader(ckpt_dir)
    metadata = fs_reader.read_metadata()

    # Load tensor data
    tensor_state_dict = {
        k: torch.empty(tp.size, dtype=tp.properties.dtype)
        for k, tp in metadata.state_dict_metadata.items()
        if type(tp).__name__ == "TensorStorageMetadata"
    }

    if tensor_state_dict:
        dcp.load(tensor_state_dict, storage_reader=fs_reader)

    # Load scheduler data
    sched_keys = [k for k, tp in metadata.state_dict_metadata.items() if "sched" in k]

    sched_state_dict = {}
    if sched_keys:
        sched_state_dict = {k: None for k in sched_keys}
        try:
            dcp.load(sched_state_dict, storage_reader=fs_reader)
        except Exception:
            sched_state_dict = {}

    return tensor_state_dict, sched_state_dict

def load_safetensors(ckpt_dir: Path | str) -> dict[str, torch.Tensor]:
    """
    Loads a safetensors checkpoint in a state dictionary from a directory.
    """
    from safetensors import safe_open

    state_dict = {}
    if not isinstance(ckpt_dir, Path):
        ckpt_dir = Path(ckpt_dir)
    with safe_open(ckpt_dir, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict

def convert_nemo_dcp_to_safetensors(
    model_dict, 
    output_dir, 
    max_shard_size_gb=5,
    target_dtype=None,
):
    import torch
    from safetensors.torch import save_file

    if target_dtype is None:
        target_dtype = torch.bfloat16

    print(f"Step 1: Mapping keys and converting to {target_dtype}...")
    state_dict = OrderedDict()
    for k, v in model_dict.items():
        if isinstance(v, torch.Tensor):
            state_dict[k] = v.to(target_dtype).cpu().contiguous()
        else:
            print(k)
            continue

    print(f"Step 2: Sharding and saving (Max {max_shard_size_gb}GB per shard)...")

    # 首先计算是否需要分片
    total_size_bytes = sum(t.nelement() * t.element_size() for t in state_dict.values())
    needs_sharding = total_size_bytes > max_shard_size_gb * 1024**3

    if not needs_sharding:
        # 情况 A: 模型较小，直接保存为单文件
        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(state_dict, save_path)
        print(f"Conversion Finished! Saved as single file: {save_path}")
        return state_dict

    # 情况 B: 模型较大，执行原有的分片逻辑
    current_shard = {}
    current_size = 0
    shard_count = 0 # 从 0 开始，方便循环内统一处理
    weight_map = {}

    for key, tensor in state_dict.items():
        tensor_size = tensor.nelement() * tensor.element_size()

        if current_size + tensor_size > max_shard_size_gb * 1024**3 and current_shard:
            shard_count += 1
            shard_name = f"model-{shard_count:05d}-of-index.safetensors"
            save_path = os.path.join(output_dir, shard_name)
            save_file(current_shard, save_path)

            for k in current_shard.keys():
                weight_map[k] = shard_name

            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    # 保存最后一个分片
    if current_shard:
        shard_count += 1
        temp_name = f"model-{shard_count:05d}-of-index.safetensors"
        save_file(current_shard, os.path.join(output_dir, temp_name))
        for k in current_shard.keys():
            weight_map[k] = temp_name

    print("Step 3: Generating index.json...")
    final_shard_count = shard_count
    actual_weight_map = {}
    total_str = f"{final_shard_count:05d}"

    # 这里的rename逻辑仅在needs_sharding为True时执行
    for k, v in weight_map.items():
        final_name = v.replace("index", total_str)
        actual_weight_map[k] = final_name

        old_path = os.path.join(output_dir, v)
        new_path = os.path.join(output_dir, final_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)

    index_data = {
        "metadata": {"total_size": total_size_bytes},
        "weight_map": actual_weight_map
    }

    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"Conversion Finished! Saved to: {output_dir}")
    return state_dict


def parse_dtype(dtype_name: str) -> torch.dtype:
    import torch

    key = dtype_name.strip().lower()
    if key not in DTYPE_CHOICES:
        supported = ", ".join(sorted(DTYPE_CHOICES))
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Expected one of: {supported}")
    return getattr(torch, DTYPE_CHOICES[key])


def copy_model_configs(source_model_dir: Path, output_dir: Path) -> None:
    if not source_model_dir.exists():
        raise FileNotFoundError(f"Source model directory does not exist: {source_model_dir}")

    patterns = ("*.json", "*.txt")
    copied_any = False
    for pattern in patterns:
        for src_path in source_model_dir.glob(pattern):
            shutil.copy2(src_path, output_dir / src_path.name)
            copied_any = True

    if not copied_any:
        raise FileNotFoundError(
            f"No tokenizer/config (*.json, *.txt) were found under: {source_model_dir}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a NeMo DCP checkpoint directory into Hugging Face-compatible safetensors shards."
    )
    parser.add_argument(
        "--dcp-dir",
        required=True,
        help="Path to the DCP checkpoint directory, typically the `model/` subdirectory under an experiment checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the converted safetensors checkpoint will be written.",
    )
    parser.add_argument(
        "--source-model-dir",
        default=None,
        help="Optional directory containing tokenizer/config to copy into the output directory.",
    )
    parser.add_argument(
        "--max-shard-size-gb",
        type=float,
        default=5.0,
        help="Maximum size of each safetensors shard in GB. Default: 5.0",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Target dtype for converted weights. Supported: fp32, fp16, bf16. Default: bfloat16",
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Do not copy tokenizer/config from `--source-model-dir`.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dcp_path = Path(args.dcp_dir)
    output_path = Path(args.output_dir)
    target_dtype = parse_dtype(args.dtype)

    if not dcp_path.exists():
        raise FileNotFoundError(f"DCP checkpoint directory does not exist: {dcp_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    if not args.skip_copy:
        if args.source_model_dir is None:
            raise ValueError("--source-model-dir is required unless --skip-copy is set.")
        copy_model_configs(Path(args.source_model_dir), output_path)

    print(f"Loading DCP from {dcp_path}...")
    restored_model_dict, _ = load_dcp(dcp_path)

    final_state_dict = convert_nemo_dcp_to_safetensors(
        model_dict=restored_model_dict,
        output_dir=str(output_path),
        max_shard_size_gb=args.max_shard_size_gb,
        target_dtype=target_dtype,
    )

    total_params = sum(p.numel() for p in final_state_dict.values())
    print(f"Total Parameters: {total_params / 1e9:.2f}B")

if __name__ == "__main__":
    main()
