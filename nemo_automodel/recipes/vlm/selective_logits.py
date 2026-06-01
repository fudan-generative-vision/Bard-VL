from __future__ import annotations

from typing import Any

import torch


def _select_idx(select_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if select_mask.ndim != 2:
        raise ValueError(f"`select_mask` must have shape [B, L], got {tuple(select_mask.shape)}")

    select_mask = select_mask.to(dtype=torch.bool)
    batch_size, _ = select_mask.shape
    counts = select_mask.sum(dim=1)
    max_selected = int(counts.max().item()) if counts.numel() > 0 else 0
    indices = torch.zeros((batch_size, max_selected), dtype=torch.long, device=select_mask.device)
    valid_mask = torch.zeros((batch_size, max_selected), dtype=torch.bool, device=select_mask.device)
    if max_selected == 0:
        return indices, valid_mask

    for sample_idx in range(batch_size):
        chosen = torch.nonzero(select_mask[sample_idx], as_tuple=False).flatten()
        if chosen.numel() == 0:
            continue
        indices[sample_idx, : chosen.numel()] = chosen
        valid_mask[sample_idx, : chosen.numel()] = True
    return indices, valid_mask


def _gather_idx(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    valid_mask: torch.Tensor,
    fill_value: Any,
) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError(f"Expected tensor with shape [B, L, ...], got {tuple(tensor.shape)}")
    if indices.shape[1] == 0:
        return tensor[:, :0]

    view_shape = indices.shape + (1,) * (tensor.ndim - 2)
    expand_shape = indices.shape + tensor.shape[2:]
    gather_index = indices.view(view_shape).expand(expand_shape).to(device=tensor.device, dtype=torch.long)
    gathered = torch.gather(tensor, dim=1, index=gather_index)
    mask = valid_mask.view(valid_mask.shape + (1,) * (tensor.ndim - 2)).to(gathered.device)
    return gathered.masked_fill(~mask, fill_value)


def _select_response(
    select_mask: torch.Tensor,
    tensors_with_fill: dict[str, tuple[torch.Tensor, Any]],
    *,
    empty_error: str,
) -> dict[str, torch.Tensor]:
    indices, valid_mask = _select_idx(select_mask)
    if indices.shape[1] == 0:
        raise ValueError(empty_error)

    selected = {"indices": indices, "valid_mask": valid_mask}
    for name, (tensor, fill_value) in tensors_with_fill.items():
        selected[name] = _gather_idx(tensor, indices, valid_mask, fill_value)
    return selected


def _check_no_cp(device_mesh, feature_name: str) -> None:
    if (
        device_mesh is not None
        and "cp" in getattr(device_mesh, "mesh_dim_names", ())
        and device_mesh["cp"].size() > 1
    ):
        raise NotImplementedError(f"{feature_name} does not support context parallel yet")


def _num_samples(batch: dict[str, Any]) -> int:
    num_samples = batch.get("num_samples", None)
    if num_samples is None:
        return int(batch["input_ids"].shape[0])
    if isinstance(num_samples, torch.Tensor):
        return int(num_samples.sum().item())
    return int(num_samples)
