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
from unittest.mock import MagicMock

import os
import re
import random
import torch
import json
import numpy as np
from types import SimpleNamespace
from nemo_automodel.shared.import_utils import MISSING_QWEN_VL_UTILS_MSG
from typing import Optional, Tuple

try:
    from qwen_vl_utils import process_vision_info
    HAVE_QWEN_VL_UTILS = True
except ImportError:
    HAVE_QWEN_VL_UTILS = False
    process_vision_info = MagicMock()

try:
    from qwen_omni_utils import process_mm_info

    HAVE_QWEN_OMNI_UTILS = True
except ImportError:
    HAVE_QWEN_OMNI_UTILS = False
    process_mm_info = MagicMock()

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

from nemo_automodel.components.datasets.vlm.utils import default_stop_tokens
from nemo_automodel.utils.time_reparam import tau_to_t


def _message_content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text") is not None:
                text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)
    return "" if content is None else str(content)


def _parse_optional_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
        if normalized in {"none", "null"}:
            return None
    return value


def _resolve_enable_thinking(messages, assistant_message, default="auto"):
    assistant_text = _message_content_to_text(assistant_message.get("content", ""))
    # 1. 只要 assistant 答案里已经有 <think> 或 </think>, 就认为这是 CoT / thinking 样本
    has_thinking_content = "<think>" in assistant_text or "</think>" in assistant_text
    if has_thinking_content:
        return True

    # 2. 如果 system message 显式写了 enable_thinking，也用它
    for message in (assistant_message, messages[0] if messages else None):
        if isinstance(message, dict) and "enable_thinking" in message:
            value = _parse_optional_bool(message.get("enable_thinking"))
            if value != "auto":
                return value

    # 3. 如果 collate_fn 配置里是 enable_thinking: auto, 那么没有 <think> 的样本默认按 non-thinking 处理
    default = _parse_optional_bool(default)
    if default == "auto":
        return False

    # 4. 否则使用 collate_fn 配置里的全局默认值
    return default


def _apply_chat_template(processor, messages, *, enable_thinking=None, **kwargs):
    if enable_thinking is not None:
        try:
            return processor.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                **kwargs,
            )
        except TypeError:
            pass
    return processor.apply_chat_template(messages, **kwargs)


def _extract_response_text(assistant_text, header_text):
    if assistant_text.startswith(header_text):
        return assistant_text[len(header_text):]

    newline_idx = header_text.find("\n")
    if newline_idx == -1:
        return None

    assistant_header = header_text[:newline_idx + 1]
    if assistant_text.startswith(assistant_header):
        return assistant_text[len(assistant_header):]
    return None


def _find_pattern_indices(template, pattern, search_start_index=0, allow_first_token_mismatch=False):
    template_len = len(template)
    pattern_len = len(pattern)
    for i in range(search_start_index, template_len - pattern_len + 1):
        match = template[i : i + pattern_len] == pattern
        if torch.all(match) or (allow_first_token_mismatch and torch.all(match[1:])):
            return i, i + pattern_len
    return -1, -1


def _extract_assistant_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")
    return ""


def build_labels(
    input_ids_batch: torch.Tensor,
    conversations: Sequence[Sequence[Dict[str, Any]]],
    processor,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Construct label and optional loss-mask tensors aligned to assistant responses."""
    tokenizer = getattr(processor, "tokenizer", processor)

    labels_list: List[torch.Tensor] = []

    for encoded, conversation in zip(input_ids_batch, conversations):
        labels = torch.full_like(encoded, -100)
        search_start_index = 0

        for message in conversation:
            if message.get("role") != "assistant":
                continue

            assistant_text = _extract_assistant_text(message)
            if not assistant_text:
                continue

            assistant_tokens = tokenizer(
                assistant_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0].to(encoded.device)

            answer_start, answer_end = _find_pattern_indices(encoded, assistant_tokens, search_start_index)

            if answer_end < len(encoded):
                next_token_str = tokenizer.decode(encoded[answer_end])
                if next_token_str.strip() in default_stop_tokens(processor):
                    answer_end += 1

            if answer_start >= 0:
                labels[answer_start:answer_end] = encoded[answer_start:answer_end]
                search_start_index = answer_end
            else:
                logger.warning(
                    (
                        "Unable to find answer segment in the tokenized conversation. "
                        "Skipping labeling for this and subsequent answers. Details:"
                        "\n- Processed Text: %s"
                        "\n- Tokens: %s"
                        "\n- Target Answer Tokens: %s"
                        "\n- Search Start Index: %d"
                    ),
                    conversation,
                    encoded,
                    assistant_tokens,
                    search_start_index,
                )
                break

        labels_list.append(labels)

    labels_tensor = torch.stack(labels_list)
    return labels_tensor


def default_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
) -> Dict[str, torch.Tensor]:
    """Default collate function for multimodal VLM datasets."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    conversations = [example["conversation"] for example in examples]
    batch = processor.apply_chat_template(
        conversations,
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    )

    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1)
        )

    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

    labels = build_labels(
        batch["input_ids"],
        conversations,
        processor,
    )
    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key in batch:
        if batch[key].shape == input_shape and key != "labels":
            batch[key] = batch[key][:, :-1]
    return batch


def get_rope_index(
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    spatial_merge_size: int = 2,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    vision_start_token_id: int = 151652,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standalone function to calculate 3D position IDs for Qwen3-VL.
    Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids.
    """
    # Since we use timestamps to seperate videos, the video_grid_thw should also be split
    if video_grid_thw is not None:
        # Avoid modifying the input tensor in-place if it's used elsewhere
        video_grid_thw = video_grid_thw.clone() 
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
 
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        image_index, video_index = 0, 0
        # Ensure attention_mask is on the same device
        attention_mask = attention_mask.to(total_input_ids.device)

        # Iterate over batch
        for i, input_ids_item in enumerate(total_input_ids):
            # Filter valid tokens using attention mask
            valid_input_ids = input_ids_item[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0

            # Find all indices of <vision_start>
            vision_start_indices = torch.argwhere(valid_input_ids == vision_start_token_id).squeeze(1)

            # Check the token immediately following <vision_start> to identify type
            # We need to ensure we don't go out of bounds if vision_start is the last token (unlikely but safe to check)
            if len(vision_start_indices) > 0:
                vision_tokens = valid_input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()

            input_tokens = valid_input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            # Iterate through each vision segment in the sequence
            for _ in range(image_nums + video_nums):
                # Find the next image token index
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1

                # Find the next video token index
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                # Determine which comes first
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                # Calculate grid sizes for LLM (downsampled by spatial_merge_size)
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )

                # Process text before this vision block
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # Process vision block position IDs
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)

                # Update start pointer
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            # Process remaining text after the last vision block
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

            # Fill the position_ids tensor
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        # Pure text case
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


from nemo_automodel.utils.scheduler import (
    CondOTScheduler,
    ConvexScheduler,
    CosineScheduler,
    LinearVPScheduler,
    PolynomialConvexScheduler,
    Scheduler,
    VPScheduler,
)
from nemo_automodel.utils.mixture import MixtureDiscreteProbPath


_NOISE_SCHEDULER_REGISTRY = {
    "condot": CondOTScheduler,
    "condotscheduler": CondOTScheduler,
    "polynomial": PolynomialConvexScheduler,
    "polynomialconvex": PolynomialConvexScheduler,
    "polynomialconvexscheduler": PolynomialConvexScheduler,
    "vp": VPScheduler,
    "vpscheduler": VPScheduler,
    "linearvp": LinearVPScheduler,
    "linearvpscheduler": LinearVPScheduler,
    "cosine": CosineScheduler,
    "cosinescheduler": CosineScheduler,
}


def _normalize_scheduler_name(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", name).lower()


def _resolve_scheduler_cls(name: str):
    normalized = _normalize_scheduler_name(name)
    scheduler_cls = _NOISE_SCHEDULER_REGISTRY.get(normalized)
    if scheduler_cls is None:
        supported = ", ".join(sorted({"CondOT", "PolynomialConvex", "VP", "LinearVP", "Cosine"}))
        raise ValueError(f"Unsupported noise scheduler '{name}'. Supported schedulers: {supported}")
    return scheduler_cls


def _build_noise_scheduler(noise_scheduler: Any) -> tuple[Optional[Scheduler], Optional[Any]]:
    if noise_scheduler is None:
        noise_scheduler = "CondOT"

    if isinstance(noise_scheduler, dict):
        scheduler_cfg = dict(noise_scheduler)
        scheduler_name = (
            scheduler_cfg.pop("name", None)
            or scheduler_cfg.pop("type", None)
            or scheduler_cfg.pop("scheduler", None)
        )
        target = scheduler_cfg.pop("_target_", None)
        if scheduler_name is None and target is not None:
            scheduler_name = str(target).rsplit(".", maxsplit=1)[-1]
        if scheduler_name is None:
            raise ValueError("Noise scheduler config must provide 'name', 'type', 'scheduler', or '_target_'.")
        scheduler = _resolve_scheduler_cls(str(scheduler_name))(**scheduler_cfg)
    elif isinstance(noise_scheduler, str):
        scheduler = _resolve_scheduler_cls(noise_scheduler)()
    elif isinstance(noise_scheduler, Scheduler):
        scheduler = noise_scheduler
    elif isinstance(noise_scheduler, MixtureDiscreteProbPath):
        return getattr(noise_scheduler, "scheduler", None), noise_scheduler
    elif hasattr(noise_scheduler, "sample"):
        return getattr(noise_scheduler, "scheduler", None), noise_scheduler
    elif hasattr(noise_scheduler, "to_dict"):
        return _build_noise_scheduler(noise_scheduler.to_dict())
    elif hasattr(noise_scheduler, "__dict__"):
        scheduler_cfg = {k: v for k, v in noise_scheduler.__dict__.items() if not k.startswith("_")}
        if scheduler_cfg:
            return _build_noise_scheduler(scheduler_cfg)
        raise TypeError("Received an empty scheduler config object.")
    else:
        raise TypeError(
            "noise_scheduler/path_scheduler must be a scheduler name, scheduler config, scheduler instance, "
            "or path object with a sample() method."
        )

    path = MixtureDiscreteProbPath(scheduler=scheduler) if isinstance(scheduler, ConvexScheduler) else None
    return scheduler, path


def _sample_discrete_noise(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    *,
    scheduler: Optional[Scheduler],
    path: Optional[Any],
) -> torch.Tensor:
    if path is not None:
        return path.sample(t=t, x_0=x_0, x_1=x_1).x_t

    if scheduler is None:
        raise ValueError("A valid noise scheduler or path object is required for Uniform corruption.")

    scheduler_output = scheduler(t)
    sigma_t = getattr(scheduler_output, "sigma_t", None)
    if sigma_t is None:
        raise ValueError(f"Noise scheduler {type(scheduler).__name__} must return sigma_t for discrete corruption.")

    sigma_t = sigma_t.to(device=x_1.device, dtype=torch.float32)
    while sigma_t.ndim < x_1.ndim:
        sigma_t = sigma_t.unsqueeze(-1)
    sigma_t = torch.broadcast_to(sigma_t, x_1.shape)

    source_indices = torch.rand(x_1.shape, device=x_1.device) < sigma_t
    return torch.where(source_indices, x_0, x_1)


class bard_vl_block_collate_fn:
    def __init__(self, processor, **kwargs):
        # no sequence packing version.
        self.processor = processor
        self.max_len = kwargs.get("max_len", 8192)
        self.model_type = kwargs.get("model_type", "bard-vl")
        self.enable_thinking = kwargs.get("enable_thinking", "auto")

        self.image_token_id = processor.tokenizer.encode("<|image_pad|>")[0]
        self.video_token_id = processor.tokenizer.encode("<|video_pad|>")[0]
        self.vision_start_token_id = processor.tokenizer.encode("<|vision_start|>")[0]
        self.mask_token_id = kwargs.get("mask_token_id", 151671)
        self.vocab_size = kwargs.get("vocab_size", 151646)

        pad_token = kwargs.get("pad_token", "<|endoftext|>")
        self.pad_token_id = processor.tokenizer.encode(pad_token)[0]
        im_end_token = kwargs.get("im_end_token", "<|im_end|>")
        self.im_end_id = processor.tokenizer.encode(im_end_token)[0]
        self.newline_token_ids = self._encode_text("\n")

        self.ignore_index = -100

        self.block_size = kwargs.get("block_size", 4)
        self.min_mask_rate = float(kwargs.get("min_mask_rate", 0.001))
        self.max_mask_rate = float(kwargs.get("max_mask_rate", 1.0))
        if not (0.0 <= self.min_mask_rate <= 1.0):
            raise ValueError(f"min_mask_rate must be in [0, 1], got {self.min_mask_rate}")
        if not (0.0 <= self.max_mask_rate <= 1.0):
            raise ValueError(f"max_mask_rate must be in [0, 1], got {self.max_mask_rate}")
        if self.min_mask_rate > self.max_mask_rate:
            raise ValueError(
                f"min_mask_rate must be <= max_mask_rate, got {self.min_mask_rate} > {self.max_mask_rate}"
            )
        noise_scheduler = kwargs.get("noise_scheduler", kwargs.get("path_scheduler", "CondOT"))
        # self.prior_dist = kwargs.get("prior_dist", "Mask")
        # assert self.prior_dist in ["Uniform", "Mask"], f"{self.prior_dist} not support"
        self.noise_scheduler, self.path = _build_noise_scheduler(noise_scheduler)

        self.semantic_top_k = None

    def _sample_semantic_or_uniform_token(self, token_id: int) -> int:
        if self.semantic_top_k is not None and 0 <= token_id < self.vocab_size:
            candidates = self.semantic_top_k[token_id]
            sampled_idx = torch.randint(low=0, high=candidates.numel(), size=(1,)).item()
            return int(candidates[sampled_idx].item())
        return int(torch.randint(low=0, high=self.vocab_size, size=(1,)).item())

    def _sample_block_times(self, num_blocks: int, device: torch.device, *, low: float, high: float) -> torch.Tensor:
        t = torch.rand(num_blocks, device=device).clamp(low, high)
        if num_blocks > 0:
            offset = torch.arange(num_blocks, device=device, dtype=t.dtype) / num_blocks
            t = t / num_blocks + offset
            t = t[torch.randperm(num_blocks, device=device)]
        return t

    def _sample_mix_noise(self, response_padded: torch.Tensor, num_res_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = response_padded.device
        if num_res_blocks == 0:
            empty = response_padded.new_empty((0,))
            return empty, torch.empty((0,), dtype=torch.float32, device=device)

        t = self._sample_block_times(num_res_blocks, device=device, low=0.001, high=1.0).repeat_interleave(self.block_size)
        noisy_ids = []
        noisy_t = []

        for idx, token in enumerate(response_padded.tolist()):
            phi = torch.pi / 2 * t[idx]
            probs = torch.stack(
                [
                    torch.clamp(1 - torch.cos(phi), 0.001, 1.0),
                    torch.clamp(torch.sin(phi) + torch.cos(phi) - 1, 0.001, 1.0),
                    torch.clamp(1 - torch.sin(phi), 0.001, 1.0),
                ]
            )
            data_type = torch.multinomial(probs, num_samples=1, replacement=True).item()

            # Mask token
            if data_type == 0:
                noisy_ids.append(self.mask_token_id)
                noisy_t.append(t[idx].item())
            # Uniform token
            elif data_type == 1:
                op_probs = torch.tensor([0, 0, 1.0], dtype=torch.float32)
                op_type = torch.multinomial(op_probs, num_samples=1, replacement=True).item()
                # delete
                if op_type == 0:
                    continue
                replacement = self._sample_semantic_or_uniform_token(token)
                # add
                if op_type == 1:
                    noisy_ids.append(replacement)
                    noisy_t.append(t[idx].item())
                    noisy_ids.append(token)
                    noisy_t.append(t[idx].item())
                # substitution
                else:
                    noisy_ids.append(replacement if replacement < self.vocab_size else token)
                    noisy_t.append(t[idx].item())
            # Clean token
            else:
                noisy_ids.append(token)
                noisy_t.append(t[idx].item())

        # target_len = response_padded.numel()
        # noisy_ids = noisy_ids[:target_len]
        # noisy_t = noisy_t[:target_len]
        # if len(noisy_ids) < target_len:
        #     pad_len = target_len - len(noisy_ids)
        #     noisy_ids.extend([self.pad_token_id] * pad_len)
        #     noisy_t.extend([1.0] * pad_len)

        return (
            torch.tensor(noisy_ids, device=device, dtype=response_padded.dtype),
            torch.tensor(noisy_t, device=device, dtype=torch.float32),
        )

    def _sample_edit_noise(self, response_padded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = response_padded.device
        alpha_t = (0.05 + torch.rand(1, device=device) * 0.15).item()
        target_len = response_padded.numel()
        num_edits = round(target_len * alpha_t)
        op_probs = torch.tensor([1.0, 1.0, 3.0], dtype=torch.float32, device=device)
        op_probs = op_probs / op_probs.sum()
        tokens = response_padded.tolist()

        for _ in range(num_edits):
            if not tokens:
                break
            op_type = torch.multinomial(op_probs, num_samples=1, replacement=True).item()
            idx = torch.randint(0, len(tokens), (1,), device=device).item()

            if op_type == 0:
                tokens.pop(idx)
            elif op_type == 1:
                tokens.insert(idx, int(torch.randint(low=0, high=self.vocab_size, size=(1,), device=device).item()))
            else:
                tokens[idx] = int(torch.randint(low=0, high=self.vocab_size, size=(1,), device=device).item())

        tokens = tokens[:target_len]
        if len(tokens) < target_len:
            tokens.extend([self.pad_token_id] * (target_len - len(tokens)))

        return (
            torch.tensor(tokens, device=device, dtype=response_padded.dtype),
            torch.full((target_len,), fill_value=alpha_t, device=device, dtype=torch.float32),
        )

    def _encode_text(self, text):
        return self.processor.tokenizer(
            text=[text],
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]

    def _mask_response_terminator(self, response_ids, noisy_response_ids):
        im_end_positions = (response_ids == self.im_end_id).nonzero(as_tuple=False).flatten()
        if im_end_positions.numel() == 0:
            return noisy_response_ids

        start = int(im_end_positions[-1].item())
        end = start + 1
        newline_len = int(self.newline_token_ids.numel())
        if newline_len > 0 and end + newline_len <= response_ids.numel():
            newline_token_ids = self.newline_token_ids.to(response_ids.device)
            if torch.equal(response_ids[end : end + newline_len], newline_token_ids):
                end += newline_len

        noisy_response_ids[start:end] = self.mask_token_id
        return noisy_response_ids

    def _assistant_idx(self, messages):
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "assistant":
                return idx
        return None

    def _split_turn(self, messages, image_inputs, video_inputs, video_kwargs, video_metadata):
        assistant_idx = self._assistant_idx(messages)
        if assistant_idx is None:
            return None, None, None

        prefix_messages = messages[:assistant_idx]
        assistant_message = messages[assistant_idx]
        enable_thinking = _resolve_enable_thinking(messages, assistant_message, self.enable_thinking)

        prefix_base = _apply_chat_template(
            self.processor,
            prefix_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        prefix_text = _apply_chat_template(
            self.processor,
            prefix_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if not prefix_text.startswith(prefix_base):
            return None, None, None

        header_text = prefix_text[len(prefix_base):]
        full_text = _apply_chat_template(
            self.processor,
            messages[:assistant_idx + 1],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        if not full_text.startswith(prefix_base):
            return None, None, None
        assistant_text = full_text[len(prefix_base):]
        response_text = _extract_response_text(assistant_text, header_text)
        if response_text is None:
            return None, None, None

        prefix_inputs = self.processor(
            text=[prefix_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
            video_metadata=video_metadata,
            **video_kwargs,
        )
        prefix_ids = prefix_inputs.input_ids[0]
        response_ids = self._encode_text(response_text)
        return prefix_ids, response_ids, prefix_inputs

    def create_attention_mask(
        self,
        prefix_len,
        res_len,
        block_size,
        max_len,
        response_valid_len=None,
        keep_noisy_suffix_latent: bool = False,
    ):
        seq_len = min(prefix_len + 2 * res_len, max_len)
        mask = torch.zeros((max_len, max_len), dtype=torch.bool)
        if seq_len <= 0:
            return mask.unsqueeze(0).unsqueeze(0)

        block_ids = torch.empty((seq_len,), dtype=torch.long)
        token_types = torch.empty((seq_len,), dtype=torch.int8)

        clean_start = min(prefix_len, seq_len)
        clean_end = min(prefix_len + res_len, seq_len)
        noisy_start = clean_end

        block_ids[:clean_start] = torch.arange(clean_start) // block_size
        token_types[:clean_start] = 0

        if clean_end > clean_start:
            num_prefix_blocks = (prefix_len + block_size - 1) // block_size
            clean_len = clean_end - clean_start
            clean_rel_positions = torch.arange(clean_len)
            clean_blocks = num_prefix_blocks + clean_rel_positions // block_size
            block_ids[clean_start:clean_end] = clean_blocks
            token_types[clean_start:clean_end] = 1

        if noisy_start < seq_len:
            num_prefix_blocks = (prefix_len + block_size - 1) // block_size
            noisy_len = seq_len - noisy_start
            noisy_rel_positions = torch.arange(noisy_len)
            noisy_blocks = num_prefix_blocks + noisy_rel_positions // block_size
            block_ids[noisy_start:seq_len] = noisy_blocks
            token_types[noisy_start:seq_len] = 2

        q_blocks = block_ids.view(-1, 1)
        k_blocks = block_ids.view(1, -1)
        q_types = token_types.view(-1, 1)
        k_types = token_types.view(1, -1)

        # block causal for prefix + clean blocks.
        base_mask = q_blocks >= k_blocks
        # attention pattern for noisy blocks.
        noisy_query_mask = q_types == 2
        noisy_visibility = (k_types == 0) | ((k_types == 1) & (k_blocks < q_blocks)) | ((k_types == 2) & (k_blocks == q_blocks))
        local_mask = torch.where(noisy_query_mask, noisy_visibility, base_mask)
        if response_valid_len is not None and response_valid_len < res_len:
            valid_positions = torch.ones((seq_len,), dtype=torch.bool)
            valid_clean_end = min(clean_start + response_valid_len, clean_end)
            if valid_clean_end < clean_end:
                valid_positions[valid_clean_end:clean_end] = False
            if not keep_noisy_suffix_latent:
                valid_noisy_end = min(noisy_start + response_valid_len, seq_len)
                if valid_noisy_end < seq_len:
                    valid_positions[valid_noisy_end:seq_len] = False
            local_mask = local_mask & valid_positions.view(1, -1) & valid_positions.view(-1, 1)
        mask[:seq_len, :seq_len] = local_mask

        return mask.unsqueeze(0).unsqueeze(0)

    def __call__(self, batch):
        batch_labels = []
        batch_input_ids = []
        batch_pos_ids = []
        batch_loss_mask = []
        batch_state_mask = []
        batch_response_mask = []
        batch_t = []
        batch_token_types = []  # 0: prefix, 1: clean, 2: noisy
        batch_block_indices = []
        actual_lens = []
        len_pre_list = []
        len_res_list = []
        len_res_valid_list = []

        batch_pixel_values, batch_image_grid_thw = [], []
        batch_pixel_values_videos, batch_video_grid_thw = [], []

        for messages in batch:
            prior_dist = messages[0].get("prior_dist", "Mask")
            return_video_metadata = True
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                return_video_metadata=return_video_metadata,
                image_patch_size=self.processor.image_processor.patch_size
            )
            video_metadata = None
            if return_video_metadata and video_inputs is not None:
                video_metadata = [_[1] for _ in video_inputs]
                video_inputs = [_[0] for _ in video_inputs]

            prefix_ids, response_ids, prefix_inputs = self._split_turn(
                messages,
                image_inputs=image_inputs,
                video_inputs=video_inputs,
                video_kwargs=video_kwargs,
                video_metadata=video_metadata,
            )
            if prefix_ids is None or response_ids is None or prefix_inputs is None or response_ids.numel() == 0:
                continue

            # block划分逻辑
            # [prefix (image + prompt) + [clean response blocks] + [noisy response blocks]]
            len_pre = len(prefix_ids)
            max_res_len = ((self.max_len - len_pre) // (2 * self.block_size)) * self.block_size
            if max_res_len <= 0:
                continue
            response_ids = response_ids[:max_res_len]

            num_res_blocks = (len(response_ids) + self.block_size - 1) // self.block_size
            response_padded = torch.full(
                (num_res_blocks * self.block_size,),
                self.pad_token_id,
                device=prefix_ids.device,
                dtype=prefix_ids.dtype,
            )
            response_padded[:len(response_ids)] = response_ids
            len_res_padded = len(response_padded)

            # 构造noisy部分
            if prior_dist == "Mix":
                noisy_response_ids, noisy_t = self._sample_mix_noise(response_padded, num_res_blocks)

            elif prior_dist == "Edit":
                noisy_response_ids, noisy_t = self._sample_edit_noise(response_padded)
            else:
                noisy_ids_list = []
                noisy_t_list = []
                # 采样noise level t
                if prior_dist == "Uniform":
                    t = self._sample_block_times(num_res_blocks, device=prefix_ids.device, low=0.0, high=0.999)
                elif prior_dist == "Mask":
                    t = self._sample_block_times(
                        num_res_blocks,
                        device=prefix_ids.device,
                        low=self.min_mask_rate,
                        high=self.max_mask_rate,
                    )
                else:
                    raise ValueError(f"Unsupported prior_dist '{prior_dist}'")

                for i in range(num_res_blocks):
                    block_clean = response_padded[i * self.block_size: (i + 1) * self.block_size]
                    if prior_dist == "Uniform": # Uniform Diffusion
                        x_0 = torch.randint_like(block_clean.unsqueeze(0), low=0, high=self.vocab_size)
                        block_noisy = _sample_discrete_noise(
                            t=t[i].unsqueeze(0),
                            x_0=x_0,
                            x_1=block_clean.unsqueeze(0),
                            scheduler=self.noise_scheduler,
                            path=self.path,
                        )[0]
                    elif prior_dist == "Mask": # Mask Diffusion
                        change_indices = torch.rand(len(block_clean), device=block_clean.device) <= t[i].repeat(len(block_clean))
                        block_noisy = block_clean.clone()
                        block_noisy = torch.where(change_indices, self.mask_token_id, block_clean)

                    noisy_ids_list.append(block_noisy)
                    noisy_t_list.append(
                        torch.full(
                            (self.block_size,),
                            fill_value=float(t[i].item()),
                            dtype=torch.float32,
                            device=prefix_ids.device,
                        )
                    )

                noisy_response_ids = torch.cat(noisy_ids_list)
                noisy_t = torch.cat(noisy_t_list)
                if prior_dist == "Mask":
                    noisy_response_ids[response_ids.numel():] = self.mask_token_id
                    noisy_response_ids = self._mask_response_terminator(response_ids, noisy_response_ids)

            # 拼接输入序列 prefix + clean + noisy
            full_input_ids = torch.cat([prefix_ids, response_padded, noisy_response_ids])
            full_labels = torch.cat([prefix_ids, response_padded, response_padded])

            if prior_dist == "Uniform":
                prefix_t = torch.full((len(prefix_ids),), fill_value=0.999, dtype=torch.float32, device=prefix_ids.device)
                clean_t = torch.full((len(response_padded),), fill_value=0.999, dtype=torch.float32, device=prefix_ids.device)
            elif prior_dist in ["Mask", "Mix", "Edit"]:
                prefix_t = torch.full((len(prefix_ids),), fill_value=0.001, dtype=torch.float32, device=prefix_ids.device)
                clean_t = torch.full((len(response_padded),), fill_value=0.001, dtype=torch.float32, device=prefix_ids.device)
            full_t = torch.cat([prefix_t, clean_t, noisy_t])

            # token types: 0=prefix, 1=clean, 2=noisy
            t_types = torch.zeros(len(full_input_ids), dtype=torch.int8, device=prefix_ids.device)
            t_types[len_pre : len_pre + len_res_padded] = 1
            t_types[len_pre + len_res_padded:] = 2

            block_ids = torch.zeros(len(full_input_ids), dtype=torch.int32, device=prefix_ids.device)
            # prefix部分的block_ids
            num_prefix_blocks = (len(prefix_ids) + self.block_size - 1) // self.block_size
            for b in range(num_prefix_blocks):
                start = b * self.block_size
                end = min((b + 1) * self.block_size, len(prefix_ids)) # 不是block size的整数倍时，最后一个block偏小
                block_ids[start: end] = b

            for b in range(num_res_blocks):
                curr_block_id = num_prefix_blocks + b
                # clean部分的block_id
                block_ids[len_pre + b * self.block_size: len_pre + (b + 1) * self.block_size] = curr_block_id
                # noisy部分的block_id
                block_ids[len_pre + len_res_padded + b * self.block_size: len_pre + len_res_padded + (b + 1) * self.block_size] = curr_block_id

            #  position_ids复制逻辑, 先获取 prefix + clean 的 3d position_ids
            pos_pre_clean, _ = get_rope_index(
                    input_ids=torch.cat([prefix_ids, response_padded]).unsqueeze(0),
                    image_grid_thw=getattr(prefix_inputs, "image_grid_thw", None),
                    video_grid_thw=getattr(prefix_inputs, "video_grid_thw", None),
                    attention_mask=torch.ones(1, len_pre + len_res_padded, device=prefix_ids.device),
                    image_token_id=self.image_token_id,
                    video_token_id=self.video_token_id,
                    vision_start_token_id=self.vision_start_token_id
            ) # [3, b=1, seq_len]
            pos_noisy = pos_pre_clean[..., len_pre:] # 复制 clean 的位置
            full_pos_ids = torch.cat([pos_pre_clean, pos_noisy], dim=-1)

            noisy_start = len_pre + len_res_padded
            noisy_valid_end = noisy_start + len(response_ids)
            valid_noisy_mask = torch.zeros(len(full_input_ids), dtype=torch.bool, device=prefix_ids.device)
            valid_noisy_mask[noisy_start:noisy_valid_end] = True

            response_mask = valid_noisy_mask.clone()

            loss_mask = torch.zeros(len(full_input_ids), dtype=torch.bool, device=prefix_ids.device)
            if prior_dist in ["Uniform", "Edit"]:   # 全部response部分计算loss
                loss_mask = valid_noisy_mask.clone()
            elif prior_dist == "Mix":
                loss_mask = torch.where(full_input_ids == full_labels, False, True) & valid_noisy_mask
            elif prior_dist == "Mask":              # 只<MASK>部分计算loss
                loss_mask = (full_input_ids == self.mask_token_id) & valid_noisy_mask

            state_mask = torch.zeros(len(full_input_ids), dtype=torch.bool, device=prefix_ids.device)
            state_mask[noisy_start : len_pre + 2 * len_res_padded] = True

            batch_labels.append(full_labels)
            batch_input_ids.append(full_input_ids)
            batch_pos_ids.append(full_pos_ids)
            batch_loss_mask.append(loss_mask)
            batch_state_mask.append(state_mask)
            batch_t.append(full_t)
            batch_token_types.append(t_types)
            batch_block_indices.append(block_ids)
            batch_response_mask.append(response_mask)
            actual_lens.append(len(full_input_ids))
            len_pre_list.append(len_pre)
            len_res_list.append(len_res_padded)
            len_res_valid_list.append(len(response_ids))

            if getattr(prefix_inputs, "pixel_values", None) is not None:
                batch_pixel_values.append(prefix_inputs.pixel_values)
                batch_image_grid_thw.append(prefix_inputs.image_grid_thw)
            if getattr(prefix_inputs, "pixel_values_videos", None) is not None:
                batch_pixel_values_videos.append(prefix_inputs.pixel_values_videos)
                batch_video_grid_thw.append(prefix_inputs.video_grid_thw)

        if len(actual_lens) == 0:
            raise ValueError("actual_lens is empty")

        max_len = min(max(actual_lens), self.max_len)
        batch_masks = []

        for idx in range(len(batch_input_ids)):
            diff_len = max_len - actual_lens[idx]

            if diff_len < 0:
                # truncation：当前序列超过 self.max_len
                batch_input_ids[idx] = batch_input_ids[idx][:max_len]
                batch_labels[idx] = batch_labels[idx][:max_len]
                batch_pos_ids[idx] = batch_pos_ids[idx][..., :max_len]
                batch_response_mask[idx] = batch_response_mask[idx][:max_len]
                batch_loss_mask[idx] = batch_loss_mask[idx][:max_len]
                batch_state_mask[idx] = batch_state_mask[idx][:max_len]
                batch_t[idx] = batch_t[idx][:max_len]
                batch_token_types[idx] = batch_token_types[idx][:max_len]
                batch_block_indices[idx] = batch_block_indices[idx][:max_len]
            elif diff_len > 0:
                # padding：当前序列短于 self.max_len
                batch_input_ids[idx] = torch.cat([
                    batch_input_ids[idx],
                    torch.full((diff_len,), fill_value=self.pad_token_id, device=batch_input_ids[idx].device, dtype=batch_input_ids[idx].dtype),
                ])
                batch_labels[idx] = torch.cat([
                    batch_labels[idx],
                    torch.full((diff_len,), fill_value=self.pad_token_id, device=batch_labels[idx].device, dtype=batch_labels[idx].dtype),
                ])
                batch_pos_ids[idx] = torch.cat([
                    batch_pos_ids[idx],
                    torch.zeros((3, 1, diff_len), device=batch_pos_ids[idx].device, dtype=batch_pos_ids[idx].dtype),
                ], dim=-1)
                batch_response_mask[idx] = torch.cat([
                    batch_response_mask[idx],
                    torch.zeros((diff_len,), device=batch_response_mask[idx].device, dtype=torch.bool),
                ])
                batch_loss_mask[idx] = torch.cat([
                    batch_loss_mask[idx],
                    torch.zeros((diff_len,), device=batch_loss_mask[idx].device, dtype=torch.bool),
                ])
                batch_state_mask[idx] = torch.cat([
                    batch_state_mask[idx],
                    torch.zeros((diff_len,), device=batch_state_mask[idx].device, dtype=torch.bool),
                ])
                batch_t[idx] = torch.cat([
                    batch_t[idx],
                    torch.full((diff_len,), fill_value=0.5, device=batch_t[idx].device, dtype=batch_t[idx].dtype),
                ])
                batch_token_types[idx] = torch.cat([
                    batch_token_types[idx],
                    torch.full((diff_len,), fill_value=-1, device=batch_token_types[idx].device, dtype=batch_token_types[idx].dtype),
                ])
                batch_block_indices[idx] = torch.cat([
                    batch_block_indices[idx],
                    torch.full((diff_len,), fill_value=-1, device=batch_block_indices[idx].device, dtype=batch_block_indices[idx].dtype),
                ])

            # 生成对应的 attention mask
            batch_masks.append(
                self.create_attention_mask(
                    len_pre_list[idx],
                    len_res_list[idx],
                    self.block_size,
                    max_len,
                    response_valid_len=len_res_valid_list[idx],
                    keep_noisy_suffix_latent=True,
                )
            )

        results = {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "position_ids": torch.cat(batch_pos_ids, dim=1).to(torch.int64),    # [3, B, max_len]
            "response_mask": torch.stack(batch_response_mask),                  # [B, seq_len]
            "loss_mask": torch.stack(batch_loss_mask).to(torch.bool),
            "state_mask": torch.stack(batch_state_mask).to(torch.bool),
            "attention_mask": torch.cat(batch_masks, dim=0),
            "t": torch.stack(batch_t),                                          # [B, max_len]
            # "block_metadata": {                                               # [B, seq_len]
            #     "token_types": torch.stack(batch_token_types),
            #     "block_ids": torch.stack(batch_block_indices),
            # },
            "block_size": self.block_size,
            "num_samples": torch.tensor(len(batch)),
        }

        if batch_pixel_values:
            results["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            results["image_grid_thw"] = torch.cat(batch_image_grid_thw, dim=0)  # [B, 3]
        if batch_pixel_values_videos:
            results["pixel_values_videos"] = torch.cat(batch_pixel_values_videos, dim=0)
            results["video_grid_thw"] = torch.cat(batch_video_grid_thw, dim=0)  # [B, 3]

        return results


class bard_block_collate_fn:
    def __init__(self, processor, **kwargs):
        # Text-only block diffusion collate for causal LMs such as Qwen3Flow.
        self.processor = getattr(processor, "tokenizer", processor)
        self.max_len = kwargs.get("max_len", 8192)
        self.model_type = kwargs.get("model_type", "bard")
        self.model_type_normalized = re.sub(r"[\s_\-]+", "", str(self.model_type).lower())
        self.enable_thinking = kwargs.get("enable_thinking", "auto")

        self.vocab_size = kwargs.get("vocab_size", 151936)

        pad_token = kwargs.get("pad_token", "<|endoftext|>")
        self.pad_token_id = self.processor.encode(pad_token)[0]
        im_end_token = kwargs.get("im_end_token", "<|im_end|>")
        self.im_end_id = self.processor.encode(im_end_token)[0]

        self.ignore_index = -100

        self.block_size = kwargs.get("block_size", 32)
        self.tau_min = float(kwargs.get("tau_min", 0.0))
        self.tau_max = float(kwargs.get("tau_max", 1.0))
        if not (0.0 <= self.tau_min <= self.tau_max <= 1.0):
            raise ValueError(
                "tau_min/tau_max must satisfy "
                f"0 <= min <= max <= 1, got {self.tau_min}, {self.tau_max}"
            )

        self.time_reparameterization = kwargs.get("time_reparameterization", kwargs.get("time_reparam", "linear"))
        self.reparam_lut_size = int(kwargs.get("reparam_lut_size", 1000))
        self.reparam_quad_points = int(kwargs.get("reparam_quad_points", 64))
        noise_scheduler = kwargs.get("noise_scheduler", kwargs.get("path_scheduler", "CondOT"))
        self.noise_scheduler, self.path = _build_noise_scheduler(noise_scheduler)

    def _to_text_messages(self, messages):
        template_messages = []
        for message in messages:
            template_message = dict(message)
            content = template_message.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("text") is not None:
                        text_parts.append(str(item["text"]))
                    elif isinstance(item, str):
                        text_parts.append(item)
                template_message["content"] = "\n".join(text_parts)
            template_messages.append(template_message)
        return template_messages

    def _encode_text(self, text):
        return self.processor(
            text=[text],
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]

    def _assistant_idx(self, messages):
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "assistant":
                return idx
        return None

    def _split_turn(self, messages):
        assistant_idx = self._assistant_idx(messages)
        if assistant_idx is None or assistant_idx == 0:
            return None, None

        prefix_messages = messages[:assistant_idx]
        assistant_message = messages[assistant_idx]
        enable_thinking = _resolve_enable_thinking(messages, assistant_message, self.enable_thinking)

        prefix_base = _apply_chat_template(
            self.processor,
            prefix_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        prefix_text = _apply_chat_template(
            self.processor,
            prefix_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if not prefix_text.startswith(prefix_base):
            return None, None

        header_text = prefix_text[len(prefix_base):]
        full_text = _apply_chat_template(
            self.processor,
            messages[:assistant_idx + 1],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        if not full_text.startswith(prefix_base):
            return None, None
        assistant_text = full_text[len(prefix_base):]
        response_text = _extract_response_text(assistant_text, header_text)
        if response_text is None:
            return None, None

        return self._encode_text(prefix_text), self._encode_text(response_text)

    def _sample_block_times(self, num_blocks: int, device: torch.device, *, low: float, high: float) -> torch.Tensor:
        tau = torch.rand(num_blocks, device=device)
        if num_blocks > 0:
            offset = torch.arange(num_blocks, device=device, dtype=tau.dtype) / num_blocks
            tau = tau / num_blocks + offset
            tau = tau[torch.randperm(num_blocks, device=device)]
        tau = low + (high - low) * tau

        if self.time_reparameterization in (None, "none", "linear"):
            return tau

        config = SimpleNamespace(
            vocab_size=self.vocab_size,
            time_reparam=self.time_reparameterization,
            reparam_lut_size=self.reparam_lut_size,
            reparam_quad_points=self.reparam_quad_points,
        )
        return tau_to_t(tau, config)

    def _sample_block_tau(self, num_blocks: int, device: torch.device, *, low: float, high: float) -> torch.Tensor:
        tau = torch.rand(num_blocks, device=device)
        if num_blocks > 0:
            offset = torch.arange(num_blocks, device=device, dtype=tau.dtype) / num_blocks
            tau = tau / num_blocks + offset
            tau = tau[torch.randperm(num_blocks, device=device)]
        return low + (high - low) * tau

    def _use_uniform_discrete_path(self, messages) -> bool:
        if self.model_type_normalized in {"qwen3uniform"}:
            return True
        if not messages:
            return False
        prior_dist = messages[0].get("prior_dist") if isinstance(messages[0], dict) else None
        return isinstance(prior_dist, str) and prior_dist.lower() == "uniform"

    def _sample_uniform_discrete_response(self, response_padded: torch.Tensor, block_t: torch.Tensor) -> torch.Tensor:
        if response_padded.numel() == 0:
            return response_padded.clone()

        noisy_blocks = []
        for block_idx in range(block_t.shape[0]):
            block_clean = response_padded[block_idx * self.block_size : (block_idx + 1) * self.block_size]
            x_0 = torch.randint_like(block_clean.unsqueeze(0), low=0, high=self.vocab_size)
            block_noisy = _sample_discrete_noise(
                x_0=x_0,
                x_1=block_clean.unsqueeze(0),
                t=block_t[block_idx].unsqueeze(0),
                scheduler=self.noise_scheduler,
                path=self.path,
            )[0]
            noisy_blocks.append(block_noisy)

        return torch.cat(noisy_blocks, dim=0)

    def create_attention_mask(
        self,
        prefix_len,
        res_len,
        block_size,
        max_len,
        response_valid_len=None,
        keep_noisy_suffix_latent: bool = False,
    ):
        seq_len = min(prefix_len + 2 * res_len, max_len)
        mask = torch.zeros((max_len, max_len), dtype=torch.bool)
        if seq_len <= 0:
            return mask.unsqueeze(0).unsqueeze(0)

        block_ids = torch.empty((seq_len,), dtype=torch.long)
        token_types = torch.empty((seq_len,), dtype=torch.int8)

        clean_start = min(prefix_len, seq_len)
        clean_end = min(prefix_len + res_len, seq_len)
        noisy_start = clean_end

        block_ids[:clean_start] = torch.arange(clean_start) // block_size
        token_types[:clean_start] = 0

        if clean_end > clean_start:
            num_prefix_blocks = (prefix_len + block_size - 1) // block_size
            clean_len = clean_end - clean_start
            clean_rel_positions = torch.arange(clean_len)
            clean_blocks = num_prefix_blocks + clean_rel_positions // block_size
            block_ids[clean_start:clean_end] = clean_blocks
            token_types[clean_start:clean_end] = 1

        if noisy_start < seq_len:
            num_prefix_blocks = (prefix_len + block_size - 1) // block_size
            noisy_len = seq_len - noisy_start
            noisy_rel_positions = torch.arange(noisy_len)
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
        if response_valid_len is not None and response_valid_len < res_len:
            valid_positions = torch.ones((seq_len,), dtype=torch.bool)
            valid_clean_end = min(clean_start + response_valid_len, clean_end)
            if valid_clean_end < clean_end:
                valid_positions[valid_clean_end:clean_end] = False
            if not keep_noisy_suffix_latent:
                valid_noisy_end = min(noisy_start + response_valid_len, seq_len)
                if valid_noisy_end < seq_len:
                    valid_positions[valid_noisy_end:seq_len] = False
            local_mask = local_mask & valid_positions.view(1, -1) & valid_positions.view(-1, 1)
        mask[:seq_len, :seq_len] = local_mask

        return mask.unsqueeze(0).unsqueeze(0)

    def __call__(self, batch):
        batch_labels = []
        batch_input_ids = []
        batch_pos_ids = []
        batch_loss_mask = []
        batch_state_mask = []
        batch_response_mask = []
        batch_t = []
        batch_timesteps = []
        batch_token_types = []
        batch_block_indices = []
        actual_lens = []
        len_pre_list = []
        len_res_list = []
        len_res_valid_list = []

        for messages in batch:
            template_messages = self._to_text_messages(messages)
            prefix_ids, response_ids = self._split_turn(template_messages)
            if prefix_ids is None or response_ids is None or response_ids.numel() == 0:
                continue
            len_pre = len(prefix_ids)

            num_res_blocks = (len(response_ids) + self.block_size - 1) // self.block_size
            response_padded = torch.full(
                (num_res_blocks * self.block_size,),
                self.pad_token_id,
                device=response_ids.device,
                dtype=response_ids.dtype,
            )
            response_padded[:len(response_ids)] = response_ids
            len_res_padded = len(response_padded)

            use_uniform_discrete = self._use_uniform_discrete_path(template_messages)
            if use_uniform_discrete:
                block_t = self._sample_block_tau(
                    num_res_blocks,
                    device=response_ids.device,
                    low=self.tau_min,
                    high=self.tau_max,
                )
                block_tau = block_t
                noisy_response_ids = self._sample_uniform_discrete_response(response_padded, block_t)
            else:
                block_tau = self._sample_block_tau(
                    num_res_blocks,
                    device=response_ids.device,
                    low=self.tau_min,
                    high=self.tau_max,
                )
                config = SimpleNamespace(
                    vocab_size=self.vocab_size,
                    time_reparam=self.time_reparameterization,
                    reparam_lut_size=self.reparam_lut_size,
                    reparam_quad_points=self.reparam_quad_points,
                )
                block_t = tau_to_t(block_tau, config)
                noisy_response_ids = response_padded

            noisy_tau = block_tau.repeat_interleave(self.block_size).to(dtype=torch.float32)
            noisy_t = block_t.repeat_interleave(self.block_size).to(dtype=torch.float32)

            full_input_ids = torch.cat([prefix_ids, response_padded, noisy_response_ids])
            full_labels = torch.cat([prefix_ids, response_padded, response_padded])

            prefix_t = torch.ones((len(prefix_ids),), dtype=torch.float32, device=prefix_ids.device)
            clean_t = torch.ones((len(response_padded),), dtype=torch.float32, device=prefix_ids.device)
            full_t = torch.cat([prefix_t, clean_t, noisy_t])
            if use_uniform_discrete:
                full_timesteps = full_t
            else:
                prefix_tau = torch.ones((len(prefix_ids),), dtype=torch.float32, device=prefix_ids.device)
                clean_tau = torch.ones((len(response_padded),), dtype=torch.float32, device=prefix_ids.device)
                full_timesteps = torch.cat([prefix_tau, clean_tau, noisy_tau])

            t_types = torch.zeros(len(full_input_ids), dtype=torch.int8, device=prefix_ids.device)
            t_types[len_pre : len_pre + len_res_padded] = 1
            t_types[len_pre + len_res_padded:] = 2

            block_ids = torch.zeros(len(full_input_ids), dtype=torch.int32, device=prefix_ids.device)
            num_prefix_blocks = (len(prefix_ids) + self.block_size - 1) // self.block_size
            for b in range(num_prefix_blocks):
                start = b * self.block_size
                end = min((b + 1) * self.block_size, len(prefix_ids))
                block_ids[start: end] = b

            for b in range(num_res_blocks):
                curr_block_id = num_prefix_blocks + b
                block_ids[len_pre + b * self.block_size: len_pre + (b + 1) * self.block_size] = curr_block_id
                block_ids[len_pre + len_res_padded + b * self.block_size: len_pre + len_res_padded + (b + 1) * self.block_size] = curr_block_id

            pos_pre_clean = torch.arange(len_pre + len_res_padded, device=response_ids.device).unsqueeze(0)
            pos_noisy = pos_pre_clean[:, len_pre:]
            full_pos_ids = torch.cat([pos_pre_clean, pos_noisy], dim=-1)

            noisy_start = len_pre + len_res_padded
            noisy_valid_end = noisy_start + len(response_ids)
            response_mask = torch.zeros(len(full_labels), dtype=torch.bool, device=response_ids.device)
            response_mask[noisy_start:noisy_valid_end] = True

            loss_mask = torch.zeros(len(full_input_ids), dtype=torch.bool, device=response_ids.device)
            loss_mask[noisy_start:noisy_valid_end] = True
            state_mask = torch.zeros(len(full_input_ids), dtype=torch.bool, device=response_ids.device)
            state_mask[noisy_start : len_pre + 2 * len_res_padded] = True

            batch_labels.append(full_labels)
            batch_input_ids.append(full_input_ids)
            batch_pos_ids.append(full_pos_ids)
            batch_loss_mask.append(loss_mask)
            batch_state_mask.append(state_mask)
            batch_t.append(full_t)
            batch_timesteps.append(full_timesteps)
            batch_token_types.append(t_types)
            batch_block_indices.append(block_ids)
            batch_response_mask.append(response_mask)
            actual_lens.append(len(full_input_ids))
            len_pre_list.append(len_pre)
            len_res_list.append(len_res_padded)
            len_res_valid_list.append(len(response_ids))

        if len(actual_lens) == 0:
            raise ValueError("actual_lens is empty")

        max_len = min(max(actual_lens), self.max_len)
        batch_masks = []

        for idx in range(len(batch_input_ids)):
            diff_len = max_len - actual_lens[idx]

            if diff_len < 0:
                batch_input_ids[idx] = batch_input_ids[idx][:max_len]
                batch_labels[idx] = batch_labels[idx][:max_len]
                batch_pos_ids[idx] = batch_pos_ids[idx][..., :max_len]
                batch_response_mask[idx] = batch_response_mask[idx][:max_len]
                batch_loss_mask[idx] = batch_loss_mask[idx][:max_len]
                batch_state_mask[idx] = batch_state_mask[idx][:max_len]
                batch_t[idx] = batch_t[idx][:max_len]
                batch_timesteps[idx] = batch_timesteps[idx][:max_len]
                batch_token_types[idx] = batch_token_types[idx][:max_len]
                batch_block_indices[idx] = batch_block_indices[idx][:max_len]
            elif diff_len > 0:
                batch_input_ids[idx] = torch.cat([
                    batch_input_ids[idx],
                    torch.full((diff_len,), fill_value=self.pad_token_id, device=batch_input_ids[idx].device, dtype=batch_input_ids[idx].dtype),
                ])
                batch_labels[idx] = torch.cat([
                    batch_labels[idx],
                    torch.full((diff_len,), fill_value=self.pad_token_id, device=batch_labels[idx].device, dtype=batch_labels[idx].dtype),
                ])
                batch_pos_ids[idx] = torch.cat([
                    batch_pos_ids[idx],
                    torch.zeros((1, diff_len), device=batch_pos_ids[idx].device, dtype=batch_pos_ids[idx].dtype),
                ], dim=-1)
                batch_response_mask[idx] = torch.cat([
                    batch_response_mask[idx],
                    torch.zeros((diff_len,), device=batch_response_mask[idx].device, dtype=torch.bool),
                ])
                batch_loss_mask[idx] = torch.cat([
                    batch_loss_mask[idx],
                    torch.zeros((diff_len,), device=batch_loss_mask[idx].device, dtype=torch.bool),
                ])
                batch_state_mask[idx] = torch.cat([
                    batch_state_mask[idx],
                    torch.zeros((diff_len,), device=batch_state_mask[idx].device, dtype=torch.bool),
                ])
                batch_t[idx] = torch.cat([
                    batch_t[idx],
                    torch.full((diff_len,), fill_value=1.0, device=batch_t[idx].device, dtype=batch_t[idx].dtype),
                ])
                batch_timesteps[idx] = torch.cat([
                    batch_timesteps[idx],
                    torch.full((diff_len,), fill_value=1.0, device=batch_timesteps[idx].device, dtype=batch_timesteps[idx].dtype),
                ])
                batch_token_types[idx] = torch.cat([
                    batch_token_types[idx],
                    torch.full((diff_len,), fill_value=-1, device=batch_token_types[idx].device, dtype=batch_token_types[idx].dtype),
                ])
                batch_block_indices[idx] = torch.cat([
                    batch_block_indices[idx],
                    torch.full((diff_len,), fill_value=-1, device=batch_block_indices[idx].device, dtype=batch_block_indices[idx].dtype),
                ])

            batch_masks.append(
                self.create_attention_mask(
                    len_pre_list[idx],
                    len_res_list[idx],
                    self.block_size,
                    max_len,
                    response_valid_len=len_res_valid_list[idx],
                    keep_noisy_suffix_latent=True,
                )
            )

        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "position_ids": torch.cat(batch_pos_ids, dim=0).to(torch.int64),
            "response_mask": torch.stack(batch_response_mask),
            "loss_mask": torch.stack(batch_loss_mask).to(torch.bool),
            "state_mask": torch.stack(batch_state_mask).to(torch.bool),
            "attention_mask": torch.cat(batch_masks, dim=0),
            "t": torch.stack(batch_t),
            "timesteps": torch.stack(batch_timesteps),
            "block_size": self.block_size,
            "num_samples": torch.tensor(len(batch)),
        }


class bard_uni_collate_fn:
    """Collate function for Bard-Uni image generation/editing tasks.

    Constructs training sequences with Block Diffusion masking on VQ code regions.
    Sequence layout: [prefix + clean_response + noisy_response]
    where response = <|img_gen_start|> [VQ_ROW_1] <|img_newline|> ... [VQ_ROW_N] <|img_gen_end|>
    """

    def __init__(self, processor, **kwargs):
        self.processor = processor
        self.max_len = kwargs.get("max_len", 8192)
        self.block_size = kwargs.get("block_size", 32)

        self.mask_token_id = kwargs.get("mask_token_id", 151671)
        self.img_gen_start_token_id = kwargs.get("img_gen_start_token_id", 151672)  # <|img_gen_start|>
        self.img_gen_end_token_id = kwargs.get("img_gen_end_token_id", 151673)      # <|img_gen_end|>  
        self.img_newline_token_id = kwargs.get("img_newline_token_id", 151674)      # <|img_newline|>
        self.img_src_start_token_id = kwargs.get("img_src_start_token_id", 151675)  # <|img_src_start|>
        self.img_src_end_token_id = kwargs.get("img_src_end_token_id", 151676)      # <|img_end_start|>
        self.uncondition_token_id = kwargs.get("uncondition_token_id", 151677)      # <|uncondition|>

        pad_token = kwargs.get("pad_token", "<|endoftext|>")
        self.pad_token_id = processor.tokenizer.encode(pad_token)[0]

        im_end_token = kwargs.get("im_end_token", "<|im_end|>")
        self.im_end_id = processor.tokenizer.encode(im_end_token)[0]

        self.use_newline = kwargs.get("use_newline", False)
        self.mask_newline = kwargs.get("mask_newline", False)

        self.min_mask_rate = float(kwargs.get("min_mask_rate", 0.0))
        self.max_mask_rate = float(kwargs.get("max_mask_rate", 1.0))
        self.cfg_uncond_drop_rate = float(kwargs.get("cfg_uncond_drop_rate", 0.1))

    def _encode_text(self, text):
        return self.processor.tokenizer(
            text=[text],
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]

    def _build_vq_sequence(self, vq_codes, code_height, code_width):
        """Build token sequence from VQ codes, optionally with newline separators."""
        codes = torch.tensor(vq_codes, dtype=torch.long).reshape(code_height, code_width)
        if not self.use_newline:
            return codes.reshape(-1)
        seq_parts = []
        for row_idx in range(code_height):
            seq_parts.append(codes[row_idx])
            if row_idx < code_height - 1:
                seq_parts.append(torch.tensor([self.img_newline_token_id], dtype=torch.long))
        return torch.cat(seq_parts)

    def _build_t2i_prefix(self, caption):
        """Build prefix for text-to-image: system + user prompt."""
        # <|im_start|>user\nGenerate an image: {caption}<|im_end|>\n<|im_start|>assistant\n
        prefix_text = (
            f"<|im_start|>user\nGenerate an image: {caption}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return self._encode_text(prefix_text)

    def _build_editing_prefix(self, src_images, instruction):
        """Build prefix for editing: user message with one or more source VQ images + instruction.

        Args:
            src_images: list of dicts, each with keys: vq_codes, code_height, code_width
            instruction: text instruction for editing

        Returns:
            prefix_ids: token tensor
            vq_regions: list of (start_offset, code_height, code_width) for each VQ image
        """
        parts = [self._encode_text("<|im_start|>user\n")]
        vq_regions = []

        for img in src_images:
            src_start_tok = torch.tensor([self.img_src_start_token_id], dtype=torch.long)
            src_seq = self._build_vq_sequence(img["vq_codes"], img["code_height"], img["code_width"])
            src_end_tok = torch.tensor([self.img_src_end_token_id], dtype=torch.long)

            # VQ region offset = current total length + 1 (for src_start token)
            cur_len = sum(len(p) for p in parts)
            vq_start = cur_len + 1
            vq_regions.append((vq_start, img["code_height"], img["code_width"]))

            parts.extend([src_start_tok, src_seq, src_end_tok])

        parts.append(self._encode_text(f"\n{instruction}<|im_end|>\n<|im_start|>assistant\n"))
        prefix_ids = torch.cat(parts)
        return prefix_ids, vq_regions

    def _build_response_ids(self, vq_codes, code_height, code_width):
        """Build response token sequence: <|img_gen_start|> [VQ] <|img_gen_end|><|im_end|>\n"""
        gen_start = torch.tensor([self.img_gen_start_token_id], dtype=torch.long)
        vq_seq = self._build_vq_sequence(vq_codes, code_height, code_width)
        gen_end = torch.tensor([self.img_gen_end_token_id], dtype=torch.long)
        im_end = self._encode_text("<|im_end|>\n")
        return torch.cat([gen_start, vq_seq, gen_end, im_end])

    def _build_vq_mrope_positions(self, code_height, code_width, start_pos):
        """Build MRoPE position IDs for a VQ response region (gen_start + vq_grid + gen_end).

        For VQ tokens at grid (row, col):
            T = start_pos (fixed, same image)
            H = start_pos + row
            W = start_pos + col
        For special tokens (gen_start, img_newline, gen_end):
            T = H = W = sequential (text-like)

        Returns: [3, response_len] position tensor
        """
        pos_t = []
        pos_h = []
        pos_w = []

        # <|img_gen_start|>
        pos_t.append(start_pos)
        pos_h.append(start_pos)
        pos_w.append(start_pos)

        # VQ rows
        for row in range(code_height):
            for col in range(code_width):
                pos_t.append(start_pos)
                pos_h.append(start_pos + row)
                pos_w.append(start_pos + col)
            if self.use_newline and row < code_height - 1:
                nl_pos = start_pos + code_height
                pos_t.append(nl_pos)
                pos_h.append(nl_pos)
                pos_w.append(nl_pos)

        # <|img_gen_end|>
        end_pos = start_pos + max(code_height, code_width)
        pos_t.append(end_pos)
        pos_h.append(end_pos)
        pos_w.append(end_pos)

        return torch.stack([
            torch.tensor(pos_t, dtype=torch.long),
            torch.tensor(pos_h, dtype=torch.long),
            torch.tensor(pos_w, dtype=torch.long),
        ])

    def _build_prefix_positions(self, prefix_len, vq_regions):
        """Build MRoPE positions for prefix, with 2D encoding for VQ regions.

        Args:
            prefix_len: total length of prefix tokens
            vq_regions: list of (vq_start_offset, code_height, code_width) for each VQ image,
                        or None/[] if prefix is pure text.
        Returns: [3, prefix_len] position tensor
        """
        if not vq_regions:
            pos = torch.arange(prefix_len, dtype=torch.long)
            return pos.unsqueeze(0).expand(3, -1)

        pos_t = torch.zeros(prefix_len, dtype=torch.long)
        pos_h = torch.zeros(prefix_len, dtype=torch.long)
        pos_w = torch.zeros(prefix_len, dtype=torch.long)

        # First pass: assign sequential text positions everywhere
        seq_pos = 0
        vq_spans = []  # (start, end) ranges that are VQ tokens
        for (vq_start, ch, cw) in vq_regions:
            vq_len = ch * cw + (ch - 1) if self.use_newline else ch * cw
            vq_spans.append((vq_start, vq_start + vq_len))

        # Build text position counter that skips over VQ regions
        cur_text_pos = 0
        i = 0
        for (vq_start, vq_end) in vq_spans:
            # Text before this VQ region
            while i < vq_start and i < prefix_len:
                pos_t[i] = cur_text_pos
                pos_h[i] = cur_text_pos
                pos_w[i] = cur_text_pos
                cur_text_pos += 1
                i += 1
            # Skip VQ region (will fill separately)
            i = min(vq_end, prefix_len)
            cur_text_pos += 1  # VQ block occupies 1 "text step" for continuity

        # Remaining text after last VQ region
        while i < prefix_len:
            pos_t[i] = cur_text_pos
            pos_h[i] = cur_text_pos
            pos_w[i] = cur_text_pos
            cur_text_pos += 1
            i += 1

        # Second pass: fill VQ regions with 2D positions
        for (vq_start, ch, cw) in vq_regions:
            # Find the text position just before this VQ region
            base_pos = pos_t[vq_start - 1].item() + 1 if vq_start > 0 else 0
            idx = vq_start
            for row in range(ch):
                for col in range(cw):
                    if idx < prefix_len:
                        pos_t[idx] = base_pos
                        pos_h[idx] = base_pos + row
                        pos_w[idx] = base_pos + col
                        idx += 1
                if self.use_newline and row < ch - 1 and idx < prefix_len:
                    nl_pos = base_pos + ch
                    pos_t[idx] = nl_pos
                    pos_h[idx] = nl_pos
                    pos_w[idx] = nl_pos
                    idx += 1

        return torch.stack([pos_t, pos_h, pos_w])

    def _apply_mask_corruption(self, response_ids, num_res_blocks):
        """Apply Block Diffusion mask corruption to VQ response tokens."""
        device = response_ids.device
        t = torch.rand(num_res_blocks, device=device) * (self.max_mask_rate - self.min_mask_rate) + self.min_mask_rate
        # Stratified sampling
        offset = torch.arange(num_res_blocks, device=device, dtype=t.dtype) / max(num_res_blocks, 1)
        t = t / max(num_res_blocks, 1) + offset
        t = t[torch.randperm(num_res_blocks, device=device)].clamp(0, 1)

        noisy_ids = response_ids.clone()
        noisy_t = torch.zeros(len(response_ids), dtype=torch.float32, device=device)

        for i in range(num_res_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, len(response_ids))
            block_len = end - start
            mask_indices = torch.rand(block_len, device=device) <= t[i]
            # 至少 mask 一个 token
            if not mask_indices.any():
                mask_indices[torch.randint(0, block_len, (1,)).item()] = True
            noisy_ids[start:end] = torch.where(mask_indices, self.mask_token_id, response_ids[start:end])
            noisy_t[start:end] = t[i]

        return noisy_ids, noisy_t

    def create_attention_mask(self, prefix_len, res_len, block_size, max_len):
        """Block causal attention mask for prefix + clean + noisy layout."""
        seq_len = min(prefix_len + 2 * res_len, max_len)
        mask = torch.zeros((max_len, max_len), dtype=torch.bool)
        if seq_len <= 0:
            return mask.unsqueeze(0).unsqueeze(0)

        block_ids = torch.empty((seq_len,), dtype=torch.long)
        token_types = torch.empty((seq_len,), dtype=torch.int8)

        clean_start = min(prefix_len, seq_len)
        clean_end = min(prefix_len + res_len, seq_len)
        noisy_start = clean_end

        block_ids[:clean_start] = torch.arange(clean_start) // block_size
        token_types[:clean_start] = 0

        if clean_end > clean_start:
            num_prefix_blocks = (prefix_len + block_size - 1) // block_size
            clean_len = clean_end - clean_start
            block_ids[clean_start:clean_end] = num_prefix_blocks + torch.arange(clean_len) // block_size
            token_types[clean_start:clean_end] = 1

        if noisy_start < seq_len:
            num_prefix_blocks = (prefix_len + block_size - 1) // block_size
            noisy_len = seq_len - noisy_start
            block_ids[noisy_start:seq_len] = num_prefix_blocks + torch.arange(noisy_len) // block_size
            token_types[noisy_start:seq_len] = 2

        q_blocks = block_ids.view(-1, 1)
        k_blocks = block_ids.view(1, -1)
        q_types = token_types.view(-1, 1)
        k_types = token_types.view(1, -1)

        base_mask = q_blocks >= k_blocks
        noisy_query_mask = q_types == 2
        noisy_visibility = (
            (k_types == 0) |
            ((k_types == 1) & (k_blocks < q_blocks)) |
            ((k_types == 2) & (k_blocks == q_blocks))
        )
        local_mask = torch.where(noisy_query_mask, noisy_visibility, base_mask)
        mask[:seq_len, :seq_len] = local_mask

        return mask.unsqueeze(0).unsqueeze(0)

    def __call__(self, batch):
        batch_input_ids = []
        batch_labels = []
        batch_pos_ids = []
        batch_loss_mask = []
        batch_response_mask = []
        batch_t = []
        batch_generation_mask = []
        batch_vq_codes = []
        batch_vq_code_mask = []
        actual_lens = []
        len_pre_list = []
        len_res_list = []

        for sample in batch:
            task_type = sample["task_type"]
            vq_codes = sample["vq_codes"]
            code_height = sample["code_height"]
            code_width = sample["code_width"]
            caption = sample["caption"]

            # CFG: drop prompt with probability cfg_uncond_drop_rate
            if random.random() < self.cfg_uncond_drop_rate:
                caption = ""

            # Build prefix
            prefix_vq_regions = None
            if task_type == "text_to_image":
                prefix_ids = self._build_t2i_prefix(caption)
            elif task_type in ("image_editing", "style_transfer", "control_generation", "subject_driven"):
                instruction = sample.get("instruction") or caption
                # Support single or multiple source images
                if "src_images" in sample:
                    src_images = sample["src_images"]
                else:
                    src_images = [{
                        "vq_codes": sample["src_vq_codes"],
                        "code_height": sample["src_code_height"],
                        "code_width": sample["src_code_width"],
                    }]
                prefix_ids, prefix_vq_regions = self._build_editing_prefix(
                    src_images, instruction,
                )
            else:
                prefix_ids = self._build_t2i_prefix(caption)

            # Build response (target VQ codes)
            response_ids = self._build_response_ids(vq_codes, code_height, code_width)

            len_pre = len(prefix_ids)
            max_res_len = ((self.max_len - len_pre) // (2 * self.block_size)) * self.block_size
            if max_res_len <= 0:
                continue
            response_ids = response_ids[:max_res_len]

            # Pad response to block boundary
            num_res_blocks = (len(response_ids) + self.block_size - 1) // self.block_size
            response_padded = torch.full(
                (num_res_blocks * self.block_size,),
                self.pad_token_id,
                dtype=prefix_ids.dtype,
            )
            response_padded[:len(response_ids)] = response_ids
            len_res_padded = len(response_padded)

            # Apply mask corruption
            noisy_response_ids, noisy_t = self._apply_mask_corruption(response_padded, num_res_blocks)
            # Padding positions → mask (model must learn nothing to generate there)
            noisy_response_ids[len(response_ids):] = self.mask_token_id
            # Terminator (<|im_end|>\n) → mask (model must learn to predict end)
            im_end_pos = (response_ids == self.im_end_id).nonzero(as_tuple=False).flatten()
            if im_end_pos.numel() > 0:
                noisy_response_ids[im_end_pos[0]:len(response_ids)] = self.mask_token_id
            # When mask_newline=False, restore newline positions so they stay visible
            if self.use_newline and not self.mask_newline:
                nl_positions = (response_padded == self.img_newline_token_id)
                noisy_response_ids[nl_positions] = self.img_newline_token_id

            # Concatenate: prefix + clean + noisy
            full_input_ids = torch.cat([prefix_ids, response_padded, noisy_response_ids])
            full_labels = torch.cat([prefix_ids, response_padded, response_padded])

            prefix_t = torch.full((len_pre,), 0.001, dtype=torch.float32)
            clean_t = torch.full((len_res_padded,), 0.001, dtype=torch.float32)
            full_t = torch.cat([prefix_t, clean_t, noisy_t])

            # Position IDs with 2D-aware MRoPE for VQ tokens
            total_len = len(full_input_ids)

            # Prefix: handle both pure text and editing (with source VQ images)
            prefix_pos_3d = self._build_prefix_positions(len_pre, prefix_vq_regions)

            # Response region: VQ grid gets 2D spatial positions
            response_start_pos = prefix_pos_3d.max().item() + 1
            vq_pos = self._build_vq_mrope_positions(code_height, code_width, start_pos=response_start_pos)
            # Pad response positions to len_res_padded (pad tokens get text-like positions)
            res_actual_len = vq_pos.shape[1]
            if len_res_padded > res_actual_len:
                pad_start = vq_pos.max().item() + 1
                pad_pos = torch.arange(pad_start, pad_start + len_res_padded - res_actual_len, dtype=torch.long)
                pad_pos_3d = pad_pos.unsqueeze(0).expand(3, -1)
                response_pos_3d = torch.cat([vq_pos, pad_pos_3d], dim=1)  # [3, len_res_padded]
            else:
                response_pos_3d = vq_pos[:, :len_res_padded]

            # Clean and noisy share the same positions (same image, same spatial layout)
            full_pos_ids = torch.cat([prefix_pos_3d, response_pos_3d, response_pos_3d], dim=1)
            full_pos_ids = full_pos_ids.unsqueeze(1)  # [3, 1, total_len]

            # Loss mask: only masked positions in noisy region
            noisy_start = len_pre + len_res_padded
            noisy_valid_end = noisy_start + len(response_ids)
            valid_noisy_mask = torch.zeros(total_len, dtype=torch.bool)
            valid_noisy_mask[noisy_start:noisy_valid_end] = True
            loss_mask = (full_input_ids == self.mask_token_id) & valid_noisy_mask

            response_mask = valid_noisy_mask.clone()

            # Generation mask: VQ region in noisy part (for dual-head routing)
            generation_mask = valid_noisy_mask.clone()

            # VQ code mask: positions in input that are VQ codes (for vq_embed routing)
            # In t2i: no source VQ in prefix. In editing: source VQ in prefix.
            vq_code_mask = torch.zeros(total_len, dtype=torch.bool)
            # Mark VQ positions in clean response (between gen_start and gen_end, excluding newlines)
            # and noisy response region
            # For simplicity, mark all non-special-token positions in response as VQ
            for region_start in [len_pre, noisy_start]:
                for pos in range(region_start, min(region_start + len_res_padded, total_len)):
                    tok = full_input_ids[pos].item()
                    if tok not in (
                        self.img_gen_start_token_id, self.img_gen_end_token_id,
                        self.img_newline_token_id, self.pad_token_id, self.mask_token_id,
                    ):
                        vq_code_mask[pos] = True

            batch_input_ids.append(full_input_ids)
            batch_labels.append(full_labels)
            batch_pos_ids.append(full_pos_ids)
            batch_loss_mask.append(loss_mask)
            batch_response_mask.append(response_mask)
            batch_t.append(full_t)
            batch_generation_mask.append(generation_mask)
            batch_vq_code_mask.append(vq_code_mask)
            actual_lens.append(total_len)
            len_pre_list.append(len_pre)
            len_res_list.append(len_res_padded)

        if len(actual_lens) == 0:
            raise ValueError("Empty batch after processing")

        max_len = min(max(actual_lens), self.max_len)
        batch_masks = []

        for idx in range(len(batch_input_ids)):
            diff_len = max_len - actual_lens[idx]
            if diff_len < 0:
                batch_input_ids[idx] = batch_input_ids[idx][:max_len]
                batch_labels[idx] = batch_labels[idx][:max_len]
                batch_pos_ids[idx] = batch_pos_ids[idx][..., :max_len]
                batch_loss_mask[idx] = batch_loss_mask[idx][:max_len]
                batch_response_mask[idx] = batch_response_mask[idx][:max_len]
                batch_t[idx] = batch_t[idx][:max_len]
                batch_generation_mask[idx] = batch_generation_mask[idx][:max_len]
                batch_vq_code_mask[idx] = batch_vq_code_mask[idx][:max_len]
            elif diff_len > 0:
                batch_input_ids[idx] = torch.cat([batch_input_ids[idx], torch.full((diff_len,), self.pad_token_id, dtype=batch_input_ids[idx].dtype)])
                batch_labels[idx] = torch.cat([batch_labels[idx], torch.full((diff_len,), self.pad_token_id, dtype=batch_labels[idx].dtype)])
                batch_pos_ids[idx] = torch.cat([batch_pos_ids[idx], torch.zeros((3, 1, diff_len), dtype=batch_pos_ids[idx].dtype)], dim=-1)
                batch_loss_mask[idx] = torch.cat([batch_loss_mask[idx], torch.zeros(diff_len, dtype=torch.bool)])
                batch_response_mask[idx] = torch.cat([batch_response_mask[idx], torch.zeros(diff_len, dtype=torch.bool)])
                batch_t[idx] = torch.cat([batch_t[idx], torch.full((diff_len,), 0.5, dtype=torch.float32)])
                batch_generation_mask[idx] = torch.cat([batch_generation_mask[idx], torch.zeros(diff_len, dtype=torch.bool)])
                batch_vq_code_mask[idx] = torch.cat([batch_vq_code_mask[idx], torch.zeros(diff_len, dtype=torch.bool)])

            batch_masks.append(
                self.create_attention_mask(len_pre_list[idx], len_res_list[idx], self.block_size, max_len)
            )

        results = {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "position_ids": torch.cat(batch_pos_ids, dim=1).to(torch.int64),
            "response_mask": torch.stack(batch_response_mask),
            "loss_mask": torch.stack(batch_loss_mask).to(torch.bool),
            "attention_mask": torch.cat(batch_masks, dim=0),
            "t": torch.stack(batch_t),
            "generation_mask": torch.stack(batch_generation_mask).to(torch.bool),
            "vq_code_mask": torch.stack(batch_vq_code_mask).to(torch.bool),
            "block_size": self.block_size,
            "num_samples": torch.tensor(len(batch)),
        }
        return results


class internvl_block_collate_fn(bard_vl_block_collate_fn):
    """Block-diffusion collate for Bard-InternVL.

    Reuses the diffusion machinery from `bard_vl_block_collate_fn` (noise sampling, block
    layout, loss/state masks, padding/truncation) and only overrides the InternVL-specific
    bits:
      - image encoding via `InternVLProcessorLite` (dynamic tiling + <IMG_CONTEXT> expansion);
      - 1D position_ids (Qwen3 RoPE) instead of 3D MRoPE;
      - FLOAT additive 4D attention mask (the Qwen3 LLM adds the mask, so a bool mask would
        corrupt attention).
    """

    def __init__(self, processor, **kwargs):
        self.processor = processor
        self.max_len = kwargs.get("max_len", 8192)
        self.model_type = kwargs.get("model_type", "bard-internvl")
        self.enable_thinking = kwargs.get("enable_thinking", "auto")

        self.image_token_id = kwargs.get("image_token_id", 151671)  # <IMG_CONTEXT>
        self.mask_token_id = kwargs.get("mask_token_id", 151935)
        self.vocab_size = kwargs.get("vocab_size", 151936)
        if self.image_token_id == self.mask_token_id:
            raise ValueError(
                f"image_token_id ({self.image_token_id}) must differ from mask_token_id "
                f"({self.mask_token_id}); the diffusion <MASK> would collide with <IMG_CONTEXT>."
            )

        pad_token = kwargs.get("pad_token", "<|endoftext|>")
        self.pad_token_id = processor.tokenizer.encode(pad_token, add_special_tokens=False)[0]
        im_end_token = kwargs.get("im_end_token", "<|im_end|>")
        self.im_end_id = processor.tokenizer.encode(im_end_token, add_special_tokens=False)[0]
        self.newline_token_ids = self._encode_text("\n")

        self.ignore_index = -100
        self.block_size = kwargs.get("block_size", 32)
        self.min_mask_rate = float(kwargs.get("min_mask_rate", 0.001))
        self.max_mask_rate = float(kwargs.get("max_mask_rate", 1.0))

        noise_scheduler = kwargs.get("noise_scheduler", kwargs.get("path_scheduler", "CondOT"))
        self.noise_scheduler, self.path = _build_noise_scheduler(noise_scheduler)
        self.semantic_top_k = None

    def _extract_images(self, messages):
        images = []
        for message in messages:
            content_list = message.get("content", [])
            if not isinstance(content_list, list):
                continue
            for content in content_list:
                if content.get("image") is not None:
                    images.append(content["image"])
        return images if images else None

    def _extract_videos(self, messages):
        videos = []
        for message in messages:
            content_list = message.get("content", [])
            if not isinstance(content_list, list):
                continue
            for content in content_list:
                if content.get("video") is not None:
                    videos.append(content["video"])
        return videos if videos else None

    def _split_turn(self, messages, image_inputs, video_inputs=None, *args, **kwargs):
        assistant_idx = self._assistant_idx(messages)
        if assistant_idx is None:
            return None, None, None

        prefix_messages = messages[:assistant_idx]
        assistant_message = messages[assistant_idx]
        enable_thinking = _resolve_enable_thinking(messages, assistant_message, self.enable_thinking)

        prefix_base = _apply_chat_template(
            self.processor, prefix_messages, tokenize=False,
            add_generation_prompt=False, enable_thinking=enable_thinking,
        )
        prefix_text = _apply_chat_template(
            self.processor, prefix_messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=enable_thinking,
        )
        if not prefix_text.startswith(prefix_base):
            return None, None, None

        header_text = prefix_text[len(prefix_base):]
        full_text = _apply_chat_template(
            self.processor, messages[:assistant_idx + 1], tokenize=False,
            add_generation_prompt=False, enable_thinking=enable_thinking,
        )
        if not full_text.startswith(prefix_base):
            return None, None, None
        assistant_text = full_text[len(prefix_base):]
        response_text = _extract_response_text(assistant_text, header_text)
        if response_text is None:
            return None, None, None

        prefix_inputs = self.processor(
            text=[prefix_text], images=image_inputs, videos=video_inputs,
            padding=False, return_tensors="pt",
        )
        prefix_ids = prefix_inputs["input_ids"][0]
        response_ids = self._encode_text(response_text)
        # Wrap so the caller can use attribute access like the Qwen path.
        prefix_obj = SimpleNamespace(
            input_ids=prefix_inputs["input_ids"],
            pixel_values=prefix_inputs.get("pixel_values", None),
        )
        return prefix_ids, response_ids, prefix_obj

    def create_attention_mask(self, prefix_len, res_len, block_size, max_len,
                              response_valid_len=None, keep_noisy_suffix_latent: bool = False):
        # Build the boolean visibility mask via the parent, then convert to a float additive
        # mask (0 keep / -inf drop) which the Qwen3 LLM consumes correctly.
        bool_mask = super().create_attention_mask(
            prefix_len, res_len, block_size, max_len,
            response_valid_len=response_valid_len,
            keep_noisy_suffix_latent=keep_noisy_suffix_latent,
        )
        float_mask = torch.zeros_like(bool_mask, dtype=torch.float32)
        float_mask.masked_fill_(~bool_mask, torch.finfo(torch.float32).min)
        return float_mask

    def _fallback_batch(self, n):
        """A trivial text-only conversation used when every sample in a batch was skipped.

        Keeps the DataLoader worker alive (avoids an empty batch crashing the rank) at the
        cost of one near-zero-loss step. n matches the original (post-skip) batch length.
        """
        msg = [
            {"role": "system", "content": "You are a helpful assistant.", "prior_dist": "Mask"},
            {"role": "user", "content": [{"type": "text", "text": "Hello."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi! How can I help you today?"}]},
        ]
        return [msg for _ in range(max(1, n))]

    def __call__(self, batch):
        batch_labels, batch_input_ids, batch_pos_ids = [], [], []
        batch_loss_mask, batch_state_mask, batch_response_mask = [], [], []
        batch_t, batch_token_types, batch_block_indices = [], [], []
        actual_lens, len_pre_list, len_res_list, len_res_valid_list = [], [], [], []
        batch_pixel_values = []

        for messages in batch:
            prior_dist = messages[0].get("prior_dist", "Mask")
            image_inputs = self._extract_images(messages)
            video_inputs = self._extract_videos(messages)

            prefix_ids, response_ids, prefix_inputs = self._split_turn(messages, image_inputs, video_inputs)
            if prefix_ids is None or response_ids is None or prefix_inputs is None or response_ids.numel() == 0:
                continue

            len_pre = len(prefix_ids)
            max_res_len = ((self.max_len - len_pre) // (2 * self.block_size)) * self.block_size
            if max_res_len <= 0:
                continue
            response_ids = response_ids[:max_res_len]

            num_res_blocks = (len(response_ids) + self.block_size - 1) // self.block_size
            response_padded = torch.full(
                (num_res_blocks * self.block_size,), self.pad_token_id,
                device=prefix_ids.device, dtype=prefix_ids.dtype,
            )
            response_padded[:len(response_ids)] = response_ids
            len_res_padded = len(response_padded)

            # ---- noise (reuse parent samplers) ----
            if prior_dist == "Mix":
                noisy_response_ids, noisy_t = self._sample_mix_noise(response_padded, num_res_blocks)
            elif prior_dist == "Edit":
                noisy_response_ids, noisy_t = self._sample_edit_noise(response_padded)
            else:
                noisy_ids_list, noisy_t_list = [], []
                if prior_dist == "Uniform":
                    t = self._sample_block_times(num_res_blocks, device=prefix_ids.device, low=0.0, high=0.999)
                elif prior_dist == "Mask":
                    t = self._sample_block_times(num_res_blocks, device=prefix_ids.device,
                                                 low=self.min_mask_rate, high=self.max_mask_rate)
                else:
                    raise ValueError(f"Unsupported prior_dist '{prior_dist}'")
                for i in range(num_res_blocks):
                    block_clean = response_padded[i * self.block_size: (i + 1) * self.block_size]
                    if prior_dist == "Uniform":
                        x_0 = torch.randint_like(block_clean.unsqueeze(0), low=0, high=self.vocab_size)
                        block_noisy = _sample_discrete_noise(
                            t=t[i].unsqueeze(0), x_0=x_0, x_1=block_clean.unsqueeze(0),
                            scheduler=self.noise_scheduler, path=self.path,
                        )[0]
                    else:  # Mask
                        change_indices = torch.rand(len(block_clean), device=block_clean.device) <= t[i].repeat(len(block_clean))
                        block_noisy = torch.where(change_indices, self.mask_token_id, block_clean)
                    noisy_ids_list.append(block_noisy)
                    noisy_t_list.append(torch.full((self.block_size,), float(t[i].item()),
                                                   dtype=torch.float32, device=prefix_ids.device))
                noisy_response_ids = torch.cat(noisy_ids_list)
                noisy_t = torch.cat(noisy_t_list)
                if prior_dist == "Mask":
                    noisy_response_ids[response_ids.numel():] = self.mask_token_id
                    noisy_response_ids = self._mask_response_terminator(response_ids, noisy_response_ids)

            full_input_ids = torch.cat([prefix_ids, response_padded, noisy_response_ids])
            full_labels = torch.cat([prefix_ids, response_padded, response_padded])

            if prior_dist == "Uniform":
                prefix_t = torch.full((len(prefix_ids),), 0.999, dtype=torch.float32, device=prefix_ids.device)
                clean_t = torch.full((len(response_padded),), 0.999, dtype=torch.float32, device=prefix_ids.device)
            else:
                prefix_t = torch.full((len(prefix_ids),), 0.001, dtype=torch.float32, device=prefix_ids.device)
                clean_t = torch.full((len(response_padded),), 0.001, dtype=torch.float32, device=prefix_ids.device)
            full_t = torch.cat([prefix_t, clean_t, noisy_t])

            t_types = torch.zeros(len(full_input_ids), dtype=torch.int8, device=prefix_ids.device)
            t_types[len_pre: len_pre + len_res_padded] = 1
            t_types[len_pre + len_res_padded:] = 2

            block_ids = torch.zeros(len(full_input_ids), dtype=torch.int32, device=prefix_ids.device)
            num_prefix_blocks = (len(prefix_ids) + self.block_size - 1) // self.block_size
            for b in range(num_prefix_blocks):
                block_ids[b * self.block_size: min((b + 1) * self.block_size, len(prefix_ids))] = b
            for b in range(num_res_blocks):
                curr = num_prefix_blocks + b
                block_ids[len_pre + b * self.block_size: len_pre + (b + 1) * self.block_size] = curr
                block_ids[len_pre + len_res_padded + b * self.block_size: len_pre + len_res_padded + (b + 1) * self.block_size] = curr

            # ---- 1D position_ids (Qwen3 RoPE), noisy copies clean positions ----
            pos_pre_clean = torch.arange(len_pre + len_res_padded, device=prefix_ids.device)
            pos_noisy = pos_pre_clean[len_pre:]
            full_pos_ids = torch.cat([pos_pre_clean, pos_noisy]).unsqueeze(0)  # [1, L]

            noisy_start = len_pre + len_res_padded
            noisy_valid_end = noisy_start + len(response_ids)
            valid_noisy_mask = torch.zeros(len(full_input_ids), dtype=torch.bool, device=prefix_ids.device)
            valid_noisy_mask[noisy_start:noisy_valid_end] = True
            response_mask = valid_noisy_mask.clone()

            loss_mask = torch.zeros(len(full_input_ids), dtype=torch.bool, device=prefix_ids.device)
            if prior_dist in ["Uniform", "Edit"]:
                loss_mask = valid_noisy_mask.clone()
            elif prior_dist == "Mix":
                loss_mask = torch.where(full_input_ids == full_labels, False, True) & valid_noisy_mask
            elif prior_dist == "Mask":
                loss_mask = (full_input_ids == self.mask_token_id) & valid_noisy_mask

            state_mask = torch.zeros(len(full_input_ids), dtype=torch.bool, device=prefix_ids.device)
            state_mask[noisy_start: len_pre + 2 * len_res_padded] = True

            batch_labels.append(full_labels)
            batch_input_ids.append(full_input_ids)
            batch_pos_ids.append(full_pos_ids)
            batch_loss_mask.append(loss_mask)
            batch_state_mask.append(state_mask)
            batch_t.append(full_t)
            batch_token_types.append(t_types)
            batch_block_indices.append(block_ids)
            batch_response_mask.append(response_mask)
            actual_lens.append(len(full_input_ids))
            len_pre_list.append(len_pre)
            len_res_list.append(len_res_padded)
            len_res_valid_list.append(len(response_ids))

            if prefix_inputs.pixel_values is not None:
                batch_pixel_values.append(prefix_inputs.pixel_values)

        if len(actual_lens) == 0:
            # local_batch_size=1 means a single skipped sample (e.g. _split_turn returns None
            # due to a template edge case or an empty assistant turn) would yield an empty
            # batch and crash this DataLoader worker -> the rank dies -> all other ranks hang on
            # the next collective. Fall back to a guaranteed-valid synthetic sample so the step
            # still runs (its loss contribution is negligible) instead of taking down training.
            if not getattr(self, "_in_fallback", False):
                self._in_fallback = True
                try:
                    return self(self._fallback_batch(len(batch)))
                finally:
                    self._in_fallback = False
            raise ValueError("actual_lens is empty even for the fallback sample")

        max_len = min(max(actual_lens), self.max_len)
        batch_masks = []
        for idx in range(len(batch_input_ids)):
            diff_len = max_len - actual_lens[idx]
            if diff_len < 0:
                batch_input_ids[idx] = batch_input_ids[idx][:max_len]
                batch_labels[idx] = batch_labels[idx][:max_len]
                batch_pos_ids[idx] = batch_pos_ids[idx][..., :max_len]
                batch_response_mask[idx] = batch_response_mask[idx][:max_len]
                batch_loss_mask[idx] = batch_loss_mask[idx][:max_len]
                batch_state_mask[idx] = batch_state_mask[idx][:max_len]
                batch_t[idx] = batch_t[idx][:max_len]
                batch_token_types[idx] = batch_token_types[idx][:max_len]
                batch_block_indices[idx] = batch_block_indices[idx][:max_len]
            elif diff_len > 0:
                dev = batch_input_ids[idx].device
                batch_input_ids[idx] = torch.cat([batch_input_ids[idx],
                    torch.full((diff_len,), self.pad_token_id, device=dev, dtype=batch_input_ids[idx].dtype)])
                batch_labels[idx] = torch.cat([batch_labels[idx],
                    torch.full((diff_len,), self.pad_token_id, device=dev, dtype=batch_labels[idx].dtype)])
                batch_pos_ids[idx] = torch.cat([batch_pos_ids[idx],
                    torch.zeros((1, diff_len), device=dev, dtype=batch_pos_ids[idx].dtype)], dim=-1)
                batch_response_mask[idx] = torch.cat([batch_response_mask[idx],
                    torch.zeros((diff_len,), device=dev, dtype=torch.bool)])
                batch_loss_mask[idx] = torch.cat([batch_loss_mask[idx],
                    torch.zeros((diff_len,), device=dev, dtype=torch.bool)])
                batch_state_mask[idx] = torch.cat([batch_state_mask[idx],
                    torch.zeros((diff_len,), device=dev, dtype=torch.bool)])
                batch_t[idx] = torch.cat([batch_t[idx],
                    torch.full((diff_len,), 0.5, device=dev, dtype=batch_t[idx].dtype)])
                batch_token_types[idx] = torch.cat([batch_token_types[idx],
                    torch.full((diff_len,), -1, device=dev, dtype=batch_token_types[idx].dtype)])
                batch_block_indices[idx] = torch.cat([batch_block_indices[idx],
                    torch.full((diff_len,), -1, device=dev, dtype=batch_block_indices[idx].dtype)])

            batch_masks.append(self.create_attention_mask(
                len_pre_list[idx], len_res_list[idx], self.block_size, max_len,
                response_valid_len=len_res_valid_list[idx], keep_noisy_suffix_latent=True,
            ))

        results = {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "position_ids": torch.cat(batch_pos_ids, dim=0).to(torch.int64),  # [B, max_len]
            "response_mask": torch.stack(batch_response_mask),
            "loss_mask": torch.stack(batch_loss_mask).to(torch.bool),
            "state_mask": torch.stack(batch_state_mask).to(torch.bool),
            "attention_mask": torch.cat(batch_masks, dim=0),
            "t": torch.stack(batch_t),
            "block_size": self.block_size,
            "num_samples": torch.tensor(len(batch)),
        }
        if batch_pixel_values:
            results["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
        return results


# Mapping of processor types to their collate functions
COLLATE_FNS = {
    # "Qwen2_5_VLProcessor": qwen2_5_collate_fn,
    # "Qwen3OmniMoeProcessor": qwen3_omni_collate_fn,
    "default": default_collate_fn,
    "BardVLProcessor": bard_vl_block_collate_fn,
    "BardUniProcessor": bard_uni_collate_fn,
    "InternVLProcessorLite": internvl_block_collate_fn,
}
