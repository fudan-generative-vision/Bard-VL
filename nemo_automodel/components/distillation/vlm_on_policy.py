import inspect
from types import SimpleNamespace

import torch

from nemo_automodel.components.distillation.vlm_kd import derive_response_segments


class AttrDict(dict):
    __getattr__ = dict.get


def _slice_position_ids(position_ids: torch.Tensor | None, batch_idx: int, length: int) -> torch.Tensor | None:
    if position_ids is None:
        return None
    if position_ids.ndim == 3:
        return position_ids[:, batch_idx : batch_idx + 1, :length]
    return position_ids[batch_idx : batch_idx + 1, :length]


def _slice_or_extend_position_ids(
    position_ids: torch.Tensor | None,
    batch_idx: int,
    target_length: int,
) -> torch.Tensor | None:
    sliced = _slice_position_ids(position_ids, batch_idx, target_length)
    if sliced is None:
        return None

    current_length = int(sliced.shape[-1])
    if current_length >= target_length:
        return sliced

    if sliced.ndim == 3:
        padded = torch.zeros(
            (sliced.shape[0], sliced.shape[1], target_length),
            dtype=sliced.dtype,
            device=sliced.device,
        )
    else:
        padded = torch.zeros(
            (sliced.shape[0], target_length),
            dtype=sliced.dtype,
            device=sliced.device,
        )

    if current_length > 0:
        padded[..., :current_length] = sliced
        extra = torch.arange(
            1,
            target_length - current_length + 1,
            dtype=sliced.dtype,
            device=sliced.device,
        )
        if sliced.ndim == 3:
            padded[..., current_length:] = sliced[..., current_length - 1 : current_length] + extra.view(1, 1, -1)
        else:
            padded[..., current_length:] = sliced[..., current_length - 1 : current_length] + extra.view(1, -1)
        return padded

    extra = torch.arange(target_length, dtype=sliced.dtype, device=sliced.device)
    if sliced.ndim == 3:
        return extra.view(1, 1, -1).expand(sliced.shape[0], sliced.shape[1], -1)
    return extra.view(1, -1).expand(sliced.shape[0], -1)


def build_prompt_batch(batch: dict[str, torch.Tensor], pad_token_id: int) -> tuple[AttrDict, torch.Tensor, torch.Tensor]:
    """从原始训练样本中裁出 prompt 部分，供 on-policy 生成使用。"""
    # 原始 batch 里通常已经是：
    #   [prompt][clean_response][noisy_response]
    # 这里要先把最前面的 prompt 单独抽出来，让 student 基于 prompt 去重新生成一段 response。
    # 返回值里的 `prompt_lens` / `response_lens` 会在后续 student/teacher batch 构造时继续复用。
    prompt_lens, response_lens, _ = derive_response_segments(batch["response_mask"])
    batch_size = batch["input_ids"].shape[0]
    max_prompt_len = int(prompt_lens.max().item()) if batch_size > 0 else 0

    prompt_input_ids = torch.full(
        (batch_size, max_prompt_len),
        fill_value=pad_token_id,
        dtype=batch["input_ids"].dtype,
        device=batch["input_ids"].device,
    )
    prompt_attention_mask = torch.zeros((batch_size, max_prompt_len), dtype=torch.long, device=batch["input_ids"].device)
    prompt_position_ids = None
    if "position_ids" in batch:
        if batch["position_ids"].ndim == 3:
            prompt_position_ids = torch.zeros(
                (batch["position_ids"].shape[0], batch_size, max_prompt_len),
                dtype=batch["position_ids"].dtype,
                device=batch["position_ids"].device,
            )
        else:
            prompt_position_ids = torch.zeros(
                (batch_size, max_prompt_len),
                dtype=batch["position_ids"].dtype,
                device=batch["position_ids"].device,
            )

    for batch_idx in range(batch_size):
        prompt_len = int(prompt_lens[batch_idx].item())
        if prompt_len <= 0:
            continue
        # 这里只拷贝 prompt 区间，不把 response 带进生成输入里。
        prompt_input_ids[batch_idx, :prompt_len] = batch["input_ids"][batch_idx, :prompt_len]
        prompt_attention_mask[batch_idx, :prompt_len] = 1
        if prompt_position_ids is not None:
            sliced = _slice_position_ids(batch["position_ids"], batch_idx, prompt_len)
            if prompt_position_ids.ndim == 3:
                prompt_position_ids[:, batch_idx : batch_idx + 1, :prompt_len] = sliced
            else:
                prompt_position_ids[batch_idx : batch_idx + 1, :prompt_len] = sliced

    prompt_batch = AttrDict(
        {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
        }
    )
    if prompt_position_ids is not None:
        prompt_batch["position_ids"] = prompt_position_ids
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if key in batch:
            prompt_batch[key] = batch[key]
    return prompt_batch, prompt_lens, response_lens


def _autoregressive_rollout(
    model,
    prompt_batch: AttrDict,
    response_lens: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    max_response_len = int(response_lens.max().item()) if response_lens.numel() > 0 else 0
    input_ids = prompt_batch["input_ids"]
    batch_size = input_ids.shape[0]
    generated = torch.full(
        (batch_size, max_response_len),
        fill_value=pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    running = input_ids
    for step in range(max_response_len):
        outputs = model(input_ids=running)
        next_token = outputs.logits[:, -1].argmax(dim=-1)
        generated[:, step] = next_token
        running = torch.cat([running, next_token.unsqueeze(-1)], dim=-1)
    return generated


def _generate_with_hf_generate(
    model,
    prompt_batch: AttrDict,
    max_response_len: int,
    generation_config: SimpleNamespace,
    pad_token_id: int,
    return_generation_logits: bool = False,
    return_rollout_state: bool = False,
) -> torch.Tensor | tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    generate_inputs = {}
    for key in (
        "input_ids",
        "attention_mask",
        "position_ids",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    ):
        if key in prompt_batch:
            generate_inputs[key] = prompt_batch[key]

    temperature = float(getattr(generation_config, "temperature", 0.0))
    top_k = int(getattr(generation_config, "top_k", 0))
    top_p = float(getattr(generation_config, "top_p", 1.0))
    do_sample = temperature > 0.0 or top_k > 0 or top_p < 1.0

    generate_kwargs = {
        "max_new_tokens": max_response_len,
        "pad_token_id": pad_token_id,
        "eos_token_id": int(getattr(generation_config, "eos_token_id", pad_token_id)),
        "do_sample": do_sample,
        "use_cache": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_k"] = top_k
        generate_kwargs["top_p"] = top_p

    generate_fn = model.generate
    use_generate_kd = (return_generation_logits or return_rollout_state) and hasattr(model, "generate_kd")
    if use_generate_kd:
        generate_fn = model.generate_kd

    generate_signature = inspect.signature(generate_fn)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in generate_signature.parameters.values()
    )
    accepts_input_ids = "input_ids" in generate_signature.parameters

    # Dstar 的自定义 `generate()` 接口接收一个整体 `inputs` 对象，而不是标准 HF 的
    # `input_ids=..., attention_mask=...` 形式；标准 HF/Transformers 模型则继续走 kwargs。
    if not accepts_var_kwargs and not accepts_input_ids and "inputs" in generate_signature.parameters:
        # 自定义 diffusion generate 接口并不接受 HF 风格的 `pad_token_id` / `do_sample`
        # / `use_cache` 等参数，这里按函数签名筛掉无关项，并补上 on-policy 配置里
        # 真正需要传给 Dstar 生成逻辑的参数。
        custom_generate_kwargs = {
            "max_new_tokens": max_response_len,
            "eos_token_id": int(getattr(generation_config, "eos_token_id", pad_token_id)),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "block_size": int(getattr(generation_config, "block_size", 4)),
            "horizon_size": int(getattr(generation_config, "horizon_size", 0)),
            "denoising_steps": int(getattr(generation_config, "denoising_steps", 4)),
            "remasking_strategy": str(
                getattr(generation_config, "remasking_strategy", "low_confidence_dynamic")
            ),
            "confidence_threshold": float(getattr(generation_config, "confidence_threshold", 0.9)),
            "pad_target_penalty": float(getattr(generation_config, "pad_target_penalty", 1.0)),
            "mask_token_id": int(getattr(generation_config, "mask_token_id", 151671)),
            "return_step_stats": bool(return_generation_logits or return_rollout_state),
            "return_rollout_state": bool(return_rollout_state),
        }
        filtered_generate_kwargs = {
            key: value
            for key, value in custom_generate_kwargs.items()
            if key in generate_signature.parameters
        }
        if use_generate_kd:
            with torch.enable_grad():
                generated = generate_fn(
                    prompt_batch,
                    **filtered_generate_kwargs,
                )
        else:
            generated = generate_fn(
                prompt_batch,
                **filtered_generate_kwargs,
            )
        if return_generation_logits or return_rollout_state:
            generation_block_step_counts = None
            if len(generated) == 5:
                (
                    generated_ids,
                    generation_payload,
                    generation_batch_indices,
                    generation_response_positions,
                    generation_block_step_counts,
                ) = generated
            else:
                generated_ids, generation_payload, generation_batch_indices, generation_response_positions = generated
            return (
                generated_ids[:, :max_response_len],
                generation_payload,
                generation_batch_indices,
                generation_response_positions,
                generation_block_step_counts,
            )
        return generated[:, :max_response_len]

    generated = model.generate(**generate_inputs, **generate_kwargs)
    prompt_len = prompt_batch["input_ids"].shape[1]
    generated = generated[:, prompt_len : prompt_len + max_response_len]

    if generated.shape[1] < max_response_len:
        padded = torch.full(
            (generated.shape[0], max_response_len),
            fill_value=pad_token_id,
            dtype=generated.dtype,
            device=generated.device,
        )
        padded[:, : generated.shape[1]] = generated
        generated = padded

    if return_generation_logits:
        return generated, None, None, None, None
    return generated


@torch.no_grad()
def sample_on_policy_responses(
    model,
    prompt_batch: AttrDict,
    response_lens: torch.Tensor,
    generation_config: SimpleNamespace,
    pad_token_id: int,
    return_generation_logits: bool = False,
    return_rollout_state: bool = False,
) -> torch.Tensor | tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """基于 prompt 做一次 on-policy 采样，返回每个样本的响应 token。"""
    # 这里的目标是先让 student 自己生成一段 response，后续 student/teacher 都围绕这段
    # response 构造各自的训练输入，因此这里返回的是“纯生成结果”，不包含 prompt 本身。
    max_response_len = int(response_lens.max().item()) if response_lens.numel() > 0 else 0
    configured_max_response_len = int(getattr(generation_config, "max_response_len", 0))
    if configured_max_response_len > 0:
        max_response_len = min(max_response_len, configured_max_response_len)
    if max_response_len <= 0:
        empty = torch.empty(
            (prompt_batch["input_ids"].shape[0], 0),
            dtype=prompt_batch["input_ids"].dtype,
            device=prompt_batch["input_ids"].device,
        )
        if return_generation_logits:
            return empty, None, None, None, None
        if return_rollout_state:
            return empty, None, None, None, None
        return empty

    if hasattr(model, "generate"):
        return _generate_with_hf_generate(
            model=model,
            prompt_batch=prompt_batch,
            max_response_len=max_response_len,
            generation_config=generation_config,
            pad_token_id=pad_token_id,
            return_generation_logits=return_generation_logits,
            return_rollout_state=return_rollout_state,
        )
    generated = _autoregressive_rollout(model, prompt_batch, response_lens, pad_token_id)
    if return_generation_logits:
        return generated, None, None, None, None
    if return_rollout_state:
        return generated, None, None, None, None
    return generated


def _sample_masked_indices(
    response_lens: torch.Tensor,
    mask_rate: float,
    device: torch.device,
) -> list[torch.Tensor]:
    masked_indices = []
    for response_len in response_lens.tolist():
        if response_len <= 0:
            masked_indices.append(torch.empty((0,), dtype=torch.bool, device=device))
            continue
        current = torch.rand(response_len, device=device) < mask_rate
        if not torch.any(current):
            current[0] = True
        masked_indices.append(current)
    return masked_indices


def _build_generic_packed_batch(
    batch: dict[str, torch.Tensor],
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
    sampled_responses: torch.Tensor,
    pad_token_id: int,
    mask_token_id: int,
    mask_rate: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch_size = batch["input_ids"].shape[0]
    device = batch["input_ids"].device
    sampled_responses = sampled_responses.to(device)
    masked_indices = _sample_masked_indices(response_lens, mask_rate=mask_rate, device=device)
    max_packed_len = int((prompt_lens + 2 * response_lens).max().item()) if batch_size > 0 else 0

    packed_input_ids = torch.full(
        (batch_size, max_packed_len),
        fill_value=pad_token_id,
        dtype=batch["input_ids"].dtype,
        device=batch["input_ids"].device,
    )
    packed_attention_mask = torch.zeros((batch_size, max_packed_len), dtype=torch.long, device=batch["input_ids"].device)
    packed_response_mask = torch.zeros((batch_size, max_packed_len), dtype=torch.bool, device=batch["input_ids"].device)
    packed_loss_mask = torch.zeros((batch_size, max_packed_len), dtype=torch.bool, device=batch["input_ids"].device)
    packed_position_ids = None
    if "position_ids" in batch:
        if batch["position_ids"].ndim == 3:
            packed_position_ids = torch.zeros(
                (batch["position_ids"].shape[0], batch_size, max_packed_len),
                dtype=batch["position_ids"].dtype,
                device=batch["position_ids"].device,
            )
        else:
            packed_position_ids = torch.zeros(
                (batch_size, max_packed_len),
                dtype=batch["position_ids"].dtype,
                device=batch["position_ids"].device,
            )
    target_tokens = []

    for batch_idx in range(batch_size):
        prompt_len = int(prompt_lens[batch_idx].item())
        response_len = int(response_lens[batch_idx].item())
        packed_len = prompt_len + 2 * response_len
        packed_attention_mask[batch_idx, :packed_len] = 1
        packed_input_ids[batch_idx, :prompt_len] = batch["input_ids"][batch_idx, :prompt_len]
        packed_input_ids[batch_idx, prompt_len : prompt_len + response_len] = sampled_responses[batch_idx, :response_len]

        noisy = sampled_responses[batch_idx, :response_len].clone()
        noisy[masked_indices[batch_idx]] = mask_token_id
        noisy_start = prompt_len + response_len
        packed_input_ids[batch_idx, noisy_start : noisy_start + response_len] = noisy
        packed_response_mask[batch_idx, noisy_start : noisy_start + response_len] = True
        packed_loss_mask[batch_idx, noisy_start : noisy_start + response_len] = masked_indices[batch_idx]
        target_tokens.append(sampled_responses[batch_idx, :response_len][masked_indices[batch_idx]])

        if packed_position_ids is not None:
            clean_pos = _slice_or_extend_position_ids(batch["position_ids"], batch_idx, prompt_len + response_len)
            if clean_pos.ndim == 3:
                packed_position_ids[:, batch_idx : batch_idx + 1, : prompt_len + response_len] = clean_pos
                packed_position_ids[:, batch_idx : batch_idx + 1, noisy_start : noisy_start + response_len] = clean_pos[
                    :, :, prompt_len : prompt_len + response_len
                ]
            else:
                packed_position_ids[batch_idx : batch_idx + 1, : prompt_len + response_len] = clean_pos
                packed_position_ids[batch_idx : batch_idx + 1, noisy_start : noisy_start + response_len] = clean_pos[
                    :, prompt_len : prompt_len + response_len
                ]

    result = {
        "input_ids": packed_input_ids,
        "attention_mask": packed_attention_mask,
        "response_mask": packed_response_mask,
        "loss_mask": packed_loss_mask,
    }
    if packed_position_ids is not None:
        result["position_ids"] = packed_position_ids
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if key in batch:
            result[key] = batch[key]
    return result, torch.cat(target_tokens, dim=0) if target_tokens else packed_input_ids.new_empty((0,))


def build_student_on_policy_batch(
    student,
    batch: dict[str, torch.Tensor],
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
    sampled_responses: torch.Tensor,
    pad_token_id: int,
    mask_token_id: int,
    mask_rate: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """把 on-policy 采样结果整理成 student 训练所需的 batch。"""
    # 这一步的核心是：
    # 1. 用 prompt + sampled response 还原一份 student 看到的序列；
    # 2. 在 response 区间内随机挑一部分 token 作为监督目标；
    # 3. 返回 student 前向需要的输入，以及这些被监督位置对应的 target token。
    if not hasattr(student, "prepare_for_rl_training"):
        return _build_generic_packed_batch(
            batch=batch,
            prompt_lens=prompt_lens,
            response_lens=response_lens,
            sampled_responses=sampled_responses,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            mask_rate=mask_rate,
        )

    batch_size = batch["input_ids"].shape[0]
    device = batch["input_ids"].device
    sampled_responses = sampled_responses.to(device)
    masked_indices = _sample_masked_indices(response_lens, mask_rate=mask_rate, device=device)
    max_len = int((prompt_lens + response_lens).max().item()) if batch_size > 0 else 0
    packed_input_ids = torch.full(
        (batch_size, max_len),
        fill_value=pad_token_id,
        dtype=batch["input_ids"].dtype,
        device=batch["input_ids"].device,
    )
    labels = torch.full((batch_size, max_len), fill_value=-100, dtype=batch["input_ids"].dtype, device=batch["input_ids"].device)
    masked_tensor = torch.zeros((batch_size, max_len), dtype=torch.bool, device=batch["input_ids"].device)
    target_tokens = []

    for batch_idx in range(batch_size):
        prompt_len = int(prompt_lens[batch_idx].item())
        response_len = int(response_lens[batch_idx].item())
        total_len = prompt_len + response_len
        packed_input_ids[batch_idx, :prompt_len] = batch["input_ids"][batch_idx, :prompt_len]
        packed_input_ids[batch_idx, prompt_len:total_len] = sampled_responses[batch_idx, :response_len]
        labels[batch_idx, prompt_len:total_len] = sampled_responses[batch_idx, :response_len]
        masked_tensor[batch_idx, prompt_len:total_len] = masked_indices[batch_idx]
        target_tokens.append(sampled_responses[batch_idx, :response_len][masked_indices[batch_idx]])

    # `prepare_for_rl_training()` 会把 packed 输入转换成 Dstar 训练期真正使用的格式，
    # 包括 attention_mask / position_ids / logits_to_keep 等辅助张量。
    prepared = student.prepare_for_rl_training(
        input_ids=packed_input_ids,
        position_ids=None if "position_ids" not in batch else batch["position_ids"][..., :max_len],
        labels=labels,
        masked_indices=masked_tensor,
        step_ids=None,
    )

    result = {
        "input_ids": prepared.input_ids,
        "attention_mask": prepared.attention_mask,
        "position_ids": prepared.position_ids,
        "response_mask": torch.zeros_like(prepared.logits_to_keep, dtype=torch.bool),
        "loss_mask": prepared.logits_to_keep.bool(),
        "logits_to_keep": prepared.logits_to_keep.bool(),
    }
    for batch_idx in range(batch_size):
        response_len = int(response_lens[batch_idx].item())
        prompt_len = int(prompt_lens[batch_idx].item())
        noisy_start = prompt_len + response_len
        result["response_mask"][batch_idx, noisy_start : noisy_start + response_len] = True
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if key in batch:
            result[key] = batch[key]
    return result, torch.cat(target_tokens, dim=0) if target_tokens else packed_input_ids.new_empty((0,))


def build_teacher_batch_from_response_tokens(
    batch: dict[str, torch.Tensor],
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
    response_tokens: torch.Tensor,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """把指定 response token 序列拼成 teacher 侧要看的输入。"""
    # teacher 不看 noisy/masked 的那一半序列，而是直接看：
    # prompt + response_tokens
    # 这样 teacher 输出的 logits 就对应“干净 response 上每个位置”的分布。
    batch_size = batch["input_ids"].shape[0]
    max_teacher_len = int((prompt_lens + response_lens).max().item()) if batch_size > 0 else 0
    teacher_input_ids = torch.full(
        (batch_size, max_teacher_len),
        fill_value=pad_token_id,
        dtype=batch["input_ids"].dtype,
        device=batch["input_ids"].device,
    )
    teacher_attention_mask = torch.zeros((batch_size, max_teacher_len), dtype=torch.long, device=batch["input_ids"].device)
    teacher_position_ids = None
    if "position_ids" in batch:
        if batch["position_ids"].ndim == 3:
            teacher_position_ids = torch.zeros(
                (batch["position_ids"].shape[0], batch_size, max_teacher_len),
                dtype=batch["position_ids"].dtype,
                device=batch["position_ids"].device,
            )
        else:
            teacher_position_ids = torch.zeros(
                (batch_size, max_teacher_len),
                dtype=batch["position_ids"].dtype,
                device=batch["position_ids"].device,
            )

    for batch_idx in range(batch_size):
        prompt_len = int(prompt_lens[batch_idx].item())
        response_len = int(response_lens[batch_idx].item())
        teacher_len = prompt_len + response_len
        # teacher 输入就是 prompt + response，没有 student 训练时那段 noisy block。
        teacher_input_ids[batch_idx, :prompt_len] = batch["input_ids"][batch_idx, :prompt_len]
        teacher_input_ids[batch_idx, prompt_len:teacher_len] = response_tokens[batch_idx, :response_len]
        teacher_attention_mask[batch_idx, :teacher_len] = 1
        if teacher_position_ids is not None:
            clean_pos = _slice_or_extend_position_ids(batch["position_ids"], batch_idx, teacher_len)
            if teacher_position_ids.ndim == 3:
                teacher_position_ids[:, batch_idx : batch_idx + 1, :teacher_len] = clean_pos
            else:
                teacher_position_ids[batch_idx : batch_idx + 1, :teacher_len] = clean_pos

    result = {
        "input_ids": teacher_input_ids,
        "attention_mask": teacher_attention_mask,
    }
    if teacher_position_ids is not None:
        result["position_ids"] = teacher_position_ids
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if key in batch:
            result[key] = batch[key]
    return result


def build_teacher_batch_from_on_policy_response(
    batch: dict[str, torch.Tensor],
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
    sampled_responses: torch.Tensor,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """把同一份 sampled response 拼成 teacher 侧要看的输入。"""
    return build_teacher_batch_from_response_tokens(
        batch=batch,
        prompt_lens=prompt_lens,
        response_lens=response_lens,
        response_tokens=sampled_responses,
        pad_token_id=pad_token_id,
    )


def build_teacher_batch_from_gold_response(
    batch: dict[str, torch.Tensor],
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    batch_size = batch["input_ids"].shape[0]
    max_response_len = int(response_lens.max().item()) if batch_size > 0 else 0
    gold_responses = torch.full(
        (batch_size, max_response_len),
        fill_value=pad_token_id,
        dtype=batch["labels"].dtype,
        device=batch["labels"].device,
    )

    for batch_idx in range(batch_size):
        prompt_len = int(prompt_lens[batch_idx].item())
        response_len = int(response_lens[batch_idx].item())
        if response_len <= 0:
            continue
        gold_responses[batch_idx, :response_len] = batch["labels"][batch_idx, prompt_len : prompt_len + response_len]

    return build_teacher_batch_from_response_tokens(
        batch=batch,
        prompt_lens=prompt_lens,
        response_lens=response_lens,
        response_tokens=gold_responses,
        pad_token_id=pad_token_id,
    )
