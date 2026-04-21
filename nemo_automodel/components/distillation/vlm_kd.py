import torch


def derive_response_segments(response_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if response_mask.ndim != 2:
        raise ValueError(f"`response_mask` must be 2D, got shape={tuple(response_mask.shape)}")

    response_lens = response_mask.long().sum(dim=-1)
    has_response = response_lens > 0
    first_response = torch.argmax(response_mask.long(), dim=-1)
    noisy_starts = torch.where(has_response, first_response, torch.zeros_like(first_response))
    prompt_lens = noisy_starts - response_lens
    return prompt_lens, response_lens, noisy_starts


def build_teacher_token_sequences(
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    student_logits: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_context_mode: str,
    teacher_pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_lens, response_lens, noisy_starts = derive_response_segments(response_mask)
    batch_size = input_ids.shape[0]
    max_response_len = int(response_lens.max().item()) if batch_size > 0 else 0
    max_teacher_len = int((prompt_lens + response_lens).max().item()) if batch_size > 0 else 0

    teacher_input_ids = torch.full(
        (batch_size, max_teacher_len),
        fill_value=teacher_pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    teacher_attention_mask = torch.zeros((batch_size, max_teacher_len), dtype=torch.long, device=input_ids.device)
    response_tokens = torch.full(
        (batch_size, max_response_len),
        fill_value=teacher_pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )

    for batch_idx in range(batch_size):
        prompt_len = int(prompt_lens[batch_idx].item())
        response_len = int(response_lens[batch_idx].item())
        noisy_start = int(noisy_starts[batch_idx].item())
        teacher_len = prompt_len + response_len
        if teacher_len <= 0:
            continue

        teacher_input_ids[batch_idx, :prompt_len] = input_ids[batch_idx, :prompt_len]
        teacher_attention_mask[batch_idx, :teacher_len] = 1

        if response_len <= 0:
            continue

        clean_tokens = labels[batch_idx, prompt_len : prompt_len + response_len]
        if teacher_context_mode == "clean":
            chosen_tokens = clean_tokens
        elif teacher_context_mode == "student_argmax":
            chosen_tokens = torch.argmax(
                student_logits[batch_idx, noisy_start : noisy_start + response_len],
                dim=-1,
            ).to(input_ids.dtype)
            chosen_tokens = torch.where(clean_tokens == teacher_pad_token_id, clean_tokens, chosen_tokens)
        else:
            raise ValueError(
                f"Unsupported teacher context mode `{teacher_context_mode}`. "
                "Expected one of: `clean`, `student_argmax`."
            )

        response_tokens[batch_idx, :response_len] = chosen_tokens
        teacher_input_ids[batch_idx, prompt_len:teacher_len] = chosen_tokens

    return teacher_input_ids, teacher_attention_mask, response_tokens, noisy_starts


def collect_aligned_kd_logits(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    _, response_lens, _ = derive_response_segments(response_mask)
    student_slices = []
    teacher_slices = []
    total_tokens = 0

    kd_mask = response_mask & loss_mask
    for batch_idx in range(student_logits.shape[0]):
        response_len = int(response_lens[batch_idx].item())
        if response_len <= 0:
            continue

        student_positions = torch.nonzero(kd_mask[batch_idx], as_tuple=False).flatten()
        if student_positions.numel() == 0:
            continue

        teacher_positions = student_positions - response_len - 1
        valid = teacher_positions >= 0
        if not valid.any():
            continue

        student_positions = student_positions[valid]
        teacher_positions = teacher_positions[valid]
        student_slices.append(student_logits[batch_idx, student_positions])
        teacher_slices.append(teacher_logits[batch_idx, teacher_positions])
        total_tokens += int(student_positions.numel())

    if not student_slices:
        return None, None, 0

    return torch.cat(student_slices, dim=0), torch.cat(teacher_slices, dim=0), total_tokens


def collect_aligned_kd_indices(
    *,
    response_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, int]:
    """收集 student/teacher 对齐后的 token 索引，用于后续抽取 logits 做 KD。"""
    # `response_mask & loss_mask` 给出了 student 真正参与 KD 的 noisy token 位置。
    # teacher 看到的是更短的 `prompt + clean_response`，因此 teacher 上对应的位置
    # 要把 noisy 段整体平移回去，下面的 `response_len + 1` 就是在做这个对齐。
    _, response_lens, _ = derive_response_segments(response_mask)
    batch_indices = []
    student_positions = []
    teacher_positions = []
    total_tokens = 0

    kd_mask = response_mask & loss_mask
    for batch_idx in range(response_mask.shape[0]):
        response_len = int(response_lens[batch_idx].item())
        if response_len <= 0:
            continue

        current_student_positions = torch.nonzero(kd_mask[batch_idx], as_tuple=False).flatten()
        if current_student_positions.numel() == 0:
            continue

        # student 序列形如：
        #   [prompt][clean_response][noisy_response]
        # teacher 序列形如：
        #   [prompt][clean_response]
        # 所以 noisy 段中某个 token 在 teacher 侧对应的位置，要减去一整个 response_len，
        # 再额外减 1 对齐到“预测下一个 token”的 logits 位置。
        current_teacher_positions = current_student_positions - response_len - 1
        valid = current_teacher_positions >= 0
        if not valid.any():
            continue

        current_student_positions = current_student_positions[valid]
        current_teacher_positions = current_teacher_positions[valid]

        batch_indices.append(
            torch.full_like(current_student_positions, fill_value=batch_idx, dtype=torch.long)
        )
        student_positions.append(current_student_positions)
        teacher_positions.append(current_teacher_positions)
        total_tokens += int(current_student_positions.numel())

    if not student_positions:
        return None, None, None, 0

    return (
        torch.cat(batch_indices, dim=0),
        torch.cat(student_positions, dim=0),
        torch.cat(teacher_positions, dim=0),
        total_tokens,
    )


def collect_generation_kd_indices(
    *,
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
    student_batch_indices: torch.Tensor,
    student_response_positions: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, int]:
    """对齐 diffusion 生成期保存的 student logits 与 teacher 的 AR logits 索引。"""
    # student_response_positions 的坐标系是“response 内部位置”：
    #   第 i 个位置表示 student 在生成 response 的第 i 个 token 真正被 unmask 时保存了 logits。
    # teacher 侧则是标准 AR：
    #   teacher_logits[prompt_len + i - 1] 预测的是 response 的第 i 个 token。
    generation_entry_indices = []
    batch_indices = []
    teacher_positions = []
    total_tokens = 0

    for entry_idx in range(student_batch_indices.shape[0]):
        batch_idx = int(student_batch_indices[entry_idx].item())
        prompt_len = int(prompt_lens[batch_idx].item())
        response_len = int(response_lens[batch_idx].item())
        response_pos = int(student_response_positions[entry_idx].item())
        if response_len <= 0 or response_pos >= response_len:
            continue

        teacher_position = response_pos + prompt_len - 1
        if teacher_position < 0:
            continue

        generation_entry_indices.append(entry_idx)
        batch_indices.append(batch_idx)
        teacher_positions.append(teacher_position)
        total_tokens += 1

    if not generation_entry_indices:
        return None, None, None, 0

    return (
        student_batch_indices.new_tensor(generation_entry_indices, dtype=torch.long),
        student_batch_indices.new_tensor(batch_indices, dtype=torch.long),
        student_batch_indices.new_tensor(teacher_positions, dtype=torch.long),
        total_tokens,
    )
