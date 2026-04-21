import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKKDLoss(nn.Module):
    def __init__(self, temperature: float = 1.0, fp32_upcast: bool = True, direction: str = "forward-kl"):
        super().__init__()
        self.temperature = temperature
        self.fp32_upcast = fp32_upcast
        self.direction = self._normalize_direction(direction)

    @staticmethod
    def _normalize_direction(direction: str) -> str:
        normalized = str(direction).strip().lower().replace("_", "-")
        alias_map = {
            "forward": "forward",
            "forward-kl": "forward",
            "reverse": "reverse",
            "reverse-kl": "reverse",
        }
        if normalized not in alias_map:
            raise ValueError(
                f"Unsupported TopKKDLoss direction `{direction}`. "
                "Expected one of: `forward-kl`, `reverse-kl`."
            )
        return alias_map[normalized]

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_topk_indices: torch.Tensor,
        teacher_topk_logits: torch.Tensor,
        num_batch_labels: int | None = None,
    ) -> torch.Tensor:
        if student_logits.numel() == 0:
            return student_logits.new_tensor(0.0)
        if teacher_topk_indices.shape != teacher_topk_logits.shape:
            raise ValueError(
                f"`teacher_topk_indices` and `teacher_topk_logits` must share the same shape, "
                f"got {tuple(teacher_topk_indices.shape)} vs {tuple(teacher_topk_logits.shape)}"
            )
        if self.direction == "reverse" and teacher_topk_indices.shape[-1] < 2:
            raise ValueError("`reverse-kl` requires teacher_topk >= 2; top-1 collapses the reverse-KL surrogate.")

        s_logits = student_logits
        t_logits = teacher_topk_logits
        if self.fp32_upcast:
            s_logits = s_logits.float()
            t_logits = t_logits.float()

        if self.temperature != 1.0:
            scale = 1.0 / self.temperature
            s_logits = s_logits * scale
            t_logits = t_logits * scale

        if self.direction == "forward":
            # Top-k truncated forward-KL surrogate: KL(q_teacher_topk || p_student).
            teacher_prob = F.softmax(t_logits, dim=-1, dtype=torch.float32)
            student_logprob = F.log_softmax(s_logits, dim=-1, dtype=torch.float32)
            gathered_student_logprob = student_logprob.gather(dim=-1, index=teacher_topk_indices.long())
            per_token = -(teacher_prob * gathered_student_logprob).sum(dim=-1)
        else:
            # Top-k truncated reverse-KL surrogate on teacher support:
            # KL(p_student_topk || q_teacher_topk), where both distributions are
            # renormalized over the teacher top-k support.
            gathered_student_logits = s_logits.gather(dim=-1, index=teacher_topk_indices.long())
            student_support_logprob = F.log_softmax(gathered_student_logits, dim=-1, dtype=torch.float32)
            student_support_prob = student_support_logprob.exp()
            teacher_support_logprob = F.log_softmax(t_logits, dim=-1, dtype=torch.float32)
            per_token = (student_support_prob * (student_support_logprob - teacher_support_logprob)).sum(dim=-1)

        if num_batch_labels is not None:
            return per_token.sum() / max(int(num_batch_labels), 1)
        return per_token.mean()
