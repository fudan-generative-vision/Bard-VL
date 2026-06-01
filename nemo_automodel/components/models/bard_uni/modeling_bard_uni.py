"""
Bard-Uni: Unified multimodal understanding, image generation, and image editing.

Extends BardVLForConditionalGeneration with:
- image_head: Linear(hidden_size → vq_codebook_size) for VQ code prediction
- vq_embed: Embedding(vq_codebook_size, hidden_size) for VQ code input
- visual_tokenizer: Frozen Emu3.5 IBQ for encode/decode (not part of training graph)
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import check_model_inputs

from nemo_automodel.components.models.bard_vl.modeling_bard_vl import (
    BardVLForConditionalGeneration,
    BardVLCausalLMOutputWithPast,
)
from .configuration_bard_uni import BardUniConfig
from .visual_tokenizer import VisualTokenizer


class BardUniForConditionalGeneration(BardVLForConditionalGeneration):
    config: BardUniConfig
    config_class = BardUniConfig

    def __init__(self, config: BardUniConfig):
        super().__init__(config)

        hidden_size = config.text_config.hidden_size
        codebook_size = config.vq_codebook_size

        # Dual head: lm_head (inherited) + image_head (new)
        self.image_head = nn.Linear(hidden_size, codebook_size, bias=False)
        self.vq_embed = nn.Embedding(codebook_size, hidden_size)

        if config.vq_weight_tying:
            self.image_head.weight = self.vq_embed.weight

        # Visual tokenizer (frozen, lazy-loaded)
        self._visual_tokenizer = None

        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs.pop("tp_size", None)
        kwargs.pop("cp_size", None)
        torch_dtype = kwargs.get("torch_dtype", None)
        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            import torch as _torch
            kwargs["torch_dtype"] = getattr(_torch, torch_dtype.replace("torch.", ""))
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @property
    def visual_tokenizer(self) -> VisualTokenizer:
        if self._visual_tokenizer is None:
            self._visual_tokenizer = VisualTokenizer(
                model_path=self.config.vq_model_path,
                device=str(self.device),
            )
        return self._visual_tokenizer

    def get_vq_input_embeddings(self, vq_codes: torch.LongTensor) -> torch.Tensor:
        """Map VQ code indices → hidden_size embeddings for LLM input."""
        return self.vq_embed(vq_codes)

    @check_model_inputs()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # Bard-Uni specific
        vq_codes: Optional[torch.LongTensor] = None,
        vq_code_mask: Optional[torch.BoolTensor] = None,
        generation_mask: Optional[torch.BoolTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BardVLCausalLMOutputWithPast]:
        """
        Extended forward with dual-head support.

        Args:
            vq_codes: [B, seq_len] VQ code indices for source image tokens in input.
                      Positions indicated by vq_code_mask use vq_embed instead of embed_tokens.
            vq_code_mask: [B, seq_len] bool mask, True where input_ids should be replaced
                          by vq_embed(vq_codes).
            generation_mask: [B, seq_len] bool mask, True for positions that use image_head
                             (VQ generation region). If None, all positions use lm_head.
        """
        # Build inputs_embeds with VQ code embeddings spliced in
        # Use torch.where instead of in-place assignment so autograd tracks vq_embed gradients
        if inputs_embeds is None and vq_code_mask is not None and vq_code_mask.any():
            text_embeds = self.model.get_input_embeddings()(input_ids)
            vq_codes_input = vq_codes if vq_codes is not None else input_ids
            # Clamp non-VQ positions to valid range for vq_embed lookup
            safe_vq_input = vq_codes_input.clamp(0, self.vq_embed.num_embeddings - 1)
            vq_embeds = self.vq_embed(safe_vq_input)
            inputs_embeds = torch.where(vq_code_mask.unsqueeze(-1), vq_embeds, text_embeds)
            input_ids = None  # use inputs_embeds path

        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Debug: log generation_mask state on first forward to diagnose routing
        if not hasattr(self, "_fwd_debug_done"):
            self._fwd_debug_done = True
            import logging as _logging
            _log = _logging.getLogger(__name__)
            _log.info(
                f"[BardUni forward] generation_mask is None={generation_mask is None}, "
                f"generation_mask.any()={generation_mask.any().item() if generation_mask is not None else 'N/A'}, "
                f"vq_code_mask is None={vq_code_mask is None}, "
                f"vq_code_mask.any()={vq_code_mask.any().item() if vq_code_mask is not None else 'N/A'}, "
                f"hidden_states.shape={hidden_states.shape}"
            )

        # Dual-head logits computation
        if generation_mask is not None and generation_mask.any():
            gen_mask = generation_mask.bool()
            if (
                isinstance(logits_to_keep, torch.Tensor)
                and logits_to_keep.ndim == 2
                and logits_to_keep.dtype != torch.bool
            ):
                gather_indices = logits_to_keep.to(device=hidden_states.device, dtype=torch.long)
                gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
                selected_hidden = torch.gather(hidden_states, dim=1, index=gather_indices).contiguous()
                text_logits = self.lm_head(selected_hidden)
                image_logits = self.image_head(selected_hidden)
                selected_gen_mask = torch.gather(gen_mask, dim=1, index=logits_to_keep.to(device=gen_mask.device, dtype=torch.long))
            else:
                text_logits = self.lm_head(hidden_states)
                image_logits = self.image_head(hidden_states)
                selected_gen_mask = gen_mask
            return BardUniOutput(
                logits=text_logits,
                image_logits=image_logits,
                generation_mask=selected_gen_mask,
                past_key_values=outputs.past_key_values,
                rope_deltas=outputs.rope_deltas,
            )
        else:
            # Pure understanding mode: only lm_head
            if (
                isinstance(logits_to_keep, torch.Tensor)
                and logits_to_keep.ndim == 2
                and logits_to_keep.dtype != torch.bool
            ):
                gather_indices = logits_to_keep.to(device=hidden_states.device, dtype=torch.long)
                gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
                selected_hidden_states = torch.gather(hidden_states, dim=1, index=gather_indices).contiguous()
                logits = self.lm_head(selected_hidden_states)
            else:
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                logits = self.lm_head(hidden_states[:, slice_indices, :])
            return BardVLCausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values,
                rope_deltas=outputs.rope_deltas,
            )

    @torch.no_grad()
    def generate_image(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        vq_codes: Optional[torch.LongTensor] = None,
        vq_code_mask: Optional[torch.BoolTensor] = None,
        target_height: int = 16,
        target_width: int = 16,
        timesteps: int = 64,
        cfg_scale: float = 4.0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        mask_token_id: int = 151671,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """MaskGit-style parallel decoding for image generation.

        Returns VQ codes [B, target_height, target_width].
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        num_vq_tokens = target_height * target_width
        use_newline = getattr(self.config, 'use_newline', False)

        if use_newline:
            num_newlines = target_height - 1
            total_gen_tokens = num_vq_tokens + num_newlines
        else:
            total_gen_tokens = num_vq_tokens

        # Initialize: all VQ positions masked
        gen_ids = torch.full(
            (batch_size, total_gen_tokens), mask_token_id, dtype=torch.long, device=device
        )

        if use_newline:
            newline_positions = []
            for row in range(target_height - 1):
                pos = (row + 1) * target_width + row
                newline_positions.append(pos)
                gen_ids[:, pos] = self.config.img_newline_token_id
            vq_positions = torch.tensor(
                [i for i in range(total_gen_tokens) if i not in newline_positions],
                device=device,
            )
        else:
            vq_positions = torch.arange(total_gen_tokens, device=device)

        assert vq_positions.shape[0] == num_vq_tokens

        # Prefill prompt KV cache
        prompt_len = input_ids.shape[1]
        block_size = 32  # match Bard-VL block size
        total_len = prompt_len + total_gen_tokens

        # Build block attention mask for full sequence
        num_prompt_blocks = (prompt_len + block_size - 1) // block_size
        num_gen_blocks = (total_gen_tokens + block_size - 1) // block_size
        total_blocks = num_prompt_blocks + num_gen_blocks

        block_ids = torch.zeros(total_len, dtype=torch.int32, device=device)
        for b_idx in range(num_prompt_blocks):
            start = b_idx * block_size
            end = min((b_idx + 1) * block_size, prompt_len)
            block_ids[start:end] = b_idx
        for b_idx in range(num_gen_blocks):
            start = prompt_len + b_idx * block_size
            end = min(prompt_len + (b_idx + 1) * block_size, total_len)
            block_ids[start:end] = num_prompt_blocks + b_idx

        q_blocks = block_ids.unsqueeze(1)
        k_blocks = block_ids.unsqueeze(0)
        block_attention_mask = (q_blocks >= k_blocks).unsqueeze(0).unsqueeze(0)

        # Prefill
        prefill_kwargs = dict(
            input_ids=input_ids,
            attention_mask=block_attention_mask[:, :, :prompt_len, :prompt_len],
            position_ids=position_ids,
            use_cache=True,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        if vq_codes is not None and vq_code_mask is not None:
            prefill_kwargs["vq_codes"] = vq_codes
            prefill_kwargs["vq_code_mask"] = vq_code_mask

        output = self(**prefill_kwargs)
        current_cache = output.past_key_values

        # Cosine schedule for mask ratio
        schedule = self._cosine_schedule(timesteps)

        for step_idx in range(timesteps):
            # Current mask: positions that are still masked (excluding newlines)
            is_masked = (gen_ids[:, vq_positions] == mask_token_id)  # [B, num_vq_tokens]
            num_masked = is_masked.sum(dim=1)  # [B]

            if num_masked.max().item() == 0:
                break

            # Number to unmask this step
            ratio = schedule[step_idx]
            num_to_unmask = torch.clamp(
                (ratio * num_vq_tokens).long(), min=1
            ).expand(batch_size)
            num_to_unmask = torch.min(num_to_unmask, num_masked)

            # Forward on generation region
            gen_pos_start = prompt_len
            gen_position_ids = torch.arange(
                gen_pos_start, gen_pos_start + total_gen_tokens, device=device
            ).unsqueeze(0).expand(batch_size, -1)
            # MRoPE: expand to 3 dims (T=pos, H=pos, W=pos for text-like)
            gen_position_ids_3d = gen_position_ids.unsqueeze(0).expand(3, -1, -1)

            # Build input embeds for gen region
            gen_embeds = self._build_gen_embeds(gen_ids, vq_positions, mask_token_id)

            output = self.model(
                inputs_embeds=gen_embeds,
                attention_mask=block_attention_mask[:, :, prompt_len:total_len, :total_len],
                position_ids=gen_position_ids_3d,
                past_key_values=current_cache,
                use_cache=False,
            )
            hidden = output[0]  # [B, total_gen_tokens, hidden_size]

            # Get logits only at VQ positions
            vq_hidden = hidden[:, vq_positions, :]  # [B, num_vq_tokens, hidden_size]
            vq_logits = self.image_head(vq_hidden)  # [B, num_vq_tokens, codebook_size]

            # CFG: if cfg_scale > 0, would need unconditional forward (omitted for simplicity in base impl)
            if temperature > 0:
                vq_logits = vq_logits / temperature

            # Sample
            probs = F.softmax(vq_logits, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, probs.shape[-1]), num_samples=1
            ).view(batch_size, num_vq_tokens)

            # Confidence for remasking
            sampled_probs = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)

            # Only unmask top-confidence among currently masked
            for b in range(batch_size):
                if num_to_unmask[b].item() == 0:
                    continue
                masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
                conf = sampled_probs[b, masked_indices]
                k = min(num_to_unmask[b].item(), len(masked_indices))
                _, top_idx = torch.topk(conf, k)
                unmask_idx = masked_indices[top_idx]
                # Write to gen_ids at the actual positions
                actual_positions = vq_positions[unmask_idx]
                gen_ids[b, actual_positions] = sampled[b, unmask_idx]

        # Extract final VQ codes (exclude newlines)
        final_codes = gen_ids[:, vq_positions].reshape(batch_size, target_height, target_width)
        return final_codes

    def _build_gen_embeds(
        self,
        gen_ids: torch.LongTensor,
        vq_positions: torch.LongTensor,
        mask_token_id: int,
    ) -> torch.Tensor:
        """Build input embeddings for the generation region.

        - Newline tokens → embed_tokens
        - Masked VQ positions → embed_tokens(mask_token_id)
        - Unmasked VQ positions → vq_embed(code)
        """
        embed_tokens = self.model.get_input_embeddings()
        batch_size, seq_len = gen_ids.shape
        device = gen_ids.device

        # Start with embed_tokens for everything (handles newlines and mask tokens)
        embeds = embed_tokens(gen_ids)

        # For unmasked VQ positions, use vq_embed
        vq_codes_at_positions = gen_ids[:, vq_positions]  # [B, num_vq]
        is_unmasked = (vq_codes_at_positions != mask_token_id)  # [B, num_vq]

        if is_unmasked.any():
            vq_embeds = self.vq_embed(vq_codes_at_positions)  # [B, num_vq, hidden]
            # Scatter back
            expanded_positions = vq_positions.unsqueeze(0).unsqueeze(-1).expand(
                batch_size, -1, embeds.shape[-1]
            )
            vq_embed_full = torch.zeros_like(embeds)
            vq_embed_full.scatter_(1, expanded_positions, vq_embeds)

            # Mask: only apply vq_embed where unmasked
            apply_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            apply_mask.scatter_(1, vq_positions.unsqueeze(0).expand(batch_size, -1), is_unmasked)
            embeds = torch.where(apply_mask.unsqueeze(-1), vq_embed_full, embeds)

        return embeds

    @staticmethod
    def _cosine_schedule(timesteps: int) -> torch.Tensor:
        """Cosine schedule: fraction of tokens to unmask at each step."""
        import math
        steps = torch.arange(timesteps + 1, dtype=torch.float32)
        # Cumulative unmask ratio at each step boundary
        ratios = torch.cos((steps / timesteps) * (math.pi / 2))
        # Per-step unmask fraction (from mask_ratio to 0)
        per_step = ratios[:-1] - ratios[1:]
        return per_step


@dataclass
class BardUniOutput(BardVLCausalLMOutputWithPast):
    """Extended output with separate image_logits and generation_mask."""

    image_logits: Optional[torch.FloatTensor] = None
    generation_mask: Optional[torch.BoolTensor] = None
