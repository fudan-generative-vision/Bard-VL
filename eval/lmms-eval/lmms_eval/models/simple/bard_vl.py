import base64
import re
import time
from io import BytesIO
from loguru import logger as eval_logger
from typing import List, Optional, Tuple, Union
import decord
import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    # Qwen3VLForConditionalGeneration,
    # Qwen3VLMoeForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")

from nemo_automodel.components.models.bard_vl import BardVLForConditionalGeneration
from lmms_eval.protocol import ChatMessages

@register_model("bard_vl")
class Bard_VL(lmms):
    def __init__(
        self,
        pretrained: str = "",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 256,
        max_pixels: int = 2048 * 2048,
        max_num_frames: int = 32,
        use_cache: Optional[bool] = False,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        block_size: Optional[int] = None,
        denoising_steps: Optional[int] = None,
        remasking_strategy: str = "low_confidence_static",
        confidence_threshold: float = 1.0,
        edit_threshold: float = 0.0,
        max_post_edit_steps: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()

        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.block_size = block_size
        self.denoising_steps = denoising_steps
        self.remasking_strategy = remasking_strategy
        self.confidence_threshold = confidence_threshold
        self.edit_threshold = edit_threshold
        self.max_post_edit_steps = max_post_edit_steps

        print(block_size, denoising_steps, remasking_strategy, confidence_threshold, edit_threshold, max_post_edit_steps)

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # check whether its an MoE model
        match = re.search(r"A\d+B", pretrained) # A3B and so on.
        # model_fn = BardVLMoeForConditionalGeneration if match else BardVLForConditionalGeneration
        model_fn = BardVLForConditionalGeneration
        self._model = model_fn.from_pretrained(pretrained, **model_kwargs).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        # self._config = self.model.config
        # self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        total_tokens = 0
        total_generate_duration = 0.0

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            vr = decord.VideoReader(visual)
                            first_frame = vr[0].asnumpy()
                            height, width = first_frame.shape[:2]
                            # max_pixels = height * width
                            processed_visuals.append({"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                        elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                            processed_visuals.append({"type": "image", "image": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:  # currently support find <image x> in the context
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)

            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
            # TODO: refactor code to allow return_video_kwargs and return_video_metadata
            image_inputs, video_inputs = process_vision_info(batched_messages, return_video_kwargs=False, image_patch_size=16, return_video_metadata=False)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Ensure unique indices if linspace produces duplicates for few frames
                indices = np.unique(indices)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                    indices = np.unique(indices)  # Ensure uniqueness again
                video_inputs[0] = video_inputs[0][indices]
            if self.batch_size > 1:
                inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, do_resize=False, padding=True, padding_side="left", return_tensors="pt")
            else:
                inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, do_resize=False, return_tensors="pt")
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            assert "max_new_tokens" in gen_kwargs or self.block_size is not None, "Neither of them can be None simultaneously"

            gen_kwargs["remasking_strategy"] = self.remasking_strategy

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024

            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0

            if "block_size" not in gen_kwargs:
                if self.block_size is not None:
                    gen_kwargs["block_size"] = self.block_size
                else:
                    gen_kwargs["block_size"] = gen_kwargs["max_new_tokens"]

            if "denoising_steps" not in gen_kwargs:
                if self.denoising_steps is not None:
                    gen_kwargs["denoising_steps"] = self.denoising_steps
                else:
                    gen_kwargs["denoising_steps"] = gen_kwargs["max_new_tokens"]

            if "top_k" not in gen_kwargs:
                gen_kwargs["top_k"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = 1.0

            gen_kwargs["confidence_threshold"] = self.confidence_threshold

            if "mask_token_id" not in gen_kwargs:
                gen_kwargs["mask_token_id"] = 151671
            if "eos_token_id" not in gen_kwargs:
                gen_kwargs["eos_token_id"] = 151645

            if "until" in gen_kwargs:
                gen_kwargs.pop("until") # ['\n\n']
            if "do_sample" in gen_kwargs:
                gen_kwargs.pop("do_sample")
            if "num_beams" in gen_kwargs:   # TODO: test-time scaling.
                gen_kwargs.pop("num_beams")

            # breakpoint()
            start_time = time.time()
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    inputs,
                    **gen_kwargs,
                )
            end_time = time.time()
            total_generate_duration += end_time - start_time

            total_tokens += (generated_ids[0] != gen_kwargs["mask_token_id"]).sum().item()

            # breakpoint()
            answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for text, answer in zip(texts, answers):
                answer = answer.strip()

                if answer.endswith("."):
                    answer = answer[:-1]

                eval_logger.debug(f"Question: {text}")
                eval_logger.debug(f"Model Response: {answer}")

                res.append(answer)
                pbar.update(1)
        res = re_ords.get_original(res)

        duration = total_generate_duration
        if hasattr(self, 'accelerator') and self.accelerator.num_processes > 1:
            num_tokens_tensor = torch.tensor(total_tokens, dtype=torch.long, device=self.device)
            duration_tensor = torch.tensor(duration, dtype=torch.float32, device=self.device)

            if dist.is_initialized():
                dist.all_reduce(num_tokens_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(duration_tensor, op=dist.ReduceOp.MAX)
            else:
                self.accelerator.wait_for_everyone()

            if self.rank == 0:
                total_tokens = num_tokens_tensor.item()
                total_duration = duration_tensor.item()
                print(f"Time taken: {total_duration:.2f} seconds")
                if total_duration > 0:
                    avg_tps_per_gpu = (total_tokens / total_duration) / self.world_size
                    total_tps = total_tokens / total_duration
                    print(f"Tokens per second (total): {total_tps:.2f}")
                    print(f"Tokens per second (per GPU, average): {avg_tps_per_gpu:.2f}")
                print(f"Total number of tokens: {total_tokens}")
                print(f"Tokens per process (average): {total_tokens / self.world_size:.0f}")
        else:
            if self.rank == 0:
                print(f"Time taken: {duration:.2f} seconds")
                if duration > 0:
                    print(f"Tokens per second: {total_tokens / duration:.2f}")
                print(f"Total number of tokens: {total_tokens}")

        pbar.close()
        # breakpoint()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
