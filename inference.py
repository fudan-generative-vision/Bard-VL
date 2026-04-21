import argparse
import torch
from transformers import AutoProcessor

from qwen_vl_utils import process_vision_info
from nemo_automodel.components.models.bard_vl import BardVLForConditionalGeneration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="pretrained_models/Bard-VL-B32-Mask-4B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--block_size", type=int, default=32,
                        help="Response block length used by block diffusion decoding.")
    parser.add_argument("--denoising_steps", type=int, default=32,
                        help="Maximum denoising steps for each response block.")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence_dynamic",
                        help="Token unmasking strategy inside each response block.")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Dynamic unmasking threshold; tokens with confidence >= threshold are unmasked.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BardVLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        _attn_implementation="sdpa",
        ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        ########### image understanding ###########
        # {
        #     "role": "user",
        #     "content": [
        #         {"type": "image", "image": "assets/puzzle.jpg", "min_pixels": 256*256, "max_pixels": 2048*2048},
        #         {"type": "text", "text": "Please describe this image"}
        #     ]
        # }
        ########### video understanding ###########
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "assets/human.mp4"},
                {"type": "text", "text": "Explain the video's components, including its characters, setting, and plot."},
            ]
        },
    ]

    return_video_metadata = True
    texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=return_video_metadata,
        image_patch_size=processor.image_processor.patch_size)

    video_metadata = None
    if return_video_metadata and video_inputs is not None:
        video_metadata = [_[1] for _ in video_inputs]
        video_inputs = [_[0] for _ in video_inputs]

    batch = processor(
        text=[texts],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
        video_metadata=video_metadata,
        **video_kwargs).to(device)

    response_ids = model.generate(
        batch,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        denoising_steps=args.denoising_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        remasking_strategy=args.remasking_strategy,
        confidence_threshold=args.confidence_threshold,
        return_step_stats=False,
    )

    print(processor.tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip())
    
