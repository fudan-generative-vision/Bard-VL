"""
Offline T2I data preprocessing: filter + VQ encode + save as Arrow.

Supports torchrun multi-GPU parallel processing.

Usage:
    # Emu3.5 IBQ (default):
    torchrun --nproc_per_node=8 tools/preprocessing/prepare_t2i_data.py \
        --input-path /inspire/qb-ilm/project/chineseculture/public/yuxuan/data/dvlm_t2i_0518/data \
        --output-path datasets/t2i_stage1_256 \
        --target-resolution 256 \
        --vq-type ibq \
        --vq-model-path pretrained_models/BAAI/Emu3.5-VisionTokenizer \
        --min-resolution 256 \
        --min-aesthetic 4.5 \
        --max-watermark 0.8 \
        --filter-nsfw

    # Lumina-DiMOO:
    torchrun --nproc_per_node=8 tools/preprocessing/prepare_t2i_data.py \
        --input-path /inspire/qb-ilm/project/chineseculture/public/yuxuan/data/dvlm_t2i_0518/data \
        --output-path datasets/t2i_stage1_256 \
        --target-resolution 256 \
        --vq-type dimoo \
        --vq-model-path pretrained_models/Alpha-VLLM/Lumina-DiMOO

    # Single GPU (no torchrun):
    python tools/preprocessing/prepare_t2i_data.py \
        --input-path /inspire/qb-ilm/project/chineseculture/public/yuxuan/data/dvlm_t2i_0518/data \
        --output-path datasets/t2i_stage1_256 \
        --target-resolution 256 \
        --vq-type ibq \
        --vq-model-path pretrained_models/BAAI/Emu3.5-VisionTokenizer

    # Background execution with nohup:
    nohup torchrun --nproc_per_node=8 tools/preprocessing/prepare_t2i_data.py \
        --input-path /inspire/qb-ilm/project/chineseculture/public/yuxuan/data/dvlm_t2i_0518/data \
        --output-path datasets/t2i_stage1_256 \
        --target-resolution 256 \
        --min-resolution 256 \
        --max-aspect-ratio 4.0 \
        --batch-size 256 \
        --vq-type ibq \
        --vq-model-path pretrained_models/BAAI/Emu3.5-VisionTokenizer \
        > logs/t2i_data_stage1.log 2>&1 &

    nohup torchrun --nproc_per_node=4 tools/preprocessing/prepare_t2i_data.py \
        --input-path /inspire/qb-ilm/project/chineseculture/public/yuxuan/data/dvlm_t2i_0518/data \
        --output-path datasets/t2i_stage2_512 \
        --target-resolution 512 \
        --min-resolution 512 \
        --batch-size 64 \
        --vq-type ibq \
        --vq-model-path pretrained_models/BAAI/Emu3.5-VisionTokenizer \
        > logs/t2i_data_stage2_512.log 2>&1 &

    nohup torchrun --nproc_per_node=8 tools/preprocessing/prepare_t2i_data.py \
        --input-path /inspire/qb-ilm/project/chineseculture/public/yuxuan/data/dvlm_t2i_0518/data \
        --output-path datasets/t2i_stage2_1024 \
        --target-resolution 1024 \
        --min-resolution 1024 \
        --max-aspect-ratio 4.0 \
        --batch-size 16 \
        --vq-type ibq \
        --vq-model-path pretrained_models/BAAI/Emu3.5-VisionTokenizer \
        > logs/t2i_data_stage2_1024.log 2>&1 &

 4 个 CPU 线程各自读不同的 parquet 文件并解码，竞争往同一个大 queue 里塞 batch;主线程只管从 queue 取数据做 GPU encode。
 内存充足时 CPU 可以持续跑满不被阻塞, GPU 利用率拉满。
"""

import argparse
import io
import json
import math
import os
import queue
import threading
import time
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][Rank %(process)d] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TOKENIZER_CONFIGS = {
    "ibq": {"codebook_size": 131072, "downsample_factor": 16},
    "dimoo": {"codebook_size": 8192, "downsample_factor": 32},
}


def parse_args():
    parser = argparse.ArgumentParser(description="T2I data preprocessing with VQ encoding")
    parser.add_argument("--input-path", type=str, required=True,
                        help="Path to raw parquet data directory")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output directory for processed Arrow data")
    parser.add_argument("--target-resolution", type=int, default=256,
                        help="Target short-edge resolution (256/512/1024)")
    parser.add_argument("--vq-type", type=str, default="ibq", choices=["ibq", "dimoo"],
                        help="VQ tokenizer type: ibq (Emu3.5) or dimoo (Lumina-DiMOO)")
    parser.add_argument("--vq-model-path", type=str, required=True,
                        help="Path to VQ tokenizer model directory")

    # Filter arguments
    parser.add_argument("--min-resolution", type=int, default=256,
                        help="Minimum short-edge pixels of original image")
    parser.add_argument("--min-aesthetic", type=float, default=4.5,
                        help="Minimum aesthetic score")
    parser.add_argument("--max-watermark", type=float, default=0.9,
                        help="Maximum watermark score")
    parser.add_argument("--filter-nsfw", action="store_true", default=True,
                        help="Filter out non-neutral NSFW content")
    parser.add_argument("--no-filter-nsfw", action="store_false", dest="filter_nsfw")
    parser.add_argument("--min-caption-words", type=int, default=5,
                        help="Minimum word count in caption")
    parser.add_argument("--max-ocr-ratio", type=float, default=0.5,
                        help="Maximum OCR text ratio")
    parser.add_argument("--exclude-sources", type=str, nargs="*",
                        default=["text-to-image-2m"],
                        help="Exclude samples from these data sources")
    parser.add_argument("--max-aspect-ratio", type=float, default=8.0,
                        help="Maximum aspect ratio (long/short), filter extreme shapes")

    # Processing arguments
    parser.add_argument("--shard-size", type=int, default=100000,
                        help="Number of samples per output shard")
    parser.add_argument("--decode-threads", type=int, default=4,
                        help="Number of CPU threads for parallel parquet read + image decode")
    parser.add_argument("--queue-depth", type=int, default=8,
                        help="Max batches buffered between CPU decode and GPU encode")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="VQ encode batch size per GPU")
    parser.add_argument("--tokenizer-path", type=str,
                        default="pretrained_models/Bard-VL-B32-Mask-4B-Instruct",
                        help="Path to tokenizer for computing sequence lengths")
    return parser.parse_args()


def nearest_multiple(value, factor):
    """Round value to the nearest multiple of factor (at least factor)."""
    return max(factor, int(math.floor(value / factor + 0.5)) * factor)


def resize_and_align(image: Image.Image, target_resolution: int, downsample_factor: int) -> Image.Image:
    """Resize short edge to target, keep aspect ratio, align both dims to downsample_factor."""
    w, h = image.size
    if w <= h:
        new_w = target_resolution
        new_h = int(h * new_w / w)
    else:
        new_h = target_resolution
        new_w = int(w * new_h / h)

    aligned_w = nearest_multiple(new_w, downsample_factor)
    aligned_h = nearest_multiple(new_h, downsample_factor)
    image = image.resize((aligned_w, aligned_h), Image.BICUBIC)
    return image


def pil_to_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert PIL RGB image to normalized [-1, 1] tensor [1, 3, H, W]."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)




class VQEncoder:
    """Wraps VQ tokenizer for batch encoding. Supports ibq (Emu3.5) and dimoo (Lumina-DiMOO)."""

    def __init__(self, model_path: str, device: torch.device, vq_type: str = "ibq"):
        self.vq_type = vq_type
        self.device = device
        self.downsample_factor = TOKENIZER_CONFIGS[vq_type]["downsample_factor"]

        if vq_type == "ibq":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            self.model.to(device).eval()
            self.dtype = next(self.model.parameters()).dtype
        elif vq_type == "dimoo":
            from diffusers import VQModel
            self.model = VQModel.from_pretrained(model_path, subfolder="vqvae")
            self.model.to(device).eval()
            self.dtype = next(self.model.parameters()).dtype
        else:
            raise ValueError(f"Unsupported vq_type: {vq_type}")

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def encode_batch(self, images: list[Image.Image], target_resolution: int) -> list[dict]:
        """Encode a batch of PIL images, return list of {vq_codes, code_height, code_width}."""
        results = []
        for img in images:
            img = resize_and_align(img, target_resolution, self.downsample_factor)
            w, h = img.size
            code_w = w // self.downsample_factor
            code_h = h // self.downsample_factor

            if self.vq_type == "ibq":
                tensor = pil_to_tensor(img, self.device, self.dtype)
                _, _, info = self.model.encode(tensor)
                codes = info[-1].reshape(code_h, code_w)
            else:
                from diffusers.image_processor import VaeImageProcessor
                vae_scale_factor = 2 ** (len(self.model.config.block_out_channels) - 1)
                processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize=False)
                x = processor.preprocess(img).to(device=self.device)
                latents = self.model.encode(x).latents
                codes = self.model.quantize(latents)[2][2].reshape(code_h, code_w)

            results.append({
                "vq_codes": codes.cpu().numpy().astype(np.int32).flatten(),
                "code_height": code_h,
                "code_width": code_w,
            })
        return results


def save_arrow_shard(records: list[dict], output_dir: Path, shard_idx: int):
    """Save a list of records as one Arrow file."""
    arrays = {
        "vq_codes": pa.array([r["vq_codes"].tolist() for r in records], type=pa.list_(pa.int32())),
        "code_height": pa.array([r["code_height"] for r in records], type=pa.int32()),
        "code_width": pa.array([r["code_width"] for r in records], type=pa.int32()),
        "caption": pa.array([r["caption"] for r in records], type=pa.string()),
        "v_tokens": pa.array([r["v_tokens"] for r in records], type=pa.int32()), # 图像被编码后的 纯 VQ token 数量
        "seq_length": pa.array([r["seq_length"] for r in records], type=pa.int32()), # prefix_len + 2 * (v_tokens + 3), 训练时完整序列长度
    }
    table = pa.table(arrays)
    out_path = output_dir / f"data-{shard_idx:05d}.arrow"
    writer = pa.ipc.new_file(str(out_path), table.schema)
    writer.write_table(table)
    writer.close()
    return out_path


def _cpu_decode_worker(file_list, filters, batch_size, out_queue, stats):
    """Background thread: read parquet files, filter, decode images, push batches to queue."""
    for pf in file_list:
        table = pq.read_table(pf)
        num_rows = len(table)
        batch_imgs = []
        batch_caps = []

        for row_idx in range(num_rows):
            row = {col: table.column(col)[row_idx].as_py() for col in table.column_names}

            if not _passes_filter(row, filters):
                stats["filtered"] += 1
                continue

            try:
                img_data = row.get("image", {})
                if not img_data or not img_data.get("bytes"):
                    stats["filtered"] += 1
                    continue
                img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                caption_raw = row.get("caption", "[]")
                caption_list = json.loads(caption_raw) if isinstance(caption_raw, str) else (caption_raw or [])
                caption = caption_list[0]["text"] if caption_list else ""
                batch_imgs.append(img)
                batch_caps.append(caption)
            except Exception:
                stats["failed"] += 1
                continue

            if len(batch_imgs) >= batch_size:
                out_queue.put((batch_imgs, batch_caps))
                batch_imgs = []
                batch_caps = []

        del table
        if batch_imgs:
            out_queue.put((batch_imgs, batch_caps))


def process(rank: int, world_size: int, args):
    """Main processing loop for one rank. Uses producer-consumer pipeline for CPU/GPU overlap."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    filters = {
        "min_resolution": args.min_resolution,
        "min_aesthetic": args.min_aesthetic,
        "max_watermark": args.max_watermark,
        "filter_nsfw": args.filter_nsfw,
        "min_caption_words": args.min_caption_words,
        "max_ocr_ratio": args.max_ocr_ratio,
        "exclude_sources": set(args.exclude_sources) if args.exclude_sources else set(),
        "max_aspect_ratio": args.max_aspect_ratio,
    }

    # Discover and shard parquet files
    data_path = Path(args.input_path)
    all_files = sorted(data_path.glob("*.parquet"))
    if not all_files:
        all_files = sorted(data_path.rglob("*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")

    my_files = all_files[rank::world_size]
    logger.info(f"Rank {rank}: assigned {len(my_files)}/{len(all_files)} parquet files")

    # Load VQ encoder
    encoder = VQEncoder(args.vq_model_path, device, vq_type=args.vq_type)
    logger.info(f"Rank {rank}: VQ encoder ({args.vq_type}) loaded on {device}")

    # Load tokenizer for sequence length computation
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    prefix_template = "<|im_start|>user\nGenerate an image: {caption}<|im_end|>\n<|im_start|>assistant\n"
    # Fixed overhead: template tokens excluding caption
    _tmpl_overhead = len(tokenizer.encode(
        prefix_template.format(caption=""), add_special_tokens=False
    ))

    # Output directory per rank
    output_dir = Path(args.output_path) / f"shard-{rank:05d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Producer-consumer: N CPU decode threads → shared queue → GPU encode main thread
    batch_queue = queue.Queue(maxsize=args.queue_depth)
    stats = {"filtered": 0, "failed": 0}
    num_threads = min(args.decode_threads, len(my_files))
    done_counter = {"count": 0}
    done_lock = threading.Lock()

    def _worker_wrapper(file_subset):
        _cpu_decode_worker(file_subset, filters, args.batch_size, batch_queue, stats)
        with done_lock:
            done_counter["count"] += 1
            if done_counter["count"] == num_threads:
                batch_queue.put(None)

    producers = []
    for t_idx in range(num_threads):
        thread_files = my_files[t_idx::num_threads]
        t = threading.Thread(target=_worker_wrapper, args=(thread_files,), daemon=True)
        t.start()
        producers.append(t)

    logger.info(f"Rank {rank}: {num_threads} decode threads, queue_depth={args.queue_depth}, "
                f"batch_size={args.batch_size}")

    records = []
    shard_idx = 0
    total_processed = 0
    last_log_time = time.time()
    t0 = last_log_time

    while True:
        item = batch_queue.get()
        if item is None:
            break
        batch_imgs, batch_caps = item

        _encode_and_collect(encoder, batch_imgs, batch_caps, args.target_resolution,
                            records, rank, stats["failed"],
                            tokenizer=tokenizer, tmpl_overhead=_tmpl_overhead)
        total_processed += len(batch_imgs)

        # Periodic progress log every 60s
        now = time.time()
        if now - last_log_time >= 60:
            elapsed = now - t0
            speed = total_processed / elapsed
            logger.info(f"Rank {rank}: encoded={total_processed}, "
                        f"speed={speed:.1f} img/s, queue={batch_queue.qsize()}, "
                        f"dropped={stats['filtered']}, unsaved={len(records)}")
            last_log_time = now

        # Save shard when buffer is full
        if len(records) >= args.shard_size:
            save_arrow_shard(records[:args.shard_size], output_dir, shard_idx)
            records = records[args.shard_size:]
            shard_idx += 1
            elapsed = time.time() - t0
            speed = total_processed / elapsed
            logger.info(f"Rank {rank}: saved shard {shard_idx-1}, "
                        f"total={total_processed}, speed={speed:.1f} samples/s")

    for t in producers:
        t.join()

    # Save remaining records
    if records:
        save_arrow_shard(records, output_dir, shard_idx)
        shard_idx += 1

    elapsed = time.time() - t0
    logger.info(f"Rank {rank}: DONE. Processed {total_processed}, filtered {stats['filtered']}, "
                f"failed {stats['failed']}, {shard_idx} shards, {elapsed:.0f}s total")


def _passes_filter(row: dict, filters: dict) -> bool:
    """Check if a row passes all filter conditions."""
    if filters["exclude_sources"]:
        source = row.get("source", "")
        if source in filters["exclude_sources"]:
            return False

    meta = _parse_json(row.get("meta"), {})
    w = meta.get("width", 0)
    h = meta.get("height", 0)
    if min(w, h) < filters["min_resolution"]:
        return False
    if min(w, h) > 0 and max(w, h) / min(w, h) > filters["max_aspect_ratio"]:
        return False

    aes = _parse_json(row.get("aes"), {})
    if aes.get("score", 0) < filters["min_aesthetic"]:
        return False

    watermark = _parse_json(row.get("watermark"), {})
    if watermark.get("score", 1.0) > filters["max_watermark"]:
        return False

    if filters["filter_nsfw"]:
        nsfw = _parse_json(row.get("nsfw"), {})
        if nsfw.get("class", "neutral") != "neutral":
            return False

    captions = _parse_json(row.get("caption"), [])
    if captions:
        best_caption = captions[0].get("text", "")
        if len(best_caption.split()) < filters["min_caption_words"]:
            return False

    ocr = _parse_json(row.get("ocr"), {})
    if ocr.get("ratio", 0) > filters["max_ocr_ratio"]:
        return False

    return True


def _parse_json(value, default):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return default
    return value if value is not None else default


def _encode_and_collect(encoder, images, captions, target_resolution, records, rank, total_failed,
                        tokenizer=None, tmpl_overhead=0):
    """VQ encode a batch of images and append results to records."""
    try:
        encoded = encoder.encode_batch(images, target_resolution)
        for i, enc in enumerate(encoded):
            v_tokens = enc["code_height"] * enc["code_width"]
            if tokenizer is not None:
                caption_tokens = len(tokenizer.encode(captions[i], add_special_tokens=False))
                prefix_len = tmpl_overhead + caption_tokens
            else:
                prefix_len = 130
            # total = prefix + gen_start + vq + gen_end + im_end (clean) + same (noisy)
            seq_length = prefix_len + 2 * (v_tokens + 3)
            records.append({
                "vq_codes": enc["vq_codes"],
                "code_height": enc["code_height"],
                "code_width": enc["code_width"],
                "caption": captions[i],
                "v_tokens": v_tokens,
                "seq_length": seq_length,
            })
    except Exception as e:
        logger.warning(f"Rank {rank}: VQ encode failed: {e}")


def main():
    args = parse_args()

    # Detect DDP environment
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    logger.info(f"Starting rank {rank}/{world_size}, target_resolution={args.target_resolution}")
    process(rank, world_size, args)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
