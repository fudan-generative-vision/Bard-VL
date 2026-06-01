# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import io
import json
import torch
from glob import glob
from pathlib import Path
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool
from tqdm import tqdm
from datasets import load_dataset, load_from_disk


def clear_none_fields(obj):
    """
    递归移除字典或列表中的 None 值字段。
    """
    if isinstance(obj, list):
        # 处理列表中的每个元素
        return [clear_none_fields(i) for i in obj if i is not None]
    elif isinstance(obj, dict):
        # 处理字典，仅保留值不为 None 的键值对
        return {
            k: clear_none_fields(v)
            for k, v in obj.items()
            if v is not None
        }
    return obj


class qwen3_dataset(torch.utils.data.Dataset):
    def __init__(self, path_or_dataset, **kwargs):
        self.ds = json.load(open(path_or_dataset))
        self.root_dir = kwargs.get("root_dir", None)
        assert self.root_dir is not None and os.path.exists(self.root_dir), f"{self.root_dir}"

        self.min_pixels = kwargs.get("min_pixels", 384*384)
        self.max_pixels = kwargs.get("max_pixels", 512*512)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]['conversation']

        content = sample[1]['content'][0]
        if content.get("image", None) is not None:
            image_path = os.path.join(self.root_dir, sample[1]['content'][0]['image'])
            assert os.path.exists(image_path), f"{image_path} not exist"
            sample[1]['content'][0]['image'] = Image.open(image_path).convert("RGB") # image字段内容转为PIL.Image
            sample[1]['content'][0]['min_pixels'] = self.min_pixels # 不设定的话，代码中默认值是4*32*32
            sample[1]['content'][0]['max_pixels'] = self.max_pixels # 不设定的话，代码中默认值是16384*32*32

        return sample


class qwen2_5_dataset(torch.utils.data.Dataset):
    def __init__(self, path_or_dataset, **kwargs):
        self.ds = json.load(open(path_or_dataset))
        self.root_dir = kwargs.get("root_dir", None)
        assert self.root_dir is not None and os.path.exists(self.root_dir), f"{self.root_dir}"

        self.min_pixels = kwargs.get("min_pixels", 384*384)
        self.max_pixels = kwargs.get("max_pixels", 512*512)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]['conversation']

        content = sample[1]['content'][0]
        if content.get("image", None) is not None:
            image_path = os.path.join(self.root_dir, sample[1]['content'][0]['image'])
            assert os.path.exists(image_path), f"{image_path} not exist"
            sample[1]['content'][0]['image'] = Image.open(image_path) # image字段内容转为PIL.Image
            # sample[1]['content'][0]['image'] = Image.open(image_path).convert("RGB") # image字段内容转为PIL.Image
            sample[1]['content'][0]['min_pixels'] = self.min_pixels # 不设定的话，代码中默认值是4*32*32
            sample[1]['content'][0]['max_pixels'] = self.max_pixels # 不设定的话，代码中默认值是16384*32*32

        return sample


class bard_vl_dataset(torch.utils.data.Dataset):
    def __init__(self, path_or_dataset, **kwargs):
        self.root_dir = kwargs.get("root_dir", None)
        assert self.root_dir is not None and os.path.exists(self.root_dir), f"{self.root_dir}"

        print(f"Loading dataset from {path_or_dataset}...")
        self.dataset = load_from_disk(path_or_dataset)

        column_names = self.dataset.column_names
        self.lengths = self.dataset.data.column("length").to_numpy() if "length" in column_names else None
        self.v_tokens = self.dataset.data.column("v_tokens").to_numpy() if "v_tokens" in column_names else None

        self.max_len = kwargs.get("max_len", 8192)
        self.prior_dist = kwargs.get("prior_dist", "Mask")
        # switch noisy prior dist for curriculum learning
        self.prior_dist_2 = kwargs.get("prior_dist_2", None)
        self.switch_prior_thresh = kwargs.get("switch_prior_thresh", 1.0)

        self.min_pixels = kwargs.get("min_pixels", 8*8*32*32)
        self.max_pixels = kwargs.get("max_pixels", 64*64*32*32)
        self.video_min_pixels = 64 * 64     # 4 tokens.
        self.video_max_pixels = 1024 * 1024 # 1024 tokens.
        self.video_total_pixels = self.max_len * 32 * 32 * 0.9
        print(f"Load {len(self.dataset)} samples")

    def get_metadata(self):
        if self.lengths is not None and self.v_tokens is not None:
            return self.lengths, self.v_tokens
        else:
            return None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            idx = idx[0]

        item = self.dataset[idx]
        messages = item['messages']
        if isinstance(messages, str):
            messages = json.loads(messages)

        # 遍历对话中的每一轮消息 (system, user, assistant)
        for message in messages:
            content_list = message.get('content', [])
            if not isinstance(content_list, list):
                continue

            # 遍历消息中的每一个内容数据块 (text, image, video)
            for content in content_list:
                # --- 图像类型 ---
                if content.get("image") is not None:
                    image_data = content['image']

                    if isinstance(image_data, dict) and 'bytes' in image_data:
                        # 处理字节流格式
                        content['image'] = Image.open(io.BytesIO(image_data['bytes'])).convert("RGB")
                    elif isinstance(image_data, str):
                        full_path = os.path.join(self.root_dir, image_data)
                        content['image'] = Image.open(full_path).convert("RGB")

                    # 移除不需要传给 processor 的元数据
                    content.pop('height', None)
                    content.pop('width', None)

                    # 注入图像分辨率配置
                    content['min_pixels'] = self.min_pixels
                    content['max_pixels'] = self.max_pixels

                # --- 视频类型 ---
                elif content.get("video") is not None:
                    video_path = content["video"]
                    full_path = os.path.join(self.root_dir, video_path)

                    content.pop('height', None)
                    content.pop('width', None)
                    content.pop('num_frames', None)

                    # 视频通常传路径，由 processor 内部的 fetch_video 处理抽帧
                    content["video"] = full_path

                    # 注入视频分辨率配置
                    content['min_pixels'] = self.video_min_pixels
                    content['max_pixels'] = self.video_max_pixels
                    # total_pixels 字段控制总预算
                    content['total_pixels'] = self.video_total_pixels

        messages[0]["prior_dist"] = self.prior_dist
        messages = clear_none_fields(messages)
        return messages


class mmfinereason_dataset(torch.utils.data.Dataset):
    def __init__(self, path_or_dataset, **kwargs):
        self.ds = load_dataset("parquet", data_files=f"{path_or_dataset}/sft-*.parquet", split="train", num_proc=32)
        self.root_dir = kwargs.get("root_dir", None)
        assert self.root_dir is not None and os.path.exists(self.root_dir), f"{self.root_dir}"

        self.min_pixels = kwargs.get("min_pixels", 8*8*32*32)
        self.max_pixels = kwargs.get("max_pixels", 64*64*32*32)

        print(f"Load {len(self.ds)} samples")

    def get_metadata(self):
        return None

    def __len__(self):
        return len(self.ds)

    def _load_image(self, image_data):
        if image_data is None:
            return None
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        if isinstance(image_data, dict):
            if image_data.get("bytes") is not None:
                return Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
            image_data = image_data.get("path") or image_data.get("image")
        if isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        if isinstance(image_data, str):
            image_path = image_data
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.root_dir, image_path)
            assert os.path.exists(image_path), f"{image_path} not exist"
            return Image.open(image_path).convert("RGB")
        return None

    def _build_user_content(self, question, image):
        question = (question or "").strip()
        parts = question.split("<image>")
        content = []

        for idx, text in enumerate(parts):
            text = text.strip()
            if text:
                content.append({"type": "text", "text": text})
            if idx < len(parts) - 1 and image is not None:
                content.append({
                    "type": "image",
                    "image": image,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                })

        if image is not None and "<image>" not in question:
            content.insert(0, {
                "type": "image",
                "image": image,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            })

        return content

    def __getitem__(self, idx):
        sample = self.ds[idx]

        question = sample.get("question", "")
        image = self._load_image(sample.get("image", None))
        assistant_response = (
            sample.get("qwen3vl_235b_thinking_response")
            or sample.get("original_answer")
            or sample.get("answer")
            or ""
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": self._build_user_content(question, image),
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(assistant_response).strip()}],
            },
        ]

        return clear_none_fields(messages)


class bard_uni_dataset(torch.utils.data.Dataset):
    """Dataset for Bard-Uni image generation/editing tasks.

    Loads pre-tokenized VQ data (Arrow IPC files from prepare_t2i_data.py).
    Uses memory-mapped IO — init is instant, data paged in on demand by OS.
    Each sample contains: vq_codes (flat int32 list), code_height, code_width, caption.

    NOTE: The table is loaded lazily (on first __getitem__) so that DataLoader workers
    with multiprocessing_context="spawn" don't need to pickle the full table.
    """

    def __init__(self, path_or_dataset, **kwargs):
        self.task_type = kwargs.get("task_type", "text_to_image")
        self.max_len = kwargs.get("max_len", 8192)
        print(f"Loading bard_uni_dataset ({self.task_type}) from {path_or_dataset}...")

        data_path = Path(path_or_dataset)
        self._arrow_files = sorted(str(f) for f in data_path.rglob("*.arrow"))
        if not self._arrow_files:
            raise FileNotFoundError(f"No .arrow files found in {data_path}")

        # Load table in main process to get metadata (lengths, total_len, column names)
        self._load_table()
        print(f"Loaded {self._total_len} samples from {len(self._arrow_files)} arrow files")

    def _load_table(self):
        import pyarrow.ipc as ipc
        tables = []
        for f in self._arrow_files:
            mmap = pa.memory_map(f, "r")
            reader = ipc.open_file(mmap)
            tables.append(reader.read_all())
            mmap.close()
        self._table = pa.concat_tables(tables)
        self._total_len = len(self._table)
        self._has_src = "src_vq_codes" in self._table.column_names
        self._has_lengths = "seq_length" in self._table.column_names

    def __getstate__(self):
        """Drop the table when pickling (for DataLoader spawn workers)."""
        state = self.__dict__.copy()
        state.pop("_table", None)
        return state

    def __setstate__(self, state):
        """Re-open mmap'd table in each worker process."""
        self.__dict__.update(state)
        self._load_table()

    def get_metadata(self):
        if not self._has_lengths:
            return None
        lengths = self._table.column("seq_length").to_numpy()
        v_tokens = self._table.column("v_tokens").to_numpy()
        return lengths, v_tokens

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        if isinstance(idx, list):
            idx = idx[0]

        row = self._table.slice(idx, 1)
        vq_codes = row.column("vq_codes")[0].as_py()
        code_height = row.column("code_height")[0].as_py()
        code_width = row.column("code_width")[0].as_py()
        caption = row.column("caption")[0].as_py()

        result = {
            "task_type": self.task_type,
            "vq_codes": vq_codes,
            "code_height": code_height,
            "code_width": code_width,
            "caption": caption,
        }

        if self._has_src:
            result["src_vq_codes"] = row.column("src_vq_codes")[0].as_py()
            result["src_code_height"] = row.column("src_code_height")[0].as_py()
            result["src_code_width"] = row.column("src_code_width")[0].as_py()
            result["instruction"] = row.column("instruction")[0].as_py()

        return result


class internvl_dataset(torch.utils.data.Dataset):
    """Bard-InternVL block-diffusion dataset.

    Same `messages` source as `bard_vl_dataset` (datasets/mixed-8192-b32-tiny). Loads image
    paths/bytes into PIL.Image and tags `prior_dist`. Unlike the Qwen path, image resolution
    is governed by InternVL dynamic tiling (min/max_patches) in the collate, so no
    min_pixels/max_pixels are injected here.
    """

    def __init__(self, path_or_dataset, **kwargs):
        self.root_dir = kwargs.get("root_dir", None)
        assert self.root_dir is not None and os.path.exists(self.root_dir), f"{self.root_dir}"

        print(f"Loading internvl_dataset from {path_or_dataset}...")
        self.dataset = load_from_disk(path_or_dataset)

        self.max_len = kwargs.get("max_len", 8192)
        self.prior_dist = kwargs.get("prior_dist", "Mask")
        print(f"Load {len(self.dataset)} samples")

    def get_metadata(self):
        # InternVL token counts differ from the stored (Qwen-computed) lengths -> disable bucketing.
        return None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            idx = idx[0]

        item = self.dataset[idx]
        messages = item['messages']
        if isinstance(messages, str):
            messages = json.loads(messages)

        for message in messages:
            content_list = message.get('content', [])
            if not isinstance(content_list, list):
                continue
            for content in content_list:
                if content.get("image") is not None:
                    image_data = content['image']
                    if isinstance(image_data, dict) and 'bytes' in image_data:
                        content['image'] = Image.open(io.BytesIO(image_data['bytes'])).convert("RGB")
                    elif isinstance(image_data, str):
                        full_path = os.path.join(self.root_dir, image_data)
                        content['image'] = Image.open(full_path).convert("RGB")
                    content.pop('height', None)
                    content.pop('width', None)
                elif content.get("video") is not None:
                    # Keep the resolved path as a string; frame sampling happens in the
                    # processor (InternVLProcessorLite._preprocess_video), same as image tiling.
                    video_data = content['video']
                    if isinstance(video_data, str):
                        content['video'] = os.path.join(self.root_dir, video_data)
                    content.pop('height', None)
                    content.pop('width', None)

        messages[0]["prior_dist"] = self.prior_dist
        messages = clear_none_fields(messages)
        return messages
