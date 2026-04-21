import os
import json
import gc
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

# ================= 配置区 =================
DATASET_PATH = "datasets/MMFineReason-1.8M"  # 你的数据集本地路径或 HuggingFace ID
SAVE_ROOT = "datasets/MMFineReason-Processed"
IMAGE_SAVE_DIR = os.path.join(SAVE_ROOT, "images")
NUM_WORKERS = 16  # 根据你的 CPU 核心数调整

# 输出目录
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
# ==========================================

# 全局变量，利用fork机制共享内存
ds_global = None

def worker_fn(idx):
    """
    格式适配
    字段: question, id, original_answer, qwen3vl_235b_thinking_response, image(PIL)
    """
    global ds_global
    try:
        item = ds_global[idx]
    except Exception:
        return None

    raw_id = item.get('id', f"{idx}")
    pil_img = item.get('image')

    # 提取文本：优先使用带思维链的 qwen3vl_235b_thinking_response
    user_query = item.get('question', '').strip()
    assistant_response = item.get('qwen3vl_235b_thinking_response', '').strip()

    if not assistant_response:
        assistant_response = item.get('original_answer', '').strip()

    if not user_query or not assistant_response:
        return None

    # 1. 图像处理
    image_rel_path = None
    if pil_img is not None:
        try:
            img_filename = f"{raw_id}.jpg"
            img_abs_path = os.path.join(IMAGE_SAVE_DIR, img_filename)

            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            # 按照要求保存
            pil_img.save(img_abs_path, format="JPEG", quality=95, optimize=True)
            width, height = pil_img.size
            image_rel_path = os.path.join("images", img_filename)
        except Exception as e:
            return None

    # 2. 构建对话格式
    # LLaVA 格式
    llava_user_text = user_query
    if image_rel_path and "<image>" not in llava_user_text:
        llava_user_text = "<image>\n" + llava_user_text

    llava_item = {
        "id": str(raw_id),
        "image": image_rel_path if image_rel_path else None,
        "conversations": [
            {"from": "human", "value": llava_user_text},
            {"from": "gpt", "value": assistant_response}
        ]
    }

    # Qwen 格式 (Block 形式)
    clean_user_text = user_query.replace("<image>", "").strip()
    qwen_user_content = []
    if image_rel_path:
        qwen_user_content.append({
            "type": "image",
            "image": image_rel_path,
            "height": height,
            "width": width
        })
    qwen_user_content.append({"type": "text", "text": clean_user_text})

    qwen_item = {
        "conversation": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": qwen_user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]}
        ]
    }

    return llava_item, qwen_item

def main():
    global ds_global
    print(f">>> 正在加载全量数据集: {DATASET_PATH}")
    # 使用 num_proc 加速加载索引
    # dataset = load_dataset("parquet", data_files="sft-*.parquet", split="train")
    ds_global = load_dataset("parquet", data_files=f"{DATASET_PATH}/sft-*.parquet", split='train', num_proc=16)
    total_items = len(ds_global)
    print(f">>> 总条目数: {total_items}")

    # llava_all = []
    qwen_all = []

    # 使用进程池处理
    with mp.Pool(processes=NUM_WORKERS) as pool:
        # 使用 imap 以节省内存，chunksize 设为 64 减少通信开销
        pbar = tqdm(pool.imap(worker_fn, range(total_items), chunksize=64), total=total_items, desc="Processing")

        for result in pbar:
            if result:
                ll_item, qw_item = result
                # llava_all.append(ll_item)
                qwen_all.append(qw_item)

    # 写入最终文件
    print(f">>> 正在保存 JSON 文件到 {SAVE_ROOT}...")

    with open(os.path.join(SAVE_ROOT, "qwen3.json"), 'w', encoding='utf-8') as f:
        json.dump(qwen_all, f, ensure_ascii=False)

    print(">>> 处理完成！")

if __name__ == "__main__":
    main()