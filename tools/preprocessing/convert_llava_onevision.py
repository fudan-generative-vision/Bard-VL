import os
import json
import copy
import gc
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
import multiprocessing as mp

# ================= 配置区 =================
DATASET_ROOT = "datasets/LLaVA-OneVision-1.5-Instruct-Data"
SAVE_ROOT = "datasets/LLaVA-OneVision"
NUM_WORKERS = 16  # 建议设为 CPU 核心数的一半左右

SUBSETS = [
    'CLEVR', 'CLEVR-Math',
    'Docmatix-part-00-of-10', 'Docmatix-part-01-of-10', 'Docmatix-part-02-of-10', 'Docmatix-part-03-of-10', 'Docmatix-part-04-of-10', 'Docmatix-part-05-of-10', 'Docmatix-part-06-of-10', 'Docmatix-part-07-of-10', 'Docmatix-part-08-of-10', 'Docmatix-part-09-of-10',
    'Evol-Instruct-GPT4-Turbo', 'FigureQA', 'GEOS', 'GeoQA+', 'Geometry3K', 'IconQA', 'OmniDocBench_train', 'PMC-VQA', 'Super-CLEVR', 'VisualWebInstruct', 'VizWiz', 'ai2d', 'alfredplpl', 'alfworld',
    'allava', 'allava_instruct_laion4v', 'allava_instruct_vflan4v', 'allenai_pixmo_docs', 'amc_aime', 'aokvqa', 'aops_forum', 'arxiv_figs', 'arxivqa',
    'cambrian', 'chart2text', 'chartqa', 'chrome_writting', 'cn_k12', 'coco', 'code_feedback_66k', 'datikz', 'diagram', 'docvqa_train', 'dvqa', 'dvqa-part-00-of-10', 'finqa',
    'geo170k_align', 'geo170k_qa', 'geo3k', 'geomverse', 'gpt4o', 'gpt4v', 'gqa', 'gsm8k', 'hateful_memes', 'hitab', 'hme100k', 'iam', 'idk', 'ifeval_like', 'iiit', 'image_textualization', 'infographic_azuregpt4v', 'infographic_vqa', 'intergps', 'invoices-and-receipts_ocr', 'laion_220k', 'latex_ocr', 'llava_cot_100k', 'llava_instruct', 'llava_wild', 'llavar', 'llrv_gpt4v', 'lrv_chart',
    'magpie_pro', 'magpie_ultra', 'mapqa', 'math', 'mathinstruct_262k', 'mathqa', 'mavis_math_metagen', 'ocr', 'ocrvqa', 'olympiads', 'oodvqa', 'open_orca', 'openmathinstruct', 'orca_994k', 'orca_agentinstruct', 'oroikon_chart_captioning', 'pathvqa',
    'plotqa-part-00-of-10', 'plotqa-part-01-of-10', 'plotqa-part-02-of-10', 'plotqa-part-03-of-10', 'plotqa-part-04-of-10', 'plotqa-part-05-of-10', 'plotqa-part-06-of-10', 'plotqa-part-07-of-10', 'plotqa-part-08-of-10', 'plotqa-part-09-of-10', 'raven', 'rects', 'rendered_text', 'robut_sqa', 'robut_wikisql', 'robut_wtq', 'rootsautomation',
    'scienceqa', 'screen2words', 'screen_qa', 'sharegpt4o', 'sherlock', 'sroie_data', 'st_vqa',
    'sharegpt4v-part-00-of-10', 'sharegpt4v-part-01-of-10', 'sharegpt4v-part-03-of-10', 'sharegpt4v-part-04-of-10', 'sharegpt4v-part-05-of-10', 'sharegpt4v-part-06-of-10', 'sharegpt4v-part-07-of-10', 'sharegpt4v-part-08-of-10', 'sharegpt4v-part-09-of-10', 'svit-part-00-of-10',
    'svit-part-01-of-10', 'svit-part-02-of-10', 'svit-part-03-of-10', 'svit-part-04-of-10', 'svit-part-05-of-10', 'svit-part-06-of-10', 'svit-part-07-of-10', 'svit-part-08-of-10', 'svit-part-09-of-10',
    'synthetic_amc', 'synthetic_math', 'tabmwp', 'tallyqa', 'tat_qa', 'textcaps', 'textocr_gpt4v', 'textvqa', 'tinychart_train', 'tqa', 'unigeo', 'ureader_cap', 'ureader_chart', 'ureader_ie', 'ureader_kg', 'ureader_ocr', 'ureader_qa', 'ureader_tr', 'vflan', 'vg', 'viquae', 'vision_flan', 'vision_oritented', 'vistext', 'visual7w', 'visual_chat', 'visualmrc', 'vqaas', 'vqarad', 'vsr', 'websight', 'wikipedia_2m',
    'wit-part-00-of-10', 'wit-part-01-of-10', 'wit-part-02-of-10', 'wit-part-03-of-10', 'wit-part-04-of-10', 'wit-part-05-of-10', 'wit-part-06-of-10', 'wit-part-07-of-10', 'wit-part-08-of-10', 'wit-part-09-of-10', 'wizardlm'
]


# ==========================================

# 全局变量，利用fork机制让子进程共享内存映射
ds_global = None

def worker_fn(args):
    """
    args: (index, subset_name, subset_image_dir)
    """
    global ds_global
    idx, subset_name, subset_image_dir = args

    try:
        item = ds_global[idx]
    except Exception:
        return None

    raw_id = item.get('id', 'unknown')
    pil_img = item.get('image')
    conversations = item.get('conversations', [])

    image_rel_path, width, height = None, 0, 0
    if pil_img is not None:
        try:
            img_filename = f"{raw_id}.jpg"
            img_path = os.path.join(subset_image_dir, img_filename)
            width, height = pil_img.size

            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            # 质量逻辑：Chart、OCR、Doc 类子集提升至 95
            quality = 95
            name_lower = subset_name.lower()
            if any(k in name_lower for k in ['chart', 'ocr', 'doc']):
                quality = 95

            pil_img.save(img_path, format="JPEG", quality=quality, optimize=True)
            # 子集本地 JSON 使用的相对路径
            image_rel_path = os.path.join("images", img_filename)
        except Exception:
            return None

    llava_convs = []
    qwen_convs = [{"role": "system", "content": "You are a helpful assistant"}]

    for i, turn in enumerate(conversations):
        # 兼容性 Key 获取
        role_raw = turn.get('from') or turn.get('role')
        value_raw = turn.get('value') or turn.get('content')
        
        if not role_raw or not value_raw:
            continue

        # 1. 处理 LLaVA 格式
        mapped_role_llava = "human" if role_raw in ["human", "user"] else "gpt"
        llava_convs.append({"from": mapped_role_llava, "value": value_raw})

        # 2. 处理 Qwen 格式
        mapped_role_qwen = "user" if role_raw in ["human", "user"] else "assistant"
        qwen_content = []
        
        if mapped_role_qwen == "user" and image_rel_path and i == 0:
            # 仅在首轮注入图片块
            qwen_content.append({
                "type": "image", 
                "image": image_rel_path, 
                "height": height, 
                "width": width
            })
            # 移除文本中的显式标记
            clean_text = value_raw.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            if clean_text:
                qwen_content.append({"type": "text", "text": clean_text})
        else:
            qwen_content.append({"type": "text", "text": value_raw})

        qwen_convs.append({"role": mapped_role_qwen, "content": qwen_content})

    llava_item = {"id": str(raw_id), "conversations": llava_convs}
    if image_rel_path:
        llava_item.update({"image": image_rel_path, "width": width, "height": height})

    qwen_item = {"conversation": qwen_convs}
    return llava_item, qwen_item


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)

    global ds_global

    for subset in SUBSETS:
        # 路径逻辑核对：SAVE_ROOT/subset/images
        subset_dir = os.path.join(SAVE_ROOT, subset)
        subset_image_dir = os.path.join(subset_dir, "images")
        os.makedirs(subset_image_dir, exist_ok=True)

        print(f"\n>>> Loading: {subset}")
        try:
            # 建立索引阶段，建议关注磁盘 I/O
            ds_global = load_dataset(DATASET_ROOT, subset, split='train', num_proc=16)
            total_items = len(ds_global)
        except Exception as e:
            print(f"!!! Fail to load {subset}: {e}")
            continue

        # 用于当前子集目录的小表
        subset_llava = []
        subset_qwen = []

        # mp.Pool 配合 imap 解决主进程分发卡顿
        with mp.Pool(processes=NUM_WORKERS) as pool:
            # 参数序列：(idx, subset_name, subset_image_dir)
            tasks = ((idx, subset, subset_image_dir) for idx in range(total_items))

            # 使用 imap 保持顺序并减少内存堆积
            for result in tqdm(pool.imap(worker_fn, tasks, chunksize=32), total=total_items, desc=f"Progress {subset}"):
                if result is not None:
                    # 解包两个格式
                    ll_item, qw_item = result

                    # A. 存入子集本地列表 (保持 images/xxx.jpg 路径)
                    subset_llava.append(ll_item)
                    subset_qwen.append(qw_item)

        # 保存子集本地 json
        with open(os.path.join(subset_dir, "llava.json"), 'w', encoding='utf-8') as f:
            json.dump(subset_llava, f, ensure_ascii=False)
        with open(os.path.join(subset_dir, "qwen3.json"), 'w', encoding='utf-8') as f:
            json.dump(subset_qwen, f, ensure_ascii=False)

        print(f"--- {subset} 处理完成。")

        # 内存回收
        ds_global = None
        del subset_llava, subset_qwen
        gc.collect()

    print(f"\n全部子集处理完毕。")

if __name__ == "__main__":
    main()