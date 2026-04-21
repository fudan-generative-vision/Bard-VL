import os
import json
import copy
import gc
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
import multiprocessing as mp

# ================= 配置区 =================
DATASET_ROOT = "datasets/FineVision"
SAVE_ROOT = "datasets/FineVision-processed"
NUM_WORKERS = 24  # 建议设为 CPU 核心数的一半左右

SUBSETS = ['CoSyn_400k_chart', 'CoSyn_400k_chemical', 'CoSyn_400k_circuit', 'CoSyn_400k_diagram', 'CoSyn_400k_document', 'CoSyn_400k_graphic', 'CoSyn_400k_math', 'CoSyn_400k_music', 'CoSyn_400k_nutrition', 'CoSyn_400k_table',
           'DoclingMatix', 'LLaVA_Instruct_150K', 'README.md', 'SynthChartNet', 'SynthCodeNet', 'SynthFormulaNet', 'Unichart', 'a_okvqa', 'aguvis-stage-1', 'ai2d_merged', 'alfworldgpt', 'allava_laion', 'allava_vflan', 'aokvqa', 'art', 'arxivqa',
           'bentham', 'blockdiagramcomputerized', 'blockdiagramhandwritten', 'cambrian(filtered)_processed', 'captcha', 'chart2text', 'chartqa', 'chinesememe', 'chrome_writting', 'clevr', 'clevr_math', 'clevr_math(mathv360k)', 'coco_colors', 'cocoqa', 'cocotext', 'ctw',
           'datik', 'datikz', 'densefusion_1m', 'diagram_image_to_text', 'docvqa', 'drivelm', 'dvqa', 'est_vqa', 'face_emotion', 'figureqa', 'figureqa(mathv360k)', 'finqa', 'funsd',
           'geo170k(align)', 'geo170k(qa)', 'geo3k', 'geometry3k(mathv360k)', 'geomverse', 'geoqa+(mathv360k)', 'geos(mathv360k)', 'google_landmarks', 'groundui',
           'handwriting_forms', 'hateful_memes', 'hitab', 'hme100k', 'hw_squad', 'iam', 'iconqa', 'iconqa(mathv360k)', 'idk', 'iiit5k', 'image_textualization(filtered)', 'imgur5k', 'indoor_qa', 'infographic(gpt4v)', 'infographic_vqa', 'infographic_vqa_llava_format', 'intergps', 'invoices_receipts',
           'k12_printing', 'laion_gpt4v', 'latex_handwritten', 'latexformulas', 'llavar_gpt4_20k', 'lnqa', 'localized_narratives', 'lrv_chart', 'lrv_normal(filtered)', 'lvis_instruct4v',
           'mapqa', 'mapqa(mathv360k)', 'maptext', 'mathwriting-google', 'mavis_math_metagen', 'mavis_math_rule_geo', 'memotion', 'mimic_cgd', 'mmc_instruct', 'mmevol', 'mmra', 'mmsoc_memotion', 'multihiertt',
           'nlvr2', 'objects365_qa', 'ocrvqa', 'olmOCR-mix-0225-books', 'olmOCR-mix-0225-documents', 'oodvqa', 'orand_car_a', 'pathvqa', 'pdfvqa', 'plotqa', 'pmc_vqa(mathv360k)',
           'raven', 'rendered_text', 'robut_sqa', 'robut_wikisql', 'robut_wtq', 'scienceqa', 'scienceqa(nona_context)', 'screen2words', 'screenqa', 'sharegpt4o', 'sharegpt4v(coco)', 'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)', 'sketchyvqa', 'slidevqa', 'spark', 'spatialsense', 'spot_the_diff', 'sroie', 'st_vqa', 'sujet_finance', 'super_clevr(mathv360k)', 'svrd', 'synthdog',
           'tabmwp', 'tabmwp(mathv360k)', 'tal_ocr_eng', 'tallyqa', 'tat_dqa', 'tat_qa', 'text_OpenMathInstruct-2', 'text_code_feedback', 'text_codefeedback_filtered_instruction', 'text_infinitymath', 'text_mathinstruct', 'text_mathqa', 'text_mathstepdpo10k', 'text_numinamath_cot', 'text_openhermes_2_5', 'text_openorca', 'text_orcamath', 'text_pythoncode25k', 'text_pythoncodealpaca', 'text_ruozhiba', 'text_theoremqa', 'text_wizardlm_evol', 'textcaps', 'textocr(gpt4v)', 'textvqa', 'tqa',
           'unigeo(mathv360k)', 'ureader_cap', 'ureader_ie', 'ureader_kg_processed', 'ureader_qa_processed',
           'vision_flan(filtered)', 'vistext', 'visual7w', 'visualmrc', 'vizwiz(mathv360k)', 'vqaonbd', 'vqarad', 'vqav2', 'vsr', 'websight', 'wildvision', 'wordart', 'yesbut']

# ==========================================

# 全局变量，利用fork机制让子进程共享内存映射
ds_global = None

def worker_fn(args):
    """
    针对新格式适配的 worker 函数
    item 格式: {'images': [PIL, ...], 'texts': [{'user': '...', 'assistant': '...'}, ...]}
    """
    global ds_global
    idx, subset_name, subset_image_dir = args

    try:
        item = ds_global[idx]
    except Exception:
        return None

    # 新格式中可能没有 'id'，使用索引作为备选 ID
    raw_id = item.get('id', f"{idx}")
    pil_images = item.get('images', [])
    text_rounds = item.get('texts', [])

    if not text_rounds:
        return None

    # 1. 批量处理多张图片
    image_info_list = [] # 存储 (rel_path, w, h)
    for img_idx, pil_img in enumerate(pil_images):
        try:
            img_filename = f"{raw_id}_{img_idx}.jpg"
            img_path = os.path.join(subset_image_dir, img_filename)

            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            width, height = pil_img.size
            quality = 95 if any(k in subset_name.lower() for k in ['chart', 'ocr', 'doc']) else 95

            pil_img.save(img_path, format="JPEG", quality=quality, optimize=True)
            image_info_list.append({
                "path": os.path.join("images", img_filename),
                "width": width,
                "height": height
            })
        except Exception:
            continue

    # 2. 构建对话格式
    llava_convs = []
    qwen_convs = [{"role": "system", "content": "You are a helpful assistant"}]

    for i, turn in enumerate(text_rounds):
        user_val = turn.get('user', '')
        assistant_val = turn.get('assistant', '')

        # --- A. 构建 LLaVA 格式 ---
        # 第一轮需要注入所有的 <image> 占位符
        if i == 0 and image_info_list:
            llava_user_text = "<image>\n" * len(image_info_list) + user_val
        else:
            llava_user_text = user_val

        llava_convs.append({"from": "human", "value": llava_user_text})
        llava_convs.append({"from": "gpt", "value": assistant_val})

        # --- B. 构建 Qwen 格式 ---
        qwen_user_content = []
        if i == 0 and image_info_list:
            # 第一轮先放图片 Block
            for img_info in image_info_list:
                qwen_user_content.append({
                    "type": "image",
                    "image": img_info["path"],
                    "height": img_info["height"],
                    "width": img_info["width"]
                })

        # 移除可能存在的旧 <image> 标签并添加文本 Block
        clean_user_text = user_val.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
        if clean_user_text:
            qwen_user_content.append({"type": "text", "text": clean_user_text})

        qwen_convs.append({"role": "user", "content": qwen_user_content})
        qwen_convs.append({"role": "assistant", "content": [{"type": "text", "text": assistant_val}]})

    llava_item = {"id": str(raw_id), "conversations": llava_convs}
    if image_info_list:
        # LLaVA-OneVision 多图通常支持列表形式的 image 字段
        llava_item["image"] = [info["path"] for info in image_info_list]

    qwen_item = {"conversation": qwen_convs}
    return llava_item, qwen_item

def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)

    global ds_global

    for subset in SUBSETS:
        # 路径逻辑：SAVE_ROOT/subset/images
        subset_dir = os.path.join(SAVE_ROOT, subset)
        if os.path.exists(os.path.join(subset_dir, "qwen3.json")):
            continue

        subset_image_dir = os.path.join(subset_dir, "images")
        os.makedirs(subset_image_dir, exist_ok=True)

        print(f"\n>>> Loading: {subset}")
        try:
            # 建立索引阶段
            # ds_global = load_dataset(DATASET_ROOT, subset, split='train', num_proc=16)
            ds_global = load_dataset("parquet", data_files=f'{DATASET_ROOT}/{subset}/*.parquet', split="train", num_proc=32)
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

        # 保存子集本地 json (indent=2)
        with open(os.path.join(subset_dir, "llava.json"), 'w', encoding='utf-8') as f:
            json.dump(subset_llava, f, ensure_ascii=False, indent=2)
        with open(os.path.join(subset_dir, "qwen3.json"), 'w', encoding='utf-8') as f:
            json.dump(subset_qwen, f, ensure_ascii=False, indent=2)

        print(f"--- {subset} 处理完成。")

        # 内存回收
        ds_global = None
        del subset_llava, subset_qwen
        gc.collect()

    print(f"\n全部子集处理完毕。")

if __name__ == "__main__":
    main()