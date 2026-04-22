<h1 align="center">BARD: Bridging AutoRegressive and Diffusion Vision-Language Models Via Highly Efficient Progressive Block Merging and Stage-Wise Distillation</h1>


<p align="center">
  <a href="https://github.com/cbyzju">Baoyou Chen</a><sup>1,3</sup> ·
  <a href="https://github.com/1ring2rta">Hanchen Xia</a><sup>1</sup> ·
  <a href="https://github.com/yhpengtu-rgb">Peng Tu</a><sup>1</sup> ·
  <a href="https://github.com/Theseus-427">Haojun Shi</a><sup>1</sup> ·
  <a href="https://github.com/AricGamma">Liwei Zhang</a><sup>1</sup> ·
  <a href="https://github.com/weihaosky">Weihao Yuan</a><sup>4</sup> ·
  <a href="https://sites.google.com/site/zhusiyucs/home">Siyu Zhu</a><sup>1,2,3,†</sup>
</p>

<p align="center">
  <sup>1</sup>Shanghai Academy of AI for Science
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>2</sup>Shanghai Innovation Institute
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>3</sup>Fudan University
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>4</sup>Nanjing University
</p>

<p align="center">
  <a href="https://fudan-generative-vision.github.io/Bard-VL"><img src="https://img.shields.io/badge/Project-HomePage-Green" alt="Project Page"></a>
  <a href="https://arxiv.org/pdf/2604.16514"><img src="https://img.shields.io/badge/Paper-Arxiv-red" alt="Paper"></a>
  <a href="https://huggingface.co/collections/fudan-generative-ai/bard-vl"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow" alt="Hugging Face"></a>
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick%20Start-0f172a?style=flat&logoColor=white" alt="Quick Start"></a>
  <a href="#training"><img src="https://img.shields.io/badge/Training-1d4ed8?style=flat&logoColor=white" alt="Training"></a>
  <a href="#inference"><img src="https://img.shields.io/badge/Inference-0f766e?style=flat&logoColor=white" alt="Inference"></a>
  <a href="#evaluation"><img src="https://img.shields.io/badge/Evaluation-7c3aed?style=flat&logoColor=white" alt="Evaluation"></a>
  <a href="#repository-layout"><img src="https://img.shields.io/badge/Repository-475569?style=flat&logoColor=white" alt="Repository"></a>
</p>


<table align='center' border="0" style="width: 100%; text-align: center; margin-top: 80px;">
    <tr>
      <td>
        <video align='center' src="https://github.com/user-attachments/assets/1b5abd60-c88c-41a7-a029-0dd143c98262" muted autoplay loop></video>
      </td>
    </tr>
  </table>

<a id="quick-start"></a>
## 🚀 Quick Start

```bash
git clone https://github.com/fudan-generative-vision/Bard-VL.git
cd Bard-VL
```

```bash
conda create -n bard-vl python=3.12 -y
conda activate bard-vl

pip install -r requirements.txt
```

<a id="training"></a>
## ⚙️ Training

### 1. Data Preparation

To prepare the training data, run:

```bash
bash tools/preprocessing.sh
```

This downloads the public source datasets, runs the local converters, and prepares the mixed metadata set used by the training configs that point to `datasets/mixed-8192-17M`.

Additional preprocessing notes are in [`tools/preprocessing/README.md`](./tools/preprocessing/README.md).


<details>
<summary><strong>Training sample format</strong></summary>

<br>

The training pipeline expects each sample to contain a `messages` field in chat format. A minimal multimodal example is shown below.

```json
{
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are a helpful assistant."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": "FineVision-processed/chart2text/images/example_chart.jpg"
        },
        {
          "type": "text",
          "text": "Please clarify the meaning conveyed by this graph."
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "The graph shows ..."
        }
      ]
    }
  ]
}
```

</details>

### 2. Stage 1: Progressive Block Merging

The PBM configs live under [`examples/vlm_finetune/bard_vl/`](./examples/vlm_finetune/bard_vl), and matching launch wrappers are available under [`scripts/`](./scripts).

| Variant | Config |
| --- | --- |
| [`B4-Mask-2B-Instruct`](https://huggingface.co/fudan-generative-ai/Bard-VL-B4-Mask-2B-Instruct) | [`examples/vlm_finetune/bard_vl/bard_vl_b4_mask_2b_instruct.yaml`](./examples/vlm_finetune/bard_vl/bard_vl_b4_mask_2b_instruct.yaml) |
| [`B4-Mask-4B-Instruct`](https://huggingface.co/fudan-generative-ai/Bard-VL-B4-Mask-4B-Instruct) | [`examples/vlm_finetune/bard_vl/bard_vl_b4_mask_4b_instruct.yaml`](./examples/vlm_finetune/bard_vl/bard_vl_b4_mask_4b_instruct.yaml) |
| [`B4-Mask-8B-Instruct`](https://huggingface.co/fudan-generative-ai/Bard-VL-B4-Mask-8B-Instruct) | [`examples/vlm_finetune/bard_vl/bard_vl_b4_mask_8b_instruct.yaml`](./examples/vlm_finetune/bard_vl/bard_vl_b4_mask_8b_instruct.yaml) |

Download the base checkpoints with:

```bash
bash tools/download_weights.sh
```

Launch training from the repository root. Example:

```bash
bash scripts/bard_vl_b4_mask_4b_instruct.sh
```

Checkpoints are written under the configured experiment directory, for example:

```text
exps/bard_vl_b4_mask_4b_instruct/epoch_0_step_19999/
```

### 3. Export Checkpoint

The distillation configs expect HuggingFace-style model directories such as `pretrained_models/Bard-VL-B4-Mask-4B-Instruct`. If your PBM training produced a checkpoint directory, export it with [`tools/consolidate_checkpoint.py`](./tools/consolidate_checkpoint.py):

```bash
python3 tools/consolidate_checkpoint.py \
  --dcp-dir exps/bard_vl_b4_mask_4b_instruct/epoch_0_step_39999/model \
  --output-dir pretrained_models/Bard-VL-B4-Mask-4B-Instruct \
  --source-model-dir pretrained_models/Qwen3-VL-Bard-4B-Instruct
```

### 4. Stage 2: Stage-Wise Distillation

The SWD configs live under [`examples/distillation/`](./examples/distillation):

| Variant | Config |
| --- | --- |
| [`B4-Mask-2B-Distil-Instruct`](https://huggingface.co/fudan-generative-ai/Bard-VL-B4-Mask-2B-Distil-Instruct) | [`examples/distillation/bard_vl_kd_diffusion_b4_mask_2b.yaml`](./examples/distillation/bard_vl_kd_diffusion_b4_mask_2b.yaml) |
| [`B8-Mask-2B-Distil-Instruct`](https://huggingface.co/fudan-generative-ai/Bard-VL-B8-Mask-2B-Distil-Instruct) | [`examples/distillation/bard_vl_kd_diffusion_b8_mask_2b.yaml`](./examples/distillation/bard_vl_kd_diffusion_b8_mask_2b.yaml) |
| [`B16-Mask-2B-Distil-Instruct`](https://huggingface.co/fudan-generative-ai/Bard-VL-B16-Mask-2B-Distil-Instruct) | [`examples/distillation/bard_vl_kd_diffusion_b16_mask_2b.yaml`](./examples/distillation/bard_vl_kd_diffusion_b16_mask_2b.yaml) |
| [`B8-Mask-4B-Distil-Instruct`](https://huggingface.co/fudan-generative-ai/Bard-VL-B8-Mask-4B-Distil-Instruct) | [`examples/distillation/bard_vl_kd_diffusion_b8_mask_4b.yaml`](./examples/distillation/bard_vl_kd_diffusion_b8_mask_4b.yaml) |
| [`B16-Mask-4B-Distil-Instruct`](https://huggingface.co/fudan-generative-ai/Bard-VL-B16-Mask-4B-Distil-Instruct) | [`examples/distillation/bard_vl_kd_diffusion_b16_mask_4b.yaml`](./examples/distillation/bard_vl_kd_diffusion_b16_mask_4b.yaml) |
| [`B32-Mask-4B-Distil-Instruct`](https://huggingface.co/fudan-generative-ai/Bard-VL-B32-Mask-4B-Distil-Instruct) | [`examples/distillation/bard_vl_kd_diffusion_b32_mask_4b.yaml`](./examples/distillation/bard_vl_kd_diffusion_b32_mask_4b.yaml) |

The repository includes launch wrappers such as [`scripts/bard_vl_kd_diffusion_b4_mask_2b.sh`](./scripts/bard_vl_kd_diffusion_b4_mask_2b.sh). Edit the environment-specific lines in those wrappers before running them.

```bash
bash scripts/bard_vl_kd_diffusion_b4_mask_2b.sh
```

SWD checkpoints are already saved as ready-to-load Hugging Face-style model directories under `exps/<project>/step-*`.

The shell launchers under [`scripts/`](./scripts) and the evaluation examples under [`eval/lmms-eval/examples/models/`](./eval/lmms-eval/examples/models/) include local conda activation, filesystem paths, and environment variables from the original training environment, so they should be treated as templates.

<a id="inference"></a>
## 🎬 Inference

[`inference.py`](./inference.py) contains minimal examples for image and video understanding. Edit the `messages` list inside the script to select the modality and prompt you want to test.

```bash
python3 inference.py \
  --model_id pretrained_models/Bard-VL-B4-Mask-4B-Instruct \
  --block_size 4 \
  --denoising_steps 4 \
  --confidence_threshold 0.6
```

Sample local assets are available under [`assets/`](./assets).

<a id="evaluation"></a>
## 📊 Evaluation

The repository vendors an LMMS-Eval setup under [`eval/lmms-eval/`](./eval/lmms-eval). The Bard-VL model wrapper is implemented in [`eval/lmms-eval/lmms_eval/models/simple/bard_vl.py`](./eval/lmms-eval/lmms_eval/models/simple/bard_vl.py).

A clean single-node evaluation example is:

```bash
cd eval/lmms-eval
export PYTHONPATH=../..:$PYTHONPATH

accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
  --model bard_vl \
  --model_args "pretrained=../../pretrained_models/Bard-VL-B4-Mask-4B-Instruct,max_pixels=4194304,block_size=4,attn_implementation=sdpa,remasking_strategy=low_confidence_static,confidence_threshold=1.0,interleave_visuals=False" \
  --tasks mme,realworldqa,mmstar,ai2d,chartqa \
  --batch_size 1 \
  --output_path eval_results/bard_vl_b4_mask_4b
```

Alternatively, you can directly use a bash script for multi-node evaluation as shown below:
```bash
cd eval/lmms-eval
bash examples/models/bard_vl.sh
```

<a id="repository-layout"></a>
## 🗂️ Repository Layout

| Path | Purpose |
| --- | --- |
| [`inference.py`](./inference.py) | Minimal generation example for image, video, and text inputs |
| [`examples/vlm_finetune/bard_vl/`](./examples/vlm_finetune/bard_vl) | Stage-1 PBM training configs |
| [`examples/distillation/`](./examples/distillation) | Stage-2 SWD configs |
| [`train/`](./train) | Stage-2 model distillation training code |
| [`tools/preprocessing.sh`](./tools/preprocessing.sh) | Public dataset download and preprocessing entrypoint |
| [`tools/consolidate_checkpoint.py`](./tools/consolidate_checkpoint.py) | Export a training checkpoint to a Hugging Face-style model directory |
| [`tools/README.md`](./tools/README.md) | Usage notes for the utilities under `tools/` |
| [`eval/lmms-eval/`](./eval/lmms-eval) | Evaluation harness with a Bard-VL model wrapper |
| [`scripts/`](./scripts) | Local launch wrappers with machine-specific settings |

<a id="citation"></a>
## 📝 Citation

If you find Bard-VL useful in your research, please cite [our paper](https://arxiv.org/abs/2604.16514):

```bibtex
@article{chen2026bard,
  title={BARD: Bridging AutoRegressive and Diffusion Vision-Language Models Via Highly Efficient Progressive Block Merging and Stage-Wise Distillation},
  author={Chen, Baoyou and Xia, Hanchen and Tu, Peng and Shi, Haojun and Mu, Shan and Yuan, Weihao and Zhu, Siyu},
  journal={arXiv preprint arXiv:2604.16514},
  year={2026}
}
```

<a id="acknowledgements"></a>
## 🙏 Acknowledgements

This repository builds on top of [NVIDIA NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel).
