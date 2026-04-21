# Tools

Standalone utilities for dataset preprocessing and checkpoint export.

## Scripts

- [`preprocessing.sh`](./preprocessing.sh)
  - Downloads the public preprocessing datasets.
  - Runs the conversion scripts under [`preprocessing/`](./preprocessing/).
  - Optionally downloads the mixed metadata set used by the training pipeline.
- [`consolidate_checkpoint.py`](./consolidate_checkpoint.py)
  - Converts a NeMo DCP checkpoint directory into Hugging Face-compatible `safetensors` shards.
  - Optionally copies tokenizer and config artifacts into the output directory.

## `preprocessing.sh`

Prepares Bard-VL training data from the public source datasets referenced by the repository.

### Common Commands

Run the full pipeline:

```bash
bash tools/preprocessing.sh
```

Run in the background with log redirection:

```bash
bash tools/preprocessing.sh --background --log-file logs/preprocessing.log
tail -f logs/preprocessing.log
```

Run a single stage:

```bash
bash tools/preprocessing.sh --download-only
bash tools/preprocessing.sh --convert-only
bash tools/preprocessing.sh --metadata-only
```

Preview commands without executing them:

```bash
bash tools/preprocessing.sh --dry-run
```

Full preprocessing notes are in [`tools/preprocessing/README.md`](./preprocessing/README.md).

## `consolidate_checkpoint.py`

Exports a training checkpoint saved in NeMo DCP format to a standard `safetensors` model directory.

### Key Arguments

- `--dcp-dir`
  - Directory containing the DCP model state, usually `.../epoch_x_step_y/model`
- `--output-dir`
  - Destination directory for the converted `safetensors` files
- `--source-model-dir`
  - Directory containing tokenizer/config artifacts such as `*.json` and `*.txt`
  - Required unless `--skip-copy-artifacts` is set

### Common Commands

Convert a checkpoint and copy tokenizer/config files from a pretrained model directory:

```bash
python3 tools/consolidate_checkpoint.py \
  --dcp-dir exps/bard_vl_b4_mask_4b_instruct/epoch_0_step_74999/model \
  --output-dir exps/bard_vl_b4_mask_4b_instruct/epoch_0_step_74999/safetensors \
  --source-model-dir pretrained_models/Qwen3-VL-Bard-4B-Instruct
```

Control shard size and target dtype:

```bash
python3 tools/consolidate_checkpoint.py \
  --dcp-dir exps/bard_vl_b4_mask_4b_instruct/epoch_0_step_74999/model \
  --output-dir exps/bard_vl_b4_mask_4b_instruct/epoch_0_step_74999/safetensors \
  --source-model-dir pretrained_models/Qwen3-VL-Bard-4B-Instruct \
  --max-shard-size-gb 2 \
  --dtype bf16
```

Export weights only:

```bash
python3 tools/consolidate_checkpoint.py \
  --dcp-dir exps/bard_vl_b4_mask_4b_instruct/epoch_0_step_74999/model \
  --output-dir exps/bard_vl_b4_mask_4b_instruct/epoch_0_step_74999/safetensors \
  --skip-copy
```

Run `bash tools/preprocessing.sh --help` or `python3 tools/consolidate_checkpoint.py --help` for the full option list.
