#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
mkdir -p pretrained_models

huggingface-cli download cbyzju/Qwen3-VL-Bard-2B-Instruct --local-dir pretrained_models/Qwen3-VL-Bard-2B-Instruct

huggingface-cli download cbyzju/Qwen3-VL-Bard-4B-Instruct --local-dir pretrained_models/Qwen3-VL-Bard-4B-Instruct

huggingface-cli download cbyzju/Qwen3-VL-Bard-8B-Instruct --local-dir pretrained_models/Qwen3-VL-Bard-8B-Instruct
