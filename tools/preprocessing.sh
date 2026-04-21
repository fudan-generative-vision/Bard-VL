#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREPROCESSING_DIR="${SCRIPT_DIR}/preprocessing"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_LOG_FILE="${REPO_ROOT}/logs/preprocessing-$(date -u +%Y%m%d-%H%M%S).log"

RUN_DOWNLOAD=1
RUN_CONVERT=1
RUN_METADATA=1
RUN_DISTILL=0
DRY_RUN=0
RUN_BACKGROUND=0
LOG_FILE="${DEFAULT_LOG_FILE}"
BACKGROUND_CHILD_ARGS=()

usage() {
  cat <<EOF
Usage:
  bash tools/preprocessing.sh [options]

Description:
  Download the public preprocessing datasets used by Bard-VL, run the
  repository conversion scripts, and optionally download the final mixed
  metadata set and distillation dataset. The script is safe to run from
  any working directory.

Options:
  --download-only   Only download source datasets.
  --convert-only    Only run conversion scripts.
  --metadata-only   Only download the final mixed metadata.
  --distill-data    Also download the distillation dataset
                    cbyzju/mixed-distil-17w.
  --distill-only    Only download the distillation dataset.
  --no-metadata     Skip the final metadata download step.
  --background      Relaunch the script in the background with stdout/stderr
                    redirected to a log file.
  --log-file PATH   Log file path for --background. Default:
                    ${DEFAULT_LOG_FILE}
  --dry-run         Print the commands that would run, then exit.
  --help, -h        Show this help message.

Environment variables:
  PYTHON_BIN        Python executable to use. Default: python3
  HF_TOKEN          Optional Hugging Face token for gated/rate-limited access.

Examples:
  bash tools/preprocessing.sh
  bash tools/preprocessing.sh --download-only
  bash tools/preprocessing.sh --distill-data
  bash tools/preprocessing.sh --distill-only
  PYTHON_BIN=python bash tools/preprocessing.sh --convert-only
  bash tools/preprocessing.sh --background
  bash tools/preprocessing.sh --background --log-file logs/preprocessing.log
EOF
}

log() {
  printf '[preprocessing] %s\n' "$*"
}

die() {
  printf '[preprocessing] ERROR: %s\n' "$*" >&2
  exit 1
}

run() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

resolve_log_file() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${REPO_ROOT}/${path}"
  fi
}

spawn_background() {
  local log_file="$1"
  shift
  local -a child_args=("$@")

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] '
    printf '%q ' nohup env PREPROCESSING_BACKGROUND_CHILD=1 bash "${SCRIPT_DIR}/preprocessing.sh"
    printf '%q ' "${child_args[@]}"
    printf '>> %q 2>&1 < /dev/null &\n' "${log_file}"
    return 0
  fi

  mkdir -p "$(dirname "${log_file}")"
  nohup env PREPROCESSING_BACKGROUND_CHILD=1 bash "${SCRIPT_DIR}/preprocessing.sh" "${child_args[@]}" >>"${log_file}" 2>&1 </dev/null &
  local pid=$!

  log "Started background preprocessing job."
  log "PID: ${pid}"
  log "Log file: ${log_file}"
  log "Follow progress with: tail -f ${log_file}"
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "Required file not found: ${path}"
}

require_python_module() {
  local module="$1"
  "${PYTHON_BIN}" - <<PY >/dev/null 2>&1 || die "Missing Python module '${module}'. Install dependencies from requirements.txt first."
import importlib
importlib.import_module("${module}")
PY
}

hf_snapshot_download() {
  local repo_id="$1"
  local local_dir="$2"

  log "Downloading ${repo_id} -> ${local_dir}"
  run "${PYTHON_BIN}" - "${repo_id}" "${local_dir}" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = sys.argv[2]
token = os.environ.get("HF_TOKEN")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    token=token,
    resume_download=True,
)
PY
}

run_converter() {
  local script_path="$1"
  require_file "${script_path}"
  log "Running $(basename "${script_path}")"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] (cd %q && %q %q)\n' "${REPO_ROOT}" "${PYTHON_BIN}" "${script_path}"
    return 0
  fi
  (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" "${script_path}"
  )
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --download-only)
        RUN_DOWNLOAD=1
        RUN_CONVERT=0
        RUN_METADATA=0
        BACKGROUND_CHILD_ARGS+=("$1")
        ;;
      --convert-only)
        RUN_DOWNLOAD=0
        RUN_CONVERT=1
        RUN_METADATA=0
        BACKGROUND_CHILD_ARGS+=("$1")
        ;;
      --metadata-only)
        RUN_DOWNLOAD=0
        RUN_CONVERT=0
        RUN_METADATA=1
        RUN_DISTILL=0
        BACKGROUND_CHILD_ARGS+=("$1")
        ;;
      --distill-data)
        RUN_DISTILL=1
        BACKGROUND_CHILD_ARGS+=("$1")
        ;;
      --distill-only)
        RUN_DOWNLOAD=0
        RUN_CONVERT=0
        RUN_METADATA=0
        RUN_DISTILL=1
        BACKGROUND_CHILD_ARGS+=("$1")
        ;;
      --no-metadata)
        RUN_METADATA=0
        BACKGROUND_CHILD_ARGS+=("$1")
        ;;
      --background)
        RUN_BACKGROUND=1
        ;;
      --log-file)
        [[ $# -ge 2 ]] || die "--log-file requires a path argument."
        shift
        LOG_FILE="$1"
        ;;
      --dry-run)
        DRY_RUN=1
        BACKGROUND_CHILD_ARGS+=("$1")
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        die "Unknown option: $1. Use --help to see supported arguments."
        ;;
    esac
    shift
  done
}

main() {
  parse_args "$@"
  LOG_FILE="$(resolve_log_file "${LOG_FILE}")"

  if [[ "${RUN_BACKGROUND}" -eq 1 && "${PREPROCESSING_BACKGROUND_CHILD:-0}" != "1" ]]; then
    spawn_background "${LOG_FILE}" "${BACKGROUND_CHILD_ARGS[@]}"
    return 0
  fi

  require_file "${PREPROCESSING_DIR}/convert_finevision.py"
  require_file "${PREPROCESSING_DIR}/convert_llava_onevision.py"
  require_file "${PREPROCESSING_DIR}/convert_mmfine.py"

  require_python_module "huggingface_hub"

  if [[ "${RUN_CONVERT}" -eq 1 ]]; then
    require_python_module "datasets"
    require_python_module "PIL"
    require_python_module "tqdm"
  fi

  mkdir -p "${REPO_ROOT}/datasets"

  log "Repository root: ${REPO_ROOT}"
  log "Python executable: ${PYTHON_BIN}"

  if [[ "${RUN_DOWNLOAD}" -eq 1 ]]; then
    hf_snapshot_download \
      "mvp-lab/LLaVA-OneVision-1.5-Instruct-Data" \
      "${REPO_ROOT}/datasets/LLaVA-OneVision-1.5-Instruct-Data"
    hf_snapshot_download \
      "HuggingFaceM4/FineVision" \
      "${REPO_ROOT}/datasets/FineVision"
    hf_snapshot_download \
      "OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking" \
      "${REPO_ROOT}/datasets/MMFineReason-1.8M"
  fi

  if [[ "${RUN_CONVERT}" -eq 1 ]]; then
    run_converter "${PREPROCESSING_DIR}/convert_finevision.py"
    run_converter "${PREPROCESSING_DIR}/convert_llava_onevision.py"
    run_converter "${PREPROCESSING_DIR}/convert_mmfine.py"
  fi

  if [[ "${RUN_METADATA}" -eq 1 ]]; then
    hf_snapshot_download "cbyzju/mixed-8192-17M" "${REPO_ROOT}/datasets"
  fi

  if [[ "${RUN_DISTILL}" -eq 1 ]]; then
    hf_snapshot_download "cbyzju/mixed-distil-17w" "${REPO_ROOT}/datasets/mixed-distil-17w"
  fi

  log "Done."
}

main "$@"
