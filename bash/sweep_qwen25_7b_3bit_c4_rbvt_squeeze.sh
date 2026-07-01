#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[rbvt-c4-sweep] FAILED at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
MODEL_LABEL="${MODEL_LABEL:-qwen25_7b}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"
BIT="${BIT:-3}"
DEVICE="${DEVICE:-cuda:0}"

DATASET="${DATASET:-c4}"
NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-2048}"
RBVT_N_CALIB="${RBVT_N_CALIB:-128}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/qwen25_7b_3bit_c4_sqllm_lnq_rbvt}"
CACHE_ROOT="${CACHE_ROOT:-cache/qwen25_7b_3bit_c4_sqllm_lnq_rbvt}"
SWEEP_ROOT="${SWEEP_ROOT:-${OUTPUT_ROOT}/rbvt_sweep}"
PPL_DIR="${PPL_DIR:-${OUTPUT_ROOT}/ppl}"

CHUNK_DIR="${CHUNK_DIR:-${OUTPUT_ROOT}/chunks}"
SQ_DIR="${SQ_DIR:-${OUTPUT_ROOT}/squeezellm_w${BIT}}"
RBVT_STATS="${RBVT_STATS:-${OUTPUT_ROOT}/rbvt_stats_${DATASET}_s${NSAMPLES}_blk${SEQLEN}_n${RBVT_N_CALIB}.pt}"

GPU_DEVICES="${GPU_DEVICES:-auto}"
GPU_MAX_DEVICES="${GPU_MAX_DEVICES:-1}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-20000}"
AUTO_GPU_CANDIDATES="${AUTO_GPU_CANDIDATES:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

RBVT_BATCH_SIZE="${RBVT_BATCH_SIZE:-4}"
RBVT_ROW_CHUNK_DEFAULT="${RBVT_ROW_CHUNK_DEFAULT:-4096}"
RBVT_GAP_FLOOR_DEFAULT="${RBVT_GAP_FLOOR_DEFAULT:-1e-8}"

PPL_BACKEND="${PPL_BACKEND:-dense_lut}"
DENSE_EVAL_DTYPE="${DENSE_EVAL_DTYPE:-float16}"
SWEEP_EVAL_STYLES="${SWEEP_EVAL_STYLES:-guidedquant}"
SWEEP_PPL_DATASETS="${SWEEP_PPL_DATASETS:-wikitext2 c4}"
GUIDEDQUANT_PPL_BATCH_SIZE="${GUIDEDQUANT_PPL_BATCH_SIZE:-4}"
NONUQ_MAX_LENGTH="${NONUQ_MAX_LENGTH:-2048}"
NONUQ_STRIDE="${NONUQ_STRIDE:-512}"
NONUQ_C4_SAMPLES="${NONUQ_C4_SAMPLES:-2000}"
LIMIT_TOKENS="${LIMIT_TOKENS:-0}"
FORCE_RBVT="${FORCE_RBVT:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
PREPARE_BASE="${PREPARE_BASE:-1}"

# Format: label:lambda:topk:row_chunk:gap_floor:allow_overshoot
RBVT_SWEEP_CONFIGS="${RBVT_SWEEP_CONFIGS:-l0p0_t0:0.0:0:${RBVT_ROW_CHUNK_DEFAULT}:${RBVT_GAP_FLOOR_DEFAULT}:0 l0p25_t0:0.25:0:${RBVT_ROW_CHUNK_DEFAULT}:${RBVT_GAP_FLOOR_DEFAULT}:0 l0p5_t0:0.5:0:${RBVT_ROW_CHUNK_DEFAULT}:${RBVT_GAP_FLOOR_DEFAULT}:0 l1p0_t0:1.0:0:${RBVT_ROW_CHUNK_DEFAULT}:${RBVT_GAP_FLOOR_DEFAULT}:0}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export ATTN_IMPLEMENTATION

mkdir -p "${SWEEP_ROOT}" "${PPL_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [rbvt-c4-sweep] $*"
}

count_files() {
  local dir="$1"
  local pattern="$2"
  if [[ ! -d "${dir}" ]]; then
    echo 0
    return
  fi
  find "${dir}" -maxdepth 1 -name "${pattern}" | wc -l | tr -d ' '
}

auto_gpu_devices() {
  local min_free_mb="$1"
  local candidates="${2:-${AUTO_GPU_CANDIDATES}}"
  local query
  query="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "${query}" ]]; then
    echo ""
    return
  fi

  local chosen=()
  local best_idx=""
  local best_free="-1"
  local idx free
  while IFS=, read -r idx free; do
    idx="${idx//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    if [[ -z "${idx}" || -z "${free}" ]]; then
      continue
    fi
    if [[ "${candidates}" != "auto" && " ${candidates} " != *" ${idx} "* ]]; then
      continue
    fi
    if (( free > best_free )); then
      best_free="${free}"
      best_idx="${idx}"
    fi
    if (( free >= min_free_mb )); then
      chosen+=("${idx}")
    fi
  done <<< "${query}"

  if [[ "${#chosen[@]}" -eq 0 && -n "${best_idx}" ]]; then
    chosen=("${best_idx}")
  fi

  local limited=()
  local i
  for i in "${!chosen[@]}"; do
    if (( i >= GPU_MAX_DEVICES )); then
      break
    fi
    limited+=("${chosen[$i]}")
  done
  echo "${limited[*]}"
}

resolve_devices() {
  if [[ "${GPU_DEVICES}" == "auto" ]]; then
    auto_gpu_devices "${GPU_MIN_FREE_MB}"
  else
    echo "${GPU_DEVICES}"
  fi
}

primary_device() {
  local devices="$1"
  local arr=()
  if [[ -n "${devices}" ]]; then
    read -r -a arr <<< "${devices}"
  fi
  if [[ "${#arr[@]}" -gt 0 ]]; then
    echo "cuda:${arr[0]}"
  else
    echo "${DEVICE}"
  fi
}

run_layer_shards_on_devices() {
  local layer_count="$1"
  local devices="$2"
  shift 2

  local arr=()
  if [[ -n "${devices}" ]]; then
    read -r -a arr <<< "${devices}"
  fi
  if [[ "${#arr[@]}" -eq 0 ]]; then
    "$@" --device "${DEVICE}"
    return
  fi

  local ngpu="${#arr[@]}"
  local shard_size=$(( (layer_count + ngpu - 1) / ngpu ))
  local pids=()
  local shard_idx
  for shard_idx in "${!arr[@]}"; do
    local start=$(( shard_idx * shard_size ))
    local end=$(( start + shard_size ))
    if (( start >= layer_count )); then
      continue
    fi
    if (( end > layer_count )); then
      end="${layer_count}"
    fi
    log "RBVT shard ${start},${end} on cuda:${arr[$shard_idx]}"
    "$@" --device "cuda:${arr[$shard_idx]}" --layer_range "${start},${end}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
}

ensure_base_artifacts() {
  local layer_count
  layer_count="$(count_files "${CHUNK_DIR}" 'layer_*.pt')"
  if [[ "${layer_count}" -gt 0 && -d "${SQ_DIR}/lut" && "$(count_files "${SQ_DIR}/lut" 'l*.pkl')" -ge "${layer_count}" ]]; then
    log "Using existing SqueezeLLM base artifacts: chunks=${CHUNK_DIR}, lut=${SQ_DIR}"
    return
  fi
  if [[ "${PREPARE_BASE}" != "1" ]]; then
    echo "Missing base artifacts. Set PREPARE_BASE=1 or run the C4 job first." >&2
    exit 1
  fi

  log "Preparing missing C4 SqueezeLLM base artifacts"
  PPL_TARGETS=squeezellm \
    PPL_DATASETS=wikitext2 \
    EVAL_STYLES=guidedquant \
    GPU_DEVICES="${GPU_DEVICES}" \
    GPU_MAX_DEVICES="${GPU_MAX_DEVICES}" \
    bash "${SCRIPT_DIR}/run_qwen25_7b_3bit_c4_sqllm_lnq_rbvt_eval.sh"
}

ensure_rbvt_stats() {
  if [[ -s "${RBVT_STATS}" ]]; then
    log "Using existing RBVT stats: ${RBVT_STATS}"
    return
  fi

  local devices device
  devices="$(resolve_devices)"
  device="$(primary_device "${devices}")"
  log "Collecting reusable RBVT stats on ${device}: ${RBVT_STATS}"
  "${PYTHON_BIN}" quantization/rbvt_squeezellm.py stats \
    --model "${MODEL}" \
    --model_chunks "${CHUNK_DIR}" \
    --input_lut "${SQ_DIR}" \
    --output_folder "${SWEEP_ROOT}/_stats_probe" \
    --model_type "${MODEL_TYPE}" \
    --dataset "${DATASET}" \
    --nsamples "${NSAMPLES}" \
    --seqlen "${SEQLEN}" \
    --cache_dir "${CACHE_ROOT}/tokens" \
    --stats_path "${RBVT_STATS}" \
    --device "${device}" \
    --n_calib "${RBVT_N_CALIB}" \
    --batch_size "${RBVT_BATCH_SIZE}" \
    --rbvt_lambda 1.0 \
    --row_chunk "${RBVT_ROW_CHUNK_DEFAULT}" \
    --gap_floor "${RBVT_GAP_FLOOR_DEFAULT}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}"
}

eval_config() {
  local label="$1"
  local lut_dir="$2"
  local style="$3"
  local dataset="$4"
  local devices device batch stride tag
  devices="$(resolve_devices)"
  device="$(primary_device "${devices}")"

  if [[ "${style}" == "guidedquant" || "${style}" == "repo" ]]; then
    style="guidedquant"
    batch="${GUIDEDQUANT_PPL_BATCH_SIZE}"
    stride="${NONUQ_MAX_LENGTH}"
    tag="guidedquant_ctx${NONUQ_MAX_LENGTH}"
  elif [[ "${style}" == "nonuquantfix" || "${style}" == "nonuq" ]]; then
    style="nonuquantfix"
    batch="1"
    stride="${NONUQ_STRIDE}"
    tag="nonuquantfix"
  else
    echo "Unsupported eval style: ${style}" >&2
    exit 1
  fi

  local out_file="${PPL_DIR}/rbvt_sweep_${label}_${dataset}_${tag}_${PPL_BACKEND}_ppl.json"
  if [[ -s "${out_file}" && "${FORCE_EVAL}" != "1" ]]; then
    log "Reusing eval ${out_file}"
    return
  fi

  log "Evaluating ${label} ${dataset} ${style} on ${device}"
  "${PYTHON_BIN}" quantization/eval_nonuquantfix_ppl.py \
    --model "${MODEL}" \
    --checkpoint "${OUTPUT_ROOT}/packed/${MODEL_LABEL}_rbvt_sweep_${label}_w${BIT}.pt" \
    --wbits "${BIT}" \
    --model_type "${MODEL_TYPE}" \
    --backend "${PPL_BACKEND}" \
    --lut_folder "${lut_dir}" \
    --dense_dtype "${DENSE_EVAL_DTYPE}" \
    --datasets "${dataset}" \
    --device "${device}" \
    --stride "${stride}" \
    --max_length "${NONUQ_MAX_LENGTH}" \
    --batch_size "${batch}" \
    --eval_style "${style}" \
    --c4_samples "${NONUQ_C4_SAMPLES}" \
    --limit_tokens "${LIMIT_TOKENS}" \
    --output_file "${out_file}"
}

ensure_base_artifacts
LAYER_COUNT="$(count_files "${CHUNK_DIR}" 'layer_*.pt')"
if [[ "${LAYER_COUNT}" -lt 1 ]]; then
  echo "No model chunks found in ${CHUNK_DIR}" >&2
  exit 1
fi
ensure_rbvt_stats

DEVICES="$(resolve_devices)"
log "Sweep devices: ${DEVICES:-${DEVICE}}"
log "Sweep configs: ${RBVT_SWEEP_CONFIGS}"

for entry in ${RBVT_SWEEP_CONFIGS}; do
  IFS=: read -r label rbvt_lambda rbvt_topk row_chunk gap_floor allow_overshoot <<< "${entry}"
  if [[ -z "${label}" || -z "${rbvt_lambda}" || -z "${rbvt_topk}" ]]; then
    echo "Bad RBVT_SWEEP_CONFIGS entry: ${entry}" >&2
    exit 1
  fi
  row_chunk="${row_chunk:-${RBVT_ROW_CHUNK_DEFAULT}}"
  gap_floor="${gap_floor:-${RBVT_GAP_FLOOR_DEFAULT}}"
  allow_overshoot="${allow_overshoot:-0}"

  out_dir="${SWEEP_ROOT}/${label}"
  mkdir -p "${out_dir}/lut"
  if [[ "$(count_files "${out_dir}/lut" 'l*.pkl')" -ge "${LAYER_COUNT}" && "${FORCE_RBVT}" != "1" ]]; then
    log "Reusing RBVT config ${label}: ${out_dir}"
  else
    log "Applying RBVT config ${label}: lambda=${rbvt_lambda}, topk=${rbvt_topk}, row_chunk=${row_chunk}, gap=${gap_floor}, overshoot=${allow_overshoot}"
    extra_args=()
    if [[ "${allow_overshoot}" == "1" ]]; then
      extra_args+=(--allow_overshoot)
    fi
    run_layer_shards_on_devices "${LAYER_COUNT}" "${DEVICES}" \
      "${PYTHON_BIN}" quantization/rbvt_squeezellm.py apply \
      --model "${MODEL}" \
      --model_chunks "${CHUNK_DIR}" \
      --input_lut "${SQ_DIR}" \
      --output_folder "${out_dir}" \
      --model_type "${MODEL_TYPE}" \
      --dataset "${DATASET}" \
      --nsamples "${NSAMPLES}" \
      --seqlen "${SEQLEN}" \
      --cache_dir "${CACHE_ROOT}/tokens" \
      --stats_path "${RBVT_STATS}" \
      --n_calib "${RBVT_N_CALIB}" \
      --batch_size "${RBVT_BATCH_SIZE}" \
      --rbvt_lambda "${rbvt_lambda}" \
      --rbvt_topk "${rbvt_topk}" \
      --row_chunk "${row_chunk}" \
      --gap_floor "${gap_floor}" \
      --attn_implementation "${ATTN_IMPLEMENTATION}" \
      "${extra_args[@]}"
  fi

  for style in ${SWEEP_EVAL_STYLES}; do
    for dataset in ${SWEEP_PPL_DATASETS}; do
      eval_config "${label}" "${out_dir}" "${style}" "${dataset}"
    done
  done
done

"${PYTHON_BIN}" - "${PPL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

ppl_dir = Path(sys.argv[1])
rows = []
for path in sorted(ppl_dir.glob("rbvt_sweep_*_ppl.json")):
    try:
        payload = json.loads(path.read_text())
    except Exception:
        continue
    results = payload.get("results", {})
    for dataset, item in results.items():
        rows.append((path.name, dataset, item.get("perplexity")))

if rows:
    print("RBVT sweep results:")
    for name, dataset, ppl in rows:
        print(f"{name}\t{dataset}\t{ppl}")
PY

log "Done. Sweep artifacts: ${SWEEP_ROOT}; PPL JSONs: ${PPL_DIR}/rbvt_sweep_*"
