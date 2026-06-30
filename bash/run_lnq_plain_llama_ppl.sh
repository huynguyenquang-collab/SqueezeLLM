#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[lnq-plain] FAILED at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
GPU_DEVICES="${GPU_DEVICES:-}"
BITS="${BITS:-3 4}"
MODEL_SPECS="${MODEL_SPECS:-llama2_7b=meta-llama/Llama-2-7b-hf;llama3_8b=meta-llama/Meta-Llama-3-8B}"
GRADIENT_SPECS="${GRADIENT_SPECS:-}"
GRADIENT_CHUNKS_SPECS="${GRADIENT_CHUNKS_SPECS:-}"
INIT_LUT_SPECS="${INIT_LUT_SPECS:-}"

DATASET="${DATASET:-redpajama}"
NSAMPLES="${NSAMPLES:-1024}"
SEQLEN="${SEQLEN:-4096}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-1}"
ACTIVATION_STORAGE="${ACTIVATION_STORAGE:-disk}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-float16}"
HESSIAN_SAVE_DTYPE="${HESSIAN_SAVE_DTYPE:-float16}"
KEEP_ACTIVATION_CACHE="${KEEP_ACTIVATION_CACHE:-0}"
LNQ_ITERATIONS="${LNQ_ITERATIONS:-3}"
LNQ_CD_CYCLES="${LNQ_CD_CYCLES:-4}"
LNQ_ROW_BLOCK="${LNQ_ROW_BLOCK:-64}"
CPU_COUNT="${CPU_COUNT:-16}"
LAYER_RANGE="${LAYER_RANGE:-}"

NONUQ_MAX_LENGTH="${NONUQ_MAX_LENGTH:-4096}"
NONUQ_STRIDE="${NONUQ_STRIDE:-512}"
NONUQ_C4_SAMPLES="${NONUQ_C4_SAMPLES:-2000}"
LIMIT_TOKENS="${LIMIT_TOKENS:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/lnq_plain_llama_ppl}"
CACHE_ROOT="${CACHE_ROOT:-cache/lnq_plain}"
mkdir -p "${OUTPUT_ROOT}" "${CACHE_ROOT}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [lnq-plain] $*"
}

layer_arg=()
if [[ -n "${LAYER_RANGE}" ]]; then
  layer_arg=(--layer_range "${LAYER_RANGE}")
fi

if [[ -n "${GPU_DEVICES}" && -n "${LAYER_RANGE}" ]]; then
  echo "Set either GPU_DEVICES for automatic layer sharding or LAYER_RANGE for a manual single shard, not both." >&2
  exit 1
fi

run_hessian_shards() {
  local model="$1"
  local output_folder="$2"
  local layer_count="$3"
  local keep_activation_args=()
  if [[ "${KEEP_ACTIVATION_CACHE}" == "1" ]]; then
    keep_activation_args+=(--keep_activation_cache)
  fi

  if [[ -z "${GPU_DEVICES}" ]]; then
    "${PYTHON_BIN}" quantization/lnq.py hessians \
      --model "${model}" \
      --dataset "${DATASET}" \
      --nsamples "${NSAMPLES}" \
      --seqlen "${SEQLEN}" \
      --cache_dir "${CACHE_ROOT}/tokens" \
      --output_folder "${output_folder}" \
      --device "${DEVICE}" \
      --calib_batch_size "${CALIB_BATCH_SIZE}" \
      --activation_storage "${ACTIVATION_STORAGE}" \
      --activation_dtype "${ACTIVATION_DTYPE}" \
      --activation_cache_dir "${CACHE_ROOT}/activations/${label}_single" \
      --hessian_save_dtype "${HESSIAN_SAVE_DTYPE}" \
      "${keep_activation_args[@]}" \
      "${layer_arg[@]}"
    return
  fi

  read -r -a gpu_array <<< "${GPU_DEVICES}"
  local ngpu="${#gpu_array[@]}"
  local shard_size=$(( (layer_count + ngpu - 1) / ngpu ))
  local pids=()
  for shard_idx in "${!gpu_array[@]}"; do
    local start=$(( shard_idx * shard_size ))
    local end=$(( start + shard_size ))
    if (( start >= layer_count )); then
      continue
    fi
    if (( end > layer_count )); then
      end="${layer_count}"
    fi
    local gpu="${gpu_array[$shard_idx]}"
    log "Hessian shard ${start},${end} on cuda:${gpu}"
    "${PYTHON_BIN}" quantization/lnq.py hessians \
      --model "${model}" \
      --dataset "${DATASET}" \
      --nsamples "${NSAMPLES}" \
      --seqlen "${SEQLEN}" \
      --cache_dir "${CACHE_ROOT}/tokens" \
      --output_folder "${output_folder}" \
      --device "cuda:${gpu}" \
      --calib_batch_size "${CALIB_BATCH_SIZE}" \
      --activation_storage "${ACTIVATION_STORAGE}" \
      --activation_dtype "${ACTIVATION_DTYPE}" \
      --activation_cache_dir "${CACHE_ROOT}/activations/${label}_${start}_${end}" \
      --hessian_save_dtype "${HESSIAN_SAVE_DTYPE}" \
      "${keep_activation_args[@]}" \
      --layer_range "${start},${end}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
}

run_quantize_shards() {
  local model_chunks="$1"
  local hessians="$2"
  local initial_lut="$3"
  local output_folder="$4"
  local bit="$5"
  local layer_count="$6"

  if [[ -z "${GPU_DEVICES}" ]]; then
    "${PYTHON_BIN}" quantization/lnq.py quantize \
      --model_chunks "${model_chunks}" \
      --hessians "${hessians}" \
      --initial_lut "${initial_lut}" \
      --output_folder "${output_folder}" \
      --model_type llama \
      --bit "${bit}" \
      --num_iterations "${LNQ_ITERATIONS}" \
      --cd_cycles "${LNQ_CD_CYCLES}" \
      --row_block "${LNQ_ROW_BLOCK}" \
      --cpu_count "${CPU_COUNT}" \
      --device "${DEVICE}" \
      "${layer_arg[@]}"
    return
  fi

  read -r -a gpu_array <<< "${GPU_DEVICES}"
  local ngpu="${#gpu_array[@]}"
  local shard_size=$(( (layer_count + ngpu - 1) / ngpu ))
  local pids=()
  local cpu_per_shard=$(( CPU_COUNT / ngpu ))
  if (( cpu_per_shard < 1 )); then
    cpu_per_shard=1
  fi
  for shard_idx in "${!gpu_array[@]}"; do
    local start=$(( shard_idx * shard_size ))
    local end=$(( start + shard_size ))
    if (( start >= layer_count )); then
      continue
    fi
    if (( end > layer_count )); then
      end="${layer_count}"
    fi
    local gpu="${gpu_array[$shard_idx]}"
    log "LNQ quantize shard ${start},${end} on cuda:${gpu}"
    "${PYTHON_BIN}" quantization/lnq.py quantize \
      --model_chunks "${model_chunks}" \
      --hessians "${hessians}" \
      --initial_lut "${initial_lut}" \
      --output_folder "${output_folder}" \
      --model_type llama \
      --bit "${bit}" \
      --num_iterations "${LNQ_ITERATIONS}" \
      --cd_cycles "${LNQ_CD_CYCLES}" \
      --row_block "${LNQ_ROW_BLOCK}" \
      --cpu_count "${cpu_per_shard}" \
      --device "cuda:${gpu}" \
      --layer_range "${start},${end}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
}

IFS=';' read -r -a MODEL_ARRAY <<< "${MODEL_SPECS}"
declare -A GRADIENT_BY_LABEL=()
declare -A GRADIENT_CHUNKS_BY_LABEL=()
declare -A INIT_LUT_BY_LABEL=()

if [[ -n "${GRADIENT_SPECS}" ]]; then
  IFS=';' read -r -a GRADIENT_ARRAY <<< "${GRADIENT_SPECS}"
  for spec in "${GRADIENT_ARRAY[@]}"; do
    [[ "${spec}" == *=* ]] || { echo "GRADIENT_SPECS entries must be label=checkpoint; got ${spec}" >&2; exit 1; }
    GRADIENT_BY_LABEL["${spec%%=*}"]="${spec#*=}"
  done
fi

if [[ -n "${GRADIENT_CHUNKS_SPECS}" ]]; then
  IFS=';' read -r -a GRADIENT_CHUNKS_ARRAY <<< "${GRADIENT_CHUNKS_SPECS}"
  for spec in "${GRADIENT_CHUNKS_ARRAY[@]}"; do
    [[ "${spec}" == *=* ]] || { echo "GRADIENT_CHUNKS_SPECS entries must be label=path; got ${spec}" >&2; exit 1; }
    GRADIENT_CHUNKS_BY_LABEL["${spec%%=*}"]="${spec#*=}"
  done
fi

if [[ -n "${INIT_LUT_SPECS}" ]]; then
  IFS=';' read -r -a INIT_LUT_ARRAY <<< "${INIT_LUT_SPECS}"
  for spec in "${INIT_LUT_ARRAY[@]}"; do
    [[ "${spec}" == *=* ]] || { echo "INIT_LUT_SPECS entries must be label=path; got ${spec}" >&2; exit 1; }
    INIT_LUT_BY_LABEL["${spec%%=*}"]="${spec#*=}"
  done
fi

for spec in "${MODEL_ARRAY[@]}"; do
  if [[ "${spec}" != *=* ]]; then
    echo "MODEL_SPECS entries must be label=checkpoint; got ${spec}" >&2
    exit 1
  fi
  label="${spec%%=*}"
  model="${spec#*=}"
  safe_model="${model//\//_}"
  model_dir="${OUTPUT_ROOT}/${label}"
  chunk_dir="${model_dir}/chunks"
  gradient_chunk_dir="${GRADIENT_CHUNKS_BY_LABEL[${label}]:-${model_dir}/gradient_chunks}"
  hessian_dir="${model_dir}/hessians_${DATASET}_s${NSAMPLES}_blk${SEQLEN}"

  mkdir -p "${model_dir}" "${chunk_dir}"

  log "Chunking ${label}: ${model}"
  "${PYTHON_BIN}" quantization/chunk_models.py \
    --model "${model}" \
    --model_type llama \
    --output_path "${chunk_dir}"
  layer_count="$(find "${chunk_dir}" -maxdepth 1 -name 'layer_*.pt' | wc -l | tr -d ' ')"

  log "Collecting plain LNQ Hessians for ${label}"
  run_hessian_shards "${model}" "${hessian_dir}" "${layer_count}"

  for bit in ${BITS}; do
    init_dir="${INIT_LUT_BY_LABEL[${label}]:-${model_dir}/squeezellm_init_w${bit}}"
    run_dir="${model_dir}/w${bit}_${DATASET}_s${NSAMPLES}_blk${SEQLEN}_iter${LNQ_ITERATIONS}_cd${LNQ_CD_CYCLES}"
    packed_ckpt="${run_dir}/lnq_plain_${safe_model}_w${bit}.pt"
    ppl_json="${run_dir}/nonuquantfix_ppl.json"
    mkdir -p "${run_dir}"

    if [[ ! -f "${init_dir}/lut/l0.pkl" && ! -f "${init_dir}/l0.pkl" ]]; then
      if [[ -n "${GRADIENT_BY_LABEL[${label}]:-}" && -z "${GRADIENT_CHUNKS_BY_LABEL[${label}]:-}" ]]; then
        log "Chunking SqueezeLLM gradients for ${label}"
        "${PYTHON_BIN}" quantization/chunk_models.py \
          --model "${GRADIENT_BY_LABEL[${label}]}" \
          --model_type llama \
          --output_path "${gradient_chunk_dir}"
      fi

      if [[ ! -d "${gradient_chunk_dir}" ]]; then
        echo "Missing SqueezeLLM init for ${label}. Provide INIT_LUT_SPECS=${label}=... or GRADIENT_CHUNKS_SPECS=${label}=... / GRADIENT_SPECS=${label}=..." >&2
        exit 1
      fi

      log "Running SqueezeLLM weighted k-means init for ${label}, ${bit}-bit"
      "${PYTHON_BIN}" quantization/nuq.py \
        --model_type llama \
        --model "${chunk_dir}" \
        --gradient "${gradient_chunk_dir}" \
        --bit "${bit}" \
        --output_folder "${init_dir}"
    else
      log "Using existing SqueezeLLM init LUT: ${init_dir}"
    fi

    log "Optimizing LNQ LUTs for ${label}, ${bit}-bit"
    run_quantize_shards "${chunk_dir}" "${hessian_dir}" "${init_dir}" "${run_dir}" "${bit}" "${layer_count}"

    log "Packing ${label}, ${bit}-bit"
    "${PYTHON_BIN}" quantization/pack.py \
      --model "${model}" \
      --wbits "${bit}" \
      --folder "${run_dir}" \
      --save "${packed_ckpt}"

    log "Running NonUQuantFix-style PPL for ${label}, ${bit}-bit"
    "${PYTHON_BIN}" quantization/eval_nonuquantfix_ppl.py \
      --model "${model}" \
      --checkpoint "${packed_ckpt}" \
      --wbits "${bit}" \
      --device "${DEVICE}" \
      --stride "${NONUQ_STRIDE}" \
      --max_length "${NONUQ_MAX_LENGTH}" \
      --c4_samples "${NONUQ_C4_SAMPLES}" \
      --limit_tokens "${LIMIT_TOKENS}" \
      --output_file "${ppl_json}"
  done
done

log "Done. Outputs are under ${OUTPUT_ROOT}"
