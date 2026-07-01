#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[qwen25-7b-3bit] FAILED at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
MODEL_LABEL="${MODEL_LABEL:-qwen25_7b}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"
BIT="${BIT:-3}"
DEVICE="${DEVICE:-cuda:0}"
GPU_DEVICES="${GPU_DEVICES:-auto}"
HESSIAN_GPU_DEVICES="${HESSIAN_GPU_DEVICES:-${GPU_DEVICES}}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-20000}"
HESSIAN_MIN_FREE_MB="${HESSIAN_MIN_FREE_MB:-30000}"
GPU_MAX_DEVICES="${GPU_MAX_DEVICES:-2}"
AUTO_GPU_CANDIDATES="${AUTO_GPU_CANDIDATES:-auto}"
HESSIAN_SAMPLE_PARALLEL="${HESSIAN_SAMPLE_PARALLEL:-0}"
HESSIAN_ACCUM_DEVICE="${HESSIAN_ACCUM_DEVICE:-cuda}"

DATASET="${DATASET:-redpajama}"
REDPAJAMA_DATASET="${REDPAJAMA_DATASET:-ZengXiangyu/RedPajama-Data-1T-Sample}"
REDPAJAMA_SPLIT="${REDPAJAMA_SPLIT:-train}"
NSAMPLES="${NSAMPLES:-1024}"
SEQLEN="${SEQLEN:-4096}"
FISHER_BATCH_SIZE="${FISHER_BATCH_SIZE:-1}"
FISHER_LAYERS_PER_PASS="${FISHER_LAYERS_PER_PASS:-1}"
FISHER_ACCUM_DEVICE="${FISHER_ACCUM_DEVICE:-cuda}"
FISHER_EMPTY_CACHE_INTERVAL="${FISHER_EMPTY_CACHE_INTERVAL:-0}"
FISHER_GRADIENT_CHECKPOINTING="${FISHER_GRADIENT_CHECKPOINTING:-on}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-1}"
HESSIAN_CALIB_BATCH_SIZE="${HESSIAN_CALIB_BATCH_SIZE:-2}"
LNQ_ITERATIONS="${LNQ_ITERATIONS:-3}"
LNQ_CD_CYCLES="${LNQ_CD_CYCLES:-4}"
LNQ_ROW_BLOCK="${LNQ_ROW_BLOCK:-64}"
CPU_COUNT="${CPU_COUNT:-16}"

RBVT_N_CALIB="${RBVT_N_CALIB:-1024}"
RBVT_BATCH_SIZE="${RBVT_BATCH_SIZE:-1}"
RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_TOPK="${RBVT_TOPK:-0}"
RBVT_ROW_CHUNK="${RBVT_ROW_CHUNK:-1024}"
RBVT_GAP_FLOOR="${RBVT_GAP_FLOOR:-1e-8}"

ACTIVATION_STORAGE="${ACTIVATION_STORAGE:-disk}"
HESSIAN_ACTIVATION_STORAGE="${HESSIAN_ACTIVATION_STORAGE:-ram}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-float16}"
HESSIAN_SAVE_DTYPE="${HESSIAN_SAVE_DTYPE:-float16}"

EVAL_DEVICE="${EVAL_DEVICE:-}"
EVAL_GPU_DEVICES="${EVAL_GPU_DEVICES:-${GPU_DEVICES}}"
EVAL_PARALLEL_DATASETS="${EVAL_PARALLEL_DATASETS:-0}"
PPL_EVAL_STYLE="${PPL_EVAL_STYLE:-nonuquantfix}"
PPL_BACKEND="${PPL_BACKEND:-dense_lut}"
DENSE_EVAL_DTYPE="${DENSE_EVAL_DTYPE:-float16}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-1}"
NONUQ_EXACT="${NONUQ_EXACT:-1}"
PPL_DATASETS="${PPL_DATASETS:-wikitext2 c4}"
PPL_TARGETS="${PPL_TARGETS:-squeezellm lnq_plain rbvt_squeeze}"
NONUQ_MAX_LENGTH="${NONUQ_MAX_LENGTH:-2048}"
NONUQ_STRIDE="${NONUQ_STRIDE:-512}"
NONUQ_C4_SAMPLES="${NONUQ_C4_SAMPLES:-2000}"
LIMIT_TOKENS="${LIMIT_TOKENS:-0}"
FORCE_REPACK="${FORCE_REPACK:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/qwen25_7b_3bit_sqllm_lnq_rbvt}"
CACHE_ROOT="${CACHE_ROOT:-cache/qwen25_7b_3bit_sqllm_lnq_rbvt}"
mkdir -p "${OUTPUT_ROOT}" "${CACHE_ROOT}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export REDPAJAMA_DATASET
export REDPAJAMA_SPLIT
export ATTN_IMPLEMENTATION

CHUNK_DIR="${OUTPUT_ROOT}/chunks"
FISHER_DIR="${OUTPUT_ROOT}/fisher_${DATASET}_s${NSAMPLES}_blk${SEQLEN}"
SQ_DIR="${OUTPUT_ROOT}/squeezellm_w${BIT}"
HESSIAN_DIR="${OUTPUT_ROOT}/lnq_hessians_${DATASET}_s${NSAMPLES}_blk${SEQLEN}"
LNQ_DIR="${OUTPUT_ROOT}/lnq_plain_w${BIT}_${DATASET}_s${NSAMPLES}_blk${SEQLEN}_iter${LNQ_ITERATIONS}_cd${LNQ_CD_CYCLES}"
RBVT_DIR="${OUTPUT_ROOT}/rbvt_squeeze_w${BIT}_lambda${RBVT_LAMBDA}"
PACK_DIR="${OUTPUT_ROOT}/packed"
PPL_DIR="${OUTPUT_ROOT}/ppl"
mkdir -p "${PACK_DIR}" "${PPL_DIR}"

SQ_CKPT="${PACK_DIR}/${MODEL_LABEL}_squeezellm_w${BIT}.pt"
LNQ_CKPT="${PACK_DIR}/${MODEL_LABEL}_lnq_plain_w${BIT}.pt"
RBVT_CKPT="${PACK_DIR}/${MODEL_LABEL}_rbvt_squeeze_w${BIT}.pt"
RBVT_STATS="${OUTPUT_ROOT}/rbvt_stats_${DATASET}_s${NSAMPLES}_blk${SEQLEN}_n${RBVT_N_CALIB}.pt"
RBVT_INPUT_LUT="${RBVT_INPUT_LUT:-${SQ_DIR}}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [qwen25-7b-3bit] $*"
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

stage_complete() {
  local dir="$1"
  local pattern="$2"
  local expected="$3"
  local count
  count="$(count_files "${dir}" "${pattern}")"
  [[ "${count}" -ge "${expected}" ]]
}

has_word() {
  local needle="$1"
  local haystack="$2"
  local item
  for item in ${haystack}; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

merge_ppl_partials() {
  local out_file="$1"
  shift
  "${PYTHON_BIN}" -c '
import json
import sys

out_file = sys.argv[1]
payload = None
results = {}
for path in sys.argv[2:]:
    with open(path, "r", encoding="utf-8") as handle:
        item = json.load(handle)
    if payload is None:
        payload = item
    results.update(item.get("results", {}))
if payload is None:
    raise SystemExit("No PPL partial files to merge.")
payload["results"] = results
with open(out_file, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
print(json.dumps(payload, indent=2))
' "${out_file}" "$@"
}

run_ppl() {
  local name="$1"
  local ckpt="$2"
  local lut_folder="$3"
  local eval_stride="${NONUQ_STRIDE}"
  local eval_style="nonuquantfix"
  local ppl_tag="nonuquantfix"
  local eval_batch_size="${PPL_BATCH_SIZE}"
  if [[ "${PPL_EVAL_STYLE}" == "guidedquant" || "${PPL_EVAL_STYLE}" == "repo" ]]; then
    eval_stride="${NONUQ_MAX_LENGTH}"
    eval_style="guidedquant"
    ppl_tag="guidedquant_ctx${NONUQ_MAX_LENGTH}"
  elif [[ "${PPL_EVAL_STYLE}" != "nonuquantfix" ]]; then
    echo "Unsupported PPL_EVAL_STYLE=${PPL_EVAL_STYLE}; expected nonuquantfix or guidedquant" >&2
    exit 1
  fi
  if [[ "${eval_style}" == "nonuquantfix" && "${NONUQ_EXACT}" == "1" ]]; then
    eval_batch_size=1
  fi
  case "${PPL_BACKEND}" in
    dense_lut)
      if [[ -z "${lut_folder}" ]]; then
        echo "PPL_BACKEND=dense_lut requires a LUT folder for ${name}" >&2
        exit 1
      fi
      ppl_tag="${ppl_tag}_dense_lut"
      ;;
    quant_cuda)
      ppl_tag="${ppl_tag}_quant_cuda"
      ;;
    *)
      echo "Unsupported PPL_BACKEND=${PPL_BACKEND}; expected dense_lut or quant_cuda" >&2
      exit 1
      ;;
  esac

  local out_file="${PPL_DIR}/${name}_${ppl_tag}_ppl.json"
  local eval_devices
  local eval_device
  local datasets=()
  read -r -a datasets <<< "${PPL_DATASETS}"
  local partial_files=()
  local dataset partial_file
  for dataset in "${datasets[@]}"; do
    case "${dataset}" in
      wikitext2|c4) ;;
      *)
        echo "Unsupported PPL dataset=${dataset}; expected wikitext2 or c4" >&2
        exit 1
        ;;
    esac
    partial_files+=("${PPL_DIR}/${name}_${dataset}_${ppl_tag}_ppl.json")
  done
  if [[ -s "${out_file}" && "${FORCE_EVAL}" != "1" ]]; then
    log "Reusing existing ${name} PPL at ${out_file}"
    return
  fi
  local all_partials_ready=1
  for partial_file in "${partial_files[@]}"; do
    if [[ ! -s "${partial_file}" || "${FORCE_EVAL}" == "1" ]]; then
      all_partials_ready=0
      break
    fi
  done
  if [[ "${all_partials_ready}" == "1" ]]; then
    log "Merging existing ${name} PPL partials into ${out_file}"
    merge_ppl_partials "${out_file}" "${partial_files[@]}"
    return
  fi

  if [[ "${EVAL_PARALLEL_DATASETS}" == "1" ]]; then
    if [[ -n "${EVAL_DEVICE}" ]]; then
      eval_devices="${EVAL_DEVICE#cuda:}"
    else
      eval_devices="$(resolve_devices "${EVAL_GPU_DEVICES}" "${GPU_MIN_FREE_MB}" "${name} PPL")"
    fi

    local eval_gpu_array=()
    if [[ -n "${eval_devices}" ]]; then
      read -r -a eval_gpu_array <<< "${eval_devices}"
    fi

    if [[ "${#eval_gpu_array[@]}" -ge "${#datasets[@]}" ]]; then
      local pids=()
      local idx gpu
      for idx in "${!datasets[@]}"; do
        dataset="${datasets[$idx]}"
        gpu="${eval_gpu_array[$idx]}"
        partial_file="${PPL_DIR}/${name}_${dataset}_${ppl_tag}_ppl.json"
        if [[ -s "${partial_file}" && "${FORCE_EVAL}" != "1" ]]; then
          log "Reusing existing ${name} ${dataset} PPL at ${partial_file}"
          continue
        fi
        log "Evaluating ${name} ${dataset} PPL on cuda:${gpu}"
        "${PYTHON_BIN}" quantization/eval_nonuquantfix_ppl.py \
          --model "${MODEL}" \
          --checkpoint "${ckpt}" \
          --wbits "${BIT}" \
          --model_type "${MODEL_TYPE}" \
          --backend "${PPL_BACKEND}" \
          --lut_folder "${lut_folder}" \
          --dense_dtype "${DENSE_EVAL_DTYPE}" \
          --datasets "${dataset}" \
          --device "cuda:${gpu}" \
          --stride "${eval_stride}" \
          --max_length "${NONUQ_MAX_LENGTH}" \
          --batch_size "${eval_batch_size}" \
          --eval_style "${eval_style}" \
          --c4_samples "${NONUQ_C4_SAMPLES}" \
          --limit_tokens "${LIMIT_TOKENS}" \
          --output_file "${partial_file}" &
        pids+=("$!")
      done
      for pid in "${pids[@]}"; do
        wait "${pid}"
      done
      merge_ppl_partials "${out_file}" "${partial_files[@]}"
      return
    fi
    log "${name} PPL: only ${#eval_gpu_array[@]} eval device(s); falling back to serial eval"
  fi

  if [[ -n "${EVAL_DEVICE}" ]]; then
    eval_device="${EVAL_DEVICE}"
  else
    eval_devices="$(resolve_devices "${GPU_DEVICES}" "${GPU_MIN_FREE_MB}" "${name} PPL")"
    eval_device="$(primary_device_from_devices "${eval_devices}")"
  fi
  local idx
  for idx in "${!datasets[@]}"; do
    dataset="${datasets[$idx]}"
    partial_file="${partial_files[$idx]}"
    if [[ -s "${partial_file}" && "${FORCE_EVAL}" != "1" ]]; then
      log "Reusing existing ${name} ${dataset} PPL at ${partial_file}"
      continue
    fi
    log "Evaluating ${name} ${dataset} PPL on ${eval_device}"
    "${PYTHON_BIN}" quantization/eval_nonuquantfix_ppl.py \
      --model "${MODEL}" \
      --checkpoint "${ckpt}" \
      --wbits "${BIT}" \
      --model_type "${MODEL_TYPE}" \
      --backend "${PPL_BACKEND}" \
      --lut_folder "${lut_folder}" \
      --dense_dtype "${DENSE_EVAL_DTYPE}" \
      --datasets "${dataset}" \
      --device "${eval_device}" \
      --stride "${eval_stride}" \
      --max_length "${NONUQ_MAX_LENGTH}" \
      --batch_size "${eval_batch_size}" \
      --eval_style "${eval_style}" \
      --c4_samples "${NONUQ_C4_SAMPLES}" \
      --limit_tokens "${LIMIT_TOKENS}" \
      --output_file "${partial_file}"
  done
  merge_ppl_partials "${out_file}" "${partial_files[@]}"
}

auto_gpu_devices() {
  local min_free_mb="$1"
  local candidates="${2:-${AUTO_GPU_CANDIDATES}}"

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    return
  fi

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
  local spec="$1"
  local min_free_mb="$2"
  local stage="$3"
  local devices
  if [[ "${spec}" == "auto" ]]; then
    devices="$(auto_gpu_devices "${min_free_mb}")"
    if [[ -z "${devices}" ]]; then
      log "${stage}: no CUDA device discovered; falling back to ${DEVICE}" >&2
    else
      log "${stage}: auto-selected CUDA devices: ${devices} (min_free=${min_free_mb}MiB)" >&2
    fi
  else
    devices="${spec}"
    log "${stage}: using configured CUDA devices: ${devices}" >&2
  fi
  echo "${devices}"
}

cpu_per_devices() {
  local devices="$1"
  local arr=()
  if [[ -n "${devices}" ]]; then
    read -r -a arr <<< "${devices}"
  fi
  local count="${#arr[@]}"
  if [[ "${count}" -lt 1 ]]; then
    count=1
  fi
  local per="$(( CPU_COUNT / count ))"
  if [[ "${per}" -lt 1 ]]; then
    per=1
  fi
  echo "${per}"
}

primary_device_from_devices() {
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
  local stage="$1"
  local layer_count="$2"
  local devices="$3"
  shift 3

  local stage_gpu_array=()
  if [[ -n "${devices}" ]]; then
    read -r -a stage_gpu_array <<< "${devices}"
  fi

  if [[ "${#stage_gpu_array[@]}" -eq 0 ]]; then
    "$@" --device "${DEVICE}"
    return
  fi

  local ngpu="${#stage_gpu_array[@]}"
  local shard_size=$(( (layer_count + ngpu - 1) / ngpu ))
  local pids=()
  for shard_idx in "${!stage_gpu_array[@]}"; do
    local start=$(( shard_idx * shard_size ))
    local end=$(( start + shard_size ))
    if (( start >= layer_count )); then
      continue
    fi
    if (( end > layer_count )); then
      end="${layer_count}"
    fi
    local gpu="${stage_gpu_array[$shard_idx]}"
    log "${stage} shard ${start},${end} on cuda:${gpu}"
    "$@" --device "cuda:${gpu}" --layer_range "${start},${end}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
}

run_layer_shards() {
  local stage="$1"
  local layer_count="$2"
  shift 2
  local devices
  devices="$(resolve_devices "${GPU_DEVICES}" "${GPU_MIN_FREE_MB}" "${stage}")"
  run_layer_shards_on_devices "${stage}" "${layer_count}" "${devices}" "$@"
}

run_missing_layer_shards_on_devices() {
  local stage="$1"
  local layer_count="$2"
  local devices="$3"
  local done_dir="$4"
  local done_prefix="$5"
  local done_suffix="$6"
  shift 6

  local stage_gpu_array=()
  if [[ -n "${devices}" ]]; then
    read -r -a stage_gpu_array <<< "${devices}"
  fi

  local missing=()
  local layer_idx
  for (( layer_idx = 0; layer_idx < layer_count; layer_idx++ )); do
    if [[ ! -s "${done_dir}/${done_prefix}${layer_idx}${done_suffix}" ]]; then
      missing+=("${layer_idx}")
    fi
  done

  if [[ "${#missing[@]}" -eq 0 ]]; then
    log "${stage} complete; skipping"
    return
  fi

  if [[ "${#stage_gpu_array[@]}" -eq 0 ]]; then
    local start="${missing[0]}"
    local end="$(( missing[${#missing[@]} - 1] + 1 ))"
    log "${stage} missing range ${start},${end} on ${DEVICE}"
    "$@" --device "${DEVICE}" --layer_range "${start},${end}"
    return
  fi

  local ngpu="${#stage_gpu_array[@]}"
  local shard_size=$(( (${#missing[@]} + ngpu - 1) / ngpu ))
  local pids=()
  local shard_idx
  for shard_idx in "${!stage_gpu_array[@]}"; do
    local first=$(( shard_idx * shard_size ))
    local last=$(( first + shard_size - 1 ))
    if (( first >= ${#missing[@]} )); then
      continue
    fi
    if (( last >= ${#missing[@]} )); then
      last="$((${#missing[@]} - 1))"
    fi
    local start="${missing[$first]}"
    local end="$(( missing[$last] + 1 ))"
    local gpu="${stage_gpu_array[$shard_idx]}"
    log "${stage} missing range ${start},${end} on cuda:${gpu}"
    "$@" --device "cuda:${gpu}" --layer_range "${start},${end}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
}

missing_layer_bounds() {
  local layer_count="$1"
  local done_dir="$2"
  local done_prefix="$3"
  local done_suffix="$4"
  local first=""
  local last=""
  local layer_idx
  for (( layer_idx = 0; layer_idx < layer_count; layer_idx++ )); do
    if [[ ! -s "${done_dir}/${done_prefix}${layer_idx}${done_suffix}" ]]; then
      if [[ -z "${first}" ]]; then
        first="${layer_idx}"
      fi
      last="${layer_idx}"
    fi
  done
  if [[ -n "${first}" ]]; then
    echo "${first},$(( last + 1 ))"
  fi
}

SQ_LUT_COUNT="$(count_files "${SQ_DIR}/lut" 'l*.pkl')"
NEEDS_LNQ_OR_RBVT=0
if has_word "lnq_plain" "${PPL_TARGETS}" || has_word "rbvt_squeeze" "${PPL_TARGETS}"; then
  NEEDS_LNQ_OR_RBVT=1
fi
SQUEEZE_EVAL_ONLY=0
if has_word "squeezellm" "${PPL_TARGETS}" && [[ "${NEEDS_LNQ_OR_RBVT}" == "0" ]]; then
  SQUEEZE_EVAL_ONLY=1
fi

if [[ "${SQUEEZE_EVAL_ONLY}" == "1" && "${SQ_LUT_COUNT}" -gt 0 ]]; then
  log "SqueezeLLM eval-only: reusing ${SQ_LUT_COUNT} LUT layers in ${SQ_DIR}/lut; skipping model chunks/Fisher"
elif [[ -d "${CHUNK_DIR}" && "$(count_files "${CHUNK_DIR}" 'layer_*.pt')" -gt 0 ]]; then
  log "Reusing existing model chunks in ${CHUNK_DIR}"
else
  log "Chunking model: ${MODEL}"
  "${PYTHON_BIN}" quantization/chunk_models.py \
    --model "${MODEL}" \
    --model_type "${MODEL_TYPE}" \
    --output_path "${CHUNK_DIR}"
fi

LAYER_COUNT="$(count_files "${CHUNK_DIR}" 'layer_*.pt')"
if [[ "${LAYER_COUNT}" -lt 1 && "${SQ_LUT_COUNT}" -gt 0 ]]; then
  LAYER_COUNT="${SQ_LUT_COUNT}"
fi
if [[ "${LAYER_COUNT}" -lt 1 ]]; then
  echo "No model chunks or SqueezeLLM LUTs found in ${CHUNK_DIR} / ${SQ_DIR}/lut" >&2
  exit 1
fi
log "Detected ${LAYER_COUNT} layers"

if stage_complete "${SQ_DIR}/lut" 'l*.pkl' "${LAYER_COUNT}"; then
  log "SqueezeLLM init LUT complete in ${SQ_DIR}/lut; skipping Fisher/SqueezeLLM build"
else
  if stage_complete "${FISHER_DIR}" 'layer_*.pt' "${LAYER_COUNT}"; then
    log "Fisher chunks complete in ${FISHER_DIR}; skipping"
  else
    log "Collecting Fisher gradient-square chunks for SqueezeLLM init"
    run_layer_shards "Fisher" "${LAYER_COUNT}" \
      "${PYTHON_BIN}" quantization/fisher.py \
      --model "${MODEL}" \
      --output_path "${FISHER_DIR}" \
      --dataset "${DATASET}" \
      --nsamples "${NSAMPLES}" \
      --seqlen "${SEQLEN}" \
      --cache_dir "${CACHE_ROOT}/tokens" \
      --batch_size "${FISHER_BATCH_SIZE}" \
      --layers_per_pass "${FISHER_LAYERS_PER_PASS}" \
      --accum_device "${FISHER_ACCUM_DEVICE}" \
      --empty_cache_interval "${FISHER_EMPTY_CACHE_INTERVAL}" \
      --gradient_checkpointing "${FISHER_GRADIENT_CHECKPOINTING}" \
      --attn_implementation "${ATTN_IMPLEMENTATION}"
  fi
  log "Building/resuming SqueezeLLM weighted k-means init LUT, dense-only, ${BIT}-bit"
  "${PYTHON_BIN}" quantization/nuq.py \
    --model_type "${MODEL_TYPE}" \
    --model "${CHUNK_DIR}" \
    --gradient "${FISHER_DIR}" \
    --bit "${BIT}" \
    --output_folder "${SQ_DIR}"
fi

if has_word "squeezellm" "${PPL_TARGETS}"; then
  if [[ "${PPL_BACKEND}" == "quant_cuda" ]]; then
    if [[ -s "${SQ_CKPT}" && "${FORCE_REPACK}" != "1" ]]; then
      log "Reusing packed SqueezeLLM checkpoint at ${SQ_CKPT}"
    else
      log "Packing SqueezeLLM init"
      "${PYTHON_BIN}" quantization/pack.py \
        --model "${MODEL}" \
        --model_type "${MODEL_TYPE}" \
        --wbits "${BIT}" \
        --folder "${SQ_DIR}" \
        --save "${SQ_CKPT}"
    fi
  fi
  run_ppl "squeezellm" "${SQ_CKPT}" "${SQ_DIR}"
else
  log "PPL_TARGETS=${PPL_TARGETS}; skipping SqueezeLLM PPL"
fi

if has_word "lnq_plain" "${PPL_TARGETS}" || { has_word "rbvt_squeeze" "${PPL_TARGETS}" && [[ "${RBVT_INPUT_LUT}" == "${LNQ_DIR}" ]]; }; then
  if stage_complete "${HESSIAN_DIR}" 'l*.pt' "${LAYER_COUNT}"; then
    log "LNQ Hessians complete in ${HESSIAN_DIR}; skipping"
  else
    log "Collecting plain LNQ Hessians"
    HESSIAN_RESOLVED_DEVICES="$(resolve_devices "${HESSIAN_GPU_DEVICES}" "${HESSIAN_MIN_FREE_MB}" "LNQ hessian")"
    if [[ "${HESSIAN_SAMPLE_PARALLEL}" == "1" ]]; then
      HESSIAN_RANGE="$(missing_layer_bounds "${LAYER_COUNT}" "${HESSIAN_DIR}" "l" ".pt")"
      if [[ -n "${HESSIAN_RANGE}" ]]; then
        HESSIAN_DEVICE="$(primary_device_from_devices "${HESSIAN_RESOLVED_DEVICES}")"
        log "LNQ hessian sample-parallel range ${HESSIAN_RANGE} on devices: ${HESSIAN_RESOLVED_DEVICES:-${HESSIAN_DEVICE}}"
        "${PYTHON_BIN}" quantization/lnq.py hessians \
          --model "${MODEL}" \
          --dataset "${DATASET}" \
          --nsamples "${NSAMPLES}" \
          --seqlen "${SEQLEN}" \
          --cache_dir "${CACHE_ROOT}/tokens" \
          --output_folder "${HESSIAN_DIR}" \
          --calib_batch_size "${HESSIAN_CALIB_BATCH_SIZE}" \
          --activation_storage "${HESSIAN_ACTIVATION_STORAGE}" \
          --activation_dtype "${ACTIVATION_DTYPE}" \
          --hessian_save_dtype "${HESSIAN_SAVE_DTYPE}" \
          --hessian_accum_device "${HESSIAN_ACCUM_DEVICE}" \
          --attn_implementation "${ATTN_IMPLEMENTATION}" \
          --device "${HESSIAN_DEVICE}" \
          --devices "${HESSIAN_RESOLVED_DEVICES}" \
          --layer_range "${HESSIAN_RANGE}"
      fi
    else
      run_missing_layer_shards_on_devices "LNQ hessian" "${LAYER_COUNT}" "${HESSIAN_RESOLVED_DEVICES}" "${HESSIAN_DIR}" "l" ".pt" \
        "${PYTHON_BIN}" quantization/lnq.py hessians \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --nsamples "${NSAMPLES}" \
        --seqlen "${SEQLEN}" \
        --cache_dir "${CACHE_ROOT}/tokens" \
        --output_folder "${HESSIAN_DIR}" \
        --calib_batch_size "${HESSIAN_CALIB_BATCH_SIZE}" \
        --activation_storage "${HESSIAN_ACTIVATION_STORAGE}" \
        --activation_dtype "${ACTIVATION_DTYPE}" \
        --hessian_save_dtype "${HESSIAN_SAVE_DTYPE}" \
        --hessian_accum_device "${HESSIAN_ACCUM_DEVICE}" \
        --attn_implementation "${ATTN_IMPLEMENTATION}"
    fi
  fi

  if stage_complete "${LNQ_DIR}/lut" 'l*.pkl' "${LAYER_COUNT}"; then
    log "LNQ plain LUT complete in ${LNQ_DIR}/lut; skipping"
  else
    log "Running/resuming LNQ plain on top of SqueezeLLM init"
    LNQ_RESOLVED_DEVICES="$(resolve_devices "${GPU_DEVICES}" "${GPU_MIN_FREE_MB}" "LNQ quantize")"
    LNQ_CPU_PER_SHARD="$(cpu_per_devices "${LNQ_RESOLVED_DEVICES}")"
    run_layer_shards_on_devices "LNQ quantize" "${LAYER_COUNT}" "${LNQ_RESOLVED_DEVICES}" \
      "${PYTHON_BIN}" quantization/lnq.py quantize \
      --model_chunks "${CHUNK_DIR}" \
      --hessians "${HESSIAN_DIR}" \
      --initial_lut "${SQ_DIR}" \
      --output_folder "${LNQ_DIR}" \
      --model_type "${MODEL_TYPE}" \
      --bit "${BIT}" \
      --num_iterations "${LNQ_ITERATIONS}" \
      --cd_cycles "${LNQ_CD_CYCLES}" \
      --row_block "${LNQ_ROW_BLOCK}" \
      --cpu_count "${LNQ_CPU_PER_SHARD}"
  fi

  if has_word "lnq_plain" "${PPL_TARGETS}" && [[ "${PPL_BACKEND}" == "quant_cuda" ]]; then
    if [[ -s "${LNQ_CKPT}" && "${FORCE_REPACK}" != "1" ]]; then
      log "Reusing packed LNQ plain checkpoint at ${LNQ_CKPT}"
    else
      log "Packing LNQ plain"
      "${PYTHON_BIN}" quantization/pack.py \
        --model "${MODEL}" \
        --model_type "${MODEL_TYPE}" \
        --wbits "${BIT}" \
        --folder "${LNQ_DIR}" \
        --save "${LNQ_CKPT}"
    fi
  fi
  if has_word "lnq_plain" "${PPL_TARGETS}"; then
    run_ppl "lnq_plain" "${LNQ_CKPT}" "${LNQ_DIR}"
  else
    log "PPL_TARGETS=${PPL_TARGETS}; skipping LNQ plain PPL"
  fi
else
  log "PPL_TARGETS=${PPL_TARGETS}; skipping LNQ/RBVT-dependent stages"
fi

if has_word "rbvt_squeeze" "${PPL_TARGETS}"; then
  log "Running dense-only RBVT-Squeeze on top of ${RBVT_INPUT_LUT}"
  RBVT_STATS_RESOLVED_DEVICES="$(resolve_devices "${GPU_DEVICES}" "${GPU_MIN_FREE_MB}" "RBVT stats")"
  RBVT_STATS_DEVICE="$(primary_device_from_devices "${RBVT_STATS_RESOLVED_DEVICES}")"
  log "Collecting/reusing RBVT activation stats on ${RBVT_STATS_DEVICE}"
  "${PYTHON_BIN}" quantization/rbvt_squeezellm.py stats \
    --model "${MODEL}" \
    --model_chunks "${CHUNK_DIR}" \
    --input_lut "${RBVT_INPUT_LUT}" \
    --output_folder "${RBVT_DIR}" \
    --model_type "${MODEL_TYPE}" \
    --dataset "${DATASET}" \
    --nsamples "${NSAMPLES}" \
    --seqlen "${SEQLEN}" \
    --cache_dir "${CACHE_ROOT}/tokens" \
    --stats_path "${RBVT_STATS}" \
    --n_calib "${RBVT_N_CALIB}" \
    --batch_size "${RBVT_BATCH_SIZE}" \
    --rbvt_lambda "${RBVT_LAMBDA}" \
    --rbvt_topk "${RBVT_TOPK}" \
    --row_chunk "${RBVT_ROW_CHUNK}" \
    --gap_floor "${RBVT_GAP_FLOOR}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}"

  if stage_complete "${RBVT_DIR}/lut" 'l*.pkl' "${LAYER_COUNT}"; then
    log "RBVT-Squeeze LUT complete in ${RBVT_DIR}/lut; skipping"
  else
    run_layer_shards "RBVT-Squeeze" "${LAYER_COUNT}" \
      "${PYTHON_BIN}" quantization/rbvt_squeezellm.py apply \
      --model "${MODEL}" \
      --model_chunks "${CHUNK_DIR}" \
      --input_lut "${RBVT_INPUT_LUT}" \
      --output_folder "${RBVT_DIR}" \
      --model_type "${MODEL_TYPE}" \
      --dataset "${DATASET}" \
      --nsamples "${NSAMPLES}" \
      --seqlen "${SEQLEN}" \
      --cache_dir "${CACHE_ROOT}/tokens" \
      --stats_path "${RBVT_STATS}" \
      --n_calib "${RBVT_N_CALIB}" \
      --batch_size "${RBVT_BATCH_SIZE}" \
      --rbvt_lambda "${RBVT_LAMBDA}" \
      --rbvt_topk "${RBVT_TOPK}" \
      --row_chunk "${RBVT_ROW_CHUNK}" \
      --gap_floor "${RBVT_GAP_FLOOR}" \
      --attn_implementation "${ATTN_IMPLEMENTATION}"
  fi

  if [[ "${PPL_BACKEND}" == "quant_cuda" ]]; then
    if [[ -s "${RBVT_CKPT}" && "${FORCE_REPACK}" != "1" ]]; then
      log "Reusing packed RBVT-Squeeze checkpoint at ${RBVT_CKPT}"
    else
      log "Packing RBVT-Squeeze"
      "${PYTHON_BIN}" quantization/pack.py \
        --model "${MODEL}" \
        --model_type "${MODEL_TYPE}" \
        --wbits "${BIT}" \
        --folder "${RBVT_DIR}" \
        --save "${RBVT_CKPT}"
    fi
  fi
  run_ppl "rbvt_squeeze" "${RBVT_CKPT}" "${RBVT_DIR}"
else
  log "PPL_TARGETS=${PPL_TARGETS}; skipping RBVT-Squeeze stages"
fi

log "Done. Results are under ${OUTPUT_ROOT}"
