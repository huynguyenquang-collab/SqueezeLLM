#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[qwen25-7b-3bit-c4] FAILED at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
export MODEL_LABEL="${MODEL_LABEL:-qwen25_7b}"
export MODEL_TYPE="${MODEL_TYPE:-qwen}"
export BIT="${BIT:-3}"

export DATASET="${DATASET:-c4}"
export NSAMPLES="${NSAMPLES:-128}"
export SEQLEN="${SEQLEN:-2048}"
export RBVT_N_CALIB="${RBVT_N_CALIB:-128}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/qwen25_7b_3bit_c4_sqllm_lnq_rbvt}"
export CACHE_ROOT="${CACHE_ROOT:-cache/qwen25_7b_3bit_c4_sqllm_lnq_rbvt}"

export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export GPU_DEVICES="${GPU_DEVICES:-auto}"
export HESSIAN_GPU_DEVICES="${HESSIAN_GPU_DEVICES:-${GPU_DEVICES}}"
export GPU_MAX_DEVICES="${GPU_MAX_DEVICES:-2}"
export GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-20000}"
export HESSIAN_MIN_FREE_MB="${HESSIAN_MIN_FREE_MB:-26000}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export FISHER_BATCH_SIZE="${FISHER_BATCH_SIZE:-1}"
export FISHER_LAYERS_PER_PASS="${FISHER_LAYERS_PER_PASS:-4}"
export FISHER_ACCUM_DEVICE="${FISHER_ACCUM_DEVICE:-cuda}"
export FISHER_EMPTY_CACHE_INTERVAL="${FISHER_EMPTY_CACHE_INTERVAL:-0}"
export FISHER_GRADIENT_CHECKPOINTING="${FISHER_GRADIENT_CHECKPOINTING:-off}"
export HESSIAN_CALIB_BATCH_SIZE="${HESSIAN_CALIB_BATCH_SIZE:-4}"
export HESSIAN_ACTIVATION_STORAGE="${HESSIAN_ACTIVATION_STORAGE:-ram}"
export ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-float16}"
export HESSIAN_SAVE_DTYPE="${HESSIAN_SAVE_DTYPE:-float16}"
export HESSIAN_ACCUM_DEVICE="${HESSIAN_ACCUM_DEVICE:-cuda}"

export LNQ_ITERATIONS="${LNQ_ITERATIONS:-3}"
export LNQ_CD_CYCLES="${LNQ_CD_CYCLES:-4}"
export LNQ_ROW_BLOCK="${LNQ_ROW_BLOCK:-64}"
export CPU_COUNT="${CPU_COUNT:-16}"

export RBVT_BATCH_SIZE="${RBVT_BATCH_SIZE:-4}"
export RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
export RBVT_TOPK="${RBVT_TOPK:-0}"
export RBVT_ROW_CHUNK="${RBVT_ROW_CHUNK:-4096}"
export RBVT_GAP_FLOOR="${RBVT_GAP_FLOOR:-1e-8}"

export PPL_TARGETS="${PPL_TARGETS:-squeezellm lnq_plain rbvt_squeeze}"
export PPL_DATASETS="${PPL_DATASETS:-wikitext2 c4}"
export PPL_BACKEND="${PPL_BACKEND:-dense_lut}"
export DENSE_EVAL_DTYPE="${DENSE_EVAL_DTYPE:-float16}"
export EVAL_PARALLEL_DATASETS="${EVAL_PARALLEL_DATASETS:-1}"
export EVAL_GPU_DEVICES="${EVAL_GPU_DEVICES:-${GPU_DEVICES}}"
export NONUQ_MAX_LENGTH="${NONUQ_MAX_LENGTH:-2048}"
export NONUQ_STRIDE="${NONUQ_STRIDE:-512}"
export NONUQ_C4_SAMPLES="${NONUQ_C4_SAMPLES:-2000}"

EVAL_STYLES="${EVAL_STYLES:-guidedquant nonuquantfix}"

for style in ${EVAL_STYLES}; do
  case "${style}" in
    guidedquant|repo)
      export PPL_EVAL_STYLE="guidedquant"
      export PPL_BATCH_SIZE="${GUIDEDQUANT_PPL_BATCH_SIZE:-${PPL_BATCH_SIZE:-4}}"
      export NONUQ_EXACT="${NONUQ_EXACT:-1}"
      ;;
    nonuquantfix|nonuq)
      export PPL_EVAL_STYLE="nonuquantfix"
      export PPL_BATCH_SIZE="${NONUQ_PPL_BATCH_SIZE:-1}"
      export NONUQ_EXACT="1"
      ;;
    *)
      echo "Unsupported eval style: ${style}; expected guidedquant or nonuquantfix" >&2
      exit 1
      ;;
  esac

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [qwen25-7b-3bit-c4] Running ${PPL_EVAL_STYLE} eval; targets=${PPL_TARGETS}; datasets=${PPL_DATASETS}"
  bash "${SCRIPT_DIR}/run_qwen25_7b_3bit_sqllm_lnq_rbvt_ppl.sh"
done
