## From-scratch Quantization

Here we provide the codes for quantizing your custom models from scratch. Follow the steps outlined below.

**NOTE:** Only dense-only quantization is supported now. We are working to release the code for the Dense-and-Sparse quantization soon.

### 0. Prerequisite

In addition to installing the dependencies required for the inference code, you will need to install additional dependencies by running the following command:
```
conda activate sqllm
pip install scikit-learn==1.3.1
```
Additionally, make sure you have your own LLaMA Huggingface checkpoint saved at `[MODEL_PATH]`.


### 1. Compute gradients (Fisher-based sensitivity score)
SqueezeLLM employs the Fisher Information matrix as a sensitivity metric.
To compute this, we offer a separate [separate framework](https://github.com/kssteven418/SqueezeLLM-gradients) where you can compute the gradient square for your target model. 
This framework will produce the gradient square in the same format as the original Huggingface model checkpoint for your target model, with the only difference being that the weight values are replaced by the gradient square.

### 2. Chunk model weights and gradients
You should now have the model checkpoint at `[MODEL_PATH]` and the gradient checkpoint computed in the previous step at `[GRADIENT_PATH]`. 
Our framework requires that both checkpoints are chunked at the layer granularity to reduce the model loading overhead. 
Run the following code to chunk both your model and gradient checkpoints:
```
python chunk_models.py --model [MODEL_PATH] --output [MODEL_CHUNKS_PATH] --model_type llama
python chunk_models.py --model [GRADIENT_PATH] --output [GRADIENT_CHUNKS_PATH] --model_type llama
```

This will save model weights and gradients in the layer granularity as `[MODEL_CHUNKS_PATH]` and `[GRADIENT_CHUNKS_PATH]`.

### 3. (Optional for D+S quantization) Outlier configuration generation
This is an optional step to generate an outlier configuration for the Dense-and-Sparse quantization.
This step creates a configuration file that defines thresholds for identifying outlier values in the weights.
Note that while SqueezeLLM extracts both outliers and sensitive values to sparse matrices, 
there's no need for a separate configuration step for sensitive values -- this is covered in step 4.
This step only generates a configuration of outlier values.

Run the following command to generate an outlier configuration:
```
python generate_outlier_config.py --model [MODEL_CHUNKS_PATH] --range [RANGE] --output [OUTLIERS_CONFIG_PATH]
```

* `--model`: Points to the chunked model weights obtained in the previous step
* `--range`: Determines the thresholds T_min and T_max as described in Section 4.2 of the paper. Adjusting this value changes the absolute values of T_min and T_max, consequently affecting the total number of outliers. A larger range value decreases the number of outliers.
* `--output`: The path where the outlier configuration will be saved, formatted as `[OUTLIERS_CONFIG_PATH]/outlier_config_o{outlier_percentage}.json`, where outlier_percentage represents the percentage of values classified as outliers. You will need to fine-tune the `range` argument to achieve a desired outlier percentage, with 1.5-2.0 being a recommended starting range.



### 4. K-means clustering
Run the following code to perform K-means clustering, which will yield the non-uniform quantization look-up table (LUT):
```
python nuq.py --bit 4 --model_type llama --model [MODEL_CHUNKS_PATH] --gradient [GRADIENT_CHUNKS_PATH] --output [LUT_PATH]
```
The `--bit` argument is the bit-precision, and can be set to either 3 or 4. 
The `--model` and `--gradient` arguments should point to the chunked model weights and gradients obtained in the previous step. 
The resulting LUT entries will be stored in `[LUT_PATH]/lut`.

To only quantize a specific range of layers, you can use the `--range` option. For instance, assigning `--range 0,10` will only compute LUT entries for layers 0 to 9.

Please note that this process is highly CPU-intensive, so it is recommended to run the code in environments with multiple and stronger CPU cores for faster computation.

**Additional arguments for D+S quantization:**
To perform Dense-and-Sparse quantization, e.g. with 0.45% outliers and 0.05% sensitive values as in the paper, you will first need to generate an outlier configuration file for 0.45% outliers by completing step 3. This file will be stored as `[OUTLIERS_CONFIG_PATH]/outlier_config_o0.45.json`.
Then, run the following command:
```
python nuq.py --bit 4 --model_type llama --model [MODEL_CHUNKS_PATH] --gradient [GRADIENT_CHUNKS_PATH] --output [LUT_PATH] --outlier_config [OUTLIERS_CONFIG_PATH]/outlier_config_o0.45.json --sensitivity 0.05
```
* `--outlier_config`: reads the outlier configuration file generated in the previous step to extract 0.45% of the outliers to the sparse matrices accordingly.
* `--sensitivity`: takes out an additional 0.05% of sensitive values to the sparse matrices. 



### 4. Packing
Finally, use the obtained LUT from the previous step to save your model into a packed format. Run the following command:
```
python pack.py --model [MODEL_PATH] --wbits 4 --folder [LUT_PATH] --save [PACKED_CKPT_PATH]
```
`[MODEL_PATH]` is the original model checkpoint, and `[LUT_PATH]` is the location where the LUT is stored from the previous step. 
The packed checkpoint will be saved at `[PACKED_CKPT_PATH]`, which can now be immediately used in your inference code.

**Additional arguments for D+S packing:**
When you performed D+S quantization in the previous step, you will obtain the outliers file as well as LUT in `[LUT_PATH]`.
In this case, you will need to  proceed with D+S packing using the following command:
```
python pack.py --model [MODEL_PATH] --wbits 4 --folder [LUT_PATH] --save [PACKED_CKPT_PATH] --include_sparse --balance
```
This command incorporates two additional arguments, `--include_sparse` and `--balance`.

## Plain LNQ, Without GuidedQuant

This repository also includes a small plain-LNQ adapter in `quantization/lnq.py`.
It follows the LNQ objective described in `document.md`: optimize a
non-uniform scalar LUT against the layer-wise output error
`||XW - XW_hat||`, using the ordinary activation Hessian `X^T X`.

Important: plain LNQ is initialized from SqueezeLLM assignments/codebooks, as in
the GuidedQuant experiments. LNQ then refines that feasible SqueezeLLM solution
with alternating closed-form codebook updates and coordinate-descent assignment
updates. It is not GuidedQuant: the Hessian is ordinary `X^T X`, not the
saliency-weighted/end-loss-guided Hessian. The output is still the standard
SqueezeLLM LUT format, so the existing `pack.py` and CUDA inference path can
stay unchanged.

### Plain LNQ steps

1. Chunk model weights:
```
python quantization/chunk_models.py \
  --model [MODEL_PATH] \
  --model_type llama \
  --output_path [MODEL_CHUNKS_PATH]
```

2. Accumulate ordinary LNQ Hessians:
```
python quantization/lnq.py hessians \
  --model [MODEL_PATH] \
  --dataset redpajama \
  --nsamples 1024 \
  --seqlen 4096 \
  --output_folder [HESSIAN_PATH] \
  --device cuda:0 \
  --calib_batch_size 1 \
  --activation_storage disk \
  --activation_dtype float16 \
  --hessian_save_dtype float16
```

3. Optimize LNQ LUTs:
```
python quantization/lnq.py quantize \
  --model_chunks [MODEL_CHUNKS_PATH] \
  --hessians [HESSIAN_PATH] \
  --initial_lut [SQUEEZELLM_LUT_PATH] \
  --output_folder [LUT_PATH] \
  --model_type llama \
  --bit 3 \
  --num_iterations 3 \
  --cd_cycles 4 \
  --row_block 64 \
  --device cuda:0
```

4. Pack and evaluate with the existing SqueezeLLM checkpoint format:
```
python quantization/pack.py \
  --model [MODEL_PATH] \
  --wbits 3 \
  --folder [LUT_PATH] \
  --save [PACKED_CKPT_PATH]

python quantization/eval_nonuquantfix_ppl.py \
  --model [MODEL_PATH] \
  --checkpoint [PACKED_CKPT_PATH] \
  --wbits 3 \
  --device cuda:0
```

For A100 40GB runs, keep the paper calibration setting
`--dataset redpajama --nsamples 1024 --seqlen 4096`. Use
`--calib_batch_size 1`, `--row_block 32` or `--row_block 64`, and keep
`--activation_storage disk` to avoid holding the full 1024x4096 hidden-state
cache in host RAM. The disk activation cache is temporary and is removed after
each Hessian pass unless `--keep_activation_cache` is set. The convenience
script defaults to the paper calibration setting and can split Hessian/LNQ work
across two A100s.

For minimum disk usage, keep `HESSIAN_SAVE_DTYPE=float16` in the script
environment. For maximum numerical fidelity, use `HESSIAN_SAVE_DTYPE=float32`;
the sample count and sequence length remain the paper values in both cases.

The convenience script below runs Llama-2-7B and Llama-3-8B, then evaluates
perplexity with a NonUQuantFix-style sliding window:
```
bash bash/run_lnq_plain_llama_ppl.sh
```

Single A100:
```
DEVICE=cuda:0 bash bash/run_lnq_plain_llama_ppl.sh
```

Two A100s:
```
GPU_DEVICES="0 1" bash bash/run_lnq_plain_llama_ppl.sh
```

Because LNQ needs SqueezeLLM initialization, provide one of these:
```
INIT_LUT_SPECS="llama2_7b=/path/to/llama2_sqllm_init;llama3_8b=/path/to/llama3_sqllm_init" \
bash bash/run_lnq_plain_llama_ppl.sh
```
or provide gradient chunks/checkpoints so the script can run `nuq.py` first:
```
GRADIENT_CHUNKS_SPECS="llama2_7b=/path/to/llama2_grad_chunks;llama3_8b=/path/to/llama3_grad_chunks" \
bash bash/run_lnq_plain_llama_ppl.sh
```

### Qwen2.5-7B 3-bit LNQ/RBVT-Squeeze Job

The Qwen job runs the full dense-only chain:
SqueezeLLM weighted k-means init LUT -> LNQ plain initialized from SqueezeLLM ->
RBVT-Squeeze initialized from LNQ -> NonUQuantFix-style PPL for LNQ and
RBVT-Squeeze. It does not pack/evaluate a separate SqueezeLLM baseline.

```
bash bash/run_qwen25_7b_3bit_sqllm_lnq_rbvt_ppl.sh
```

Defaults:
```
MODEL=Qwen/Qwen2.5-7B
BIT=3
DATASET=redpajama
REDPAJAMA_DATASET=ZengXiangyu/RedPajama-Data-1T-Sample
GPU_DEVICES="0 1"
NSAMPLES=1024
SEQLEN=4096
```

For A100 40GB keep batch sizes at 1:
```
FISHER_BATCH_SIZE=1 CALIB_BATCH_SIZE=1 RBVT_BATCH_SIZE=1 \
bash bash/run_qwen25_7b_3bit_sqllm_lnq_rbvt_ppl.sh
```

The job uses both GPUs whenever the stage can be split by layer:
Fisher collection for SqueezeLLM init, LNQ Hessian collection, LNQ
assignment/codebook optimization, and RBVT-Squeeze assignment correction are
sharded across `GPU_DEVICES`. Packing and PPL evaluation remain single-process
and use `DEVICE` (default `cuda:0`).
