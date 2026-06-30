import argparse
import gc
import json
import pickle
import random
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoTokenizer

from llama import load_quant


def load_tokenizer(model, token=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            use_fast=True,
            token=token,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_wikitext2(cache_dir, seed, token=None):
    cache_file = cache_dir / f"wikitext2_test_seed{seed}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as handle:
            return pickle.load(handle)
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        token=token,
    )
    texts = ["\n".join([text for text in dataset["text"] if text])]
    with open(cache_file, "wb") as handle:
        pickle.dump(texts, handle)
    return texts


def load_c4(cache_dir, seed, n_samples, token=None):
    cache_file = cache_dir / f"c4_validation_n{n_samples}_seed{seed}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as handle:
            return pickle.load(handle)
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        streaming=True,
        token=token,
    )
    texts = []
    for item in tqdm(dataset, total=n_samples, desc="Collecting C4"):
        if len(texts) >= n_samples:
            break
        text = item["text"].strip()
        if len(text) > 500:
            texts.append(text)
    result = ["\n\n".join(texts)]
    with open(cache_file, "wb") as handle:
        pickle.dump(result, handle)
    return result


def torch_load_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_guidedquant_input_tokens(dataset_name, tokenizer, chunk_size, cache_dir, token=None):
    cache_file = cache_dir / f"guidedquant_{dataset_name}_ctx{chunk_size}.pt"
    if cache_file.exists():
        return torch_load_cache(cache_file)

    if dataset_name == "wikitext2":
        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
            token=token,
        )
        input_tokens = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    elif dataset_name == "c4":
        valdata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            token=token,
        )
        random.seed(0)
        chunks = []
        for _ in tqdm(range(256), desc="GuidedQuant C4 chunks"):
            while True:
                idx = random.randint(0, len(valdata) - 1)
                enc = tokenizer(valdata[idx]["text"], return_tensors="pt").input_ids
                if enc.shape[1] >= chunk_size:
                    break
            if enc.shape[1] == chunk_size:
                chunks.append(enc)
            else:
                start = random.randint(0, enc.shape[1] - chunk_size - 1)
                chunks.append(enc[:, start : start + chunk_size])
        ids = torch.hstack(chunks)
        input_tokens = BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    else:
        raise ValueError(f"Unsupported GuidedQuant PPL dataset: {dataset_name}")

    torch.save(input_tokens, cache_file)
    return input_tokens


def register_linear_input_cast_hooks(module):
    handles = []

    def hook(linear, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        if torch.is_tensor(x) and x.dtype != linear.weight.dtype:
            return (x.to(dtype=linear.weight.dtype), *inputs[1:])
        return inputs

    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            handles.append(submodule.register_forward_pre_hook(hook))
    return handles


@torch.inference_mode()
def evaluate_guidedquant(model, tokenizer, dataset_name, device, chunk_size, batch_size, cache_dir, token=None):
    model.eval()
    input_tokens = load_guidedquant_input_tokens(dataset_name, tokenizer, chunk_size, cache_dir, token)
    input_ids = input_tokens.input_ids.to(device)
    seq_len = input_ids.shape[1]
    nsamples = seq_len // chunk_size
    if nsamples < 1:
        raise RuntimeError(f"No full GuidedQuant chunks for {dataset_name}.")

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_chunks = 0
    for start in tqdm(range(0, nsamples, batch_size), desc=f"GuidedQuant {dataset_name}", leave=False):
        end = min(start + batch_size, nsamples)
        chunks = [
            input_ids[:, idx * chunk_size : (idx + 1) * chunk_size]
            for idx in range(start, end)
        ]
        batch = torch.cat(chunks, dim=0)
        if "gemma" in model.config.architectures[0].lower():
            batch[:, 0] = tokenizer.bos_token_id
        out = model(batch, labels=batch)
        total_loss += out.loss.float() * (end - start)
        total_chunks += end - start

    return {
        "perplexity": torch.exp(total_loss / total_chunks).item(),
        "total_tokens": int(total_chunks * chunk_size),
        "chunks": int(total_chunks),
    }


@torch.inference_mode()
def evaluate(model, tokenizer, texts, device, max_length, stride, batch_size, limit_tokens=None):
    model.eval()
    nlls = []
    total_tokens = 0

    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        if tokenizer.bos_token_id is not None:
            if input_ids.shape[1] == 0 or input_ids[0, 0].item() != tokenizer.bos_token_id:
                bos = torch.tensor([[tokenizer.bos_token_id]], device=input_ids.device)
                input_ids = torch.cat([bos, input_ids], dim=1)
        if limit_tokens and input_ids.shape[1] > limit_tokens:
            input_ids = input_ids[:, :limit_tokens]

        input_ids = input_ids.to(device)
        seq_len = input_ids.shape[1]
        if seq_len < 2:
            continue

        windows = []
        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            trg_len = end - prev_end
            windows.append((begin, end, trg_len))
            prev_end = end
            if end == seq_len:
                break

        for start in tqdm(range(0, len(windows), batch_size), desc=f"Windows ({seq_len:,} toks)", leave=False):
            batch_windows = windows[start : start + batch_size]
            batch_len = max(end - begin for begin, end, _ in batch_windows)
            chunks = []
            targets = []
            valid_tokens = 0
            for begin, end, trg_len in batch_windows:
                chunk = input_ids[:, begin:end]
                target = chunk.clone()
                if begin > 0:
                    target[:, :-trg_len] = -100
                if chunk.shape[1] < batch_len:
                    pad_len = batch_len - chunk.shape[1]
                    pad_ids = torch.full(
                        (1, pad_len),
                        tokenizer.pad_token_id,
                        dtype=chunk.dtype,
                        device=device,
                    )
                    chunk = torch.cat([chunk, pad_ids], dim=1)
                    target = torch.cat([target, torch.full_like(pad_ids, -100)], dim=1)
                chunks.append(chunk)
                targets.append(target)
                valid_tokens += trg_len
            out = model(torch.cat(chunks, dim=0), labels=torch.cat(targets, dim=0))
            nlls.append(out.loss.float() * valid_tokens)
        total_tokens += seq_len

    if not nlls:
        raise RuntimeError("No tokens were evaluated.")
    total_nll = torch.stack(nlls).sum()
    return {
        "perplexity": torch.exp(total_nll / total_tokens).item(),
        "total_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="NonUQuantFix-style sliding-window PPL for SqueezeLLM checkpoints.")
    parser.add_argument("--model", required=True, help="Base HF model/tokenizer.")
    parser.add_argument("--checkpoint", required=True, help="Packed SqueezeLLM checkpoint.")
    parser.add_argument("--wbits", type=int, required=True, choices=[3, 4])
    parser.add_argument("--datasets", nargs="+", default=["wikitext2", "c4"], choices=["wikitext2", "c4"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include_sparse", action="store_true")
    parser.add_argument("--num_dense_channels", type=int, default=10)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--c4_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default="cache/nonuquantfix_ppl")
    parser.add_argument("--limit_tokens", type=int, default=0)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--output_file", default="")
    parser.add_argument("--eval_style", default="nonuquantfix", choices=["nonuquantfix", "guidedquant"])
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model, args.hf_token)
    model = load_quant(
        args.model,
        args.checkpoint,
        args.wbits,
        args.include_sparse,
        args.num_dense_channels,
    ).to(args.device)

    results = {}
    dtype_hooks = register_linear_input_cast_hooks(model)
    try:
        for dataset in args.datasets:
            if args.eval_style == "guidedquant":
                results[dataset] = evaluate_guidedquant(
                    model,
                    tokenizer,
                    dataset,
                    args.device,
                    args.max_length,
                    args.batch_size,
                    cache_dir,
                    args.hf_token,
                )
            else:
                if dataset == "wikitext2":
                    texts = load_wikitext2(cache_dir, args.seed, args.hf_token)
                else:
                    texts = load_c4(cache_dir, args.seed, args.c4_samples, args.hf_token)
                print(f"Evaluating {args.checkpoint} on {dataset}")
                results[dataset] = evaluate(
                    model,
                    tokenizer,
                    texts,
                    args.device,
                    args.max_length,
                    args.stride,
                    args.batch_size,
                    args.limit_tokens or None,
                )
            print(
                f"{dataset}: ppl={results[dataset]['perplexity']:.6f} "
                f"tokens={results[dataset]['total_tokens']:,}"
            )
    finally:
        for handle in dtype_hooks:
            handle.remove()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    payload = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "wbits": args.wbits,
        "stride": args.stride,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "eval_style": args.eval_style,
        "results": results,
    }
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
