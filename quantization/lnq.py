import argparse
import gc
import json
import os
import pickle
import random
import shutil
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from squeezellm.model_parse import get_layers, get_module_names, get_modules, parse_model
from squeezellm.outliers import remove_outliers


def parse_range(value):
    if value is None:
        return None
    start, end = [int(x) for x in value.split(",")]
    return range(start, end)


def torch_dtype(name):
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def numpy_dtype(name):
    if name == "float32":
        return np.float32
    return np.float16


def load_causal_lm(model_name, torch_dtype, attn_implementation=None):
    kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if attn_implementation:
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        except (ImportError, TypeError, ValueError) as exc:
            print(f"Falling back from attn_implementation={attn_implementation}: {exc}", flush=True)
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def load_calibration_tokens(model_name, dataset, nsamples, seqlen, seed, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    explicit_cache = os.environ.get("CALIB_TOKENS_PATH", "")
    if explicit_cache and os.path.exists(explicit_cache):
        return normalize_token_cache(torch.load(explicit_cache, map_location="cpu"), seqlen)

    cache_file = os.path.join(
        cache_dir, f"{dataset}_{model_name.replace('/', '_')}_s{nsamples}_blk{seqlen}_seed{seed}.pt"
    )
    if os.path.exists(cache_file):
        return normalize_token_cache(torch.load(cache_file, map_location="cpu"), seqlen)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    rng = random.Random(seed)

    if dataset == "wikitext2":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(raw["text"])
        enc = tokenizer(text, return_tensors="pt").input_ids
        tokens = []
        for _ in range(nsamples):
            i = rng.randint(0, enc.shape[1] - seqlen - 1)
            tokens.append(enc[:, i : i + seqlen])
    elif dataset == "c4":
        raw = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        tokens = []
        for _ in range(nsamples):
            while True:
                sample = raw[rng.randint(0, len(raw) - 1)]["text"]
                enc = tokenizer(sample, return_tensors="pt").input_ids
                if enc.shape[1] >= seqlen:
                    break
            i = rng.randint(0, enc.shape[1] - seqlen - 1)
            tokens.append(enc[:, i : i + seqlen])
    elif dataset == "redpajama":
        raw = load_redpajama_source()
        tokens = sample_redpajama_tokens(raw, tokenizer, nsamples, seqlen, rng)
    else:
        raise ValueError(f"Unsupported calibration dataset: {dataset}")

    torch.save(tokens, cache_file)
    return tokens


def normalize_token_cache(tokens, seqlen):
    if isinstance(tokens, torch.Tensor):
        if tokens.ndim == 2:
            return [tokens[i : i + 1, :seqlen].long() for i in range(tokens.shape[0])]
        if tokens.ndim == 3 and tokens.shape[1] == 1:
            return [tokens[i, :, :seqlen].long() for i in range(tokens.shape[0])]
        raise ValueError(f"Unsupported token tensor shape: {tuple(tokens.shape)}")
    if isinstance(tokens, (list, tuple)):
        out = []
        for item in tokens:
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"Unsupported token item type: {type(item).__name__}")
            item = item.detach().cpu().long()
            if item.ndim == 1:
                item = item.unsqueeze(0)
            if item.ndim != 2 or item.shape[0] != 1:
                raise ValueError(f"Unsupported token item shape: {tuple(item.shape)}")
            out.append(item[:, :seqlen])
        return out
    raise TypeError(f"Unsupported token cache type: {type(tokens).__name__}")


def load_redpajama_source():
    dataset_name = os.environ.get("REDPAJAMA_DATASET", "ZengXiangyu/RedPajama-Data-1T-Sample")
    config = os.environ.get("REDPAJAMA_CONFIG", "")
    split = os.environ.get("REDPAJAMA_SPLIT", "train")
    streaming = os.environ.get("REDPAJAMA_STREAMING", "0") == "1"
    print(
        "Loading RedPajama calibration mirror: "
        f"dataset={dataset_name}, config={config or '<none>'}, split={split}, streaming={streaming}",
        flush=True,
    )
    args = [dataset_name]
    if config:
        args.append(config)
    return load_dataset(*args, split=split, streaming=streaming, trust_remote_code=True)


def sample_redpajama_tokens(raw, tokenizer, nsamples, seqlen, rng):
    tokens = []
    is_streaming = not hasattr(raw, "__len__")
    if is_streaming:
        for item in raw:
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt").input_ids
            if enc.shape[1] < seqlen:
                continue
            i = rng.randint(0, enc.shape[1] - seqlen)
            tokens.append(enc[:, i : i + seqlen])
            if len(tokens) >= nsamples:
                return tokens
    else:
        seen = set()
        while len(tokens) < nsamples:
            idx = rng.randint(0, len(raw) - 1)
            if idx in seen and len(seen) < len(raw):
                continue
            seen.add(idx)
            item = raw[idx]
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt").input_ids
            if enc.shape[1] < seqlen:
                continue
            i = rng.randint(0, enc.shape[1] - seqlen)
            tokens.append(enc[:, i : i + seqlen])
        return tokens

    raise RuntimeError(f"Only collected {len(tokens)} RedPajama samples; expected {nsamples}.")


class ActivationBuffer:
    def __init__(self, shape, dtype, storage, path=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.storage = storage
        self.path = path
        if storage == "ram":
            self.data = torch.empty(self.shape, dtype=dtype, device="cpu")
        else:
            np_dtype = np.float32 if dtype == torch.float32 else np.float16
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.data = np.memmap(path, dtype=np_dtype, mode="w+", shape=self.shape)

    def get(self, start, end):
        if self.storage == "ram":
            return self.data[start:end]
        arr = np.asarray(self.data[start:end])
        return torch.from_numpy(arr)

    def set(self, start, end, value):
        value = value.detach().cpu()
        if self.storage == "ram":
            self.data[start:end].copy_(value.to(self.dtype))
        else:
            self.data[start:end] = value.to(self.dtype).numpy()

    def flush(self):
        if self.storage == "disk":
            self.data.flush()


def save_hessians(hessians, path, save_dtype):
    dtype = torch_dtype(save_dtype)
    payload = {name: value.to(dtype=dtype, device="cpu") for name, value in hessians.items()}
    torch.save(payload, path)


def first_floating_dtype(module):
    for param in module.parameters(recurse=True):
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def move_forward_arg(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, tuple):
        return tuple(move_forward_arg(item, device) for item in value)
    if isinstance(value, list):
        return [move_forward_arg(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_forward_arg(item, device) for key, item in value.items()}
    return value


def register_linear_input_cast_hooks(module):
    handles = []

    def hook(_module, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        if torch.is_tensor(x) and x.dtype != _module.weight.dtype:
            return (x.to(dtype=_module.weight.dtype), *inputs[1:])
        return inputs

    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            handles.append(submodule.register_forward_pre_hook(hook))
    return handles


@torch.no_grad()
def capture_layer_inputs(model, tokens, device, args):
    model_type = parse_model(model)
    layers = get_layers(model, model_type)
    embeddings = model.model.embed_tokens if hasattr(model, "model") else None
    if embeddings is None:
        raise ValueError("LNQ activation capture currently expects HF causal models with model.embed_tokens.")

    embeddings = embeddings.to(device)
    layers[0] = layers[0].to(device)
    model_dtype = next(model.parameters()).dtype
    dtype = torch_dtype(args.activation_dtype) if args.activation_storage == "disk" else model_dtype
    hidden_size = model.config.hidden_size
    shape = (len(tokens), tokens[0].shape[-1], hidden_size)
    inps = ActivationBuffer(
        shape=shape,
        dtype=dtype,
        storage=args.activation_storage,
        path=os.path.join(args.activation_cache_dir, "activations_a.dat"),
    )
    cache = {"i": 0, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.set(cache["i"], cache["i"] + 1, inp)
            cache["i"] += 1
            cache["kwargs"] = {k: move_forward_arg(v, "cpu") for k, v in kwargs.items()}
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in tqdm(tokens, desc="Capturing first-layer inputs"):
        try:
            model(batch.to(device), use_cache=False)
        except ValueError:
            pass
    layers[0] = layers[0].module.cpu()
    embeddings.cpu()
    inps.flush()
    torch.cuda.empty_cache()
    return inps, cache["kwargs"] or {}


class HessianCollector:
    def __init__(self, names, modules, device, accum_device="cuda"):
        self.names = names
        self.device = device
        self.accum_device = device if accum_device == "cuda" and torch.device(device).type == "cuda" else torch.device("cpu")
        self.xtx = {}
        self.count = {}
        self.handles = []
        for name, module in zip(names, modules):
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(_module, inp, _out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            xtx = x.t().matmul(x)
            if name not in self.xtx:
                self.xtx[name] = torch.zeros_like(xtx, device=self.accum_device)
            self.xtx[name].add_(xtx.to(self.accum_device, non_blocking=True))
            self.count[name] = self.count.get(name, 0) + x.shape[0]

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def result(self):
        return {name: (self.xtx[name] / max(1, self.count[name])).cpu() for name in self.xtx}


def parse_devices(value, fallback):
    if not value:
        return [torch.device(fallback)]
    devices = []
    for item in value.replace(",", " ").split():
        if item.startswith("cuda") or item == "cpu":
            devices.append(torch.device(item))
        else:
            devices.append(torch.device(f"cuda:{item}"))
    return devices or [torch.device(fallback)]


def split_sample_ranges(nsamples, nchunks):
    nchunks = max(1, min(nchunks, nsamples))
    chunk = (nsamples + nchunks - 1) // nchunks
    ranges = []
    for idx in range(nchunks):
        start = idx * chunk
        end = min(start + chunk, nsamples)
        if start < end:
            ranges.append((start, end))
    return ranges


@torch.no_grad()
def forward_layer_chunk(
    layer,
    model_type,
    module_names,
    inps,
    outs,
    sample_start,
    sample_end,
    batch_size,
    device,
    forward_kwargs,
    collect_hessian,
    hessian_accum_device,
    desc,
):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    layer = layer.to(device)
    layer_dtype = first_floating_dtype(layer)
    dtype_handles = register_linear_input_cast_hooks(layer)
    current_batch_size = max(1, int(batch_size))

    try:
        while True:
            collector = None
            if collect_hessian:
                collector = HessianCollector(
                    module_names,
                    get_modules(layer, model_type),
                    device,
                    accum_device=hessian_accum_device,
                )

            try:
                with tqdm(total=sample_end - sample_start, desc=f"{desc} bs={current_batch_size}", leave=False) as pb:
                    start = sample_start
                    while start < sample_end:
                        end = min(start + current_batch_size, sample_end)
                        batch = inps.get(start, end).to(device=device, dtype=layer_dtype)
                        kwargs = {key: move_forward_arg(value, device) for key, value in forward_kwargs.items()}
                        out = layer(batch, **kwargs)[0]
                        outs.set(start, end, out)
                        pb.update(end - start)
                        start = end
                        del batch, kwargs, out
                if collector is None:
                    return {}
                return {
                    name: (collector.xtx[name], collector.count[name])
                    for name in collector.xtx
                }
            except torch.cuda.OutOfMemoryError:
                if collector is not None:
                    collector.close()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if current_batch_size <= 1:
                    raise
                next_batch_size = max(1, current_batch_size // 2)
                print(
                    f"{desc}: CUDA OOM with batch_size={current_batch_size}; "
                    f"retrying whole chunk with batch_size={next_batch_size}",
                    flush=True,
                )
                current_batch_size = next_batch_size
            finally:
                if collector is not None:
                    collector.close()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        for handle in dtype_handles:
            handle.remove()


def merge_hessian_states(states, module_names):
    merged = {}
    counts = {}
    for state in states:
        for name, (xtx, count) in state.items():
            if name not in merged:
                merged[name] = xtx.float()
                counts[name] = count
            else:
                merged[name].add_(xtx.float())
                counts[name] += count
    return {name: merged[name] / max(1, counts[name]) for name in module_names if name in merged}


@torch.no_grad()
def forward_layer(model_type, layer, layer_idx, inps, outs, forward_kwargs, collect_hessian, args, devices):
    module_names = get_module_names(model_type)
    active_ranges = split_sample_ranges(inps.shape[0], len(devices))
    active_devices = devices[: len(active_ranges)]
    if len(active_devices) == 1:
        states = [
            forward_layer_chunk(
                layer,
                model_type,
                module_names,
                inps,
                outs,
                active_ranges[0][0],
                active_ranges[0][1],
                args.calib_batch_size,
                active_devices[0],
                forward_kwargs,
                collect_hessian,
                args.hessian_accum_device,
                f"Layer {layer_idx} forward",
            )
        ]
        return merge_hessian_states(states, module_names) if collect_hessian else None

    from torch.nn.parallel import parallel_apply, replicate

    layer = layer.to(active_devices[0])
    replicas = replicate(layer, devices=active_devices, detach=True)
    replicas[0] = layer
    funcs = [forward_layer_chunk for _ in active_devices]
    inputs = []
    for replica, dev, (start, end) in zip(replicas, active_devices, active_ranges):
        inputs.append(
            (
                replica,
                model_type,
                module_names,
                inps,
                outs,
                start,
                end,
                args.calib_batch_size,
                dev,
                forward_kwargs,
                collect_hessian,
                args.hessian_accum_device,
                f"Layer {layer_idx} cuda:{dev.index} samples {start}:{end}",
            )
        )
    states = parallel_apply(funcs, inputs, devices=active_devices)
    del replicas
    return merge_hessian_states(states, module_names) if collect_hessian else None


@torch.no_grad()
def accumulate_hessians(model, tokens, output_folder, args):
    os.makedirs(output_folder, exist_ok=True)
    if args.activation_cache_dir is None:
        args.activation_cache_dir = os.path.join(output_folder, f"activation_cache_{os.getpid()}")
    if args.activation_storage == "disk":
        os.makedirs(args.activation_cache_dir, exist_ok=True)

    device = torch.device(args.device)
    devices = parse_devices(args.devices, args.device)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model_type = parse_model(model)
    layers = get_layers(model, model_type)
    selected_layers = sorted(parse_range(args.layer_range) or range(len(layers)))
    layer_range = set(selected_layers)
    last_layer_needed = selected_layers[-1] if selected_layers else -1

    try:
        inps, forward_kwargs = capture_layer_inputs(model, tokens, device, args)
        outs = ActivationBuffer(
            shape=inps.shape,
            dtype=inps.dtype,
            storage=args.activation_storage,
            path=os.path.join(args.activation_cache_dir, "activations_b.dat"),
        )

        for layer_idx, layer in enumerate(layers):
            if layer_idx > last_layer_needed:
                break
            out_file = os.path.join(output_folder, f"l{layer_idx}.pt")
            collect_hessian = layer_idx in layer_range and not os.path.exists(out_file)
            hess = forward_layer(model_type, layer, layer_idx, inps, outs, forward_kwargs, collect_hessian, args, devices)

            outs.flush()
            if hess is not None:
                save_hessians(hess, out_file, args.hessian_save_dtype)
                print(f"Saved Hessians to {out_file} ({args.hessian_save_dtype})")

            layers[layer_idx] = layer.cpu()
            del layer, hess
            inps, outs = outs, inps
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        if args.activation_storage == "disk" and not args.keep_activation_cache:
            shutil.rmtree(args.activation_cache_dir, ignore_errors=True)


def kmeans_fit(task):
    weights, n_cluster, random_state = task
    km = KMeans(n_clusters=n_cluster, random_state=random_state, n_init="auto", max_iter=50)
    km.fit(weights.reshape(-1, 1))
    return km.cluster_centers_.reshape(-1).astype(np.float32), km.labels_.astype(np.uint8)


def initial_lut(module_weight, bit, random_state, cpu_count):
    n_cluster = 2**bit
    rows = [module_weight[i].astype(np.float32) for i in range(module_weight.shape[0])]
    tasks = [(row, n_cluster, random_state) for row in rows]
    if cpu_count == 1:
        results = [kmeans_fit(task) for task in tqdm(tasks, desc="KMeans init")]
    else:
        with Pool(cpu_count) as pool:
            results = list(tqdm(pool.imap(kmeans_fit, tasks), total=len(tasks), desc="KMeans init"))
    centers = np.stack([item[0] for item in results], axis=0)
    labels = np.stack([item[1] for item in results], axis=0)
    return labels, centers


def load_squeezellm_init_layer(initial_lut, layer_idx, module_names):
    lut_file = os.path.join(initial_lut, "lut", f"l{layer_idx}.pkl")
    if not os.path.exists(lut_file):
        lut_file = os.path.join(initial_lut, f"l{layer_idx}.pkl")
    if not os.path.exists(lut_file):
        raise FileNotFoundError(
            f"Missing SqueezeLLM initialization LUT for layer {layer_idx}: "
            f"expected {initial_lut}/lut/l{layer_idx}.pkl or {initial_lut}/l{layer_idx}.pkl"
        )
    with open(lut_file, "rb") as handle:
        init_layer = pickle.load(handle)

    parsed = {}
    for name in module_names:
        rows = init_layer[name]
        centers = []
        labels = []
        for row in rows:
            row_centers, row_labels = row[0]
            centers.append(np.asarray(row_centers, dtype=np.float32))
            labels.append(np.asarray(row_labels, dtype=np.uint8))
        parsed[name] = (np.stack(labels, axis=0), np.stack(centers, axis=0))
    return parsed


@torch.no_grad()
def objective(W, H, labels, C, row_block):
    total = 0.0
    count = 0
    for start in range(0, W.shape[0], row_block):
        end = min(start + row_block, W.shape[0])
        row_ids = torch.arange(start, end, device=W.device)
        q = C[start:end].gather(1, labels[start:end].long())
        e = q - W[start:end]
        val = torch.einsum("bi,ij,bj->b", e, H, e).sum()
        total += val.item()
        count += end - start
    return total / max(1, count)


def regularize_hessian(H, percdamp=1e-5):
    H = torch.nan_to_num(H.float())
    H = 0.5 * (H + H.t())
    diag = torch.arange(H.shape[0], device=H.device)
    avg_diag = torch.diag(H).abs().mean().clamp_min(1e-8)
    prev = 0.0
    damp = percdamp
    while True:
        try:
            torch.linalg.cholesky(H)
            return H
        except Exception as exc:
            H[diag, diag] += (damp - prev) * avg_diag
            prev = damp
            damp *= 10
            if damp > 10:
                raise RuntimeError("Could not make LNQ Hessian positive definite with damping.") from exc


def damp_cholesky(H):
    H = regularize_hessian(H, percdamp=1e-6)
    return torch.linalg.cholesky(H)


@torch.no_grad()
def update_assignments(W, H, labels, C, cd_cycles, row_block):
    cols = W.shape[1]
    cd_block_size = 128

    labels = labels.long()
    Q = torch.gather(C.unsqueeze(1).expand(-1, cols, -1), 2, labels.unsqueeze(-1)).squeeze(-1)
    Hn = H / torch.diag(H).clamp_min(1e-12).reshape(1, -1)
    Hn = torch.tril(Hn, diagonal=-1)

    with tqdm(total=max(1, cd_cycles) * cols, desc="CD assignments") as pb:
        for _ in range(max(1, cd_cycles)):
            B = (Q - W).matmul(Hn)
            for block_start in range(0, cols, cd_block_size):
                block_end = min(block_start + cd_block_size, cols)
                for idx in range(block_start, block_end):
                    sol = W[:, idx : idx + 1] - B[:, idx : idx + 1]
                    new_labels = torch.abs(sol - C).argmin(dim=1).long()
                    labels[:, idx] = new_labels
                    Q[:, idx] = C.gather(1, new_labels.reshape(-1, 1)).reshape(-1)
                    if idx + 1 < block_end:
                        delta = Q[:, idx : idx + 1] - W[:, idx : idx + 1]
                        B[:, idx + 1 : block_end] += delta.matmul(Hn[idx : idx + 1, idx + 1 : block_end])
                    pb.update(1)
                if block_end < cols:
                    delta = Q[:, block_start:block_end] - W[:, block_start:block_end]
                    B[:, block_end:] += delta.matmul(Hn[block_start:block_end, block_end:])
    return labels


@torch.no_grad()
def update_centroids(W, H, labels, C, row_block):
    L = damp_cholesky(H)
    X = L.t()
    n_cluster = C.shape[1]
    new_C = torch.empty_like(C)
    eye = torch.eye(n_cluster, device=W.device, dtype=W.dtype) * 1e-4
    zeros = torch.zeros(n_cluster, 1, device=W.device, dtype=W.dtype)

    for start in tqdm(range(0, W.shape[0], row_block), desc="Centroids"):
        end = min(start + row_block, W.shape[0])
        labels_b = labels[start:end].long()
        one_hot = torch.nn.functional.one_hot(labels_b, num_classes=n_cluster).to(W.dtype)
        A = torch.einsum("si,bik->bsk", X, one_hot)
        b = X.matmul(W[start:end].t()).t().unsqueeze(-1)
        A = torch.cat([A, eye.unsqueeze(0).expand(end - start, -1, -1)], dim=1)
        b = torch.cat([b, zeros.unsqueeze(0).expand(end - start, -1, -1)], dim=1)
        new_C[start:end] = torch.linalg.lstsq(A, b).solution.squeeze(-1)
    return new_C


@torch.no_grad()
def lnq_module(weight, H, bit, num_iterations, cd_cycles, row_block, random_state, cpu_count, device):
    labels_np, centers_np = initial_lut(weight, bit, random_state, cpu_count)
    return lnq_module_from_init(
        weight,
        H,
        labels_np,
        centers_np,
        num_iterations,
        cd_cycles,
        row_block,
        device,
    )


@torch.no_grad()
def lnq_module_from_init(weight, H, labels_np, centers_np, num_iterations, cd_cycles, row_block, device):
    W = torch.from_numpy(weight.astype(np.float32)).to(device)
    H = regularize_hessian(H.float().to(device))
    labels = torch.from_numpy(labels_np).long().to(device)
    C = torch.from_numpy(centers_np).to(device)

    best_labels = labels.clone()
    best_C = C.clone()
    best_obj = objective(W, H, labels, C, row_block)
    log = {"objective": [best_obj]}
    print(f"Initial LNQ objective: {best_obj:.6f}")

    for iteration in range(num_iterations):
        start = time.time()
        if iteration > 0:
            labels = update_assignments(W, H, labels, C, cd_cycles, row_block)
            log["objective"].append(objective(W, H, labels, C, row_block))
        C = update_centroids(W, H, labels, C, row_block)
        current_obj = objective(W, H, labels, C, row_block)
        log["objective"].append(current_obj)
        print(f"Iteration {iteration + 1}: objective={current_obj:.6f} time={time.time() - start:.1f}s")
        if current_obj <= best_obj:
            best_obj = current_obj
            best_labels = labels.clone()
            best_C = C.clone()
        else:
            print("Objective did not improve; keeping previous best and stopping early.")
            break

    return best_C.cpu().numpy().astype(np.float32), best_labels.cpu().numpy().astype(np.uint8), log


def quantize_luts(args):
    os.makedirs(os.path.join(args.output_folder, "lut"), exist_ok=True)
    if args.outlier_config is not None:
        os.makedirs(os.path.join(args.output_folder, "outliers"), exist_ok=True)
        with open(args.outlier_config, "r") as handle:
            outlier_payload = json.load(handle)
        outlier_threshold = outlier_payload["outlier_threshold"]
        outlier_config = outlier_payload["outlier_config"]
    else:
        outlier_threshold = 0
        outlier_config = None

    model_type = args.model_type or "llama"
    layer_ids = sorted(
        int(name[len("layer_") : -len(".pt")])
        for name in os.listdir(args.model_chunks)
        if name.startswith("layer_") and name.endswith(".pt")
    )
    selected = set(parse_range(args.layer_range) or layer_ids)
    device = torch.device(args.device)

    for layer_idx in layer_ids:
        if layer_idx not in selected:
            continue
        lut_file = os.path.join(args.output_folder, "lut", f"l{layer_idx}.pkl")
        if os.path.exists(lut_file) and not args.overwrite:
            print(f"Skipping layer {layer_idx}; {lut_file} already exists.")
            continue

        print(f"LNQ quantizing layer {layer_idx}")
        model_layer = torch.load(os.path.join(args.model_chunks, f"layer_{layer_idx}.pt"), map_location="cpu")
        hessian_layer = torch.load(os.path.join(args.hessians, f"l{layer_idx}.pt"), map_location="cpu")
        module_names = get_module_names(model_type)
        init_layer = None
        if args.initial_lut:
            init_layer = load_squeezellm_init_layer(args.initial_lut, layer_idx, module_names)
        elif not args.kmeans_init:
            raise ValueError(
                "LNQ follows GuidedQuant/document.md experiments by initializing from SqueezeLLM assignments. "
                "Pass --initial_lut [SQUEEZELLM_LUT_PATH]. Use --kmeans_init only for debugging."
            )

        outliers = None
        if outlier_config is not None or args.sensitivity > 0:
            gradient_layer = None
            if args.sensitivity > 0:
                if args.gradient_chunks is None:
                    raise ValueError("--sensitivity requires --gradient_chunks, matching SqueezeLLM nuq.py")
                gradient_layer = torch.load(
                    os.path.join(args.gradient_chunks, f"layer_{layer_idx}.pt"),
                    map_location="cpu",
                )
            outliers = remove_outliers(
                model=model_layer,
                sensitivity=args.sensitivity,
                outlier_config=outlier_config[layer_idx],
                gradients=gradient_layer,
            )

        config = {}
        logs = {}
        for name in module_names:
            print(f"Layer {layer_idx} module {name}")
            weight = model_layer[name].float().numpy()
            if init_layer is None:
                centers, labels, log = lnq_module(
                    weight=weight,
                    H=hessian_layer[name],
                    bit=args.bit,
                    num_iterations=args.num_iterations,
                    cd_cycles=args.cd_cycles,
                    row_block=args.row_block,
                    random_state=args.seed,
                    cpu_count=args.cpu_count,
                    device=device,
                )
            else:
                init_labels, init_centers = init_layer[name]
                if init_centers.shape[1] != 2**args.bit:
                    raise ValueError(
                        f"Initial LUT for layer {layer_idx} module {name} has "
                        f"{init_centers.shape[1]} centers, expected {2**args.bit} for {args.bit}-bit."
                    )
                centers, labels, log = lnq_module_from_init(
                    weight=weight,
                    H=hessian_layer[name],
                    labels_np=init_labels,
                    centers_np=init_centers,
                    num_iterations=args.num_iterations,
                    cd_cycles=args.cd_cycles,
                    row_block=args.row_block,
                    device=device,
                )
            config[name] = [[(centers[i], labels[i])] for i in range(labels.shape[0])]
            logs[name] = log
            del weight, centers, labels
            if name in hessian_layer:
                del hessian_layer[name]
            if name in model_layer:
                del model_layer[name]
            if init_layer is not None and name in init_layer:
                del init_layer[name]
            gc.collect()
            torch.cuda.empty_cache()

        with open(lut_file, "wb") as handle:
            pickle.dump(config, handle)
        torch.save(logs, os.path.join(args.output_folder, "lut", f"log_l{layer_idx}.pt"))
        print(f"Saved layer LUT to {lut_file}")

        if outliers is not None:
            outlier_file = os.path.join(args.output_folder, "outliers", f"l{layer_idx}.pkl")
            with open(outlier_file, "wb") as handle:
                pickle.dump([x.to_sparse() for x in outliers[0]], handle)


def main():
    parser = argparse.ArgumentParser(description="Plain LNQ for SqueezeLLM LUTs (no GuidedQuant saliency).")
    sub = parser.add_subparsers(dest="mode", required=True)

    h = sub.add_parser("hessians", help="Collect ordinary LNQ Hessians X^T X.")
    h.add_argument("--model", required=True)
    h.add_argument("--dataset", default="redpajama", choices=["wikitext2", "c4", "redpajama"])
    h.add_argument("--nsamples", type=int, default=1024)
    h.add_argument("--seqlen", type=int, default=4096)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--cache_dir", default="cache/lnq_tokens")
    h.add_argument("--output_folder", required=True)
    h.add_argument("--device", default="cuda:0")
    h.add_argument("--devices", default=None, help="Optional comma/space separated CUDA devices for sample-parallel Hessian.")
    h.add_argument("--calib_batch_size", type=int, default=1)
    h.add_argument("--layer_range", default=None, help="start,end")
    h.add_argument("--activation_storage", default="disk", choices=["disk", "ram"])
    h.add_argument("--activation_cache_dir", default=None)
    h.add_argument("--activation_dtype", default="float16", choices=["float16", "float32"])
    h.add_argument("--keep_activation_cache", action="store_true")
    h.add_argument("--hessian_save_dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    h.add_argument("--hessian_accum_device", default="cuda", choices=["cuda", "cpu"])
    h.add_argument("--attn_implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"))

    q = sub.add_parser("quantize", help="Optimize LNQ assignments/codebooks and write SqueezeLLM LUTs.")
    q.add_argument("--model_chunks", required=True)
    q.add_argument("--hessians", required=True)
    q.add_argument("--initial_lut", default=None, help="SqueezeLLM LUT folder used to initialize LNQ.")
    q.add_argument("--output_folder", required=True)
    q.add_argument("--model_type", default="llama", choices=["llama", "mistral", "opt", "qwen"])
    q.add_argument("--bit", type=int, default=3, choices=[3, 4])
    q.add_argument("--num_iterations", type=int, default=3)
    q.add_argument("--cd_cycles", type=int, default=4)
    q.add_argument("--row_block", type=int, default=64)
    q.add_argument("--cpu_count", type=int, default=max(1, os.cpu_count() or 1))
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--layer_range", default=None, help="start,end")
    q.add_argument("--overwrite", action="store_true")
    q.add_argument("--kmeans_init", action="store_true", help="Debug fallback; paper experiments initialize from SqueezeLLM.")
    q.add_argument("--outlier_config", default=None)
    q.add_argument("--gradient_chunks", default=None)
    q.add_argument("--sensitivity", type=float, default=0)

    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.mode == "hessians":
        model = load_causal_lm(args.model, torch_dtype="auto", attn_implementation=args.attn_implementation)
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.eval()
        tokens = load_calibration_tokens(
            args.model, args.dataset, args.nsamples, args.seqlen, args.seed, args.cache_dir
        )
        accumulate_hessians(model, tokens, args.output_folder, args)
    else:
        quantize_luts(args)


if __name__ == "__main__":
    main()
