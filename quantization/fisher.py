import argparse
import gc
import os

import torch
from tqdm import tqdm

from quantization.lnq import load_calibration_tokens, load_causal_lm
from squeezellm.model_parse import get_layers, get_module_names, get_modules, parse_model


def parse_layer_range(value, layer_count):
    if value is None:
        return range(layer_count)
    start, end = [int(x) for x in value.split(",")]
    return range(start, min(end, layer_count))


def layer_groups(layer_ids, layers_per_pass):
    if layers_per_pass <= 0:
        return [list(layer_ids)]
    return [layer_ids[i : i + layers_per_pass] for i in range(0, len(layer_ids), layers_per_pass)]


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cuda_oom(error):
    message = str(error).lower()
    return isinstance(error, torch.cuda.OutOfMemoryError) or "cuda out of memory" in message


def maybe_empty_cache(args, batch_idx):
    if args.empty_cache_interval > 0 and batch_idx % args.empty_cache_interval == 0:
        cleanup_cuda()


def collect_fisher_group(model, layers, model_type, module_names, layer_ids, tokens, limit, args, device):
    pending_layer_ids = []
    for layer_idx in layer_ids:
        path = os.path.join(args.output_path, f"layer_{layer_idx}.pt")
        if os.path.exists(path) and not args.overwrite:
            print(f"Skipping existing Fisher chunk: {path}")
            continue
        pending_layer_ids.append(layer_idx)

    if not pending_layer_ids:
        return

    accum_device = args.accum_device
    if accum_device == "cuda" and "cuda" not in args.device:
        accum_device = "cpu"

    accum = {}
    modules_by_layer = {}
    selected_modules = []
    handles = []

    try:
        for layer_idx in pending_layer_ids:
            modules = list(zip(module_names, get_modules(layers[layer_idx], model_type)))
            modules_by_layer[layer_idx] = modules
            for module_name, module in modules:
                module.weight.requires_grad_(True)
                module.weight.grad = None
                selected_modules.append(module)

                if accum_device == "cpu":
                    accum.setdefault(layer_idx, {})[module_name] = torch.zeros_like(
                        module.weight, dtype=torch.float32, device="cpu"
                    )

                    def make_cpu_hook(target_layer_idx, target_name):
                        def hook(grad):
                            accum[target_layer_idx][target_name].add_(grad.detach().float().cpu().square())
                            return torch.zeros_like(grad)

                        return hook

                    handles.append(module.weight.register_hook(make_cpu_hook(layer_idx, module_name)))
                else:

                    def square_grad_hook(grad):
                        return grad.detach().pow(2)

                    handles.append(module.weight.register_hook(square_grad_hook))

        group_label = (
            f"{pending_layer_ids[0]}"
            if len(pending_layer_ids) == 1
            else f"{pending_layer_ids[0]}-{pending_layer_ids[-1]}"
        )
        desc = f"Fisher layers {group_label} ({accum_device})"
        model.zero_grad(set_to_none=True)
        batch_starts = range(0, limit, args.batch_size)
        for batch_idx, start in enumerate(tqdm(batch_starts, desc=desc), start=1):
            batch = torch.cat(tokens[start : start + args.batch_size], dim=0).to(device)
            if accum_device == "cpu":
                model.zero_grad(set_to_none=True)
            out = model(input_ids=batch, labels=batch, use_cache=False)
            out.loss.backward()
            if accum_device == "cpu":
                for module in selected_modules:
                    module.weight.grad = None
            del batch, out
            maybe_empty_cache(args, batch_idx)

        for layer_idx in pending_layer_ids:
            payload = {}
            for module_name, module in modules_by_layer[layer_idx]:
                if accum_device == "cpu":
                    payload[module_name] = accum[layer_idx][module_name]
                else:
                    if module.weight.grad is None:
                        raise RuntimeError(f"No Fisher gradient collected for layer {layer_idx} module {module_name}")
                    payload[module_name] = module.weight.grad.detach().float().cpu()
            path = os.path.join(args.output_path, f"layer_{layer_idx}.pt")
            torch.save(payload, path)
            print(f"Saved Fisher chunk: {path}")
            del payload
    finally:
        for handle in handles:
            handle.remove()
        for module in selected_modules:
            module.weight.requires_grad_(False)
            module.weight.grad = None
        del handles, selected_modules, modules_by_layer, accum
        cleanup_cuda()


def collect_fisher_group_with_retry(model, layers, model_type, module_names, layer_ids, tokens, limit, args, device):
    try:
        collect_fisher_group(model, layers, model_type, module_names, layer_ids, tokens, limit, args, device)
    except RuntimeError as error:
        if not is_cuda_oom(error) or len(layer_ids) <= 1:
            raise
        cleanup_cuda()
        midpoint = len(layer_ids) // 2
        left = layer_ids[:midpoint]
        right = layer_ids[midpoint:]
        print(
            "CUDA OOM while collecting Fisher for layers "
            f"{layer_ids[0]}-{layer_ids[-1]}; retrying as {left[0]}-{left[-1]} and {right[0]}-{right[-1]}"
        )
        collect_fisher_group_with_retry(model, layers, model_type, module_names, left, tokens, limit, args, device)
        collect_fisher_group_with_retry(model, layers, model_type, module_names, right, tokens, limit, args, device)


def collect_fisher(args):
    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device(args.device)
    model = load_causal_lm(
        args.model,
        torch_dtype=torch.float16 if "cuda" in args.device else torch.float32,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing == "on" and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.train()
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.eval()
    model.to(device)

    model_type = parse_model(model)
    layers = get_layers(model, model_type)
    module_names = get_module_names(model_type)
    layer_ids = list(parse_layer_range(args.layer_range, len(layers)))
    tokens = load_calibration_tokens(
        args.model,
        args.dataset,
        args.nsamples,
        args.seqlen,
        args.seed,
        args.cache_dir,
    )

    for param in model.parameters():
        param.requires_grad_(False)

    limit = min(args.nsamples, len(tokens))
    for group in layer_groups(layer_ids, args.layers_per_pass):
        collect_fisher_group_with_retry(model, layers, model_type, module_names, group, tokens, limit, args, device)

    model.cpu()
    del model
    cleanup_cuda()


def main():
    parser = argparse.ArgumentParser(description="Collect SqueezeLLM Fisher gradient-square chunks.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dataset", default="redpajama", choices=["redpajama", "wikitext2", "c4"])
    parser.add_argument("--nsamples", type=int, default=1024)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", default="cache/lnq_plain/tokens")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--layers_per_pass",
        type=int,
        default=1,
        help="Collect Fisher for this many layers per full calibration pass. Use 0 for all layers in the shard.",
    )
    parser.add_argument(
        "--accum_device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Accumulate squared gradients on CUDA for speed or CPU for lower VRAM.",
    )
    parser.add_argument(
        "--empty_cache_interval",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache() every N calibration batches; 0 disables per-batch cache clears.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default="on",
        choices=["on", "off"],
        help="Use HF gradient checkpointing during Fisher collection. 'off' is faster but needs more VRAM.",
    )
    parser.add_argument("--layer_range", default=None, help="Optional start,end layer shard.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--attn_implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"))
    args = parser.parse_args()
    collect_fisher(args)


if __name__ == "__main__":
    main()
