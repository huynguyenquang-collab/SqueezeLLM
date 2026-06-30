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


def collect_fisher(args):
    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device(args.device)
    model = load_causal_lm(
        args.model,
        torch_dtype=torch.float16 if "cuda" in args.device else torch.float32,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
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
    for layer_idx in layer_ids:
        path = os.path.join(args.output_path, f"layer_{layer_idx}.pt")
        if os.path.exists(path) and not args.overwrite:
            print(f"Skipping existing Fisher chunk: {path}")
            continue

        layer = layers[layer_idx]
        modules = list(zip(module_names, get_modules(layer, model_type)))
        accum = {
            name: torch.zeros_like(module.weight, dtype=torch.float32, device="cpu")
            for name, module in modules
        }
        handles = []

        for module_name, module in modules:
            module.weight.requires_grad_(True)

            def make_hook(name):
                def hook(grad):
                    accum[name].add_(grad.detach().float().cpu().square())
                    return grad

                return hook

            handles.append(module.weight.register_hook(make_hook(module_name)))

        try:
            desc = f"Fisher layer {layer_idx}"
            for start in tqdm(range(0, limit, args.batch_size), desc=desc):
                batch = torch.cat(tokens[start : start + args.batch_size], dim=0).to(device)
                model.zero_grad(set_to_none=True)
                out = model(input_ids=batch, labels=batch, use_cache=False)
                out.loss.backward()
                for _, module in modules:
                    module.weight.grad = None
                del batch, out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            for handle in handles:
                handle.remove()
            for _, module in modules:
                module.weight.requires_grad_(False)
                module.weight.grad = None

        payload = {name: accum[name] for name in module_names}
        torch.save(payload, path)
        print(f"Saved Fisher chunk: {path}")
        del accum, payload, handles
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    parser.add_argument("--layer_range", default=None, help="Optional start,end layer shard.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--attn_implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"))
    args = parser.parse_args()
    collect_fisher(args)


if __name__ == "__main__":
    main()
