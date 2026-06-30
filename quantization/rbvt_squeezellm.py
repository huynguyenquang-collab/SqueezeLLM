import argparse
import gc
import os
import pickle
from dataclasses import dataclass

import torch
from tqdm import tqdm

from quantization.lnq import load_calibration_tokens, load_causal_lm
from squeezellm.model_parse import get_layers, get_module_names, get_modules, parse_model


@dataclass
class RBVTStats:
    flips: int = 0
    channels: int = 0
    candidates: int = 0
    boundary_kept: int = 0
    bias_before: float = 0.0
    bias_after: float = 0.0
    objective_before: float = 0.0
    objective_after: float = 0.0
    variance_increase: float = 0.0

    def add(self, other):
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))


class ActivationStatsCollector:
    def __init__(self, module_map, want_var):
        self.module_map = module_map
        self.want_var = want_var
        self.sum = {}
        self.sumsq = {}
        self.count = {}
        self.handles = []

    def _hook(self, name):
        def hook(_module, inp, _out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            s = x.sum(dim=0).cpu()
            if name not in self.sum:
                self.sum[name] = s
                self.count[name] = x.shape[0]
                if self.want_var:
                    self.sumsq[name] = (x * x).sum(dim=0).cpu()
            else:
                self.sum[name] += s
                self.count[name] += x.shape[0]
                if self.want_var:
                    self.sumsq[name] += (x * x).sum(dim=0).cpu()

        return hook

    def register(self):
        for name, module in self.module_map.items():
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def results(self):
        means = {}
        variances = {}
        for name, total in self.sum.items():
            count = max(1, self.count[name])
            mean = total / count
            means[name] = mean
            if self.want_var:
                ex2 = self.sumsq[name] / count
                variances[name] = (ex2 - mean * mean).clamp(min=0.0)
        return means, variances


def build_module_map(model, model_type):
    layers = get_layers(model, model_type)
    module_names = get_module_names(model_type)
    module_map = {}
    for layer_idx, layer in enumerate(layers):
        for short_name, module in zip(module_names, get_modules(layer, model_type)):
            module_map[f"{layer_idx}:{short_name}"] = module
    return module_map


def collect_activation_stats(args):
    if args.stats_path and os.path.exists(args.stats_path) and not args.overwrite_stats:
        payload = torch.load(args.stats_path, map_location="cpu")
        return payload["means"], payload["variances"]

    device = torch.device(args.device)
    model = load_causal_lm(
        args.model,
        torch_dtype=torch.float16 if "cuda" in args.device else torch.float32,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.eval().to(device)
    model_type = parse_model(model)
    collector = ActivationStatsCollector(build_module_map(model, model_type), args.rbvt_lambda > 0.0)
    tokens = load_calibration_tokens(
        args.model,
        args.dataset,
        args.nsamples,
        args.seqlen,
        args.seed,
        args.cache_dir,
    )
    collector.register()
    try:
        limit = min(args.n_calib, len(tokens))
        for start in tqdm(range(0, limit, args.batch_size), desc="Collecting RBVT activation stats"):
            batch = torch.cat(tokens[start : start + args.batch_size], dim=0).to(device)
            with torch.inference_mode():
                model(input_ids=batch, use_cache=False)
            del batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        collector.remove()

    means, variances = collector.results()
    if args.stats_path:
        os.makedirs(os.path.dirname(args.stats_path), exist_ok=True)
        torch.save({"means": means, "variances": variances}, args.stats_path)
    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return means, variances


def dequantize(labels, centers):
    row_ids = torch.arange(labels.shape[0], device=labels.device).view(-1, 1)
    return centers[row_ids, labels.long()]


@torch.no_grad()
def apply_rbvt_indices(W_fp, labels, centers, mu, sigma_ii, args):
    device = W_fp.device
    out_features, _ = W_fp.shape
    levels = centers.to(device).float()
    idx_full = labels.to(device).long().clone()
    Wq_full = dequantize(idx_full, levels).float()
    mu = mu.to(device).float()
    sigma_ii = torch.zeros_like(mu) if sigma_ii is None else sigma_ii.to(device).float()

    stats = RBVTStats(channels=out_features)
    num_levels = levels.shape[-1]
    eps = 1e-12

    for r0 in range(0, out_features, args.row_chunk):
        r1 = min(r0 + args.row_chunk, out_features)
        rc = r1 - r0
        Wr = W_fp[r0:r1].float()
        Wq = Wq_full[r0:r1]
        idx = idx_full[r0:r1]
        row_ids = torch.arange(rc, device=device).unsqueeze(1)

        e = Wq - Wr
        e_sign = torch.sign(e)
        b = e @ mu

        left_idx = (idx - 1).clamp(min=0)
        right_idx = (idx + 1).clamp(max=num_levels - 1)
        cur = levels[r0:r1][row_ids, idx]
        left = levels[r0:r1][row_ids, left_idx]
        right = levels[r0:r1][row_ids, right_idx]

        move_down = e_sign > 0
        gap = torch.where(move_down, (cur - left).abs(), (right - cur).abs())
        target_idx = torch.where(move_down, left_idx, right_idx)
        feasible = torch.where(move_down, idx > 0, idx < num_levels - 1)

        v = mu.unsqueeze(0) * e_sign * gap
        r = v.abs()
        q = sigma_ii.unsqueeze(0) * (gap.square() - 2.0 * gap * e.abs()).clamp(min=0.0)
        admissible = feasible & (gap > args.gap_floor) & ((b.unsqueeze(1) * v) > 0) & (r > eps)
        rho = q / (r + eps)

        for rr in range(rc):
            T = float(abs(b[rr].item()))
            base_obj = T * T
            stats.bias_before += base_obj
            stats.objective_before += base_obj
            if T <= eps:
                stats.bias_after += base_obj
                stats.objective_after += base_obj
                continue

            cand = torch.nonzero(admissible[rr], as_tuple=False).squeeze(1)
            stats.candidates += int(cand.numel())
            if cand.numel() == 0:
                stats.bias_after += base_obj
                stats.objective_after += base_obj
                continue

            cand = cand[torch.argsort(rho[rr, cand], descending=False)]
            if args.rbvt_topk > 0 and cand.numel() > args.rbvt_topk:
                cand = cand[: args.rbvt_topk]

            r_cand = r[rr, cand]
            q_cand = q[rr, cand]
            limit = T if not args.allow_overshoot else 2.0 * T
            cum_r = torch.cumsum(r_cand, dim=0)
            cum_q = torch.cumsum(q_cand, dim=0)
            zero = torch.zeros(1, device=device, dtype=r_cand.dtype)
            s_prev = torch.cat([zero, cum_r[:-1]], dim=0)
            q_prev = torch.cat([zero, cum_q[:-1]], dim=0)

            upper = ((limit - s_prev) / (r_cand + eps)).clamp(min=0.0, max=1.0)
            gamma_star = (T - s_prev - args.rbvt_lambda * q_cand / (2.0 * (r_cand + eps))) / (r_cand + eps)
            gamma = torch.minimum(torch.maximum(gamma_star, torch.zeros_like(gamma_star)), upper)
            relaxed = (T - s_prev - gamma * r_cand).square() + args.rbvt_lambda * (q_prev + gamma * q_cand)
            relaxed = torch.where(upper > 0.0, relaxed, torch.full_like(relaxed, float("inf")))

            best_val, best_pos = relaxed.min(dim=0)
            if float(best_val.item()) >= base_obj:
                stats.bias_after += base_obj
                stats.objective_after += base_obj
                continue

            best_pos_i = int(best_pos.item())
            prefix_r = float(s_prev[best_pos_i].item())
            prefix_q = float(q_prev[best_pos_i].item())
            drop_obj = (T - prefix_r) ** 2 + args.rbvt_lambda * prefix_q
            chosen_count = best_pos_i
            final_r = prefix_r
            final_q = prefix_q

            if float(gamma[best_pos_i].item()) > 0.0:
                keep_r = float((prefix_r + r_cand[best_pos_i]).item())
                if keep_r <= limit + 1e-8:
                    keep_q = float((prefix_q + q_cand[best_pos_i]).item())
                    keep_obj = (T - keep_r) ** 2 + args.rbvt_lambda * keep_q
                    if keep_obj < drop_obj:
                        chosen_count = best_pos_i + 1
                        final_r = keep_r
                        final_q = keep_q
                        stats.boundary_kept += 1

            if chosen_count > 0:
                chosen = cand[:chosen_count]
                idx_full[r0 + rr, chosen] = target_idx[rr, chosen]
                stats.flips += chosen_count

            stats.bias_after += (T - final_r) ** 2
            stats.objective_after += (T - final_r) ** 2 + args.rbvt_lambda * final_q
            stats.variance_increase += final_q

    return idx_full.cpu().numpy().astype("uint8"), stats


def load_lut_layer(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def apply_rbvt(args):
    os.makedirs(os.path.join(args.output_folder, "lut"), exist_ok=True)
    means, variances = collect_activation_stats(args)
    device = torch.device(args.device)
    model_type = args.model_type
    module_names = get_module_names(model_type)
    totals = RBVTStats()

    layer_ids = sorted(
        int(name[len("layer_") : -len(".pt")])
        for name in os.listdir(args.model_chunks)
        if name.startswith("layer_") and name.endswith(".pt")
    )
    selected = set(range(layer_ids[0], layer_ids[-1] + 1))
    if args.layer_range:
        start, end = [int(x) for x in args.layer_range.split(",")]
        selected = set(range(start, end))

    for layer_idx in tqdm(layer_ids, desc="Applying RBVT-Squeeze"):
        if layer_idx not in selected:
            continue
        out_file = os.path.join(args.output_folder, "lut", f"l{layer_idx}.pkl")
        if os.path.exists(out_file) and not args.overwrite:
            continue
        model_layer = torch.load(os.path.join(args.model_chunks, f"layer_{layer_idx}.pt"), map_location="cpu")
        lut_layer = load_lut_layer(os.path.join(args.input_lut, "lut", f"l{layer_idx}.pkl"))
        out_layer = {}
        for module_name in module_names:
            rows = lut_layer[module_name]
            centers = torch.from_numpy(
                __import__("numpy").stack([row[0][0] for row in rows]).astype("float32")
            )
            labels = torch.from_numpy(
                __import__("numpy").stack([row[0][1] for row in rows]).astype("uint8")
            )
            stat_key = f"{layer_idx}:{module_name}"
            new_labels, stats = apply_rbvt_indices(
                W_fp=model_layer[module_name].to(device),
                labels=labels.to(device),
                centers=centers.to(device),
                mu=means[stat_key],
                sigma_ii=variances.get(stat_key),
                args=args,
            )
            totals.add(stats)
            centers_np = centers.cpu().numpy()
            out_layer[module_name] = [[(centers_np[i], new_labels[i])] for i in range(new_labels.shape[0])]
            del centers, labels, new_labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        with open(out_file, "wb") as handle:
            pickle.dump(out_layer, handle)
    print(
        "RBVT-Squeeze summary: "
        f"flips={totals.flips} candidates={totals.candidates} "
        f"bias={totals.bias_before:.6e}->{totals.bias_after:.6e} "
        f"objective={totals.objective_before:.6e}->{totals.objective_after:.6e}"
    )


def main():
    parser = argparse.ArgumentParser(description="Dense-only RBVT correction for SqueezeLLM/LNQ LUTs.")
    parser.add_argument("mode", nargs="?", default="all", choices=["all", "stats", "apply"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_chunks", required=True)
    parser.add_argument("--input_lut", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--model_type", default="qwen", choices=["llama", "mistral", "qwen"])
    parser.add_argument("--dataset", default="redpajama", choices=["redpajama", "wikitext2", "c4"])
    parser.add_argument("--nsamples", type=int, default=1024)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", default="cache/lnq_plain/tokens")
    parser.add_argument("--stats_path", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_calib", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rbvt_lambda", type=float, default=1.0)
    parser.add_argument("--rbvt_topk", type=int, default=0)
    parser.add_argument("--row_chunk", type=int, default=1024)
    parser.add_argument("--gap_floor", type=float, default=1e-8)
    parser.add_argument("--allow_overshoot", action="store_true")
    parser.add_argument("--layer_range", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_stats", action="store_true")
    parser.add_argument("--attn_implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"))
    args = parser.parse_args()
    if args.mode == "stats":
        collect_activation_stats(args)
    else:
        apply_rbvt(args)


if __name__ == "__main__":
    main()
