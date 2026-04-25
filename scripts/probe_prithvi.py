#!/usr/bin/env python
"""
scripts/probe_prithvi.py
One-shot probe for terratorch's Prithvi-EO-2.0-300M-TL backbone.

Empirically discovers the things we need to lock before writing
forecast/prithvi.py and embeddings_v1.parquet's schema:

  1. terratorch + torch versions actually present in this env
  2. model resolution under both 'terratorch_prithvi_eo_v2_300_tl' and
     'prithvi_eo_v2_300_tl' (docs say the prefix is optional)
  3. forward(...) signature and __doc__ -- specifically: does it accept
     lat/lon/temporal kwargs, and what are they named?
  4. expected input tensor shape ( B, C=6, T, H, W )
  5. forward output shape -- list-of-tensors or single tensor; we want
     the embedding (D-dimensional vector) per sample
  6. embedding dimension D after spatial+temporal mean-pool
  7. VRAM at batch=1, 2, 4, 8 in fp16 to size the inference run
  8. throughput (samples/sec) at the batch size that fits comfortably

Output: prints everything to stdout AND writes a JSON summary to
runs/prithvi_probe_<timestamp>.json so we can refer back to it from
forecast/prithvi.py without re-running the probe.

Run:
    python scripts/probe_prithvi.py
    python scripts/probe_prithvi.py --no-gpu       # CPU-only, faster sanity
    python scripts/probe_prithvi.py --batch 1 2 4  # custom batch sizes

This is a discovery script -- it is expected to print warnings and
fail-soft on probes that don't apply. The 'final summary' block at the
bottom is the contract.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

# Make `forecast/` importable so we can import hls_common constants
# (ensures we're using the same 6-band order in the probe input)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# -----------------------------------------------------------------------------
# Section helpers
# -----------------------------------------------------------------------------
def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def kv(label: str, value) -> None:
    print(f"  {label:<32s} {value}")


def safe_run(label: str, fn):
    """Run a probe step, printing any exception but never crashing the script.
    Returns the function's return value or None on failure."""
    print()
    print(f"--- {label} ---")
    try:
        return fn()
    except Exception as e:
        print(f"  ! {label} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return None


# -----------------------------------------------------------------------------
# 1. Environment probe
# -----------------------------------------------------------------------------
def probe_environment() -> dict:
    section("1. ENVIRONMENT")
    info: dict = {}
    for pkg in ("torch", "torchvision", "terratorch", "rasterio", "geopandas",
                "earthaccess", "xgboost", "numpy", "pandas"):
        try:
            v = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            v = "NOT INSTALLED"
        info[pkg] = v
        kv(pkg, v)
    info["python"] = sys.version.split()[0]
    kv("python", info["python"])

    import torch
    info["cuda_available"] = torch.cuda.is_available()
    kv("torch.cuda.is_available()", torch.cuda.is_available())
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"]  = torch.cuda.get_device_capability(0)
        info["cuda_vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        kv("cuda device", info["cuda_device_name"])
        kv("compute capability", f"sm_{info['cuda_capability'][0]}{info['cuda_capability'][1]}")
        kv("total VRAM (GB)", f"{info['cuda_vram_gb']:.2f}")

    return info


# -----------------------------------------------------------------------------
# 2. Model registration probe
# -----------------------------------------------------------------------------
def probe_registry() -> dict:
    section("2. TERRATORCH BACKBONE REGISTRY")
    info: dict = {}
    from terratorch import BACKBONE_REGISTRY  # noqa: I001

    # List all prithvi-related models
    all_names = list(BACKBONE_REGISTRY)
    prithvi = [n for n in all_names if "prithvi" in n.lower()]
    info["registry_size"] = len(all_names)
    info["prithvi_models_visible"] = prithvi
    kv("total models in registry", len(all_names))
    print(f"  prithvi-related ({len(prithvi)}):")
    for name in prithvi:
        print(f"    {name}")

    # Try resolving the 300_tl model under both naming conventions
    for candidate in ("terratorch_prithvi_eo_v2_300_tl", "prithvi_eo_v2_300_tl"):
        present = candidate in BACKBONE_REGISTRY
        kv(f"  {candidate!r} in registry", present)
        info[f"resolves_{candidate}"] = present

    return info


# -----------------------------------------------------------------------------
# 3. Model build + signature inspection
# -----------------------------------------------------------------------------
def build_model(model_name: str, pretrained: bool = True):
    """Return the loaded torch.nn.Module."""
    from terratorch import BACKBONE_REGISTRY
    print(f"  building {model_name!r} (pretrained={pretrained}) ...")
    print("  this triggers a ~1.2 GB download to ~/.cache/terratorch/ on first run.")
    t0 = _now_s()
    model = BACKBONE_REGISTRY.build(model_name, pretrained=pretrained)
    elapsed = _now_s() - t0
    print(f"  built in {elapsed:.1f}s")
    return model


def probe_signature(model) -> dict:
    section("3. MODEL SIGNATURE & STRUCTURE")
    info: dict = {}

    info["model_class"] = type(model).__name__
    info["model_module"] = type(model).__module__
    kv("class", info["model_class"])
    kv("module", info["model_module"])

    # Param count
    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["params_total"] = n_params
    info["params_trainable"] = n_train
    kv("total params (M)", f"{n_params / 1e6:.1f}")
    kv("trainable params (M)", f"{n_train / 1e6:.1f}")

    # Forward signature
    try:
        sig = inspect.signature(model.forward)
        info["forward_signature"] = str(sig)
        kv("forward signature", str(sig))
    except (TypeError, ValueError) as e:
        info["forward_signature"] = f"(could not inspect: {e})"
        kv("forward signature", info["forward_signature"])

    # Forward docstring
    doc = (inspect.getdoc(model.forward) or "").strip()
    info["forward_doc"] = doc[:400] + ("..." if len(doc) > 400 else "")
    if doc:
        print()
        print("  forward.__doc__:")
        for line in doc.splitlines()[:15]:
            print(f"    {line}")
        if len(doc.splitlines()) > 15:
            print(f"    ... [{len(doc.splitlines()) - 15} more lines]")

    # Top-level submodule structure (one level deep)
    print()
    print("  top-level submodules:")
    for name, child in model.named_children():
        kind = type(child).__name__
        n_sub = sum(p.numel() for p in child.parameters())
        print(f"    {name:<24s} {kind:<28s} {n_sub / 1e6:>6.2f}M params")
    info["children"] = [(name, type(c).__name__) for name, c in model.named_children()]

    # Look for the attributes most likely to tell us about expected input shape
    print()
    print("  candidate attributes (input-shape hints):")
    for attr in ("img_size", "patch_size", "in_chans", "embed_dim", "num_frames",
                 "num_classes", "num_features"):
        if hasattr(model, attr):
            v = getattr(model, attr)
            kv(f"    .{attr}", v)
            info[f"attr_{attr}"] = v if isinstance(v, (int, float, str, bool, list, tuple)) else str(v)

    return info


# -----------------------------------------------------------------------------
# 4. Forward pass probe
# -----------------------------------------------------------------------------
def probe_forward(model, info_so_far: dict, device: str) -> dict:
    """Try a forward pass with synthetic input. Tries multiple input shapes
    and kwarg combinations -- the TL variant accepts location+temporal but
    they may be optional or named differently."""
    section("4. FORWARD PASS PROBE")
    import torch

    info: dict = {}

    # Try to read expected dims from earlier probes; fall back to known defaults
    img_size = info_so_far.get("attr_img_size", 224)
    if isinstance(img_size, (list, tuple)):
        img_size = img_size[-1]
    img_size = int(img_size or 224)
    in_chans = int(info_so_far.get("attr_in_chans", 6) or 6)
    num_frames = info_so_far.get("attr_num_frames", None)
    if isinstance(num_frames, (list, tuple)):
        num_frames = num_frames[0]
    # Prithvi-EO-2.0 is multi-temporal; default to T=3 to match our chip-pick design
    T_default = int(num_frames or 3)

    info["assumed_C"] = in_chans
    info["assumed_H"] = img_size
    info["assumed_W"] = img_size
    info["assumed_T"] = T_default
    kv("assumed input shape", f"(B, C={in_chans}, T={T_default}, H={img_size}, W={img_size})")

    model = model.to(device).eval()

    # We'll try shape (B, C, T, H, W) first (standard 3D conv) and (B, T, C, H, W)
    # second as a fallback. Synthetic input: random reflectance-scaled values
    # in a realistic range (HLS reflectance / 10000 -> ~0..0.5 typical).
    #
    # We want to know whether forward accepts kwargs for lat/lon/temporal.
    # Common terratorch names: 'temporal_coords', 'location_coords'. We'll
    # try with and without.

    B = 2  # tiny batch for the probe -- just to validate the call works
    rng = np.random.default_rng(0)

    def make_input(shape):
        return torch.tensor(
            rng.uniform(0.0, 0.5, size=shape).astype(np.float32),
            device=device,
        )

    candidates = [
        ("BCTHW",  (B, in_chans, T_default, img_size, img_size)),
        ("BTCHW",  (B, T_default, in_chans, img_size, img_size)),
    ]

    info["forward_attempts"] = []

    success = False
    chosen_shape = None
    chosen_layout = None
    out = None
    last_kwargs = {}

    # For the TL model, try both with and without temporal/location kwargs.
    kwarg_sets = [
        {},
        # temporal_coords: (B, T, 2) for (year, day_of_year)
        # location_coords: (B, 2)    for (lat, lon)
        {
            "temporal_coords": torch.tensor(
                np.tile([[2018, 213]], (B, T_default, 1)).astype(np.float32),
                device=device,
            ),
            "location_coords": torch.tensor(
                np.tile([[42.0, -93.0]], (B, 1)).astype(np.float32),
                device=device,
            ),
        },
    ]

    for layout_name, shape in candidates:
        for kw_idx, kwargs in enumerate(kwarg_sets):
            attempt_label = f"{layout_name} kwargs#{kw_idx}"
            try:
                x = make_input(shape)
                with torch.no_grad():
                    o = model(x, **kwargs)
                # 'o' may be a tensor, list of tensors, or a tuple
                kind = type(o).__name__
                shapes = _output_shapes(o)
                print(f"  ✓ {attempt_label:<22s} -> {kind} shapes={shapes}")
                info["forward_attempts"].append({
                    "layout": layout_name, "kwargs_idx": kw_idx,
                    "input_shape": list(shape),
                    "kwargs_keys": list(kwargs.keys()),
                    "output_kind": kind,
                    "output_shapes": shapes,
                })
                if not success:
                    success = True
                    chosen_shape = shape
                    chosen_layout = layout_name
                    out = o
                    last_kwargs = kwargs
            except Exception as e:
                print(f"  ✗ {attempt_label:<22s} -> {type(e).__name__}: {str(e)[:120]}")
                info["forward_attempts"].append({
                    "layout": layout_name, "kwargs_idx": kw_idx,
                    "input_shape": list(shape),
                    "kwargs_keys": list(kwargs.keys()),
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                })

    info["forward_succeeded"] = success
    info["chosen_input_layout"] = chosen_layout
    info["chosen_input_shape"]  = list(chosen_shape) if chosen_shape else None
    info["chosen_kwargs"]       = list(last_kwargs.keys())

    if success:
        # Inspect output to figure out where the embedding lives
        info["output_shapes"] = _output_shapes(out)
        # Try to find a (B, ?) or (B, ?, D)-shaped tensor we can pool to (B, D)
        embedded = _find_pooled_embedding(out, B)
        if embedded is not None:
            info["pooled_embedding_dim_D"] = int(embedded.shape[-1])
            kv("pooled embedding dim D", info["pooled_embedding_dim_D"])
        else:
            kv("pooled embedding dim D", "could not infer; inspect output_shapes")

    return info


def _output_shapes(o):
    """Return a serializable description of the forward output shape(s)."""
    import torch
    if isinstance(o, torch.Tensor):
        return list(o.shape)
    if isinstance(o, (list, tuple)):
        return [_output_shapes(x) for x in o]
    if isinstance(o, dict):
        return {k: _output_shapes(v) for k, v in o.items()}
    return f"<{type(o).__name__}>"


def _find_pooled_embedding(o, batch_size: int):
    """Walk the output structure and return a tensor that can be pooled to
    (B, D). For prithvi backbones this is typically the last hidden state of
    shape (B, num_tokens, D) or a list of such tensors."""
    import torch
    if isinstance(o, torch.Tensor):
        if o.dim() == 2 and o.shape[0] == batch_size:
            return o
        if o.dim() == 3 and o.shape[0] == batch_size:
            # mean-pool tokens
            return o.mean(dim=1)
        if o.dim() >= 4 and o.shape[0] == batch_size:
            # spatially-extended -> mean over all non-batch, non-channel dims
            # (B, D, ...) -> (B, D)
            return o.mean(dim=tuple(range(2, o.dim())))
        return None
    if isinstance(o, (list, tuple)):
        # Try each element; prefer the last (deepest layer)
        for el in reversed(o):
            res = _find_pooled_embedding(el, batch_size)
            if res is not None:
                return res
        return None
    if isinstance(o, dict):
        for k in ("last_hidden_state", "hidden_states", "features"):
            if k in o:
                res = _find_pooled_embedding(o[k], batch_size)
                if res is not None:
                    return res
        return None
    return None


# -----------------------------------------------------------------------------
# 5. VRAM + throughput probe
# -----------------------------------------------------------------------------
def probe_vram_and_throughput(model, info_so_far: dict, batch_sizes: list[int],
                              device: str) -> dict:
    section("5. VRAM USE + THROUGHPUT (fp16)")
    info: dict = {"per_batch": {}}

    if device == "cpu":
        kv("skipping", "CPU-only mode; VRAM probe not applicable")
        return info

    if not info_so_far.get("forward_succeeded"):
        kv("skipping", "forward probe failed; cannot run VRAM probe")
        return info

    import torch

    layout = info_so_far["chosen_input_layout"]
    shape_template = info_so_far["chosen_input_shape"]
    use_kwargs = bool(info_so_far["chosen_kwargs"])

    # cast model to fp16
    model_fp16 = model.to(device).half().eval()

    rng = np.random.default_rng(1)
    for B in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Build a (B, ...) shape with the same C/T/H/W as the probe shape
        new_shape = list(shape_template)
        new_shape[0] = B
        x = torch.tensor(
            rng.uniform(0.0, 0.5, size=new_shape).astype(np.float32),
            device=device,
        ).half()
        kwargs = {}
        if use_kwargs:
            T = new_shape[1] if layout == "BTCHW" else new_shape[2]
            kwargs = {
                "temporal_coords": torch.tensor(
                    np.tile([[2018, 213]], (B, T, 1)).astype(np.float32),
                    device=device,
                ).half(),
                "location_coords": torch.tensor(
                    np.tile([[42.0, -93.0]], (B, 1)).astype(np.float32),
                    device=device,
                ).half(),
            }

        # warm-up
        try:
            with torch.no_grad():
                _ = model_fp16(x, **kwargs)
        except Exception as e:
            print(f"  B={B}: warm-up failed ({type(e).__name__}: {str(e)[:80]})")
            info["per_batch"][str(B)] = {"error": str(e)[:200]}
            continue

        # timed run (10 iterations)
        torch.cuda.synchronize()
        t0 = _now_s()
        N_ITERS = 10
        with torch.no_grad():
            for _ in range(N_ITERS):
                _ = model_fp16(x, **kwargs)
        torch.cuda.synchronize()
        elapsed = _now_s() - t0

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        sps = (B * N_ITERS) / elapsed
        per_batch_ms = elapsed / N_ITERS * 1000
        info["per_batch"][str(B)] = {
            "peak_vram_gb": peak_gb,
            "samples_per_sec": sps,
            "per_batch_ms": per_batch_ms,
        }
        print(f"  B={B:<3d} peak VRAM = {peak_gb:.2f} GB    "
              f"per-batch = {per_batch_ms:6.1f} ms    "
              f"throughput = {sps:6.1f} samples/sec")

    # cast model back to fp32 for any later probes
    model.to(device).float()

    return info


# -----------------------------------------------------------------------------
# 6. Embedding-extraction throughput estimation
# -----------------------------------------------------------------------------
def estimate_embedding_run(probe: dict) -> dict:
    section("6. EMBEDDING-RUN ESTIMATE (full grid)")
    info: dict = {}

    # We have ~ 388 GEOIDs × 12 years × 4 forecast_dates = 18,624 queries
    # times T=3 chips per query = 55,872 chip forward passes (one per chip);
    # but we batch them, so what matters is (n_queries / batch_size).
    n_queries = 388 * 12 * 4
    info["n_queries_estimated"] = n_queries
    kv("estimated queries (388 GEOIDs × 12 years × 4 dates)", n_queries)

    per_batch = probe.get("per_batch", {})
    if not per_batch:
        kv("skipping", "no VRAM probe results")
        return info

    # Pick the largest batch size that fit within ~10 GB (leaving 2 GB for OS)
    candidates = [(int(k), v) for k, v in per_batch.items()
                  if "peak_vram_gb" in v and v["peak_vram_gb"] < 10.0]
    if not candidates:
        kv("skipping", "no batch size fit in <10 GB VRAM")
        return info
    best_B, best_v = max(candidates, key=lambda kv: kv[0])
    sps = best_v["samples_per_sec"]
    info["recommended_batch_size"] = best_B
    info["recommended_throughput_sps"] = sps
    kv("recommended batch size", best_B)
    kv("throughput @ recommended", f"{sps:.1f} samples/sec")

    # estimated wall-clock for full embedding run
    eta_sec = n_queries / sps
    info["estimated_full_run_seconds"] = eta_sec
    info["estimated_full_run_minutes"] = eta_sec / 60
    kv("estimated full run", f"{eta_sec / 60:.1f} minutes ({eta_sec / 3600:.2f} hours)")

    return info


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def _now_s() -> float:
    import time
    return time.monotonic()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="prithvi_eo_v2_300_tl",
                    help="Backbone registry name (default: prithvi_eo_v2_300_tl)")
    ap.add_argument("--no-gpu", action="store_true",
                    help="Force CPU; skip the VRAM probe")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="Batch sizes to probe (default: 1 2 4 8)")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="Skip the pretrained-weights download (for fast struct probe)")
    ap.add_argument("--out-dir", default="runs",
                    help="Where to write the JSON summary")
    args = ap.parse_args()

    print("probe_prithvi.py starting")
    print(f"args = {vars(args)}")

    summary: dict = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat()}

    # 1. environment
    env_info = safe_run("environment", probe_environment) or {}
    summary["env"] = env_info

    # Decide device
    if args.no_gpu:
        device = "cpu"
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print()
    kv("DEVICE FOR PROBE", device)
    summary["device"] = device

    # 2. registry
    reg_info = safe_run("registry", probe_registry) or {}
    summary["registry"] = reg_info

    # 3. build model
    model = safe_run(
        f"build {args.model_name!r}",
        lambda: build_model(args.model_name, pretrained=not args.no_pretrained),
    )
    if model is None:
        # Try the alternate name once
        alt = ("terratorch_prithvi_eo_v2_300_tl"
               if args.model_name == "prithvi_eo_v2_300_tl"
               else "prithvi_eo_v2_300_tl")
        print(f"\n  retrying with {alt!r} ...")
        model = safe_run(f"build {alt!r}",
                         lambda: build_model(alt, pretrained=not args.no_pretrained))
        if model is not None:
            args.model_name = alt
    summary["model_name_used"] = args.model_name

    if model is None:
        print()
        print("CRITICAL: model build failed under both names; aborting deeper probes.")
        _write_summary(summary, args.out_dir)
        return

    # 4. signature
    sig_info = safe_run("signature", lambda: probe_signature(model)) or {}
    summary["signature"] = sig_info

    # 5. forward
    fwd_info = safe_run("forward", lambda: probe_forward(model, sig_info, device)) or {}
    summary["forward"] = fwd_info

    # 6. VRAM + throughput
    vram_info = safe_run(
        "vram + throughput",
        lambda: probe_vram_and_throughput(model, fwd_info, args.batch, device),
    ) or {}
    summary["vram"] = vram_info

    # 7. estimate
    est_info = safe_run("embedding-run estimate",
                        lambda: estimate_embedding_run(vram_info)) or {}
    summary["estimate"] = est_info

    # 8. final summary block
    section("FINAL SUMMARY (the contract for forecast/prithvi.py)")
    kv("model_name_used",          summary.get("model_name_used"))
    kv("device",                   summary.get("device"))
    kv("input layout",             fwd_info.get("chosen_input_layout"))
    kv("input shape (probe)",      fwd_info.get("chosen_input_shape"))
    kv("kwargs to pass",           fwd_info.get("chosen_kwargs"))
    kv("embedding dim D",          fwd_info.get("pooled_embedding_dim_D"))
    kv("recommended batch size",   est_info.get("recommended_batch_size"))
    kv("throughput (sps)",         est_info.get("recommended_throughput_sps"))
    kv("est. full run (min)",      est_info.get("estimated_full_run_minutes"))

    _write_summary(summary, args.out_dir)


def _write_summary(summary: dict, out_dir: str) -> None:
    summary["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir_p / f"prithvi_probe_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print()
    print(f"wrote summary -> {out_path}")


if __name__ == "__main__":
    main()
