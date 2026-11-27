"""Strict B0 vs B2 equivalence for grouped batches (2/4/8).

Requires existing, passing B1 equivalence.json and unchanged verified sources.
Verifies both directional permutations / RNG, loss, input gradient, model
parameter gradient and unchanged variance regularization. No training/eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "extension_experiments"
B_DIR = Path(__file__).resolve().parent
for d in (PROJECT_ROOT, EXPERIMENT_ROOT, B_DIR):
    sys.path.insert(0, str(d))

from Model import MyModel  # noqa: E402
from model_b2 import MyModelB2  # noqa: E402
from verify_b1 import (  # noqa: E402
    RTOL, ATOL, SIZES, PARAM_SIZES,
    rng_snapshot, same_rng, seeded_embeddings, run_embedding_variant,
    run_parameter_variant, compare, ring_edges,
)

SOURCE_FILES = (
    "Model.py",
    "extension_experiments/efficiency_B/model_b1.py",
    "extension_experiments/efficiency_B/model_b2.py",
    "extension_experiments/efficiency_B/verify_b1.py",
    "extension_experiments/efficiency_B/verify_b2.py",
    "extension_experiments/vr_regularization.py",
)
GROUPS = (2, 4, 8)


def parse_args():
    parser = argparse.ArgumentParser(description="B0/B2 grouped-loss equivalence test")
    parser.add_argument("--b1-verification-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default=None)
    parser.add_argument("--small-only", action="store_true", help="Diagnostic only, cannot pass formal verification")
    args = parser.parse_args()
    if args.batch_size != 256:
        parser.error("Target B2 experiment uses batch-size 256")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        parser.error("Run on the same CUDA GPU as the B1 verification")
    return args


def sha256_files():
    return {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def check_b1_verification(path, device):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not doc.get("passed") or doc.get("small_only"):
        raise RuntimeError("B1 source verification must be a complete PASS")
    if doc.get("batch_size") != 256 or doc.get("gpu_name") != torch.cuda.get_device_name(device):
        raise RuntimeError("B1 verification was performed on different GPU/batch settings")
    if doc.get("device") != str(device):
        raise RuntimeError("B1 verification CUDA device differs")
    for name, old_hash in doc.get("source_sha256", {}).items():
        if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != old_hash:
            raise RuntimeError(f"B1 verified source changed after verification: {name}")
    return doc


def embedding_case(n, device, seed, batch_size):
    z1, z2 = seeded_embeddings(n, device, seed)
    before = rng_snapshot()
    baseline = run_embedding_variant(MyModel, z1, z2, batch_size, before)
    checks = []
    for g in GROUPS:
        candidate = run_embedding_variant(partial(MyModelB2, groups_per_call=g), z1, z2, batch_size, before)
        comparisons = [compare("loss", baseline[0], candidate[0]),
                       compare("grad_z1", baseline[1], candidate[1]),
                       compare("grad_z2", baseline[2], candidate[2])]
        rng_ok = same_rng(baseline[4], candidate[4])
        perms_ok = len(baseline[3]) == len(candidate[3]) == 2 and baseline[3] == candidate[3]
        checks.append({"groups_per_call": g,
                       "two_randperm_draws_identical": perms_ok,
                       "rng_after_loss_identical": rng_ok,
                       "comparisons": comparisons,
                       "passed": rng_ok and perms_ok and all(c["passed"] for c in comparisons)})
        del candidate
    del baseline, z1, z2
    return {"kind": "embedding_and_input_gradient", "nodes": n,
            "last_batch_nodes": n % batch_size or batch_size,
            "groups": checks, "passed": all(v["passed"] for v in checks)}


def parameter_case(n, device, seed, batch_size):
    generator = torch.Generator(device="cpu").manual_seed(seed + n + 100_000)
    x1 = torch.randn((n, 32), generator=generator).to(device)
    x2 = torch.randn((n, 32), generator=generator).to(device)
    edges = ring_edges(n, device)
    torch.manual_seed(seed + n)
    base = MyModel(32, 256, 256, 0.5).to(device)
    state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    del base
    before = rng_snapshot()
    baseline = run_parameter_variant(MyModel, state, x1, x2, edges, batch_size, before)
    checks = []
    for g in GROUPS:
        candidate = run_parameter_variant(partial(MyModelB2, groups_per_call=g), state, x1, x2, edges, batch_size, before)
        comps = [compare("contrastive_loss", baseline[0], candidate[0]),
                 compare("variance_penalty_unchanged", baseline[1], candidate[1]),
                 compare("total_VR_loss", baseline[2], candidate[2])]
        for name, orig_grad in baseline[3].items():
            cand_grad = candidate[3].get(name)
            if orig_grad is None or cand_grad is None:
                comps.append({"name": "grad_" + name, "passed": orig_grad is None and cand_grad is None,
                              "error": "unexpected missing gradient"})
            else:
                comps.append(compare("grad_" + name, orig_grad, cand_grad))
        rng_ok = same_rng(baseline[5], candidate[5])
        perms_ok = len(baseline[4]) == len(candidate[4]) == 2 and baseline[4] == candidate[4]
        checks.append({"groups_per_call": g, "comparisons": comps,
                       "two_randperm_draws_identical": perms_ok,
                       "rng_after_loss_identical": rng_ok,
                       "passed": rng_ok and perms_ok and all(c["passed"] for c in comps)})
        del candidate
    del baseline, x1, x2, edges, state
    return {"kind": "model_parameter_gradients_with_VR", "nodes": n,
            "last_batch_nodes": n % batch_size or batch_size,
            "groups": checks, "passed": all(v["passed"] for v in checks)}


def main():
    args = parse_args()
    device = torch.device(args.device)
    check_b1_verification(args.b1_verification_json, device)
    out = Path(args.out) if args.out else (B_DIR / "outputs" /
        (datetime.now().strftime("%Y%m%d_%H%M%S") + "_B2_verify_seed_" + str(args.seed)))
    out.mkdir(parents=True, exist_ok=False)
    report_path = out / "equivalence_b2.json"
    data = {
        "study": "B0 vs B2x2/B2x4/B2x8 fixed-input loss/gradient equivalence",
        "seed": args.seed, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device), "torch_version": str(torch.__version__),
        "rtol": RTOL, "atol": ATOL, "batch_size": args.batch_size,
        "groups": list(GROUPS), "small_only": args.small_only,
        "b1_verification": str(Path(args.b1_verification_json).resolve()),
        "source_sha256": sha256_files(), "checks": [], "passed": False,
    }
    print("B2 grouped-loss equivalence. Output:", out, flush=True)
    try:
        sizes = [n for n in SIZES if not args.small_only or n < 34493]
        for n in sizes:
            print(f"Fixed embeddings + input gradients: N={n}", flush=True)
            result = embedding_case(n, device, args.seed, args.batch_size)
            data["checks"].append(result)
            report_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            print("  ", "PASS" if result["passed"] else "FAIL", flush=True)
        for n in PARAM_SIZES:
            print(f"Model parameter gradients + original VR: N={n}", flush=True)
            result = parameter_case(n, device, args.seed, args.batch_size)
            data["checks"].append(result)
            report_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            print("  ", "PASS" if result["passed"] else "FAIL", flush=True)
        data["passed"] = (not args.small_only and
                          len(data["checks"]) == len(SIZES) + len(PARAM_SIZES) and
                          all(case["passed"] for case in data["checks"]))
    except Exception as exc:
        data["error"] = repr(exc)
        print("B2 verification aborted:", repr(exc), flush=True)
    finally:
        report_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print("Overall:", "PASS" if data["passed"] else "NOT PASSED", flush=True)
        print("Results:", report_path, flush=True)
    if not data["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
