"""Strict B0/B1 equivalence checks; run from the EPAGCL project root.

Uses fixed *identical tensors* and restores CPU/CUDA RNG state between variants.
Checks input gradients, model parameter gradients, and both randperm draws.
No formal experiment files, test labels, optimizer state, or splits are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "extension_experiments"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Model import MyModel  # noqa: E402
from model_b1 import MyModelB1  # noqa: E402
from vr_regularization import variance_floor_loss  # noqa: E402

RTOL = 1e-4
ATOL = 1e-5
SIZES = (64, 256, 257, 1025, 34493)
PARAM_SIZES = (64, 256, 257, 1025)
SOURCE_FILES = (
    "Model.py",
    "extension_experiments/efficiency_B/model_b1.py",
    "extension_experiments/efficiency_B/verify_b1.py",
    "extension_experiments/vr_regularization.py",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Verify unchanged B0 vs B1 loss and gradients")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--small-only", action="store_true", help="Debug only; skips 34493-node case and cannot pass final verification")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not (PROJECT_ROOT / "Model.py").is_file():
        parser.error("Place scripts under EPAGCL/extension_experiments/efficiency_B/")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; verify in the existing EPAGCL GPU environment")
    return args


def source_hashes():
    return {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def rng_snapshot():
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def same_rng(a, b):
    return torch.equal(a["cpu"], b["cpu"]) and (
        len(a["cuda"]) == len(b["cuda"]) and
        all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"]))
    )


@contextmanager
def capture_randperm():
    original = torch.randperm
    records = []

    def intercept(*args, **kwargs):
        result = original(*args, **kwargs)
        # Hash exact permutation data without printing all 34493 node indices.
        permutation = result.detach().to("cpu", dtype=torch.int64).contiguous()
        records.append(hashlib.sha256(permutation.numpy().tobytes()).hexdigest())
        return result

    torch.randperm = intercept
    try:
        yield records
    finally:
        torch.randperm = original


def compare(name, baseline, candidate):
    a, b = baseline.detach(), candidate.detach()
    if a.shape != b.shape:
        return {"name": name, "passed": False, "error": "shape mismatch", "b0_shape": list(a.shape), "b1_shape": list(b.shape)}
    diff = (a - b).abs()
    rel = diff / a.abs().clamp_min(ATOL)  # Stabilized near zero; see max_abs as well.
    close = torch.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=False)
    return {
        "name": name,
        "passed": bool(close.all().item()),
        "max_absolute_difference": float(diff.max().item()),
        "max_relative_difference_clamped_denominator": float(rel.max().item()),
        "elements_outside_tolerance": int((~close).sum().item()),
        "element_count": a.numel(),
    }


def seeded_embeddings(num_nodes, device, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed + num_nodes)
    z1 = torch.randn((num_nodes, 256), generator=generator, dtype=torch.float32).to(device)
    z2 = torch.randn((num_nodes, 256), generator=generator, dtype=torch.float32).to(device)
    return z1, z2


def run_embedding_variant(cls, z1, z2, batch_size, rng):
    model = cls(32, 256, 256, 0.5).to(z1.device)
    a = z1.detach().clone().requires_grad_(True)
    b = z2.detach().clone().requires_grad_(True)
    restore_rng(rng)
    with capture_randperm() as permutations:
        loss = model.loss(a, b, batch_compute=True, batch_size=batch_size)
    state_after = rng_snapshot()
    grad_a, grad_b = torch.autograd.grad(loss, (a, b))
    result = (loss.detach().clone(), grad_a.detach().clone(), grad_b.detach().clone(), list(permutations), state_after)
    del a, b, model, loss, grad_a, grad_b
    return result


def embedding_case(num_nodes, device, seed, batch_size):
    z1, z2 = seeded_embeddings(num_nodes, device, seed)
    before = rng_snapshot()
    b0 = run_embedding_variant(MyModel, z1, z2, batch_size, before)
    b1 = run_embedding_variant(MyModelB1, z1, z2, batch_size, before)
    checked = [compare("loss", b0[0], b1[0]), compare("d_loss_d_z1", b0[1], b1[1]), compare("d_loss_d_z2", b0[2], b1[2])]
    perms_match = len(b0[3]) == len(b1[3]) == 2 and b0[3] == b1[3]
    rng_match = same_rng(b0[4], b1[4])
    return {
        "kind": "embedding_and_input_gradient",
        "nodes": num_nodes,
        "last_batch_nodes": num_nodes % batch_size or batch_size,
        "batch_size": batch_size,
        "two_directional_randperm_draws_identical": perms_match,
        "rng_state_after_loss_identical": rng_match,
        "comparisons": checked,
        "passed": perms_match and rng_match and all(c["passed"] for c in checked),
    }


def ring_edges(n, device):
    v = torch.arange(n, dtype=torch.long, device=device)
    w = (v + 1) % n
    return torch.stack((torch.cat((v, w)), torch.cat((w, v))), dim=0)


def run_parameter_variant(cls, state, x1, x2, edge_index, batch_size, rng):
    model = cls(x1.shape[1], 256, 256, 0.5).to(x1.device)
    model.load_state_dict(state, strict=True)
    model.train()
    out1, z1 = model(x1, edge_index)
    out2, z2 = model(x2, edge_index)
    restore_rng(rng)
    with capture_randperm() as permutations:
        contrastive = model.loss(z1, z2, batch_compute=True, batch_size=batch_size)
    after = rng_snapshot()
    vr = variance_floor_loss(out1, out2, gamma=0.002)
    total = contrastive + 10.0 * vr  # identical to B0's VR objective
    total.backward()
    grads = {k: p.grad.detach().clone() if p.grad is not None else None for k, p in model.named_parameters()}
    result = (contrastive.detach().clone(), vr.detach().clone(), total.detach().clone(), grads, list(permutations), after)
    del model, out1, out2, z1, z2, contrastive, vr, total
    return result


def parameter_case(num_nodes, device, seed, batch_size):
    generator = torch.Generator(device="cpu").manual_seed(seed + num_nodes + 100_000)
    x1 = torch.randn((num_nodes, 32), generator=generator).to(device)
    x2 = torch.randn((num_nodes, 32), generator=generator).to(device)
    edges = ring_edges(num_nodes, device)
    torch.manual_seed(seed + num_nodes)
    base_model = MyModel(32, 256, 256, 0.5).to(device)
    state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
    del base_model
    before = rng_snapshot()
    b0 = run_parameter_variant(MyModel, state, x1, x2, edges, batch_size, before)
    b1 = run_parameter_variant(MyModelB1, state, x1, x2, edges, batch_size, before)
    checked = [compare("contrastive_loss", b0[0], b1[0]), compare("variance_penalty_unchanged", b0[1], b1[1]), compare("total_VR_loss", b0[2], b1[2])]
    for name in b0[3]:
        a, b = b0[3][name], b1[3].get(name)
        if a is None or b is None:
            checked.append({"name": "grad_" + name, "passed": a is None and b is None, "error": "unexpected missing gradient"})
        else:
            checked.append(compare("grad_" + name, a, b))
    perms_match = len(b0[4]) == len(b1[4]) == 2 and b0[4] == b1[4]
    rng_match = same_rng(b0[5], b1[5])
    return {
        "kind": "model_parameter_gradients_with_VR",
        "nodes": num_nodes,
        "last_batch_nodes": num_nodes % batch_size or batch_size,
        "two_directional_randperm_draws_identical": perms_match,
        "rng_state_after_loss_identical": rng_match,
        "comparisons": checked,
        "passed": perms_match and rng_match and all(c["passed"] for c in checked),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device), flush=True)
    output = Path(args.out) if args.out else (
        EXPERIMENT_ROOT / "efficiency_B" / "outputs" /
        (datetime.now().strftime("%Y%m%d_%H%M%S") + "_B1_verify_seed_" + str(args.seed))
    )
    output.mkdir(parents=True, exist_ok=False)
    report_file = output / "equivalence.json"
    sizes = [n for n in SIZES if not args.small_only or n < 34493]
    report = {
        "study": "B0 vs B1 fixed-input numerical/gradient equivalence",
        "seed": args.seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "rtol": RTOL,
        "atol": ATOL,
        "batch_size": args.batch_size,
        "all_required_sizes": list(SIZES),
        "small_only": args.small_only,
        "source_sha256": source_hashes(),
        "checks": [],
        "passed": False,
    }
    print("B1 independent verification. Output:", output, flush=True)
    try:
        for n in sizes:
            print(f"Checking fixed embeddings/input gradients: N={n}", flush=True)
            result = embedding_case(n, device, args.seed, args.batch_size)
            report["checks"].append(result)
            report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            print("   ", "PASS" if result["passed"] else "FAIL", flush=True)
        for n in PARAM_SIZES:
            print(f"Checking actual model parameter gradients + unchanged VR: N={n}", flush=True)
            result = parameter_case(n, device, args.seed, args.batch_size)
            report["checks"].append(result)
            report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            print("   ", "PASS" if result["passed"] else "FAIL", flush=True)
        report["passed"] = (not args.small_only and
                            len(report["checks"]) == len(SIZES) + len(PARAM_SIZES) and
                            all(c["passed"] for c in report["checks"]))
    except Exception as exc:
        report["error"] = repr(exc)
        print("Verification aborted:", repr(exc), flush=True)
    finally:
        report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("Overall:", "PASS" if report["passed"] else "NOT PASSED", flush=True)
        print("Results:", report_file, flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
