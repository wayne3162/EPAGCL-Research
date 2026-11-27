"""Read-only E3 audit of completed B0/B2x8 paired 1000-epoch runs."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover(root: Path):
    results = {}
    for folder in sorted(root.glob("*_E3_formal_*")):
        if not folder.is_dir():
            continue
        args_file = folder / "args.json"
        result_files = list(folder.glob("run_01_seed_*/final.json"))
        if not args_file.exists() or not result_files:
            print(f"[INCOMPLETE] {folder.name}")
            continue
        if len(result_files) != 1:
            raise RuntimeError(f"Unexpected final.json count under {folder}: {len(result_files)}")
        args = read_json(args_file)
        result = read_json(result_files[0])
        env = read_json(folder / "environment.json")
        hashes = read_json(folder / "source_sha256.json")
        if args["phase"] != "formal" or result["phase"] != "formal":
            continue
        key = (int(result["seed"]), result["variant"])
        if key in results:
            raise RuntimeError(
                f"Multiple completed results for {key}. Resolve duplicate runs "
                f"without post-hoc cherry-picking: {results[key]['path']} and {folder}"
            )
        results[key] = dict(path=str(folder), args=args, final=result, env=env, hashes=hashes)
    return results


def same_protocol(a, b, seed):
    # Only the variant (loss implementation) and output destination may differ.
    x = {k: v for k, v in a["args"].items() if k not in ("variant", "output_root")}
    y = {k: v for k, v in b["args"].items() if k not in ("variant", "output_root")}
    if x != y:
        raise RuntimeError(f"Different protocol/args at seed={seed}")
    if a["hashes"] != b["hashes"]:
        raise RuntimeError(f"Different source SHA256 versions at seed={seed}")
    for field in ("split_sha256", "dataset_nodes", "gpu_name", "torch_version", "device"):
        if a["env"].get(field) != b["env"].get(field):
            raise RuntimeError(f"Mismatched {field} at seed={seed}")
    for item in (a, b):
        if item["final"]["completed_epochs"] != 1000 or item["final"]["probe_count"] != 20:
            raise RuntimeError(f"Formal protocol not complete: {item['path']}")
        if item["final"]["shadow_candidate_epoch_100"] is not None:
            raise RuntimeError(f"Shadow policy changed: {item['path']}")


def pct_reduction(base: float, fast: float) -> float:
    return 100 * (base - fast) / base


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit completed E3 B0 / B2x8 formal pairs")
    parser.add_argument("--root", default=None)
    parser.add_argument("--seeds", default="20261129,20261130,20261201")
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser().resolve() if args.root else Path(__file__).resolve().parent / "outputs"
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    results = discover(root)
    pairs = []
    all_hashes = []
    print("E3 paired formal audit (read-only)")
    print("Results root:", root)
    print("seed | B0 acc% | B2x8 acc% | delta pp | main reduction% | full reduction% | mem B0 / B2 (MiB)")
    for seed in seeds:
        b = results.get((seed, "b0"))
        f = results.get((seed, "b2x8"))
        if not b or not f:
            print(f"{seed} | NOT COMPLETE (B0={'yes' if b else 'no'}, B2x8={'yes' if f else 'no'})")
            continue
        same_protocol(b, f, seed)
        all_hashes.append(b["hashes"])
        b0, fast = b["final"], f["final"]
        pairs.append((seed, b0, fast))
        print(
            f"{seed} | {b0['final_test_accuracy_pct']:.4f} | "
            f"{fast['final_test_accuracy_pct']:.4f} | "
            f"{fast['final_test_accuracy_pct'] - b0['final_test_accuracy_pct']:+.4f} | "
            f"{pct_reduction(b0['main_model_seconds'], fast['main_model_seconds']):.2f} | "
            f"{pct_reduction(b0['elapsed_seconds'], fast['elapsed_seconds']):.2f} | "
            f"{b0['peak_allocated_memory_mib']:.1f} / {fast['peak_allocated_memory_mib']:.1f}"
        )
    if len(pairs) != len(seeds):
        print(f"NOT FINAL: {len(pairs)}/{len(seeds)} complete pairs. No aggregate claimed.")
        return
    if any(h != all_hashes[0] for h in all_hashes[1:]):
        raise RuntimeError("Source code changed between formal seeds; do not pool these as same protocol")
    bacc = [b["final_test_accuracy_pct"] for _, b, _ in pairs]
    facc = [f["final_test_accuracy_pct"] for _, _, f in pairs]
    bmain = [b["main_model_seconds"] for _, b, _ in pairs]
    fmain = [f["main_model_seconds"] for _, _, f in pairs]
    bfull = [b["elapsed_seconds"] for _, b, _ in pairs]
    ffull = [f["elapsed_seconds"] for _, _, f in pairs]
    def stat(vals):
        return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}" if len(vals)>1 else str(vals[0])
    print("\nCompleted all paired formal runs; reporting descriptive sample statistics.")
    print("Test accuracy (%) B0 mean ± sample std:", stat(bacc))
    print("Test accuracy (%) B2x8 mean ± sample std:", stat(facc))
    print("Mean paired delta accuracy (percentage points):", f"{statistics.mean([a-b for a,b in zip(facc,bacc)]):+.4f}")
    print("Aggregate main-model time reduction (%):", f"{pct_reduction(sum(bmain),sum(fmain)):.2f}")
    print("Aggregate full-run time reduction incl. probes (%):", f"{pct_reduction(sum(bfull),sum(ffull)):.2f}")
    print("E3 uses synchronized per-epoch timing; do not compare those times directly to the B0 Profiler wall clock or historical E1 totals.")
    print("Numeric/gradient equivalence is a separate prerequisite; paired classification quality is empirical, not guaranteed exact.")


if __name__ == "__main__":
    main()
