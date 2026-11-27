"""E3 exploratory same-seed replay for seed 20261129, two B0/B2x8 pairs.

This is diagnostic, not additional independent-seed confirmatory evidence.
Usage:
  python -u extension_experiments/efficiency_B/run_e3_20261129_replay.py --dry-run
  python -u extension_experiments/efficiency_B/run_e3_20261129_replay.py

Results are intentionally NESTED below outputs/e3_same_seed_replay_20261129
so summarize_e3.py scanning top-level formal runs will not consume replays.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
TRAIN = HERE / "train_e3.py"
OUTPUTS = HERE / "outputs"
SEED = 20261129
DATASET = "Coauthor-Phy"
GPU = "cuda:0"
REPLAY_ROOT = OUTPUTS / f"e3_same_seed_replay_{SEED}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def check_finished(folder: Path, variant: str, hashes: dict, split_digest: str) -> dict:
    r = folder / f"run_01_seed_{SEED}"
    needed = [folder / n for n in ("args.json", "environment.json", "source_sha256.json", "summary.json")]
    needed += [r / n for n in ("loss.csv", "validation.csv", "epoch_timing.csv", "final.json", "best_by_validation.pt")]
    if (folder / "FAILED.json").exists() or any(not n.is_file() for n in needed):
        raise RuntimeError(f"Incomplete/failed run: {folder}")
    args = load_json(folder / "args.json")
    env = load_json(folder / "environment.json")
    source = load_json(folder / "source_sha256.json")
    final = load_json(r / "final.json")
    if source != hashes:
        raise RuntimeError(f"Source hashes changed: {folder}")
    for name, data in (("args", args), ("final", final)):
        if data.get("variant") != variant or data.get("seed") != SEED or data.get("phase") != "formal":
            raise RuntimeError(f"{name}: unexpected identity: {folder}")
    for key, value in (("epoch", 1000), ("eval", 50), ("eval_epoch", 3000),
                       ("vr_lambda", 10.0), ("vr_gamma", 0.002),
                       ("batch_compute", True), ("shadow_stop", False)):
        if args.get(key) != value:
            raise RuntimeError(f"Bad hyperparameter {key}: {folder}")
    if final.get("completed_epochs") != 1000 or final.get("probe_count") != 20:
        raise RuntimeError(f"Not a complete 1000-epoch run: {folder}")
    if (env.get("split_sha256") != split_digest or final.get("split_sha256") != split_digest):
        raise RuntimeError(f"Fixed split checksum mismatch: {folder}")
    if env.get("gpu_name") != "NVIDIA GeForce RTX 2070" or env.get("device") != GPU:
        raise RuntimeError(f"GPU environment differs: {folder}")
    with (r / "loss.csv").open(newline="", encoding="utf-8") as f:
        losses = list(csv.DictReader(f))
    with (r / "validation.csv").open(newline="", encoding="utf-8") as f:
        vals = list(csv.DictReader(f))
    if [int(row["epoch"]) for row in losses] != list(range(1, 1001)):
        raise RuntimeError(f"Missing/duplicate loss epochs: {folder}")
    if [int(row["encoder_epoch"]) for row in vals] != list(range(50, 1001, 50)):
        raise RuntimeError(f"Missing/duplicate validation epochs: {folder}")
    with (r / "epoch_timing.csv").open(newline="", encoding="utf-8") as f:
        times = list(csv.DictReader(f))
    if [int(row["epoch"]) for row in times] != list(range(1, 1001)):
        raise RuntimeError(f"Missing/duplicate timing epochs: {folder}")
    # Descriptive diagnostic: first epoch after which all remaining losses are <6.22.
    contrastive = [float(row["contrastive_loss"]) for row in losses]
    stable = next((i + 1 for i in range(1000) if max(contrastive[i:]) < 6.22), None)
    return {
        "variant": variant,
        "folder": str(folder),
        "accuracy_pct": float(final["final_test_accuracy_pct"]),
        "selected_validation_pct": float(final["selected_validation_accuracy_pct"]),
        "selected_encoder_epoch": int(final["encoder_selected_epoch"]),
        "final_contrastive_loss": float(final["final_contrastive_loss"]),
        "stable_below_6_22_epoch": stable,
        "main_model_seconds": float(final["main_model_seconds"]),
        "full_run_seconds": float(final["elapsed_seconds"]),
        "mean_feature_std": float(final["diagnostics"]["mean_feature_std"]),
        "mean_pairwise_cosine": float(final["diagnostics"]["mean_pairwise_cosine"]),
    }


def locate_original(variant: str) -> Path:
    pattern = f"*_E3_formal_{variant}_{DATASET}_seed_{SEED}"
    candidates = [p for p in sorted(OUTPUTS.glob(pattern)) if p.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one original formal run for {variant}: {candidates}")
    return candidates[0]


def existing_replay(repeat: int, variant: str, hashes: dict, split_digest: str):
    subroot = REPLAY_ROOT / f"repeat_{repeat:02d}" / variant
    candidates = [p for p in sorted(subroot.glob(f"*_E3_formal_{variant}_{DATASET}_seed_{SEED}")) if p.is_dir()]
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple replay folders; refusing selection: {candidates}")
    if candidates:
        return check_finished(candidates[0], variant, hashes, split_digest)
    return None


def run_command(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        raise RuntimeError(f"Refusing to overwrite existing log without a finished run: {log}")
    print("COMMAND:", " ".join(command), flush=True)
    with log.open("x", encoding="utf-8") as out:
        out.write("COMMAND: " + " ".join(command) + "\n")
        out.flush()
        proc = subprocess.Popen(command, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                start_new_session=True)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                out.write(line)
                out.flush()
            code = proc.wait()
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait()
            raise
        if code != 0:
            raise RuntimeError(f"Child failed with exit code {code}; inspect {log}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="E3 exploratory paired replays; original formal results untouched")
    parser.add_argument("--dry-run", action="store_true", help="Preview plans without running training")
    parser.add_argument("--repetitions", type=int, choices=(1, 2), default=2,
                        help="Use 1 for a first pair, then 2 to complete the pre-set plan")
    args = parser.parse_args(argv)
    if not TRAIN.exists():
        raise FileNotFoundError(TRAIN)
    references = {v: locate_original(v) for v in ("b0", "b2x8")}
    hashes = load_json(references["b0"] / "source_sha256.json")
    if load_json(references["b2x8"] / "source_sha256.json") != hashes:
        raise RuntimeError("Original two formal models have different source versions")
    for name, expected_hash in hashes.items():
        path = REPO_ROOT / name
        if not path.is_file() or digest(path) != expected_hash:
            raise RuntimeError(f"Source file does not match original formal run: {name}")
    original_env = {v: load_json(references[v] / "environment.json") for v in references}
    split_digest = original_env["b0"]["split_sha256"]
    if original_env["b2x8"]["split_sha256"] != split_digest:
        raise RuntimeError("Original formal pair split mismatch")
    split_path = Path(original_env["b0"]["split_path"])
    if not split_path.is_file() or digest(split_path) != split_digest:
        raise RuntimeError("Original fixed split file changed or missing")
    original = {v: check_finished(references[v], v, hashes, split_digest) for v in references}
    plan = [(1, "b0"), (1, "b2x8"), (2, "b2x8"), (2, "b0")]
    plan = [(r, v) for r, v in plan if r <= args.repetitions]
    cached = {}
    print("E3 same-seed DIAGNOSTIC; results NOT added to the 5 formal independent seeds")
    print(f"Seed={SEED} | reps={args.repetitions} | GPU={GPU} | result_root={REPLAY_ROOT}")
    print(f"Original B0={original['b0']['accuracy_pct']:.4f}% B2x8={original['b2x8']['accuracy_pct']:.4f}%")
    for rep, variant in plan:
        found = existing_replay(rep, variant, hashes, split_digest)
        cached[(rep, variant)] = found
        print(f"Repeat {rep}: {variant} -> {'SKIP (verified complete)' if found else 'RUN'}")
    if args.dry_run:
        print("Dry run: no training or output files created.")
        return
    REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
    for rep, variant in plan:
        if cached[(rep, variant)] is not None:
            continue
        subroot = REPLAY_ROOT / f"repeat_{rep:02d}" / variant
        subroot.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", str(TRAIN), "--variant", variant,
               "--phase", "formal", "--seed", str(SEED),
               "--device", GPU, "--output-root", str(subroot)]
        log = REPLAY_ROOT / "logs" / f"repeat_{rep:02d}_{variant}.log"
        run_command(cmd, log)
        cached[(rep, variant)] = existing_replay(rep, variant, hashes, split_digest)
        if cached[(rep, variant)] is None:
            raise RuntimeError(f"Training succeeded but result missing: rep={rep} {variant}")
    all_rows = [{"repeat": 0, "origin": "original_formal", **original[v]} for v in ("b0", "b2x8")]
    for rep, variant in plan:
        all_rows.append({"repeat": rep, "origin": "exploratory_replay", **cached[(rep, variant)]})
    pairs = []
    for rep in (0, *range(1, args.repetitions + 1)):
        vs = {row["variant"]: row for row in all_rows if row["repeat"] == rep}
        if len(vs) != 2:
            raise RuntimeError(f"Missing one variant in rep {rep}")
        a, b = vs["b0"], vs["b2x8"]
        pairs.append({"repeat": rep, "origin": a["origin"],
                      "delta_accuracy_pp": b["accuracy_pct"] - a["accuracy_pct"],
                      "full_time_reduction_pct": 100 * (1-b["full_run_seconds"]/a["full_run_seconds"]),
                      "b0_stable_epoch": a["stable_below_6_22_epoch"],
                      "b2_stable_epoch": b["stable_below_6_22_epoch"]})
    payload = {
        "purpose": "Exploratory non-determinism/late-learning diagnosis, not new independent seeds",
        "seed": SEED,
        "original_formal_results_preserved": True,
        "warning": "Do not merge replays into five-seed confirmatory sample or replace original low result.",
        "shared_original_source_sha256": hashes,
        "runs": all_rows, "pairs": pairs,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    report = REPLAY_ROOT / "replay_comparison.json"
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n=== REPLAY DIAGNOSTIC RESULTS ===")
    for row in pairs:
        print(f"repeat={row['repeat']} ({row['origin']}) "
              f"delta_acc={row['delta_accuracy_pp']:+.4f} pp, "
              f"full_time_reduction={row['full_time_reduction_pct']:.2f}%, "
              f"stable_loss_epochs B0/B2x8={row['b0_stable_epoch']}/{row['b2_stable_epoch']}")
    print("Diagnostic JSON:", report)


if __name__ == "__main__":
    main()
