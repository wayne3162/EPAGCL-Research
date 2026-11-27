"""Run four *additional* E3 formal runs after the original six, with safe resume.

Copy this file into extension_experiments/efficiency_B/ alongside train_e3.py.
Usage from the EPAGCL project root:
    python -u extension_experiments/efficiency_B/run_e3_extra_2seeds.py
    python -u extension_experiments/efficiency_B/run_e3_extra_2seeds.py --dry-run

No original files are modified. Each formal run creates its own timestamped
output. Incomplete/failed attempts are preserved, not silently overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

B_ROOT = Path(__file__).resolve().parent
REPO_ROOT = B_ROOT.parent.parent
OUTPUTS = B_ROOT / "outputs"
TRAIN = B_ROOT / "train_e3.py"
SUMMARIZER = B_ROOT / "summarize_e3.py"
GPU = "cuda:0"
DATASET = "Coauthor-Phy"

# Fixed, pre-registered add-on order, preserving all original six runs.
OLD_SEEDS = (20261129, 20261130, 20261201)
NEW_SEEDS = (20261202, 20261203)
PLAN = (
    (20261202, "b2x8"),
    (20261202, "b0"),
    (20261203, "b0"),
    (20261203, "b2x8"),
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output(folder: Path, seed: int, variant: str, *, check_code: bool = True) -> dict:
    """Accept only a genuinely completed run, not an existing empty directory."""
    run = folder / f"run_01_seed_{seed}"
    needed = [
        folder / "args.json", folder / "environment.json",
        folder / "source_sha256.json", folder / "summary.json",
        run / "final.json", run / "best_by_validation.pt",
        run / "loss.csv", run / "validation.csv", run / "epoch_timing.csv",
    ]
    if (folder / "FAILED.json").exists() or any(not p.is_file() for p in needed):
        raise ValueError("failure marker or missing files")
    args = read_json(folder / "args.json")
    env = read_json(folder / "environment.json")
    hashes = read_json(folder / "source_sha256.json")
    result = read_json(run / "final.json")
    summary = read_json(folder / "summary.json")

    for name, obj in (("args", args), ("final", result), ("summary", summary)):
        if obj.get("seed") != seed or obj.get("variant") != variant or obj.get("phase") != "formal":
            raise ValueError(f"{name}: mismatched seed, variant, or phase")
    expected = {"epoch": 1000, "eval": 50, "eval_epoch": 3000,
                "vr_lambda": 10.0, "vr_gamma": 0.002,
                "batch_compute": True, "shadow_stop": False}
    for field, value in expected.items():
        if args.get(field) != value:
            raise ValueError(f"args: expected {field}={value!r}")
    if result.get("completed_epochs") != 1000 or result.get("probe_count") != 20:
        raise ValueError("not 1000 epochs and 20 probes")
    if result.get("shadow_candidate_epoch_100") is not None:
        raise ValueError("unexpected early-stop/shadow flag")
    if env.get("gpu_name") != "NVIDIA GeForce RTX 2070" or env.get("device") != GPU:
        raise ValueError("GPU mismatch")
    if result.get("split_sha256") != env.get("split_sha256"):
        raise ValueError("different split checksums")
    if result.get("split_path") != env.get("split_path"):
        raise ValueError("different split paths")
    split_path = Path(env["split_path"])
    if not split_path.is_file() or sha256_file(split_path) != env["split_sha256"]:
        raise ValueError("fixed split was changed or is missing")
    for field in ("main_model_seconds", "linear_probe_seconds", "elapsed_seconds",
                  "final_test_accuracy_pct"):
        value = result.get(field)
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"invalid {field}")
    if result["elapsed_seconds"] <= 0 or result["main_model_seconds"] <= 0:
        raise ValueError("invalid elapsed time")
    if abs(result["final_test_accuracy_pct"] - summary.get("test_mean_pct", -999)) > 1e-6:
        raise ValueError("summary/final accuracy mismatch")
    for filename, expected_epochs in (
        ("loss.csv", list(range(1, 1001))),
        ("validation.csv", list(range(50, 1001, 50))),
        ("epoch_timing.csv", list(range(1, 1001))),
    ):
        with (run / filename).open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        column = "encoder_epoch" if filename == "validation.csv" else "epoch"
        epochs = [int(row[column]) for row in rows]
        if epochs != expected_epochs:
            raise ValueError(f"{filename}: missing/duplicate/unexpected epoch rows")
    if check_code:
        for name, digest in hashes.items():
            path = REPO_ROOT / name
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"source version changed: {name}")
    return {"folder": str(folder), "args": args, "env": env,
            "hashes": hashes, "final": result}


def completed(seed: int, variant: str) -> dict | None:
    """Never select among multiple complete runs after inspecting outcomes."""
    found = []
    glob = f"*_E3_formal_{variant}_{DATASET}_seed_{seed}"
    for folder in sorted(OUTPUTS.glob(glob)):
        if not folder.is_dir():
            continue
        try:
            valid = validate_output(folder, seed, variant)
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
            # Never silently rerun a completed run with incompatible code, GPU,
            # data split, or protocol; that would create post-hoc duplicates.
            if any(folder.glob("run_01_seed_*/final.json")):
                raise RuntimeError(
                    f"Existing formal run cannot be safely reused: {folder}. "
                    f"Reason: {exc}. Investigate rather than rerun selectively."
                ) from exc
            print(f"[INCOMPLETE / NOT REUSABLE] {folder.name}: {exc}", flush=True)
            continue
        found.append(valid)
    if len(found) > 1:
        raise RuntimeError(f"Duplicate complete formal results for seed={seed}, variant={variant}. "
                           "Do not select or overwrite results post hoc.")
    return found[0] if found else None


def check_consistency(found: dict[tuple[int, str], dict]) -> None:
    """Check all completed formal runs use unchanged sources and same protocol."""
    if not found:
        return
    reference = next(iter(found.values()))
    ref_args = {k: v for k, v in reference["args"].items() if k not in ("variant", "output_root")}
    for (seed, variant), item in found.items():
        args = {k: v for k, v in item["args"].items() if k not in ("variant", "output_root")}
        # Seed changes by design, and its corresponding split changes too.
        for key in ("seed",):
            ref_args.pop(key, None)
            args.pop(key, None)
        if args != ref_args:
            raise RuntimeError(f"Formal hyperparameters differ at {seed}/{variant}")
        if item["hashes"] != reference["hashes"]:
            raise RuntimeError(f"Formal code SHA256 differs at {seed}/{variant}")
        for field in ("gpu_name", "torch_version", "device", "dataset_nodes"):
            if item["env"].get(field) != reference["env"].get(field):
                raise RuntimeError(f"Environment {field} differs at {seed}/{variant}")
    for seed in OLD_SEEDS + NEW_SEEDS:
        a, b = found.get((seed, "b0")), found.get((seed, "b2x8"))
        if a and b and a["env"]["split_sha256"] != b["env"]["split_sha256"]:
            raise RuntimeError(f"Different fixed split in pair seed={seed}")


def smoke_exists() -> bool:
    pattern = "*_E3_smoke_b2x8_Coauthor-Phy_seed_20261128"
    for folder in OUTPUTS.glob(pattern):
        try:
            result = read_json(folder / "run_01_seed_20261128" / "final.json")
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (result.get("phase"), result.get("variant"), result.get("seed"),
            result.get("completed_epochs"), result.get("probe_count")) == (
            "smoke", "b2x8", 20261128, 1000, 20
        ):
            return True
    return False


def tee_process(command: list[str], logfile: Path) -> int:
    with logfile.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait()
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential E3 B0/B2x8 formal add-on 2-seed run")
    parser.add_argument("--dry-run", action="store_true", help="Preview four add-on runs without training")
    args = parser.parse_args()
    if not TRAIN.is_file() or not SUMMARIZER.is_file():
        parser.error("Place this script alongside train_e3.py and summarize_e3.py")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if not args.dry_run and not smoke_exists():
        parser.error("Completed Seed 20261128 B2x8 smoke result not found under outputs/. "
                     "Do not launch formal runs until the 1000-epoch smoke is confirmed.")

    # The original six official runs form a frozen protocol/code reference.
    found = {}
    for old_seed in OLD_SEEDS:
        for old_variant in ("b0", "b2x8"):
            reference = completed(old_seed, old_variant)
            if reference is None:
                parser.error(f"Missing original reference: {old_seed}/{old_variant}. Do not launch add-on runs.")
            found[(old_seed, old_variant)] = reference
    for seed, variant in PLAN:
        value = completed(seed, variant)
        if value:
            found[(seed, variant)] = value
    check_consistency(found)
    print("=== E3 additional 2-seed fixed formal plan ===")
    for i, (seed, variant) in enumerate(PLAN, start=1):
        status = "SKIP (verified complete)" if (seed, variant) in found else "RUN"
        print(f"{i}/4  seed={seed}  variant={variant:5}  {status}")
    if args.dry_run:
        return 0

    session_dir = OUTPUTS / "e3_extra_batch_logs" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_pid{os.getpid()}")
    session_dir.mkdir(parents=True, exist_ok=False)
    print(f"Batch logs: {session_dir}", flush=True)
    manifest = {"fixed_plan": [{"seed": s, "variant": v} for s, v in PLAN],
                "device": GPU, "session_dir": str(session_dir), "runs": []}

    def save_manifest() -> None:
        temp = session_dir / "manifest.tmp"
        temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temp.replace(session_dir / "manifest.json")

    try:
        for index, (seed, variant) in enumerate(PLAN, start=1):
            # Recheck after each subprocess and before each subsequent run.
            current = completed(seed, variant)
            if current:
                found[(seed, variant)] = current
                check_consistency(found)
                print(f"\n[{index}/4] SKIP completed seed={seed} {variant}: {current['folder']}", flush=True)
                manifest["runs"].append({"seed": seed, "variant": variant,
                                         "status": "verified_complete", "output": current["folder"]})
                save_manifest()
                continue
            print(f"\n[{index}/4] START seed={seed} {variant} (1000 epochs, 20 probes)", flush=True)
            command = [sys.executable, "-u", str(TRAIN), "--variant", variant,
                       "--phase", "formal", "--seed", str(seed), "--device", GPU]
            log_path = session_dir / f"{index:02d}_{seed}_{variant}.log"
            entry = {"seed": seed, "variant": variant,
                     "status": "running", "log": str(log_path)}
            manifest["runs"].append(entry)
            save_manifest()
            status = tee_process(command, log_path)
            if status != 0:
                entry["status"] = "failed"
                entry["exit_code"] = status
                save_manifest()
                raise RuntimeError(f"Run {index}/4 failed (exit code {status}); STOP. "
                                   f"See {log_path}. Re-run this script to resume.")
            current = completed(seed, variant)
            if current is None:
                entry["status"] = "invalid_output"
                save_manifest()
                raise RuntimeError(f"Run {index}/4 exited 0 but outputs failed verification; STOP.")
            found[(seed, variant)] = current
            check_consistency(found)
            entry["status"] = "complete"
            entry["output"] = current["folder"]
            save_manifest()
            print(f"[{index}/4] VERIFIED COMPLETE: {current['folder']}", flush=True)

        print("\n=== All four add-on E3 formal runs complete; combined five-seed audit ===", flush=True)
        summary_command = [sys.executable, "-u", str(SUMMARIZER),
                           "--root", str(OUTPUTS), "--seeds", "20261129,20261130,20261201,20261202,20261203"]
        summary_status = tee_process(summary_command, session_dir / "paired_summary.log")
        if summary_status != 0:
            raise RuntimeError("Four add-on runs succeeded, but five-seed summarize_e3.py failed. "
                               "Inspect paired_summary.log; no data was deleted.")
        manifest["status"] = "completed"
        save_manifest()
        print(f"\nDONE. Batch logs and summary: {session_dir}", flush=True)
        return 0
    except (RuntimeError, KeyboardInterrupt) as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = str(exc)
        save_manifest()
        print(f"\nSTOP: {exc}", file=sys.stderr, flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    sys.exit(main())
