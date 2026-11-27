# Isolated EPAGCL extension experiments (2026-11-19)

This folder belongs **inside the existing EPAGCL project root**:

```text
EPAGCL/
  main.py                    # ORIGINAL: unchanged
  Linear_evaluation.py       # ORIGINAL: unchanged
  Model.py                   # ORIGINAL: unchanged
  Augmentation.py            # ORIGINAL: unchanged
  Load_data.py               # ORIGINAL: unchanged
  Utils.py                   # ORIGINAL: unchanged
  extension_experiments/     # NEW: all extension source and results
    run.py
    fixed_eval.py
    vr_regularization.py
    tests/
    splits/
    formal_A/outputs/
```

The runner imports the original `Model.py`, `Augmentation.py`, `Load_data.py`,
and `Utils.py` directly, without editing or copying them. It never imports the
original `Arguments.py` (which has immediate folder-writing side effects) or
original `Linear_evaluation.py` (which selects checkpoints using test labels).
New results are ONLY written to `extension_experiments/formal_A/outputs/` and splits to
`extension_experiments/splits/`. Pre-existing dataset caches are read through
the same `Load_data.py` API used by the original script.

## A. Installation

No new Python packages are needed. Continue using your existing WSL `uv`
environment and the dependencies that already run the original EPAGCL project.
Extract this folder into `~/projects/EPAGCL/` so that `run.py` is found at
`~/projects/EPAGCL/extension_experiments/run.py`.

## B. Check existing environment (no model training)

```bash
cd ~/projects/EPAGCL
python -m unittest discover -s extension_experiments/tests -v
python extension_experiments/run.py --help
```

## C. Stage 1 smoke test: fixed evaluation, baseline EPAGCL

```bash
cd ~/projects/EPAGCL
python extension_experiments/run.py \
  --method baseline --dataset Coauthor-Phy \
  --epoch 20 --repeat 1 --eval 10 --eval-epoch 30 \
  --seed 20261119
```

This is only a functionality smoke test, **NOT** an accuracy result.
It does not modify original project files or past experimental outputs.

## D. Stage 2 baseline (later, after smoke-test verification)

```bash
python extension_experiments/run.py \
  --method baseline --dataset Coauthor-Phy \
  --epoch 1000 --repeat 5 --eval 50 --eval-epoch 3000 \
  --seed 20261119
```

Five seeds will be `20261119` through `20261123` and the same split indices
can be reused by the VR method. **This is NOT run automatically.**

## E. VR-EPAGCL (only after choosing VR parameters without test labels)

```bash
python extension_experiments/run.py \
  --method vr --vr-lambda YOUR_LAMBDA --vr-gamma YOUR_GAMMA \
  --dataset Coauthor-Phy \
  --epoch 20 --repeat 1 --eval 10 --eval-epoch 30 \
  --seed 20261119
```

Replace `YOUR_LAMBDA` and `YOUR_GAMMA` with separately validated positive
numeric values. No VR parameter values are silently guessed in this package.
VR penalizes low feature-wise standard deviation in the GCN encoder output of
both augmented views. This is an experimental change to the loss function.

## F. Shadow early-stop monitoring

The historical Coauthor-Physics candidate checks **raw contrastive loss** at
encoder epochs 50/75/100 against 6.22. Shadow mode only logs a recommendation.
It **never stops** training. This rule is not independently verified on new
seeds and cannot be claimed as an actual speedup yet. For VR, its relevance
may differ even when only raw contrastive loss is monitored.

## G. Corrected evaluation protocol

1. Each run uses a saved, fixed 10% train / 10% valid / 80% test split.
2. Every linear probe is trained on train nodes and selected on validation
   nodes only. Validation ties keep the first best classifier epoch.
3. The graph encoder checkpoint is selected on validation accuracy only.
4. **Held-out test labels are accessed once per run, after encoder and probe
   selection are complete.** Test accuracy is never used for selection.
5. Baseline and VR can share run seeds and split files for paired comparison.
6. Existing historical results use a different evaluation protocol and must
   not be merged directly with this protocol's new test statistics.

## H. Outputs and reproducibility

Each new run directory has:

- `args.json`, `original_source_sha256.json`
- `run_XX_seed_.../loss.csv` (raw contrastive, variance and total loss)
- `run_XX_seed_.../validation.csv` (validation scores only)
- `run_XX_seed_.../best_by_validation.pt` (selected encoder + classifier)
- `run_XX_seed_.../final.json` (one final held-out test result)
- `summary.json` and intermediate `partial_summary.json`

The original model can be nondeterministic on CUDA for some operations; the
seed controls known RNGs but does not guarantee bitwise reproducibility.

## I. Scope and limitations

- Only the EPAGCL single-add-view augmentation is implemented; it is not a
  replacement for the six-strategy Q3 experiment suite.
- The active early-stop method is intentionally NOT implemented because the
  retrospective 30-run check has not been independently validated.
- The original `Model.MyModel.loss(..., batch_compute=True)` is reused as-is.
- Exact VR effect, optimal lambda/gamma and speedup remain unverified.
- Always keep both methods' test sets held out until final selection.
