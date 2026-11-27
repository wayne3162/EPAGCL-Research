# EPAGCL Research Extensions

[繁體中文](README.md) · **English**

These experiments extend EPAGCL along three research directions: representation stability, computational efficiency, and the relationship between edge perturbation and EPR. A, B, and C are separately testable studies. B retains A's variance regularization (VR); C compares methods without VR. A combined A+B+C training study has not been performed.

## Experiment index

| Study | Research question | Formal observation | Documentation |
| --- | --- | --- | --- |
| A: Variance regularization (VR) | Do representations approach collapse? | Coauthor-Phy: 59.148 ± 19.087% → 94.046 ± 0.340% | [中文](formal_A/README.md) / [EN](formal_A/README.en.md) |
| B: Computational efficiency | Can we reduce computation while retaining the contrastive objective? | B2×8 reduces short-step pooled median time by 77.27% and mean full-run time by 38.89% | [中文](efficiency_B/README.md) / [EN](efficiency_B/README.en.md) |
| C: Reverse edge dropping | Does reversing the drop ranking preserve original-graph EPR? | Cora: EPR is closer to the original graph, with fewer isolates; no mean accuracy gain | [中文](reverse_drop_C/README.md) / [EN](reverse_drop_C/README.en.md) |

## Shared evaluation protocol

A, B, and C use seeded, fixed 10% / 10% / 80% train/validation/test node splits. The encoder is frozen for linear evaluation. Training labels fit the classifier; validation selects the classifier and encoder checkpoints; test evaluation occurs once after selection. Formal comparisons report test accuracy (%) and sample standard deviation. A difference in percentage points (pp) is not a relative percentage change. Historical Q3 uses a different evaluation protocol and is not pooled with these results.

## Code and data layout

```text
extension_experiments/
├── run.py                      # A training entry point
├── fixed_eval.py               # Shared fixed splits and linear evaluation
├── vr_regularization.py        # VR loss used by A and B
├── tests/test_basic.py
├── formal_A/                   # A docs, formal script, historical source hashes
│   ├── outputs/                # Local only
│   └── logs/                   # Local only
├── efficiency_B/               # B1, B2, benchmarks, E3 training and diagnostics
│   └── outputs/                # Local only
├── reverse_drop_C/             # C training and offline EPR analysis
│   ├── outputs/                # Local only
│   └── epr_outputs/            # Local only
└── splits/                     # Shared fixed splits; .pt is local only
```

GitHub contains code, documentation, and [curated research evidence](../research_results/). A now writes to `formal_A/outputs/`; B and C retain their own output roots. All `logs/`, `log/`, `*_logs/`, `*_log/` directories, `.log` files (including rotated logs), raw outputs, weights, and split tensors remain local. Structured evidence such as `loss.csv` and `validation.csv` is retained. See [manifest.csv](../research_results/manifest.csv) for source paths and SHA-256 digests. The [export tool](../scripts/export_epagcl_results.py) previews or creates a new export; it refuses to overwrite an existing results directory.

## Entry points

Run from the repository root in the configured Python environment. Datasets are normally stored outside the repository in `../datasets/`. See the [upstream README](../README_upstream.md) and [project README](../README.md) for environment guidance. B's benchmark records use an RTX 2070 and PyTorch 2.3.0+cu121; the timings do not guarantee performance on other hardware.

```bash
python extension_experiments/run.py --help
python extension_experiments/efficiency_B/train_e3.py --help
python extension_experiments/reverse_drop_C/run.py --help
python extension_experiments/reverse_drop_C/epr_analysis.py --help
python -m unittest discover -s extension_experiments/tests -v
```

Clones contain no `.pt`, `.pth`, `.ckpt`, or `.safetensors` files. Recreate fixed splits and required weights when rerunning; identical seeds do not guarantee bitwise reproducibility across software versions, GPUs, or node ordering. See each experiment's documentation for settings, formal seeds, and evidence. Historical hashes identify the original source versions and are not rewritten when output directories move.
