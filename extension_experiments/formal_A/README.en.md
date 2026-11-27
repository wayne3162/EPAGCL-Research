# Experiment A: Variance Regularization and Representation Collapse

[繁體中文](README.md) · **English**

[Extension overview](../README.en.md) · [Experiment B](../efficiency_B/README.en.md) · [Evidence](../../research_results/A_variance_regularization/)

## 1. Research question

Some historical Coauthor-Phy ablation runs yielded accuracy near 50%. A investigates whether encoder representations become nearly identical across nodes, rather than diagnosing representation collapse from accuracy alone. The goal is to increase across-node variation and improve training stability while retaining EPAGCL's encoder and contrastive learning architecture.

## 2. Method

Baseline retains the original contrastive loss. VR adds variance regularization to the two augmented-view encoder outputs `out1` and `out2`, not the projected representations `z1` and `z2`. Let $H^{(v)}\in\mathbb{R}^{N\times D}$ be encoder output for view $v$, $N$ the number of nodes, $D$ the feature dimension, and $\bar H_j^{(v)}$ the mean of feature $j$. Here $\epsilon=10^{-8}$, $\gamma$ is a standard-deviation floor, and $\lambda$ weights the penalty.

$$
\sigma_j^{(v)}=\sqrt{\frac{1}{N}\sum_{i=1}^N (H_{ij}^{(v)}-\bar H_j^{(v)})^2+\epsilon}
$$

$$
\mathcal L_{\mathrm{VR}}=\frac{1}{2D}\sum_{v=1}^{2}\sum_{j=1}^{D}\max(0,\gamma-\sigma_j^{(v)}),\qquad
\mathcal L_{\mathrm{total}}=\mathcal L_{\mathrm{contrastive}}+\lambda\mathcal L_{\mathrm{VR}}.
$$

A feature incurs a penalty only if its standard deviation is below $\gamma$. The implementation uses population variance (`unbiased=False`). VR uses no class labels and does not change the original edge-augmentation strategy. Shadow monitoring records potential early high-loss stagnation without stopping training; the historical threshold of 6.22 is not a universally validated rule. Formal comparisons disable this monitoring.

## 3. Formal design

| Setting | Value |
| --- | --- |
| Dataset | Coauthor-Phy |
| Device | cuda:0 |
| Seed | 20261123–20261127 |
| Epoch / repeat | 1000 / 1 per seed |
| Eval interval / epochs | 50 / 3000 |
| Optimizer / lr / weight decay | Adam / 0.01 / 0.001 |
| Eval optimizer / lr / weight decay | Adam / 0.01 / 0 |
| Temperature | 0.5 |
| Edge add / drop (1, 2) | 0.1 / (0.1, 0.4) |
| Feature drop (1, 2) | (0, 0) |
| Feature / projection hidden dim | 256 / 256 |
| Batch compute | true |
| VR λ / γ | 10 / 0.002 (VR only) |
| Shadow stop | false |
| Loss log / save model every | 1 / 100 |

Both methods use paired seeds and fixed 10/10/80 train/validation/test splits. Linear evaluation occurs every 50 epochs (20 probes). Validation selects classifier and encoder checkpoints; the test set is evaluated only after selection. This evaluation protocol differs from historical Q3, so their measurements are not pooled.

## 4. Formal results (five paired seeds)

| Seed | Baseline (%) | VR (%) | VR − Baseline (pp) |
| --- | --- | --- | --- |
| 20261123 | 50.632 | 94.437 | +43.805 |
| 20261124 | 50.527 | 94.224 | +43.696 |
| 20261125 | 93.292 | 94.151 | +0.859 |
| 20261126 | 50.661 | 93.850 | +43.189 |
| 20261127 | 50.629 | 93.568 | +42.939 |
| Mean ± SD | 59.148 ± 19.087 | 94.046 ± 0.340 | +34.898 |

| Diagnostic (mean across five runs) | Baseline | VR |
| --- | --- | --- |
| Mean feature std | 0.002348 | 0.009197 |
| Mean pairwise cosine | 0.919926 | 0.600476 |
| Elapsed time (s) | 3457.43 | 3484.91 |

Four low-accuracy Baseline runs also have mean feature standard deviations of approximately 0.000119–0.000151 and mean pairwise cosine similarity above 0.99995, supporting a diagnosis of near-collapse. The fifth Baseline run does not show this pattern. None of the five VR runs shows the same combination. Mean accuracy increases by 34.898 pp; mean elapsed training time rises by about 0.8% (excluding preprocessing). This evidence concerns Coauthor-Phy, the specified configuration, and five seeds; it is not a cross-dataset conclusion or a formally thresholded binary collapse test.

## 5. Implementation and usage

| File | Purpose |
| --- | --- |
| [../run.py](../run.py) | A training, diagnostics, and output management |
| [../vr_regularization.py](../vr_regularization.py) | VR loss |
| [../fixed_eval.py](../fixed_eval.py) | Shared splits and validation-only selection |
| [run_remaining.sh](run_remaining.sh) | Historical formal script (remaining four seeds) |
| [source_sha256.txt](source_sha256.txt) | Historical source hashes; not rewritten |
| [../tests/test_basic.py](../tests/test_basic.py) | Basic unit tests |

Run from the repository root. Check `python extension_experiments/run.py --help` against the installed version first. The following commands illustrate the first seed; substitute each remaining seed and check all recorded `args.json` settings, including Adam, dimensions, batching, and evaluation optimizer. The historical batch script is not an instruction to overwrite existing results. New runs create timestamped directories.

```bash
# Baseline
python extension_experiments/run.py --method baseline --dataset Coauthor-Phy \
  --device cuda:0 --epoch 1000 --repeat 1 --seed 20261123 \
  --eval 50 --eval-epoch 3000 --lr 0.01 --weight-decay 0.001 \
  --temperature 0.5 --edge-add-rate 0.1 --edge-drop-rate-1 0.1 \
  --edge-drop-rate-2 0.4 --feat-drop-rate-1 0 --feat-drop-rate-2 0 \
  --loss-log 1 --save-model-every 100 --no-shadow-stop

# VR
python extension_experiments/run.py --method vr --dataset Coauthor-Phy \
  --device cuda:0 --epoch 1000 --repeat 1 --seed 20261123 \
  --eval 50 --eval-epoch 3000 --lr 0.01 --weight-decay 0.001 \
  --temperature 0.5 --edge-add-rate 0.1 --edge-drop-rate-1 0.1 \
  --edge-drop-rate-2 0.4 --feat-drop-rate-1 0 --feat-drop-rate-2 0 \
  --loss-log 1 --save-model-every 100 --no-shadow-stop \
  --vr-lambda 10 --vr-gamma 0.002
```

## 6. New output location

```text
extension_experiments/formal_A/
├── outputs/<timestamp>_<method>_<dataset>/
│   ├── args.json
│   ├── original_source_sha256.json
│   ├── partial_summary.json
│   ├── summary.json
│   └── run_01_seed_<seed>/
│       ├── final.json
│       ├── loss.csv
│       ├── validation.csv
│       └── ... (weights local only, if saved)
└── logs/                      # local only
```

All raw output and log directories remain local. Curated evidence is in [research_results/A_variance_regularization/](../../research_results/A_variance_regularization/), excluding weights and fixed split `.pt` files. Shared splits remain under `extension_experiments/splits/`. Relocating outputs does not require retraining and does not rewrite historical configuration or source hashes.

## 7. Formal evidence index

Only the five formal pairs appear below. Exploratory runs remain in the evidence directory but are excluded from the formal statistics. Each `final.json` has its corresponding `args.json`, `summary.json`, and source hashes in the experiment's parent directory.

| Seed | Baseline final.json | VR final.json |
| --- | --- | --- |
| 20261123 | [20261119_183936_883401_baseline_Coauthor-Phy](../../research_results/A_variance_regularization/20261119_183936_883401_baseline_Coauthor-Phy/run_01_seed_20261123/final.json) | [20261119_193855_310924_vr_Coauthor-Phy](../../research_results/A_variance_regularization/20261119_193855_310924_vr_Coauthor-Phy/run_01_seed_20261123/final.json) |
| 20261124 | [20261119_204656_996415_baseline_Coauthor-Phy](../../research_results/A_variance_regularization/20261119_204656_996415_baseline_Coauthor-Phy/run_01_seed_20261124/final.json) | [20261119_214445_679460_vr_Coauthor-Phy](../../research_results/A_variance_regularization/20261119_214445_679460_vr_Coauthor-Phy/run_01_seed_20261124/final.json) |
| 20261125 | [20261119_224250_775434_baseline_Coauthor-Phy](../../research_results/A_variance_regularization/20261119_224250_775434_baseline_Coauthor-Phy/run_01_seed_20261125/final.json) | [20261119_234044_122133_vr_Coauthor-Phy](../../research_results/A_variance_regularization/20261119_234044_122133_vr_Coauthor-Phy/run_01_seed_20261125/final.json) |
| 20261126 | [20261120_003915_198680_baseline_Coauthor-Phy](../../research_results/A_variance_regularization/20261120_003915_198680_baseline_Coauthor-Phy/run_01_seed_20261126/final.json) | [20261120_013602_065693_vr_Coauthor-Phy](../../research_results/A_variance_regularization/20261120_013602_065693_vr_Coauthor-Phy/run_01_seed_20261126/final.json) |
| 20261127 | [20261120_023322_567168_baseline_Coauthor-Phy](../../research_results/A_variance_regularization/20261120_023322_567168_baseline_Coauthor-Phy/run_01_seed_20261127/final.json) | [20261120_033221_112832_vr_Coauthor-Phy](../../research_results/A_variance_regularization/20261120_033221_112832_vr_Coauthor-Phy/run_01_seed_20261127/final.json) |

Source: [results manifest](../../research_results/manifest.csv). Statistics are calculated from the ten formal `final.json` records with sample standard deviation. The new output path was changed and inspected, but a clean-clone full training run has not been repeated.
