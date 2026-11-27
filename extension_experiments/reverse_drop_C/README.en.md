# Experiment C: Reverse Edge Dropping and EPR Analysis

[繁體中文](README.md) · **English**

[Extension overview](../README.en.md) · [Evidence](../../research_results/C_reverse_edge_dropping/)

## 1. Research question

EPAGCL's degree-guided dropping favors edges with particular structural properties. C investigates whether reversing the drop-weight ranking keeps augmented-graph EPR (Error Passing Rate, following the graph metric implemented in the original `EPR.py`) closer to that of the original graph and whether downstream node classification improves. It separates paired training from independent offline EPR analysis, without A's VR or B's computational optimization.

## 2. Reverse dropping and edge alignment

For an existing edge $(u,v)$, let $d_u,d_v$ be node degrees in the undirected graph and $\bar w$ the mean edge weight. Baseline and Reverse Drop use the following normalized scores:

$$
w_{uv}=\frac{1}{\sqrt{d_ud_v}},\quad
q_{uv}^{\mathrm{base}}=\frac{w_{uv}-w_{\min}}{\bar w-w_{\min}},\quad
q_{uv}^{\mathrm{rev}}=\frac{w_{\max}-w_{uv}}{w_{\max}-\bar w}.
$$

Reverse Drop reverses the score ranking and retains the edge-addition process. Equal nominal drop rates do not guarantee equal realized deletion rates after undirecting. The implementation computes weights using the canonical undirected graph, then aligns them to the original sampling index using ordered endpoint pair mapping, preserving the original sampling order. On Cora, both original and canonical edge indices contain 10,556 entries with no duplicates or missing reverse entries, but their ordering differs. Both C variants use the alignment correction; C's Baseline is therefore not an exact replica of the historical uncorrected implementation. The isolated effect of the previous misalignment on accuracy has not been quantified.

## 3. Paired node classification

| Setting | Value |
| --- | --- |
| Dataset | Cora |
| Device | cuda:0 |
| Seed / repeat | 20261119 / 5 |
| Epoch / eval interval / eval epochs | 500 / 50 / 3000 |
| Optimizer / lr / weight decay | Adam / 0.001 / 0.0001 |
| Eval optimizer / lr / weight decay | Adam / 0.01 / 0 |
| Temperature | 0.3 |
| Edge add / drop (1, 2) | 0.1 / (0.2, 0.3) |
| Feature drop (1, 2) | (0.1, 0.1) |
| Feature / projection hidden dim | 256 / 256 |
| Batch compute | true |
| Loss log / save model every | 1 / 0 |
| VR | not used |

Fixed 10/10/80 splits, validation-only checkpoint selection, and test evaluation after selection. No VR, shadow monitoring, or early stopping. Both variants use five seeds, 20261119–20261123.

| Seed | Baseline (%) | Reverse Drop (%) | Δ (pp) |
| --- | --- | --- | --- |
| 20261119 | 82.749 | 82.149 | -0.600 |
| 20261120 | 82.242 | 82.426 | +0.185 |
| 20261121 | 83.902 | 83.256 | -0.646 |
| 20261122 | 83.533 | 82.426 | -1.107 |
| 20261123 | 83.210 | 83.441 | +0.231 |
| Mean ± SD | 83.127 ± 0.652 | 82.740 ± 0.571 | -0.387 |

The mean difference is −0.387 pp, with no observed mean accuracy improvement. Sample standard deviation is recomputed from the five test values; original summaries use population standard deviation.

Sources: [comparison.csv](../../research_results/C_reverse_edge_dropping/training/20261125_012440_810530_both_Cora/comparison.csv), [summary.json](../../research_results/C_reverse_edge_dropping/training/20261125_012440_810530_both_Cora/summary.json), [args.json](../../research_results/C_reverse_edge_dropping/training/20261125_012440_810530_both_Cora/args.json), [edge_alignment.json](../../research_results/C_reverse_edge_dropping/training/20261125_012440_810530_both_Cora/edge_alignment.json).

## 4. Offline EPR analysis

Following the original [EPR.py](../../EPR.py), EPR is averaged over non-isolated nodes after undirecting the graph. Let $V_+$ be the non-isolated nodes, $\mathcal N(u)$ the neighbors of node $u$, and $y_u$ its class label. The metric uses a degree-weighted fraction of cross-class edges:

$$
\mathrm{EPR}(G)=\frac{1}{|V_+|}\sum_{u\in V_+}
\frac{\sum_{v\in\mathcal N(u)}\mathbf 1[y_u\ne y_v]/\sqrt{d_ud_v}}
{\sum_{v\in\mathcal N(u)}1/\sqrt{d_ud_v}}.
$$

True labels are used only for the offline metric, not for augmentation sampling or self-supervised training. This analysis is not a replay of the training RNG trajectory. Original Cora EPR is 0.168300, with zero isolated nodes. The formal CPU analysis uses five seeds and 500 samples per seed, giving 2500 valid pairs per stage; a separate ten-sample single-seed smoke test is retained. Nominal addition is 0.1 and drop rates are 0.2/0.3; recorded addition sampling uses p=0.5 and threshold=0.9.

| Stage | Pairs | Baseline EPR | Reverse Drop EPR | Δ |
| --- | --- | --- | --- | --- |
| view1_drop | 2500 | 0.176169 | 0.168368 | -0.007800 |
| view1_final | 2500 | 0.177468 | 0.169595 | -0.007874 |
| view2_drop | 2500 | 0.181466 | 0.168486 | -0.012980 |

`view1_drop` is view 1 after dropping; `view1_final` is view 1 after dropping and adding; `view2_drop` is view 2 after dropping. The following table also reports mean absolute EPR change from the original, isolates, and realized undirected drop rates. B means Baseline and R means Reverse Drop. The final add stage has no separate drop-rate entry (—).

| Stage | B mean abs ΔEPR | R mean abs ΔEPR | B isolates | R isolates | B undirected drop | R undirected drop |
| --- | --- | --- | --- | --- | --- | --- |
| view1_drop | 0.007869 | 0.000778 | 194.5500 | 8.2356 | 6.133% | 4.155% |
| view1_final | 0.009169 | 0.001396 | 194.5500 | 8.2356 | — | — |
| view2_drop | 0.013167 | 0.001219 | 341.9652 | 20.2464 | 12.624% | 9.335% |

Reverse dropping keeps EPR closer to the original graph and reduces isolated nodes. Mean EPR at all three stages is still slightly above the original. Realized directed-entry deletion rates are broadly similar, but undirected deletion rates differ, and EPR excludes isolated nodes. The observations cannot be attributed solely to the reversed ranking. Smaller graph perturbations do not necessarily improve classification.

Sources: [formal comparison](../../research_results/C_reverse_edge_dropping/epr_analysis/20261125_020958_891111_Cora_original_epr/comparison.csv), [summary.csv](../../research_results/C_reverse_edge_dropping/epr_analysis/20261125_020958_891111_Cora_original_epr/summary.csv), [per-sample data](../../research_results/C_reverse_edge_dropping/epr_analysis/20261125_020958_891111_Cora_original_epr/epr_samples.csv), [seed summaries](../../research_results/C_reverse_edge_dropping/epr_analysis/20261125_020958_891111_Cora_original_epr/seed_summary.csv), [original_epr.json](../../research_results/C_reverse_edge_dropping/epr_analysis/20261125_020958_891111_Cora_original_epr/original_epr.json), and [alignment](../../research_results/C_reverse_edge_dropping/epr_analysis/20261125_020958_891111_Cora_original_epr/edge_alignment.json).

## 5. Implementation and usage

| File | Purpose |
| --- | --- |
| [augmentation_c.py](augmentation_c.py) | Weights, endpoint alignment, reverse dropping |
| [run.py](run.py) | Paired training |
| [epr_analysis.py](epr_analysis.py) | Independent offline EPR analysis |
| [run_cora.sh](run_cora.sh) | Historical Cora launch script |
| [../fixed_eval.py](../fixed_eval.py) | Shared fixed splits and linear evaluation |

Run from the repository root. Reconstruct training settings from the table and `args.json`, checking the installed CLI. Historical scripts may contain machine-specific restrictions or paths. The following offline analysis entry point does not require a pretrained model.

```bash
python extension_experiments/reverse_drop_C/run.py --help
python extension_experiments/reverse_drop_C/epr_analysis.py --help

python extension_experiments/reverse_drop_C/epr_analysis.py \
  --dataset Cora --device cpu --samples 500 --repeat 5 --seed 20261119 \
  --edge-add-rate 0.1 --edge-drop-rate-1 0.2 --edge-drop-rate-2 0.3
```

Training writes to `extension_experiments/reverse_drop_C/outputs/` and offline analysis to `extension_experiments/reverse_drop_C/epr_outputs/`. Raw outputs, logs, weights, and split `.pt` files remain local. GitHub retains [curated evidence](../../research_results/C_reverse_edge_dropping/) and [manifest.csv](../../research_results/manifest.csv). Multi-dataset tests and a controlled comparison at matched realized deletion rates remain unperformed.
