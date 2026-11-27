# E3: Full Training of B0 and B2×8

[繁體中文](README_E3.md) · **English**

[Extension overview](../README.en.md) · [B overview](README.en.md) · [B evidence](../../research_results/B_computational_efficiency/) · [Two additional seeds](README_E3_extra_2seeds.en.md)

## 1. Design

Short-run benchmarks cannot establish full-training time or representation quality. E3 compares B0 (original contrastive loss) with B2×8 (batched multiplication plus normalization reuse) under A's VR configuration. Both retain the training duration, linear-probe count, and validation-only selection protocol; each method has five formal runs. B1, B2×2, and B2×4 are not part of this formal full-training comparison.

| Setting | Value |
| --- | --- |
| Dataset | Coauthor-Phy |
| Device | cuda:0 |
| Seed | 20261129–20261203 |
| Epoch / repeat | 1000 / 1 per seed |
| Eval interval / epochs | 50 / 3000 |
| Optimizer / lr / weight decay | Adam / 0.01 / 0.001 |
| Eval optimizer / lr / weight decay | Adam / 0.01 / 0 |
| Temperature | 0.5 |
| Edge add / drop (1, 2) | 0.1 / (0.1, 0.4) |
| Feature drop (1, 2) | (0, 0) |
| Feature / projection hidden dim | 256 / 256 |
| Batch compute | true |
| VR λ / γ | 10 / 0.002 (both variants) |
| Shadow stop | false |
| Loss log / save model every | 1 / 0 |

Fixed splits use 10/10/80 proportions. Recorded `split_sha256` values match within each pair. The historical expected GPU is an RTX 2070; consult each `environment.json` for the recorded software and hardware environment. Numeric seeds are RNG identifiers, not experiment dates.

## 2. Formal results

| Seed | B0 (%) | B2×8 (%) | Δ (pp) | B0 time (s) | B2×8 time (s) |
| --- | --- | --- | --- | --- | --- |
| 20261129 | 93.839 | 92.024 | -1.816 | 3350.55 | 2051.94 |
| 20261130 | 94.202 | 94.590 | +0.388 | 3371.34 | 2045.17 |
| 20261201 | 94.213 | 94.398 | +0.185 | 3392.03 | 2012.10 |
| 20261202 | 94.311 | 94.822 | +0.511 | 3316.92 | 2093.47 |
| 20261203 | 93.807 | 94.176 | +0.370 | 3418.74 | 2093.33 |
| Mean ± SD | 94.074 ± 0.233 | 94.002 ± 1.131 | -0.072 | 3369.92 | 2059.20 |

The last row reports accuracy as mean ± sample SD and time as the mean across five runs. Timing and percentage-point differences are descriptive statistics of the paired records.

| Mean timing metric | B0 | B2×8 | Time reduction |
| --- | --- | --- | --- |
| Elapsed time (s) | 3369.92 | 2059.20 | 38.89% |
| Main-model time (s) | 3275.21 | 1965.29 | 39.99% |
| Linear-probe time (s) | 94.53 | 93.67 | 0.92% |

- Full-training speedup is about 1.637×; main-model time falls by about 39.99%. Its timing scope differs from the short-run benchmark.
- B2×8 has higher accuracy in four of five pairs, but loses approximately 1.816 pp on the first seed. The mean difference is −0.072 pp and sample SD increases from 0.233 to 1.131.
- Mean feature std across five runs: 0.008577 for B0 and 0.011435 for B2×8. Mean pairwise cosine: 0.607347 and 0.607306. Maximum peak allocated memory: approximately 5836.76 and 5838.67 MiB.

These results support faster full training and similar mean accuracy, but not bitwise equivalence, identical training quality, or lower GPU memory use. No statistical equivalence test establishes that the two methods are equivalent.

## 3. Implementation and diagnostics

| File | Purpose |
| --- | --- |
| [train_e3.py](train_e3.py) | Full training and per-epoch timing |
| [run_e3_formal_batch.py](run_e3_formal_batch.py) | Historical initial formal batch |
| [run_e3_extra_2seeds.py](run_e3_extra_2seeds.py) | Additional formal seeds |
| [summarize_e3.py](summarize_e3.py) | Result aggregation |
| [diagnose_e3_phases.py](diagnose_e3_phases.py) | Phase timing diagnostics |
| [diagnose_e3_parity_control.py](diagnose_e3_parity_control.py) | Training parity control |
| [diagnose_e3_first_divergence.py](diagnose_e3_first_divergence.py) / [diagnose_e3_first_step_pipeline.py](diagnose_e3_first_step_pipeline.py) | First divergence and first-step pipeline |
| [diagnose_e3_conv1_components.py](diagnose_e3_conv1_components.py) / [diagnose_e3_aggregation_determinism_control.py](diagnose_e3_aggregation_determinism_control.py) | First GCN layer and aggregation determinism diagnostics |
| [run_e3_20261129_replay.py](run_e3_20261129_replay.py) | Same-seed replay |

Diagnostic and replay evidence is retained in [curated B results](../../research_results/B_computational_efficiency/), including `e3_same_seed_replay_20261129/`. This document does not infer a unique cause of numerical divergence from diagnostic results that were not individually reviewed, nor treat replay as additional independent formal seeds. Raw batch-log directories are no longer committed.

## 4. Usage and output

```bash
python extension_experiments/efficiency_B/train_e3.py --help
```

Run from the repository root, reconstructing each recorded `args.json` configuration. Check the installed CLI, hardware restrictions, dataset location, and output settings before launching historical batch entry points. Raw outputs are under `extension_experiments/efficiency_B/outputs/`; weights and fixed splits remain local. Formal evidence includes `args.json`, `environment.json`, `source_sha256.json`, `summary.json`, and each run's `final.json`, `epoch_timing.csv`, `loss.csv`, and `validation.csv`.

## 5. Formal evidence index (ten top-level runs only)

| Seed | B0 final.json | B2×8 final.json |
| --- | --- | --- |
| 20261129 | [20261120_222038_490934_E3_formal_b0_Coauthor-Phy_seed_20261129](../../research_results/B_computational_efficiency/20261120_222038_490934_E3_formal_b0_Coauthor-Phy_seed_20261129/run_01_seed_20261129/final.json) | [20261120_231652_685412_E3_formal_b2x8_Coauthor-Phy_seed_20261129](../../research_results/B_computational_efficiency/20261120_231652_685412_E3_formal_b2x8_Coauthor-Phy_seed_20261129/run_01_seed_20261129/final.json) |
| 20261130 | [20261121_002556_627724_E3_formal_b0_Coauthor-Phy_seed_20261130](../../research_results/B_computational_efficiency/20261121_002556_627724_E3_formal_b0_Coauthor-Phy_seed_20261130/run_01_seed_20261130/final.json) | [20261120_235127_223317_E3_formal_b2x8_Coauthor-Phy_seed_20261130](../../research_results/B_computational_efficiency/20261120_235127_223317_E3_formal_b2x8_Coauthor-Phy_seed_20261130/run_01_seed_20261130/final.json) |
| 20261201 | [20261121_012230_707333_E3_formal_b0_Coauthor-Phy_seed_20261201](../../research_results/B_computational_efficiency/20261121_012230_707333_E3_formal_b0_Coauthor-Phy_seed_20261201/run_01_seed_20261201/final.json) | [20261121_021927_029066_E3_formal_b2x8_Coauthor-Phy_seed_20261201](../../research_results/B_computational_efficiency/20261121_021927_029066_E3_formal_b2x8_Coauthor-Phy_seed_20261201/run_01_seed_20261201/final.json) |
| 20261202 | [20261121_163628_703921_E3_formal_b0_Coauthor-Phy_seed_20261202](../../research_results/B_computational_efficiency/20261121_163628_703921_E3_formal_b0_Coauthor-Phy_seed_20261202/run_01_seed_20261202/final.json) | [20261121_160112_086020_E3_formal_b2x8_Coauthor-Phy_seed_20261202](../../research_results/B_computational_efficiency/20261121_160112_086020_E3_formal_b2x8_Coauthor-Phy_seed_20261202/run_01_seed_20261202/final.json) |
| 20261203 | [20261121_173211_732025_E3_formal_b0_Coauthor-Phy_seed_20261203](../../research_results/B_computational_efficiency/20261121_173211_732025_E3_formal_b0_Coauthor-Phy_seed_20261203/run_01_seed_20261203/final.json) | [20261121_182935_626683_E3_formal_b2x8_Coauthor-Phy_seed_20261203](../../research_results/B_computational_efficiency/20261121_182935_626683_E3_formal_b2x8_Coauthor-Phy_seed_20261203/run_01_seed_20261203/final.json) |

Source: [manifest.csv](../../research_results/manifest.csv). The `final.json` files directly support the timing and accuracy values; the corresponding experiment directories contain `args.json`. Sanity runs, smoke tests, diagnostics, and replay are excluded from formal statistics. Full-training commands have not been rerun in a clean clone.
