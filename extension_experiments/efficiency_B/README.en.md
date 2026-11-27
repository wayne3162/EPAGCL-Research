# Experiment B: Contrastive-Loss Computational Efficiency

[繁體中文](README.md) · **English**

[Extension overview](../README.en.md) · [Experiment A](../formal_A/README.en.md) · [Evidence](../../research_results/B_computational_efficiency/)

## 1. Objective and stages

B builds on A's VR-EPAGCL training configuration. It targets repeated normalization and repeated small matrix multiplications in the contrastive loss. The goal is to retain the mathematical objective, positive/negative definitions, independent directional random batching, and training schedule, rather than claim speedups from fewer epochs, nodes, or linear probes.

| Stage | Purpose | Documentation |
| --- | --- | --- |
| B0 | Original contrastive loss with VR; timing reference | This page |
| B1 | Reuse L2-normalized projected representations | [中文](README_B1.md) / [EN](README_B1.en.md) |
| B2 | Batch matrix multiplications over independent groups | [中文](README_B2.md) / [EN](README_B2.en.md) |
| E3 | Paired full training of B0 and B2×8 | [中文](README_E3.md) / [EN](README_E3.en.md) |
| E3 additional seeds | Two additional pairs beyond the initial three; part of the five-seed analysis | [中文](README_E3_extra_2seeds.md) / [EN](README_E3_extra_2seeds.en.md) |

The historical [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) records the original plan, not a claim that all planned work was completed. These READMEs prioritize recorded results.

## 2. B0 profiling and short-run benchmarks

`profile_b0.py` profiles B0 on Coauthor-Phy with an RTX 2070 and PyTorch 2.3.0+cu121, using five warmup and 20 timed steps. Its median step time is approximately 1.7620 s and peak allocated memory is approximately 5555 MiB. The main comparison uses `benchmark_b012.py` without a profiler: ten warmup and 20 timed steps per trial, three rounds, excluding linear evaluation and preprocessing.

| Variant | Median (s) | Mean (s) | P90 (s) | Reduction vs B0 |
| --- | --- | --- | --- | --- |
| B0 | 1.6023 | 1.5497 | 1.8779 | 0.00% |
| B1 | 0.9525 | 1.1317 | 1.3768 | 40.56% |
| B2x2 | 0.5145 | 0.6998 | 0.9792 | 67.89% |
| B2x4 | 0.4324 | 0.6235 | 0.9026 | 73.01% |
| B2x8 | 0.3642 | 0.5540 | 0.8456 | 77.27% |

B2×8 achieves approximately 4.40× pooled-median speedup, which is distinct from full-training speedup. An earlier five-warmup test had one slower B2×8 round; both records are retained. Peak allocated GPU memory remains near 5555 MiB across variants, without a clear reduction.

Sources: [B0 timing.json](../../research_results/B_computational_efficiency/20261120_111127_B0_seed_20261123/timing.json), [comparison_b012.json](../../research_results/B_computational_efficiency/20261120_183748_B012_compare/comparison_b012.json), and [five-warmup comparison](../../research_results/B_computational_efficiency/20261120_181707_B012_compare/comparison_b012.json).

## 3. E3 full-training results

Five formal pairs (Coauthor-Phy, seeds 20261129–20261203), 1000 epochs and 20 linear probes. Both B0 and B2×8 retain VR.

| Metric | B0 | B2×8 |
| --- | ---: | ---: |
| Test accuracy (%), mean ± sample SD | 94.074 ± 0.233 | 94.002 ± 1.131 |
| Mean elapsed time (s) | 3369.92 | 2059.20 |
| Mean main-model time (s) | 3275.21 | 1965.29 |

Mean elapsed training time falls by 38.89% (approximately 1.637× speedup) and mean accuracy differs by −0.072 pp. B2×8 has higher across-seed standard deviation. Passing fixed-input numerical checks does not establish bitwise-identical full training trajectories or equivalent classification quality.

## 4. Entry points and data

Run from the repository root and inspect the installed CLI options first:

```bash
python extension_experiments/efficiency_B/profile_b0.py --help
python extension_experiments/efficiency_B/verify_b1.py --help
python extension_experiments/efficiency_B/benchmark_b01.py --help
python extension_experiments/efficiency_B/verify_b2.py --help
python extension_experiments/efficiency_B/benchmark_b012.py --help
python extension_experiments/efficiency_B/train_e3.py --help
```

`model_b1.py` and `model_b2.py` implement the optimizations. `train_e3.py` handles full training. `run_e3_formal_batch.py` and `run_e3_extra_2seeds.py` are historical batch entry points; `summarize_e3.py` summarizes runs; `diagnose_e3_*.py` and `run_e3_20261129_replay.py` support divergence diagnostics and same-seed replay. Inspect hardware restrictions, seeds, paths, and current settings before launching historical scripts.

Raw output is under `extension_experiments/efficiency_B/outputs/`; neither raw output nor log directories are committed. Curated evidence contains benchmarks, numerical checks, formal training, exploratory runs, and diagnostics. Formal E3 statistics exclude replay. See [manifest.csv](../../research_results/manifest.csv).
