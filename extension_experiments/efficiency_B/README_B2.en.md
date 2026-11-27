# B2: Batched Matrix Multiplication

[繁體中文](README_B2.md) · **English**

[Extension overview](../README.en.md) · [B overview](README.en.md) · [B evidence](../../research_results/B_computational_efficiency/) · [B1](README_B1.en.md)

## 1. Optimization

[model_b2.py](model_b2.py) retains B1's normalization reuse and combines multiple independent similarity multiplications into `torch.bmm` (batched matrix multiplication). For batch size $b=256$, feature dimension $D$, and $g$ groups per call ($g\in\{2,4,8\}$), multiplying $[g,b,D]$ by $[g,D,b]$ yields $[g,b,b]$ similarities, not a cross-batch $[gb,gb]$ matrix.

Each batch retains its own positives and negatives, original batch size, and independent directional `randperm` draws. An incomplete final batch is processed separately without padding or dropping nodes. Coauthor-Phy has 34,493 nodes: 134 full batches plus a tail of 189. The mathematical objective is unchanged from B1, but floating-point accumulation order may differ.

## 2. Numerical checks

[verify_b2.py](verify_b2.py) checks group sizes 2, 4, and 8. Fixed-input loss and input gradients pass for node counts 64, 256, 257, 1025, and 34493; model parameter-gradient tests with VR cover the first four sizes. The tolerances are `rtol=1e-4` and `atol=1e-5`. Both directional permutations and RNG state are checked for agreement. Passing these tests does not establish identical full training trajectories.

Source: [equivalence_b2.json](../../research_results/B_computational_efficiency/20261120_181453_B2_verify_seed_20261123/equivalence_b2.json).

## 3. Unified benchmark

Coauthor-Phy, RTX 2070, PyTorch 2.3.0+cu121; ten warmup and 20 timed steps per trial, three rounds, no profiler, excluding linear evaluation and preprocessing. P90 denotes the 90th-percentile step time, distinct from the mean and median.

| Variant | Median (s) | Mean (s) | P90 (s) | Reduction vs B0 |
| --- | --- | --- | --- | --- |
| B0 | 1.6023 | 1.5497 | 1.8779 | 0.00% |
| B1 | 0.9525 | 1.1317 | 1.3768 | 40.56% |
| B2x2 | 0.5145 | 0.6998 | 0.9792 | 67.89% |
| B2x4 | 0.4324 | 0.6235 | 0.9026 | 73.01% |
| B2x8 | 0.3642 | 0.5540 | 0.8456 | 77.27% |

B2×8 reduces pooled median time by 77.27% relative to B0 (about 4.40× speedup). Its median is faster in all three paired rounds. In the earlier five-warmup test, B2×8's first-round median was 2.3194 s versus B0's 1.6247 s, about 42.76% slower. Both the earlier and main records are retained. Peak allocated memory is around 5555 MiB, without a clear reduction.

Sources: [comparison_b012.json](../../research_results/B_computational_efficiency/20261120_183748_B012_compare/comparison_b012.json), [five-warmup benchmark](../../research_results/B_computational_efficiency/20261120_181707_B012_compare/comparison_b012.json).

## 4. Entry points

```bash
python extension_experiments/efficiency_B/verify_b2.py --help
python extension_experiments/efficiency_B/benchmark_b012.py --help
```

B2×8 advances to [E3 full training](README_E3.en.md). A 4.40× short-step speedup must not substitute for the approximately 1.637× full-training measurement. Raw outputs and logs remain local; curated verification and comparison JSON files are in the research results directory.
