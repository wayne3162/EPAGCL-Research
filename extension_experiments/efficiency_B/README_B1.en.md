# B1: Reusing L2-Normalized Representations

[繁體中文](README_B1.md) · **English**

[Extension overview](../README.en.md) · [B overview](README.en.md) · [B evidence](../../research_results/B_computational_efficiency/)

## 1. Optimization

[model_b1.py](model_b1.py) normalizes the projected representations of each view once before EPAGCL's contrastive computation (`F.normalize`, p=2, dim=1, eps=1e-12). Subsequent within-view and cross-view similarities reuse the normalized tensors without detaching them, preserving gradient flow. Let $I$ be the node indices in one independent batch and $\tau$ the temperature. One directional loss is:

$$
\widehat z_i^{(v)}=\frac{z_i^{(v)}}{\max(\lVert z_i^{(v)}\rVert_2,\epsilon)},\quad
s_{ij}^{vw}=\exp\left(\frac{\widehat z_i^{(v)}\cdot\widehat z_j^{(w)}}{\tau}\right).
$$

$$
\ell_i^{1\to2}=-\log\frac{s_{ii}^{12}}{\sum_{j\in I}s_{ij}^{11}+\sum_{j\in I}s_{ij}^{12}-s_{ii}^{11}}.
$$

The reverse direction swaps views; both directions still draw their own original random permutation (`randperm`). B1 does not expand negative sets, change batch size, alter VR, or modify linear evaluation.

## 2. Fixed-input numerical checks

[verify_b1.py](verify_b1.py) checks contrastive loss and input gradients with `rtol=1e-4`, `atol=1e-5`, and batch size 256. Node counts 64, 256, 257, 1025, and 34493 pass. Model parameter gradients with VR pass at the first four sizes; that check is not run at 34493 nodes. Both directional permutations and the post-loss RNG state match. `passed=true` establishes agreement within the tested tolerances, not bitwise equality.

Source: [equivalence.json](../../research_results/B_computational_efficiency/20261120_172257_B1_verify_seed_20261123/equivalence.json).

## 3. B0/B1 short-run benchmarks

Coauthor-Phy on an RTX 2070, without a profiler and excluding linear evaluation and preprocessing.

| Record | Rounds | Warmup | Timed | B0 median (s) | B1 median (s) | Reduction |
| --- | --- | --- | --- | --- | --- | --- |
| 20261120_172552_B01_compare | 3 | 5 | 20 | 1.6266 | 1.1393 | 29.95% |
| 20261120_180316_B01_compare | 1 | 5 | 20 | 1.4464 | 1.0622 | 26.56% |

In the first test, round three's B0 median is 4.7069 s versus 0.9882 s for B1, indicating substantial variation across trials. The separate rerun still shows a 26.56% median reduction. These are paired measurements, not a guaranteed fixed speedup. The unified ten-warmup B0/B1/B2 comparison is documented in [B2](README_B2.en.md).

Sources: [initial comparison](../../research_results/B_computational_efficiency/20261120_172552_B01_compare/comparison.json), [rerun](../../research_results/B_computational_efficiency/20261120_180316_B01_compare/comparison.json).

## 4. Entry points

```bash
python extension_experiments/efficiency_B/verify_b1.py --help
python extension_experiments/efficiency_B/benchmark_b01.py --help
```

Use the recorded JSON for exact configuration and environment. Entry points and raw outputs remain in the B directory; logs are not committed. B1 is the foundation of B2. [E3](README_E3.en.md) compares full B0/B2×8 training, not standalone full B1 training.
