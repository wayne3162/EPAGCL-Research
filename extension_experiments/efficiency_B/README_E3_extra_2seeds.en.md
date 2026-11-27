# E3: Two Additional Formal Seeds

[繁體中文](README_E3_extra_2seeds.md) · **English**

[Extension overview](../README.en.md) · [B overview](README.en.md) · [B evidence](../../research_results/B_computational_efficiency/) · [Full E3 statistics](README_E3.en.md)

## 1. Purpose

The initial formal E3 comparison used three seeds (20261129, 20261130, and 20261201). Two additional B0/B2×8 pairs, 20261202 and 20261203, add four full-training runs to broaden across-seed observations. They are included in the five formal E3 pairs, not counted as additional samples beyond those five.

## 2. Controls

The additional runs retain E3's Coauthor-Phy dataset, 1000 epochs, evaluation every 50 epochs, 3000 linear-evaluation epochs, VR λ=10 and γ=0.002, fixed splits, and validation-only selection. Recorded `split_sha256` values match within each seed pair. See [E3](README_E3.en.md) and the original `args.json` files for the full settings. Numeric seeds are identifiers, not calendar dates.

## 3. Additional results

| Seed | B0 (%) | B2×8 (%) | Δ (pp) | B0 time (s) | B2×8 time (s) |
| --- | --- | --- | --- | --- | --- |
| 20261202 | 94.311 | 94.822 | +0.511 | 3316.92 | 2093.47 |
| 20261203 | 93.807 | 94.176 | +0.370 | 3418.74 | 2093.33 |

Both additional pairs have shorter elapsed training time and slightly higher test accuracy for B2×8. Assess the method using all five formal pairs, not these two alone. The first formal seed still loses approximately 1.816 pp with B2×8. Across all five pairs, the mean difference is −0.072 pp and mean elapsed time falls by 38.89%.

## 4. Script and evidence

Historical entry point: [run_e3_extra_2seeds.py](run_e3_extra_2seeds.py). Before running it, inspect [train_e3.py](train_e3.py) with `--help` and check recorded settings and GPU restrictions. Do not pool same-seed replay with the additional formal seeds. All batch logs and raw outputs remain local; GitHub does not retain `e3_extra_batch_logs/`.

| Seed | B0 final.json | B2×8 final.json |
| --- | --- | --- |
| 20261202 | [20261121_163628_703921_E3_formal_b0_Coauthor-Phy_seed_20261202](../../research_results/B_computational_efficiency/20261121_163628_703921_E3_formal_b0_Coauthor-Phy_seed_20261202/run_01_seed_20261202/final.json) | [20261121_160112_086020_E3_formal_b2x8_Coauthor-Phy_seed_20261202](../../research_results/B_computational_efficiency/20261121_160112_086020_E3_formal_b2x8_Coauthor-Phy_seed_20261202/run_01_seed_20261202/final.json) |
| 20261203 | [20261121_173211_732025_E3_formal_b0_Coauthor-Phy_seed_20261203](../../research_results/B_computational_efficiency/20261121_173211_732025_E3_formal_b0_Coauthor-Phy_seed_20261203/run_01_seed_20261203/final.json) | [20261121_182935_626683_E3_formal_b2x8_Coauthor-Phy_seed_20261203](../../research_results/B_computational_efficiency/20261121_182935_626683_E3_formal_b2x8_Coauthor-Phy_seed_20261203/run_01_seed_20261203/final.json) |

The four `final.json` records above and corresponding `args.json` files support this page. See [full E3 documentation](README_E3.en.md) for the five-seed analysis and diagnostics.
