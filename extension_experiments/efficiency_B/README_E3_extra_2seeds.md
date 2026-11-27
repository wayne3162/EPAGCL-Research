# E3：追加兩組正式種子

**繁體中文** · [English](README_E3_extra_2seeds.en.md)

[延伸實驗總覽](../README.md) · [B 總覽](README.md) · [B 結果](../../research_results/B_computational_efficiency/) · [E3 完整統計](README_E3.md)

## 1. 追加實驗的目的

初始正式 E3 使用三組種子（20261129、20261130、20261201）。追加 20261202、20261203 兩組 B0/B2×8 配對，共四次完整訓練，以增加跨種子觀察。這兩組已納入 E3 的五組正式配對，不是五組之外再額外計入的樣本。

## 2. 控制條件

延續 E3 的 Coauthor-Phy、1000 epoch、每 50 epoch 評估、每次線性評估 3000 epoch、VR λ=10、γ=0.002，固定切分與驗證集選擇流程不變。兩變體同種子的 `split_sha256` 紀錄一致。完整設定請見 [E3](README_E3.md) 與原始 `args.json`。種子數字是識別值而不是日期。

## 3. 追加結果

| 種子 (Seed) | B0 (%) | B2×8 (%) | Δ (pp) | B0 time (s) | B2×8 time (s) |
| --- | --- | --- | --- | --- | --- |
| 20261202 | 94.311 | 94.822 | +0.511 | 3316.92 | 2093.47 |
| 20261203 | 93.807 | 94.176 | +0.370 | 3418.74 | 2093.33 |

兩組追加實驗中 B2×8 完整訓練均較快，測試準確率亦略高於配對 B0。應以完整五組配對評估方法，而非只採追加兩組的結果。第一個正式種子仍有約 1.816 pp 的 B2×8 準確率下降；五組整體平均差 −0.072 pp、完整訓練時間減少 38.89%。

## 4. 腳本與紀錄

歷史追加批次入口：[run_e3_extra_2seeds.py](run_e3_extra_2seeds.py)。執行前先查看 [train_e3.py](train_e3.py) 的 `--help`，核對記錄的完整參數及 GPU 條件。不要將同種子重播混入追加統計。所有 batch logs 及原始輸出仍留在本機；GitHub 不保留 `e3_extra_batch_logs/`。

| 種子 (Seed) | B0 final.json | B2×8 final.json |
| --- | --- | --- |
| 20261202 | [20261121_163628_703921_E3_formal_b0_Coauthor-Phy_seed_20261202](../../research_results/B_computational_efficiency/20261121_163628_703921_E3_formal_b0_Coauthor-Phy_seed_20261202/run_01_seed_20261202/final.json) | [20261121_160112_086020_E3_formal_b2x8_Coauthor-Phy_seed_20261202](../../research_results/B_computational_efficiency/20261121_160112_086020_E3_formal_b2x8_Coauthor-Phy_seed_20261202/run_01_seed_20261202/final.json) |
| 20261203 | [20261121_173211_732025_E3_formal_b0_Coauthor-Phy_seed_20261203](../../research_results/B_computational_efficiency/20261121_173211_732025_E3_formal_b0_Coauthor-Phy_seed_20261203/run_01_seed_20261203/final.json) | [20261121_182935_626683_E3_formal_b2x8_Coauthor-Phy_seed_20261203](../../research_results/B_computational_efficiency/20261121_182935_626683_E3_formal_b2x8_Coauthor-Phy_seed_20261203/run_01_seed_20261203/final.json) |

資料來源為上述四個 `final.json` 及同目錄的 `args.json`。五組正式結果與其他診斷見 [E3 完整文件](README_E3.md)。
