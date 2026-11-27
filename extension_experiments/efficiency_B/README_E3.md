# E3：B0 與 B2×8 完整訓練

**繁體中文** · [English](README_E3.en.md)

[延伸實驗總覽](../README.md) · [B 總覽](README.md) · [B 結果](../../research_results/B_computational_efficiency/) · [追加兩組種子](README_E3_extra_2seeds.md)

## 1. 研究設計

短期測速無法回答整個訓練的時間與表示品質是否維持，因此 E3 使用 A 的 VR 設定比較 B0（原對比損失）與 B2×8（批次矩陣乘法 + 正規化重用）。正式比較保留原訓練長度、線性評估次數與驗證集選擇流程，兩組各五次；B1、B2×2、B2×4 不屬此正式完整訓練對照。

| 參數 | 值 |
| --- | --- |
| 資料集 (Dataset) | Coauthor-Phy |
| 裝置 (Device) | cuda:0 |
| 種子 (Seed) | 20261129–20261203 |
| 訓練輪數 / 重複次數 | 1000 / 每個種子一次 |
| 評估間隔 / 評估輪數 | 50 / 3000 |
| 最佳化器 / 學習率 / 權重衰減 | Adam / 0.01 / 0.001 |
| 評估最佳化器 / 學習率 / 權重衰減 | Adam / 0.01 / 0 |
| 溫度參數 | 0.5 |
| 加邊率 / 刪邊率 (視圖 1, 2) | 0.1 / (0.1, 0.4) |
| 特徵遮蔽率 (視圖 1, 2) | (0, 0) |
| 表示 / 投影隱藏維度 | 256 / 256 |
| 分批計算 | true |
| VR λ / γ | 10 / 0.002 （兩組皆使用） |
| 影子停止監測 | false |
| 損失記錄 / 權重儲存間隔 | 1 / 0 |

固定切分 10/10/80；配對的 `split_sha256` 紀錄一致。歷史設定的 GPU 期望值為 RTX 2070；完整軟硬體環境應查閱每次 `environment.json`。種子數字是 RNG 識別值，不是實驗日期。

## 2. 正式結果

| 種子 (Seed) | B0 (%) | B2×8 (%) | Δ (pp) | B0 time (s) | B2×8 time (s) |
| --- | --- | --- | --- | --- | --- |
| 20261129 | 93.839 | 92.024 | -1.816 | 3350.55 | 2051.94 |
| 20261130 | 94.202 | 94.590 | +0.388 | 3371.34 | 2045.17 |
| 20261201 | 94.213 | 94.398 | +0.185 | 3392.03 | 2012.10 |
| 20261202 | 94.311 | 94.822 | +0.511 | 3316.92 | 2093.47 |
| 20261203 | 93.807 | 94.176 | +0.370 | 3418.74 | 2093.33 |
| 平均 ± 樣本標準差 | 94.074 ± 0.233 | 94.002 ± 1.131 | -0.072 | 3369.92 | 2059.20 |

最後一列的準確率為平均 ± 樣本標準差，時間為五次平均。時間差與百分點差皆為配對資料的描述統計。

| 平均時間指標 | B0 | B2×8 | 時間減少 |
| --- | --- | --- | --- |
| 完整訓練 (s) | 3369.92 | 2059.20 | 38.89% |
| 主模型訓練 (s) | 3275.21 | 1965.29 | 39.99% |
| 線性評估 (s) | 94.53 | 93.67 | 0.92% |

- 完整訓練加速約 1.637 倍；主模型時間減少約 39.99%。計時範圍與短期測速不同。
- B2×8 五組中四組準確率較高，但第一組下降約 1.816 pp；平均差異 −0.072 pp，樣本標準差從 0.233 增至 1.131。
- 五次平均 feature std：B0 0.008577、B2×8 0.011435；mean pairwise cosine：0.607347、0.607306。最大峰值 allocated memory 約 5836.76、5838.67 MiB。

這些結果支持完整訓練加速及接近的平均準確率，但不保證逐位元等價、完全相同的訓練品質或更低的顯示記憶體。未以統計等價檢定證明兩方法相同。

## 3. 程式與診斷

| 檔案 | 用途 |
| --- | --- |
| [train_e3.py](train_e3.py) | 完整訓練與每 epoch 時間紀錄 |
| [run_e3_formal_batch.py](run_e3_formal_batch.py) | 歷史初始正式批次 |
| [run_e3_extra_2seeds.py](run_e3_extra_2seeds.py) | 追加正式種子 |
| [summarize_e3.py](summarize_e3.py) | 結果彙整 |
| [diagnose_e3_phases.py](diagnose_e3_phases.py) | 階段耗時診斷 |
| [diagnose_e3_parity_control.py](diagnose_e3_parity_control.py) | 訓練一致性控制 |
| [diagnose_e3_first_divergence.py](diagnose_e3_first_divergence.py) / [diagnose_e3_first_step_pipeline.py](diagnose_e3_first_step_pipeline.py) | 首次差異與首步流程 |
| [diagnose_e3_conv1_components.py](diagnose_e3_conv1_components.py) / [diagnose_e3_aggregation_determinism_control.py](diagnose_e3_aggregation_determinism_control.py) | GCN 第一層與聚合決定性診斷 |
| [run_e3_20261129_replay.py](run_e3_20261129_replay.py) | 同種子重播 |

診斷與重播紀錄保留於 [B 精選結果](../../research_results/B_computational_efficiency/)（含 `e3_same_seed_replay_20261129/`）。本文件不從未逐項核對的診斷結果推斷唯一的數值差異原因，不把重播當作新的獨立正式種子。原始 batch log 目錄不再提交。

## 4. 執行與輸出

```bash
python extension_experiments/efficiency_B/train_e3.py --help
```

從專案根目錄執行，依每組 `args.json` 重建相同參數；檢查實際 CLI、硬體限制、資料位置、輸出設定後再使用歷史批次入口。完整原始輸出在 `extension_experiments/efficiency_B/outputs/`，模型權重及固定切分留在本機。正式資料夾保留 `args.json`、`environment.json`、`source_sha256.json`、`summary.json`、每次的 `final.json`、`epoch_timing.csv`、`loss.csv`、`validation.csv`。

## 5. 正式資料索引（只取頂層十組）

| 種子 (Seed) | B0 final.json | B2×8 final.json |
| --- | --- | --- |
| 20261129 | [20261120_222038_490934_E3_formal_b0_Coauthor-Phy_seed_20261129](../../research_results/B_computational_efficiency/20261120_222038_490934_E3_formal_b0_Coauthor-Phy_seed_20261129/run_01_seed_20261129/final.json) | [20261120_231652_685412_E3_formal_b2x8_Coauthor-Phy_seed_20261129](../../research_results/B_computational_efficiency/20261120_231652_685412_E3_formal_b2x8_Coauthor-Phy_seed_20261129/run_01_seed_20261129/final.json) |
| 20261130 | [20261121_002556_627724_E3_formal_b0_Coauthor-Phy_seed_20261130](../../research_results/B_computational_efficiency/20261121_002556_627724_E3_formal_b0_Coauthor-Phy_seed_20261130/run_01_seed_20261130/final.json) | [20261120_235127_223317_E3_formal_b2x8_Coauthor-Phy_seed_20261130](../../research_results/B_computational_efficiency/20261120_235127_223317_E3_formal_b2x8_Coauthor-Phy_seed_20261130/run_01_seed_20261130/final.json) |
| 20261201 | [20261121_012230_707333_E3_formal_b0_Coauthor-Phy_seed_20261201](../../research_results/B_computational_efficiency/20261121_012230_707333_E3_formal_b0_Coauthor-Phy_seed_20261201/run_01_seed_20261201/final.json) | [20261121_021927_029066_E3_formal_b2x8_Coauthor-Phy_seed_20261201](../../research_results/B_computational_efficiency/20261121_021927_029066_E3_formal_b2x8_Coauthor-Phy_seed_20261201/run_01_seed_20261201/final.json) |
| 20261202 | [20261121_163628_703921_E3_formal_b0_Coauthor-Phy_seed_20261202](../../research_results/B_computational_efficiency/20261121_163628_703921_E3_formal_b0_Coauthor-Phy_seed_20261202/run_01_seed_20261202/final.json) | [20261121_160112_086020_E3_formal_b2x8_Coauthor-Phy_seed_20261202](../../research_results/B_computational_efficiency/20261121_160112_086020_E3_formal_b2x8_Coauthor-Phy_seed_20261202/run_01_seed_20261202/final.json) |
| 20261203 | [20261121_173211_732025_E3_formal_b0_Coauthor-Phy_seed_20261203](../../research_results/B_computational_efficiency/20261121_173211_732025_E3_formal_b0_Coauthor-Phy_seed_20261203/run_01_seed_20261203/final.json) | [20261121_182935_626683_E3_formal_b2x8_Coauthor-Phy_seed_20261203](../../research_results/B_computational_efficiency/20261121_182935_626683_E3_formal_b2x8_Coauthor-Phy_seed_20261203/run_01_seed_20261203/final.json) |

來源：[manifest.csv](../../research_results/manifest.csv)。`final.json` 是上述時間與準確率的直接來源；各組的 `args.json` 位於對應實驗目錄。早期 sanity、smoke、診斷與 replay 排除於正式統計之外。完整訓練指令未在新 clone 環境重跑。
