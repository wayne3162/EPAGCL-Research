# 實驗 B：對比損失計算效率

**繁體中文** · [English](README.en.md)

[延伸實驗總覽](../README.md) · [實驗 A](../formal_A/README.md) · [研究結果](../../research_results/B_computational_efficiency/)

## 1. 研究目標與實驗階段

B 延續 A 的 VR-EPAGCL 訓練設定，針對對比損失的重複正規化及多次小型矩陣乘法進行運算改良。目標是保留原本的數學損失、正負樣本定義、雙方向獨立隨機分批與完整訓練規格，不透過減少 epoch、節點或線性評估次數取得表面加速。

| 階段 | 目的 | 詳細文件 |
| --- | --- | --- |
| B0 | 原對比損失 + VR，建立測速基準 | 本頁 |
| B1 | 重用 L2 正規化後的投影表示 | [中文](README_B1.md) / [EN](README_B1.en.md) |
| B2 | 將獨立批次的相似度運算改為批次矩陣乘法 | [中文](README_B2.md) / [EN](README_B2.en.md) |
| E3 | B0 與 B2×8 完整配對訓練 | [中文](README_E3.md) / [EN](README_E3.en.md) |
| E3 追加種子 | 前三組之外再追加兩組，納入五組正式統計 | [中文](README_E3_extra_2seeds.md) / [EN](README_E3_extra_2seeds.en.md) |

歷史 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) 保留原規劃，不代表所有規劃均已完成。本系列 README 以實際紀錄為主。

## 2. B0 基準與短期測速

B0 使用 `profile_b0.py` 建立運算基準，Coauthor-Phy、RTX 2070、PyTorch 2.3.0+cu121，5 次預熱與 20 次計時；單步中位數約 1.7620 秒、峰值 allocated memory 約 5555 MiB。正式比較採無 profiler 的 `benchmark_b012.py`，每輪 10 次預熱、20 次計時，共三輪，不含線性評估、資料前處理。

| Variant | Median (s) | Mean (s) | P90 (s) | Reduction vs B0 |
| --- | --- | --- | --- | --- |
| B0 | 1.6023 | 1.5497 | 1.8779 | 0.00% |
| B1 | 0.9525 | 1.1317 | 1.3768 | 40.56% |
| B2x2 | 0.5145 | 0.6998 | 0.9792 | 67.89% |
| B2x4 | 0.4324 | 0.6235 | 0.9026 | 73.01% |
| B2x8 | 0.3642 | 0.5540 | 0.8456 | 77.27% |

B2×8 的合併中位數加速約 4.40 倍，與完整訓練時間的加速不同。早期 5 次預熱測試存在一輪 B2×8 較慢；保留兩份測速結果，不只展示較快的測試。峰值 allocated GPU memory 約 5555 MiB，各變體沒有明顯降低。

來源：[B0 timing.json](../../research_results/B_computational_efficiency/20261120_111127_B0_seed_20261123/timing.json)、[comparison_b012.json](../../research_results/B_computational_efficiency/20261120_183748_B012_compare/comparison_b012.json)、[5 次預熱比較](../../research_results/B_computational_efficiency/20261120_181707_B012_compare/comparison_b012.json)。

## 3. E3 完整訓練結果

五組正式配對（Coauthor-Phy，種子 20261129–20261203），1000 epoch、20 次線性評估。B0 與 B2×8 均保留 VR。

| Metric | B0 | B2×8 |
| --- | ---: | ---: |
| Test accuracy (%), mean ± sample SD | 94.074 ± 0.233 | 94.002 ± 1.131 |
| Mean elapsed time (s) | 3369.92 | 2059.20 |
| Mean main-model time (s) | 3275.21 | 1965.29 |

完整訓練平均減少 38.89%（約 1.637 倍加速），平均準確率差 −0.072 pp；B2×8 的跨種子標準差增加。固定輸入的數值驗證通過，不等於完整訓練軌跡逐位元相同或分類品質完全等價。

## 4. 程式入口與資料

在專案根目錄執行，先查看實際版本的參數：

```bash
python extension_experiments/efficiency_B/profile_b0.py --help
python extension_experiments/efficiency_B/verify_b1.py --help
python extension_experiments/efficiency_B/benchmark_b01.py --help
python extension_experiments/efficiency_B/verify_b2.py --help
python extension_experiments/efficiency_B/benchmark_b012.py --help
python extension_experiments/efficiency_B/train_e3.py --help
```

`model_b1.py`、`model_b2.py` 實作兩項改良；`train_e3.py` 負責完整訓練；`run_e3_formal_batch.py` 與 `run_e3_extra_2seeds.py` 為歷史批次入口；`summarize_e3.py` 彙整結果；`diagnose_e3_*.py` 與 `run_e3_20261129_replay.py` 用於差異診斷及同種子重播。執行歷史批次腳本前應檢查硬體限制、種子、路徑及現行參數。

完整原始輸出在 `extension_experiments/efficiency_B/outputs/`，Log 與原始輸出不提交。儲存庫中的精選結果包含測速、等價性驗證、正式訓練、探索及診斷紀錄，正式 E3 統計不混入 replay。詳見 [manifest.csv](../../research_results/manifest.csv)。
