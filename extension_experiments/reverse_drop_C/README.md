# 實驗 C：反向刪邊與 EPR 分析

**繁體中文** · [English](README.en.md)

[延伸實驗總覽](../README.md) · [研究結果](../../research_results/C_reverse_edge_dropping/)

## 1. 研究問題

原版 EPAGCL 的度數導向刪邊會偏向特定結構的邊。C 研究反轉刪邊權重排序方向，是否能讓增強圖的 EPR（錯誤傳播率，Error Passing Rate；沿用原版 `EPR.py` 所定義的圖結構指標）更接近原圖，並觀察下游節點分類準確率是否隨之改善。此研究分為配對訓練與獨立離線 EPR 分析，不加入 A 的 VR 或 B 的運算改良。

## 2. 反向刪邊與索引對齊

令既有邊 $(u,v)$ 兩端的無向化圖節點度數為 $d_u,d_v$，$\bar w$ 為邊權重平均。Baseline 與 Reverse Drop 的正規化權重如下：

$$
w_{uv}=\frac{1}{\sqrt{d_ud_v}},\quad
q_{uv}^{\mathrm{base}}=\frac{w_{uv}-w_{\min}}{\bar w-w_{\min}},\quad
q_{uv}^{\mathrm{rev}}=\frac{w_{\max}-w_{uv}}{w_{\max}-\bar w}.
$$

反向刪邊反轉權重排序，保留加邊流程；名義刪邊率相同不保證實際無向圖刪除率相同。程式先依無向化圖計算權重，再使用有序端點配對映射（Ordered Endpoint Pair Mapping）對齊原抽樣邊索引，保留原始抽樣排列。Cora 的紀錄顯示原始與 canonical 無向化索引各 10,556 欄、無重複欄或缺少反向欄，但排序不同。對齊修正同時套用於兩組，因此 C Baseline 不是未修正歷史實作的完全複製；未量化舊錯位對準確率的單獨影響。

## 3. 配對節點分類

| 參數 | 值 |
| --- | --- |
| 資料集 (Dataset) | Cora |
| 裝置 (Device) | cuda:0 |
| 種子 / 重複次數 | 20261119 / 5 |
| 訓練輪數 / 評估間隔 / 評估輪數 | 500 / 50 / 3000 |
| 最佳化器 / 學習率 / 權重衰減 | Adam / 0.001 / 0.0001 |
| 評估最佳化器 / 學習率 / 權重衰減 | Adam / 0.01 / 0 |
| 溫度參數 | 0.3 |
| 加邊率 / 刪邊率 (視圖 1, 2) | 0.1 / (0.2, 0.3) |
| 特徵遮蔽率 (視圖 1, 2) | (0.1, 0.1) |
| 表示 / 投影隱藏維度 | 256 / 256 |
| 分批計算 | true |
| 損失記錄 / 權重儲存間隔 | 1 / 0 |
| VR | 不使用 |

固定切分 10/10/80，驗證集選擇檢查點，測試集只在選擇完成後評估；沒有 VR、影子停止或提前終止。種子 20261119–20261123，兩組各五次。

| 種子 (Seed) | Baseline (%) | Reverse Drop (%) | Δ (pp) |
| --- | --- | --- | --- |
| 20261119 | 82.749 | 82.149 | -0.600 |
| 20261120 | 82.242 | 82.426 | +0.185 |
| 20261121 | 83.902 | 83.256 | -0.646 |
| 20261122 | 83.533 | 82.426 | -1.107 |
| 20261123 | 83.210 | 83.441 | +0.231 |
| 平均 ± 樣本標準差 | 83.127 ± 0.652 | 82.740 ± 0.571 | -0.387 |

平均差 −0.387 pp，未觀察到平均分類準確率提升。樣本標準差由五個測試值重算；原始 summary 的 population std 口徑不同。

來源：[comparison.csv](../../research_results/C_reverse_edge_dropping/training/20261125_012440_810530_both_Cora/comparison.csv)、[summary.json](../../research_results/C_reverse_edge_dropping/training/20261125_012440_810530_both_Cora/summary.json)、[args.json](../../research_results/C_reverse_edge_dropping/training/20261125_012440_810530_both_Cora/args.json)、[edge_alignment.json](../../research_results/C_reverse_edge_dropping/training/20261125_012440_810530_both_Cora/edge_alignment.json)。

## 4. 離線 EPR 分析

依據原版 [EPR.py](../../EPR.py) 的定義，在無向化後圖上計算非孤立節點的平均 EPR。令 $V_+$ 為非孤立節點集合，$\mathcal N(u)$ 為節點鄰居，$y_u$ 為類別標籤。以度數權重計算跨類鄰邊占比：

$$
\mathrm{EPR}(G)=\frac{1}{|V_+|}\sum_{u\in V_+}
\frac{\sum_{v\in\mathcal N(u)}\mathbf 1[y_u\ne y_v]/\sqrt{d_ud_v}}
{\sum_{v\in\mathcal N(u)}1/\sqrt{d_ud_v}}.
$$

真實標籤只用於離線指標，不參與增強抽樣或自監督訓練；此分析也不重播訓練時的 RNG 軌跡。原始 Cora EPR = 0.168300、孤立節點 0。正式分析使用 CPU、五組種子，每組 500 次，兩種方法各階段有 2500 筆有效配對；另保留 10 次單種子 smoke test。名義加邊率 0.1、兩視圖刪邊率 0.2/0.3，記錄的加邊抽樣 p=0.5、threshold=0.9。

| Stage | Pairs | Baseline EPR | Reverse Drop EPR | Δ |
| --- | --- | --- | --- | --- |
| view1_drop | 2500 | 0.176169 | 0.168368 | -0.007800 |
| view1_final | 2500 | 0.177468 | 0.169595 | -0.007874 |
| view2_drop | 2500 | 0.181466 | 0.168486 | -0.012980 |

`view1_drop` 是視圖一刪邊後；`view1_final` 是視圖一完成加刪邊；`view2_drop` 是視圖二刪邊後。下表另列與原圖 EPR 的平均絕對差、孤立節點及無向化後實際刪邊率。B 表示 Baseline、R 表示 Reverse Drop；加邊後階段無單獨刪邊率，記為 —。

| Stage | B mean abs ΔEPR | R mean abs ΔEPR | B isolates | R isolates | B undirected drop | R undirected drop |
| --- | --- | --- | --- | --- | --- | --- |
| view1_drop | 0.007869 | 0.000778 | 194.5500 | 8.2356 | 6.133% | 4.155% |
| view1_final | 0.009169 | 0.001396 | 194.5500 | 8.2356 | — | — |
| view2_drop | 0.013167 | 0.001219 | 341.9652 | 20.2464 | 12.624% | 9.335% |

反向刪邊使 EPR 更接近原圖，也大幅減少孤立節點；三個階段的平均 EPR 仍略高於原圖。原始有向欄位的實際刪邊率大致接近，但無向化後刪邊率不同，且 EPR 排除孤立節點，因此目前不能將差異單獨歸因於排序方向。圖結構擾動較小不必然提升分類表現。

來源：[正式比較](../../research_results/C_reverse_edge_dropping/epr_analysis/20261125_020958_891111_Cora_original_epr/comparison.csv)、[summary.csv](../../research_results/C_reverse_edge_dropping/epr_analysis/20261125_020958_891111_Cora_original_epr/summary.csv)、[逐次抽樣](../../research_results/C_reverse_edge_dropping/epr_analysis/20261125_020958_891111_Cora_original_epr/epr_samples.csv)、[逐種子彙整](../../research_results/C_reverse_edge_dropping/epr_analysis/20261125_020958_891111_Cora_original_epr/seed_summary.csv)、[original_epr.json](../../research_results/C_reverse_edge_dropping/epr_analysis/20261125_020958_891111_Cora_original_epr/original_epr.json)、[對齊紀錄](../../research_results/C_reverse_edge_dropping/epr_analysis/20261125_020958_891111_Cora_original_epr/edge_alignment.json)。

## 5. 程式與執行

| 檔案 | 用途 |
| --- | --- |
| [augmentation_c.py](augmentation_c.py) | 權重、端點對齊、反向刪邊 |
| [run.py](run.py) | 配對訓練 |
| [epr_analysis.py](epr_analysis.py) | 獨立離線 EPR 分析 |
| [run_cora.sh](run_cora.sh) | 歷史 Cora 執行腳本 |
| [../fixed_eval.py](../fixed_eval.py) | 共用固定切分及線性評估 |

從專案根目錄執行。訓練請依上表及 `args.json` 核對實際 CLI；歷史腳本可能含硬體或本機路徑設定。以下提供離線正式分析入口，不需要預訓練模型。

```bash
python extension_experiments/reverse_drop_C/run.py --help
python extension_experiments/reverse_drop_C/epr_analysis.py --help

python extension_experiments/reverse_drop_C/epr_analysis.py \
  --dataset Cora --device cpu --samples 500 --repeat 5 --seed 20261119 \
  --edge-add-rate 0.1 --edge-drop-rate-1 0.2 --edge-drop-rate-2 0.3
```

訓練輸出 `extension_experiments/reverse_drop_C/outputs/`，離線分析輸出 `extension_experiments/reverse_drop_C/epr_outputs/`。原始輸出、Log、權重與切分 `.pt` 留在本機，GitHub 保存 [精選研究結果](../../research_results/C_reverse_edge_dropping/) 及 [manifest.csv](../../research_results/manifest.csv)。本研究尚未進行多資料集或等實際刪邊率的控制比較。
