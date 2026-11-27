# 實驗 A：變異數正規化與表示崩潰

**繁體中文** · [English](README.en.md)

[延伸實驗總覽](../README.md) · [實驗 B](../efficiency_B/README.md) · [研究結果](../../research_results/A_variance_regularization/)

## 1. 研究問題

在歷史 Coauthor-Phy 消融實驗中，部分訓練的分類準確率接近 50%。A 進一步檢查編碼器表示是否趨於相同，而不只依賴準確率判斷崩潰（Representation Collapse）。研究目標是在保留 EPAGCL 編碼器與對比學習架構的前提下，增加表示的跨節點變異，改善訓練穩定性。

## 2. 方法

基準組（Baseline）沿用原對比損失；VR 組在兩個增強視圖的編碼器輸出 `out1`、`out2`（不是投影頭輸出 `z1`、`z2`）加入變異數正規化（Variance Regularization, VR）。令 $H^{(v)}\in\mathbb{R}^{N\times D}$ 為視圖 $v$ 的編碼器輸出，$N$ 為節點數，$D$ 為特徵維度，$\bar H_j^{(v)}$ 為第 $j$ 維平均值；$\epsilon=10^{-8}$、$\gamma$ 為標準差下限、$\lambda$ 為正規化係數。

$$
\sigma_j^{(v)}=\sqrt{\frac{1}{N}\sum_{i=1}^N (H_{ij}^{(v)}-\bar H_j^{(v)})^2+\epsilon}
$$

$$
\mathcal L_{\mathrm{VR}}=\frac{1}{2D}\sum_{v=1}^{2}\sum_{j=1}^{D}\max(0,\gamma-\sigma_j^{(v)}),\qquad
\mathcal L_{\mathrm{total}}=\mathcal L_{\mathrm{contrastive}}+\lambda\mathcal L_{\mathrm{VR}}.
$$

當某維特徵標準差低於 $\gamma$ 時才產生懲罰；實作採母體變異數（`unbiased=False`）。VR 不使用類別標籤、不改變原加刪邊策略。影子停止監測（shadow monitoring）只記錄早期高損失停滯的候選訊號，不終止訓練；歷史門檻 6.22 不視為普遍有效規則，正式比較關閉此功能。

## 3. 正式實驗設計

| 參數 | 值 |
| --- | --- |
| 資料集 (Dataset) | Coauthor-Phy |
| 裝置 (Device) | cuda:0 |
| 種子 (Seed) | 20261123–20261127 |
| 訓練輪數 / 重複次數 | 1000 / 每個種子一次 |
| 評估間隔 / 評估輪數 | 50 / 3000 |
| 最佳化器 / 學習率 / 權重衰減 | Adam / 0.01 / 0.001 |
| 評估最佳化器 / 學習率 / 權重衰減 | Adam / 0.01 / 0 |
| 溫度參數 | 0.5 |
| 加邊率 / 刪邊率 (視圖 1, 2) | 0.1 / (0.1, 0.4) |
| 特徵遮蔽率 (視圖 1, 2) | (0, 0) |
| 表示 / 投影隱藏維度 | 256 / 256 |
| 分批計算 | true |
| VR λ / γ | 10 / 0.002 （僅 VR 組） |
| 影子停止監測 | false |
| 損失記錄 / 權重儲存間隔 | 1 / 100 |

兩組使用對應種子與固定切分；訓練／驗證／測試比例為 10/10/80。每 50 epoch 線性評估一次，共 20 次；由驗證集選擇分類器及編碼器，最後才測試。這是 A 的新評估協定，不能直接與歷史 Q3 的評估值混合比較。

## 4. 正式結果（五組配對）

| 種子 (Seed) | Baseline (%) | VR (%) | VR − Baseline (pp) |
| --- | --- | --- | --- |
| 20261123 | 50.632 | 94.437 | +43.805 |
| 20261124 | 50.527 | 94.224 | +43.696 |
| 20261125 | 93.292 | 94.151 | +0.859 |
| 20261126 | 50.661 | 93.850 | +43.189 |
| 20261127 | 50.629 | 93.568 | +42.939 |
| 平均 ± 樣本標準差 | 59.148 ± 19.087 | 94.046 ± 0.340 | +34.898 |

| 診斷指標（五次平均） | Baseline | VR |
| --- | --- | --- |
| 特徵標準差平均 | 0.002348 | 0.009197 |
| 節點間餘弦相似度平均 | 0.919926 | 0.600476 |
| 完整訓練時間 (s) | 3457.43 | 3484.91 |

Baseline 的四次低準確率執行同時呈現約 0.000119–0.000151 的特徵標準差平均與大於 0.99995 的節點間餘弦相似度平均，支持近似表示崩潰的診斷。第五次 Baseline 未出現此組合。VR 五次皆未出現同樣的診斷組合。平均準確率增加 34.898 pp，平均完整訓練時間增加約 0.8%（不含資料前處理）。這是 Coauthor-Phy、指定參數與五個種子的觀察；尚無跨資料集的普遍結論，也未設定正式的二元崩潰判定門檻。

## 5. 程式與執行方式

| 檔案 | 用途 |
| --- | --- |
| [../run.py](../run.py) | A 主訓練流程、診斷、輸出管理 |
| [../vr_regularization.py](../vr_regularization.py) | VR 損失 |
| [../fixed_eval.py](../fixed_eval.py) | 共用切分及驗證集選擇 |
| [run_remaining.sh](run_remaining.sh) | 歷史正式實驗腳本（剩餘四組種子） |
| [source_sha256.txt](source_sha256.txt) | 當時的程式來源雜湊，不改寫 |
| [../tests/test_basic.py](../tests/test_basic.py) | 基本單元測試 |

以下在專案根目錄執行，先用 `python extension_experiments/run.py --help` 核對實際版本。示範第一組種子；其他種子請逐次替換，並對照原始 `args.json` 的完整設定（包含 Adam、維度、分批與評估最佳化器）。不要直接執行歷史批次腳本來覆寫結果；新執行會建立新時間戳記目錄。

```bash
# Baseline
python extension_experiments/run.py --method baseline --dataset Coauthor-Phy \
  --device cuda:0 --epoch 1000 --repeat 1 --seed 20261123 \
  --eval 50 --eval-epoch 3000 --lr 0.01 --weight-decay 0.001 \
  --temperature 0.5 --edge-add-rate 0.1 --edge-drop-rate-1 0.1 \
  --edge-drop-rate-2 0.4 --feat-drop-rate-1 0 --feat-drop-rate-2 0 \
  --loss-log 1 --save-model-every 100 --no-shadow-stop

# VR
python extension_experiments/run.py --method vr --dataset Coauthor-Phy \
  --device cuda:0 --epoch 1000 --repeat 1 --seed 20261123 \
  --eval 50 --eval-epoch 3000 --lr 0.01 --weight-decay 0.001 \
  --temperature 0.5 --edge-add-rate 0.1 --edge-drop-rate-1 0.1 \
  --edge-drop-rate-2 0.4 --feat-drop-rate-1 0 --feat-drop-rate-2 0 \
  --loss-log 1 --save-model-every 100 --no-shadow-stop \
  --vr-lambda 10 --vr-gamma 0.002
```

## 6. 新輸出位置

```text
extension_experiments/formal_A/
├── outputs/<timestamp>_<method>_<dataset>/
│   ├── args.json
│   ├── original_source_sha256.json
│   ├── partial_summary.json
│   ├── summary.json
│   └── run_01_seed_<seed>/
│       ├── final.json
│       ├── loss.csv
│       ├── validation.csv
│       └── ... (weights local only, if saved)
└── logs/                      # local only
```

完整原始輸出與所有 Log 留在本機；精選結果位於 [research_results/A_variance_regularization/](../../research_results/A_variance_regularization/)，不包含權重與固定切分 `.pt`。共用切分仍在 `extension_experiments/splits/`。移動輸出資料夾不需要重新訓練，舊 JSON 的歷史參數與來源雜湊保持原樣。

## 7. 正式資料索引

下表只列出正式五組配對，其他探索資料保留於同一精選結果目錄但不納入正式統計。每個 `final.json` 的上層實驗目錄提供 `args.json`、`summary.json` 及來源雜湊。

| 種子 (Seed) | Baseline final.json | VR final.json |
| --- | --- | --- |
| 20261123 | [20261119_183936_883401_baseline_Coauthor-Phy](../../research_results/A_variance_regularization/20261119_183936_883401_baseline_Coauthor-Phy/run_01_seed_20261123/final.json) | [20261119_193855_310924_vr_Coauthor-Phy](../../research_results/A_variance_regularization/20261119_193855_310924_vr_Coauthor-Phy/run_01_seed_20261123/final.json) |
| 20261124 | [20261119_204656_996415_baseline_Coauthor-Phy](../../research_results/A_variance_regularization/20261119_204656_996415_baseline_Coauthor-Phy/run_01_seed_20261124/final.json) | [20261119_214445_679460_vr_Coauthor-Phy](../../research_results/A_variance_regularization/20261119_214445_679460_vr_Coauthor-Phy/run_01_seed_20261124/final.json) |
| 20261125 | [20261119_224250_775434_baseline_Coauthor-Phy](../../research_results/A_variance_regularization/20261119_224250_775434_baseline_Coauthor-Phy/run_01_seed_20261125/final.json) | [20261119_234044_122133_vr_Coauthor-Phy](../../research_results/A_variance_regularization/20261119_234044_122133_vr_Coauthor-Phy/run_01_seed_20261125/final.json) |
| 20261126 | [20261120_003915_198680_baseline_Coauthor-Phy](../../research_results/A_variance_regularization/20261120_003915_198680_baseline_Coauthor-Phy/run_01_seed_20261126/final.json) | [20261120_013602_065693_vr_Coauthor-Phy](../../research_results/A_variance_regularization/20261120_013602_065693_vr_Coauthor-Phy/run_01_seed_20261126/final.json) |
| 20261127 | [20261120_023322_567168_baseline_Coauthor-Phy](../../research_results/A_variance_regularization/20261120_023322_567168_baseline_Coauthor-Phy/run_01_seed_20261127/final.json) | [20261120_033221_112832_vr_Coauthor-Phy](../../research_results/A_variance_regularization/20261120_033221_112832_vr_Coauthor-Phy/run_01_seed_20261127/final.json) |

來源：[結果 manifest](../../research_results/manifest.csv)。正式數值由上述十次訓練的 `final.json` 計算；標準差為樣本標準差。新路徑已修改並檢查，未在新 clone 中重新執行完整訓練。
