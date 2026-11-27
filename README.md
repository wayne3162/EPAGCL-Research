# EPAGCL：論文重現與延伸研究

延伸實驗雙語說明：[繁體中文](extension_experiments/README.md) · [English](extension_experiments/README.en.md)

本專案以 AAAI 2025 論文 **Why Does Dropping Edges Usually Outperform Adding Edges in Graph Contrastive Learning?** 的 EPAGCL（Error-PAssing-based Graph Contrastive Learning，錯誤傳播率導向的圖對比學習）為研究基礎。

研究目標是重現原作者的方法與部分實驗，分析圖增強（Graph Augmentation）策略的影響，並從表示學習穩定性、運算效率及圖結構擾動三個方向進行延伸。此儲存庫記錄本機實驗成果，而非原作者發布的新版本；A、B、C 是各有比較基準的研究實驗，尚未整合為單一新模型。

- 原始程式碼：[hyzhang98/EPAGCL](https://github.com/hyzhang98/EPAGCL)
- 本機原版基礎版本：`1cfcf61`（Git commit）
- 原始論文：[AAAI 2025](https://doi.org/10.1609/aaai.v39i20.35488)；[arXiv 延伸版本](https://arxiv.org/abs/2412.08128v4)
- 原作者使用說明：[README_upstream.md](README_upstream.md)

## 1. 原版方法與研究問題

EPAGCL 以錯誤傳播率（Error Passing Rate, EPR）分析圖結構與訊息傳遞的關係，並以節點度數導出的權重進行加邊、刪邊。原方法建立兩個增強視圖，其中一個僅刪邊，另一個同時加邊與刪邊；兩個視圖亦可進行特徵遮蔽（Feature Masking）。共享參數的圖卷積網路（Graph Convolutional Network, GCN）與投影頭（Projection Head）將節點映射至表示空間，以同一節點在不同視圖中的表示形成正樣本對，再透過對比損失進行自監督訓練。最後在原始圖取得節點嵌入（Node Embedding），進行線性分類評估。

原方法以 EPR 理論設計圖增強策略，不需要在每次訓練中使用真實類別標籤計算 EPR 作為損失。本專案主要探討以下問題：

| 研究項目 | 問題 | 實作方向 |
| --- | --- | --- |
| 原版重現、Q3 | 不同加邊、刪邊策略如何影響分類結果？ | 重現節點分類與六種圖增強設定，繪製 Figure 3 類型的 t-SNE 圖 |
| A：Variance Regularization (VR) | 為何部分 Coauthor-Phy 實驗的節點表示近似崩潰？ | 在編碼器輸出加入變異數正規化，維持表示的特徵變化 |
| B：Computational Efficiency | 能否減少對比損失的計算成本，同時維持分類表現？ | 重用正規化後的投影表示、合併多個獨立批次的矩陣運算 |
| C：Reverse Edge Dropping | 反轉刪邊權重後，EPR、孤立節點與分類結果如何改變？ | 比較反向刪邊、修正邊權重索引對齊、進行離線 EPR 分析 |
| 節點預測與視覺化 | 訓練後的表示如何用於節點分類？ | 凍結編碼器、訓練線性分類器，追蹤節點表示與預測結果 |

## 2. 相對原版的程式修改

### 2.1 原版核心檔案的修改

以下為本機相對 `1cfcf61` 的主要已確認修改。A、B、C 則以獨立模組實作，避免直接覆蓋原版模型與增強函式。

| 檔案 | 修改內容 | 目的 |
| --- | --- | --- |
| `Arguments.py` | 增加 `--random_drop_edge`、`--not_drop_feature_random` 選項 | 支援圖增強實驗設定 |
| `main.py` | 隨機刪邊時使用全 1 邊權重；將 `ac, z = train(...)` 改為 `ac = train(...)`；移除未使用的特徵增強匯入 | 支援隨機刪邊對照，修正回傳值使用方式 |
| `Linear_evaluation.py` | 複製評估參數，將 `eval_optim`、`eval_lr`、`eval_weight_decay` 對應至 `get_optim` 所需欄位 | 修正線性評估的最佳化器介面 |

`Model.py`、`Augmentation.py`、`EPR.py` 等原版核心模組在此次工作樹比對中未直接修改。上述根目錄工程修正與研究方法的改動應分開理解；新增命令列旗標本身不代表已驗證其所有執行路徑。

### 2.2 獨立研究模組

| 實作 | 主要功能 |
| --- | --- |
| [`extension_experiments/run.py`](extension_experiments/run.py) | A 實驗訓練入口，支援 `baseline` 與 `vr` |
| [`extension_experiments/vr_regularization.py`](extension_experiments/vr_regularization.py) | VR 損失與高損失停滯監測 |
| [`extension_experiments/fixed_eval.py`](extension_experiments/fixed_eval.py) | 固定資料切分與依驗證集選擇模型的評估流程 |
| [`extension_experiments/efficiency_B/model_b1.py`](extension_experiments/efficiency_B/model_b1.py) | B1：投影表示正規化重用 |
| [`extension_experiments/efficiency_B/model_b2.py`](extension_experiments/efficiency_B/model_b2.py) | B2：以批次矩陣乘法處理多組獨立相似度矩陣 |
| [`extension_experiments/efficiency_B/train_e3.py`](extension_experiments/efficiency_B/train_e3.py) | B0 與 B2×8 完整訓練比較 |
| [`extension_experiments/reverse_drop_C/augmentation_c.py`](extension_experiments/reverse_drop_C/augmentation_c.py) | 反向刪邊權重及端點配對索引對齊 |
| [`extension_experiments/reverse_drop_C/run.py`](extension_experiments/reverse_drop_C/run.py) | C 實驗配對訓練 |
| [`extension_experiments/reverse_drop_C/epr_analysis.py`](extension_experiments/reverse_drop_C/epr_analysis.py) | 以原版 EPR.py 定義分析圖增強，不訓練模型 |
| `diagnose_coauthor_collapse.py` | Coauthor-Phy 訓練異常診斷工具 |
| `epagcl_predict_nodes.py`、`epagcl_trace_visualize.py`、`plot_figure3.py` | 節點預測、局部結構與表示追蹤、t-SNE 視覺化 |

## 3. 原版重現與 Q3 消融實驗

### 3.1 原版重現

下列數據來自本機已完成的五次重複實驗，並非原論文公布的結果。標準差沿用原始 `result.txt` 的統計口徑。

| 資料集 | Epoch | 特徵遮蔽率（兩視圖） | Accuracy (%) |
| --- | ---: | --- | ---: |
| Cora | 500 | 0.1 / 0.1 | 84.48 ± 0.41 |
| Cora | 500 | 0 / 0 | 83.86 ± 0.63 |
| CiteSeer | 500 | 0.1 / 0.1 | 70.63 ± 0.37 |

結果及參數：[research_results/reproduction/](research_results/reproduction/)。Figure 3 類型的 Cora、CiteSeer t-SNE 視覺化：[figure3_output/](figure3_output/)。這些圖是本機重現產物，不宣稱與論文圖形或數值完全相同。

### 3.2 Q3 六種圖增強策略

| 方法 | 實際設定 |
| --- | --- |
| EPAGCL | 權重導向刪邊，僅一個視圖加邊 |
| DropOnly | 權重導向刪邊，不加邊 |
| AddOnly | 不刪邊，僅一個視圖加邊 |
| AddBoth | 權重導向刪邊，兩個視圖都加邊 |
| RandomDrop | 隨機刪邊，不加邊 |
| RandomAdd | 隨機刪邊、隨機加邊，兩個視圖都允許加邊 |

| 方法 | Cora | WikiCS | Amazon-Photo | Coauthor-Phy |
| --- | ---: | ---: | ---: | ---: |
| EPAGCL | 83.86 ± 0.63* | 81.56 ± 0.11 | 88.45 ± 0.60 | 68.17 ± 21.29 |
| DropOnly | 84.34 ± 0.42 | 81.69 ± 0.10 | 88.38 ± 0.74 | 50.74 ± 0.04 |
| AddOnly | 79.92 ± 0.96 | 80.41 ± 0.22 | 48.11 ± 3.80 | 56.63 ± 11.89 |
| AddBoth | 84.38 ± 0.25 | 81.82 ± 0.24 | 88.10 ± 0.57 | 59.42 ± 17.37 |
| RandomDrop | 83.51 ± 0.37 | 81.71 ± 0.07 | 87.58 ± 1.10 | 76.40 ± 20.97 |
| RandomAdd | 83.95 ± 0.72 | 81.28 ± 0.09 | 86.40 ± 1.68 | 59.21 ± 17.01 |

單位為 Accuracy (%)；每組重複五次，Cora、WikiCS、Amazon-Photo、Coauthor-Phy 分別訓練 500、3000、1000、1000 Epoch。* Cora 的 EPAGCL 取自特徵遮蔽率為 0 的原版重現結果，並非獨立的 Q3 EPAGCL 執行；其程式版本及種子是否與其他方法完全相同尚未確認。

Q3 中 Amazon-Photo 的 AddOnly 結果明顯較低，Coauthor-Phy 則出現多組不穩定的分類結果。Coauthor-Phy 的 EPAGCL 五次實驗中，有三次約 50.7–50.8%，其餘兩次約 94%。這些現象構成後續研究表示崩潰的動機，但僅憑分類準確率不足以確定崩潰原因。

原始依據：[research_results/q3_ablation/](research_results/q3_ablation/) 內的 `args.txt`、`result.txt`、`loss.txt`、`eval.txt`。各資料集的歷史程式補丁與基礎版本保留在根目錄 `EPAGCL_Q3_Cora_changes.patch` 及 `logs/` (local only)；執行腳本為 `run_amazon_photo_q3_all.sh`、`run_coauthor_phy_q3_all.sh`、`run_wikics_q3_all.sh`。歷史腳本可能含本機路徑與舊版設定，執行前需對照補丁與參數。

**評估口徑：** 原版重現與 Q3 沿用歷史評估紀錄；A、B、C 使用新的固定切分與驗證集選擇流程。兩者不可直接混合計算方法改善幅度。

## 4. 實驗 A：變異數正規化（VR-EPAGCL）

### 4.1 方法

為避免不同節點的編碼器輸出趨於相同（Representation Collapse，表示崩潰），在兩個增強視圖的編碼器輸出 `out1`、`out2` 上加入變異數正規化（Variance Regularization, VR）。此懲罰項不是作用在投影頭輸出 `z1`、`z2`，也不需要真實標籤。

令 $H^{(v)}\in\mathbb{R}^{N\times D}$ 為第 $v$ 個視圖的編碼器輸出，$N$ 為節點數、$D$ 為特徵維度，$\bar H_j^{(v)}$ 為第 $j$ 維特徵的節點平均值。程式使用母體變異數、$\epsilon=10^{-8}$ 與標準差下限 $\gamma$：

$$
\sigma_j^{(v)}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(H_{ij}^{(v)}-\bar H_j^{(v)})^2+\epsilon}
$$

$$
\mathcal{L}_{\mathrm{VR}}=\frac{1}{2D}\sum_{v=1}^{2}\sum_{j=1}^{D}\max(0,\gamma-\sigma_j^{(v)})
$$

$$
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{contrastive}}+\lambda\mathcal{L}_{\mathrm{VR}}
$$

其中 $\lambda$ 控制正規化強度。基準組僅使用原對比損失；VR 組加入上述懲罰項。另有高損失停滯的影子監測（Shadow Monitoring）：只在指定早期 Epoch 記錄候選停止訊號，不實際提前終止訓練。正式比較關閉此功能。

### 4.2 正式結果

Coauthor-Phy，五組配對種子 20261123–20261127，1000 Epoch，每 50 Epoch 進行一次線性評估，每次評估 3000 Epoch；VR 設定 $\lambda=10$、$\gamma=0.002$。兩組使用相同資料切分流程和對應種子，僅 VR 設定不同。

| Seed | Baseline (%) | VR (%) |
| --- | ---: | ---: |
| 20261123 | 50.632 | 94.437 |
| 20261124 | 50.527 | 94.224 |
| 20261125 | 93.292 | 94.151 |
| 20261126 | 50.661 | 93.850 |
| 20261127 | 50.629 | 93.568 |
| 平均 ± 樣本標準差 | 59.148 ± 19.087 | 94.046 ± 0.340 |

| 診斷指標（五次平均） | Baseline | VR |
| --- | ---: | ---: |
| Mean Feature Std（特徵標準差平均） | 0.002348 | 0.009197 |
| Mean Pairwise Cosine（節點間餘弦相似度平均） | 0.919926 | 0.600476 |
| 完整訓練時間 (s) | 3457.43 | 3484.91 |

VR 的平均分類準確率增加 34.898 個百分點（percentage points, pp），完整訓練平均時間增加約 0.8%。Baseline 有四組同時呈現低分類準確率、極低特徵標準差與接近 1 的餘弦相似度；VR 五組均未呈現此組合。這支持 VR 在此設定下緩解近似表示崩潰的效果，尚不代表所有資料集或參數皆有相同改善。

來源：[research_results/A_variance_regularization/](research_results/A_variance_regularization/)；正式結果由五組 Baseline 及五組 VR 的 `final.json` 計算，其他探索結果也保留，但不納入上述正式統計。詳細設計：[實驗 A](extension_experiments/formal_A/README.md)。

## 5. 實驗 B：對比損失計算效率

B 以 A 的 VR 訓練流程為基礎，研究在不刻意縮短訓練 Epoch 或減少線性評估的前提下，如何降低對比損失計算時間。

- B0：VR-EPAGCL，沿用原對比損失實作。
- B1：每個視圖的投影表示只做一次 L2 正規化，後續相似度運算重用結果，保留原正負樣本定義及兩個方向各自的隨機分批。
- B2×2 / B2×4 / B2×8：基於 B1，將 2、4 或 8 組互相獨立的相似度矩陣合併成 `torch.bmm`（批次矩陣乘法）運算。未將不同批次的節點混為同一負樣本集合，尾端不足 256 個節點的批次獨立處理。

### 5.1 數值驗證與短期測速

B1、B2 的對比損失、輸入梯度及包含 VR 的模型參數梯度，在實作所測試的輸入規模與容許誤差（`rtol=1e-4`、`atol=1e-5`）內通過驗證。此結果不保證浮點數逐位元相同，也不代表完整訓練軌跡必然相同。

Coauthor-Phy，RTX 2070、PyTorch 2.3.0+cu121；下表使用每輪 10 次預熱、20 次計時，共三輪的主模型訓練步驟測速結果，不含線性評估與前處理。

| 方法 | 單步中位數 (s) | 單步平均 (s) | 相對 B0 中位數時間減少 |
| --- | ---: | ---: | ---: |
| B0 | 1.6023 | 1.5497 | — |
| B1 | 0.9525 | 1.1317 | 40.56% |
| B2×2 | 0.5145 | 0.6998 | 67.89% |
| B2×4 | 0.4324 | 0.6235 | 73.01% |
| B2×8 | 0.3642 | 0.5540 | 77.27% |

B2×8 的合併中位數加速約 4.40 倍；另一份 5 次預熱測試存在一輪 B2×8 反而較慢的結果，因此兩份測速紀錄均保留。測速未顯示峰值 GPU allocated memory 明顯降低。

來源：[B 的完整輸出目錄](research_results/B_computational_efficiency/)；主要測速為 [`20261120_183748_B012_compare/comparison_b012.json`](research_results/B_computational_efficiency/20261120_183748_B012_compare/comparison_b012.json)，等價性紀錄為 `equivalence.json` 與 `equivalence_b2.json`。

### 5.2 E3 完整訓練比較

Coauthor-Phy，五組配對種子 20261129–20261203，維持 A 的 VR 設定，1000 Epoch 與 20 次線性評估。各組配對的資料切分 SHA-256 紀錄一致。

| Seed | B0 (%) | B2×8 (%) |
| --- | ---: | ---: |
| 20261129 | 93.839 | 92.024 |
| 20261130 | 94.202 | 94.590 |
| 20261201 | 94.213 | 94.398 |
| 20261202 | 94.311 | 94.822 |
| 20261203 | 93.807 | 94.176 |
| 平均 ± 樣本標準差 | 94.074 ± 0.233 | 94.002 ± 1.131 |

| 時間指標（五次平均） | B0 | B2×8 |
| --- | ---: | ---: |
| 完整訓練 (s) | 3369.92 | 2059.20 |
| 主模型訓練 (s) | 3275.21 | 1965.29 |
| 完整訓練加速倍率 | 1.00× | 1.637× |

B2×8 的完整訓練時間減少約 38.89%，平均測試準確率差異為 −0.072 pp，但樣本標準差增加，且其中一個種子的準確率下降約 1.82 pp。因此本實驗顯示效率改善與接近的平均分類表現，不主張完整訓練品質、穩定性或數值結果完全等價。短期測速的 4.40 倍與完整訓練的 1.637 倍為不同測量範圍，不可混用。

來源：[research_results/B_computational_efficiency/](research_results/B_computational_efficiency/)，正式統計只使用頂層十個 `*_E3_formal_*` 實驗的 `final.json`，不混入同種子重播與診斷結果。詳細執行計畫與說明見 [EXPERIMENT_PLAN.md](extension_experiments/efficiency_B/EXPERIMENT_PLAN.md)、[README_E3.md](extension_experiments/efficiency_B/README_E3.md) 及 [README_E3_extra_2seeds.md](extension_experiments/efficiency_B/README_E3_extra_2seeds.md)。

## 6. 實驗 C：反向刪邊與離線 EPR 分析

### 6.1 方法與索引修正

對既有邊 $(u,v)$，先根據無向化圖的節點度數 $d_u,d_v$ 計算 $w_{uv}=1/\sqrt{d_ud_v}$。在原版與反向刪邊比較中，分別使用下列權重正規化方式（$\bar w$ 為邊權重平均）：

$$
\text{Baseline:}\quad \frac{w_{uv}-w_{\min}}{\bar w-w_{\min}}
$$

$$
\text{Reverse Drop:}\quad \frac{w_{\max}-w_{uv}}{w_{\max}-\bar w}
$$

兩組保留原本的加邊流程，反轉的是刪邊權重排序方向，不改寫 EPR 定義。兩組均不加入 VR。使用端點配對（Ordered Endpoint Pair Mapping）將無向化後計算的權重對齊原始邊索引，保留原始抽樣邊的排列。實際 Cora 紀錄顯示原始與無向化邊索引均有 10,556 欄、無重複邊或缺少反向邊，但順序不同；因此此修正同時套用於 C 的兩個比較組，並非 C 的 Baseline 與歷史未對齊版本完全相同。

### 6.2 節點分類

Cora，500 Epoch、五組配對種子 20261119–20261123，每 50 Epoch 評估一次，線性評估 3000 Epoch。兩視圖名義刪邊率為 0.2、0.3，候選加邊率為 0.1。

| 方法 | Test Accuracy (%)（平均 ± 樣本標準差） |
| --- | ---: |
| Baseline | 83.127 ± 0.652 |
| Reverse Drop | 82.740 ± 0.571 |

平均配對差異為 −0.387 pp，未觀察到平均分類準確率提升。來源：[C training/comparison.csv](research_results/C_reverse_edge_dropping/training/20261125_012440_810530_both_Cora/comparison.csv) 與 [summary.json](research_results/C_reverse_edge_dropping/training/20261125_012440_810530_both_Cora/summary.json)。原始 JSON 為母體標準差，本表轉為樣本標準差以統一 A、B、C 口徑。

### 6.3 離線 EPR 與圖結構分析

另外以原版 GitHub `EPR.py` 的定義，在無向化後的圖上計算非孤立節點的平均 EPR。此分析使用真實類別標籤，但標籤僅用於離線分析，不回傳至增強抽樣或自監督訓練；抽樣也不是對既有訓練 RNG 軌跡的逐步重播。

原始 Cora 圖 EPR 為 0.168300。正式分析使用五組種子、每組 500 次增強抽樣，每種方法每個階段有 2500 筆有效觀測值。

| 階段 | Baseline 平均 EPR | Reverse Drop 平均 EPR | 差異 |
| --- | ---: | ---: | ---: |
| View 1 刪邊 | 0.176169 | 0.168368 | −0.007800 |
| View 1 加刪邊 | 0.177468 | 0.169595 | −0.007874 |
| View 2 刪邊 | 0.181466 | 0.168486 | −0.012980 |

| 階段 | Baseline 平均孤立節點 | Reverse Drop 平均孤立節點 | Baseline 無向化後實際刪邊率 | Reverse Drop 無向化後實際刪邊率 |
| --- | ---: | ---: | ---: | ---: |
| View 1 刪邊 | 194.5500 | 8.2356 | 6.133% | 4.155% |
| View 2 刪邊 | 341.9652 | 20.2464 | 12.624% | 9.335% |

反向刪邊使 EPR 更接近原始圖，且減少孤立節點；但分類準確率未隨之提升。名義刪邊率相同不代表無向化後保留的圖結構相同，實際刪除率亦有差異，因此目前不能將所有結果完全歸因於權重排序方向。此結果提供後續研究 EPR、圖結構擾動強度與表示品質之關係的依據。

來源：[C 正式 EPR 分析](research_results/C_reverse_edge_dropping/epr_analysis/20261125_020958_891111_Cora_original_epr/)，包含 `epr_samples.csv`、`summary.csv`、`comparison.csv`、`seed_summary.csv`、`edge_alignment.json` 與原始 EPR 紀錄。詳細方法：[extension_experiments/reverse_drop_C/README.md](extension_experiments/reverse_drop_C/README.md)。

## 7. 節點預測與視覺化

`epagcl_predict_nodes.py` 以訓練好的本機模型權重為輸入，凍結編碼器並在原始圖計算嵌入，再以訓練節點的表示與標籤訓練線性分類器。驗證集用於模型選擇，最後才評估測試集，輸出節點真實類別、預測類別、正確性與 softmax 信心分數。這是傳導式（Transductive）節點分類設定，不等於在訓練分類器時使用測試節點標籤。可選擇原始特徵的線性分類作為對照。

- [EPAGCL_prediction_test_README.md](EPAGCL_prediction_test_README.md)：預測流程、參數及輸出格式。
- [figure3_output/](figure3_output/)：Cora、CiteSeer 的本機 t-SNE 圖形。

此處保留已生成的圖表與紀錄；尚未重新核對所有視覺化圖表與模型權重的對應，故不從圖表額外宣稱新的分類成效。模型權重未納入儲存庫，預測工具必須使用自行訓練或經授權取得的權重。

## 8. 使用方式與實驗重現

### 8.1 環境與資料集

原作者 README 記載 Python 3.11.7、PyTorch 2.3.0、PyTorch Geometric 2.5.3，以及 NumPy、SciPy、OGB、THOP、tqdm 等依賴。這是原版參考環境，而非本機所有實驗環境的完整鎖定清單。B 的測速紀錄使用 RTX 2070 與 PyTorch 2.3.0+cu121；完整環境資訊見 B 輸出中的 `environment.json`。不同 GPU、CUDA、PyTorch 版本可能產生不同速度及浮點數結果。

在獨立的 Python 環境中配置符合 GPU 與 PyTorch 版本的依賴。資料集依原本程式的路徑設定準備，歷史實驗使用專案外的 `../datasets/` 目錄，例如 `../datasets/Cora` 與 `../datasets/Coauthor-Phy`。原始資料集與虛擬環境不包含在儲存庫中。其他機器應調整歷史記錄中的絕對路徑，而非照搬 `/home/user/...`。

### 8.2 入口指令

在專案根目錄及已配置的環境下執行。以下為原作者指令與本機研究模組的使用入口；完整正式參數以各次結果中的 `args.txt`、`args.json` 及研究模組說明為準。

```bash
# 原版風格：單視圖加邊與分批計算
python main.py --add_single --batch_compute

# A：查閱實驗選項及執行單元測試
python extension_experiments/run.py --help
python -m unittest discover -s extension_experiments/tests -v

# B：完整訓練入口
python extension_experiments/efficiency_B/train_e3.py --help

# C：配對訓練及離線 EPR 分析入口
python extension_experiments/reverse_drop_C/run.py --help
python extension_experiments/reverse_drop_C/epr_analysis.py --help

# 節點預測工具（需本機模型權重）
python epagcl_predict_nodes.py --help
```

A 正式參數參考（五組種子應分別執行；以下只示範第一組 VR 實驗）：

```bash
python extension_experiments/run.py --method vr --dataset Coauthor-Phy \
  --epoch 1000 --repeat 1 --seed 20261123 --eval 50 --eval-epoch 3000 \
  --lr 0.01 --weight-decay 0.001 --temperature 0.5 \
  --edge-add-rate 0.1 --edge-drop-rate-1 0.1 --edge-drop-rate-2 0.4 \
  --feat-drop-rate-1 0 --feat-drop-rate-2 0 --vr-lambda 10 --vr-gamma 0.002
```

對應 Baseline 使用 `--method baseline` 並移除兩個 VR 參數。完整重現應逐次對照 `args.json`（包括裝置、維度、最佳化器及分批設定）；上述範例仍依賴程式的其餘預設值，未在新 clone 環境重新執行驗證。

C 正式離線 EPR 分析的主要參數為 `--dataset Cora --device cpu --samples 500 --repeat 5 --seed 20261119 --edge-add-rate 0.1 --edge-drop-rate-1 0.2 --edge-drop-rate-2 0.3`，搭配 `extension_experiments/reverse_drop_C/epr_analysis.py` 使用。該程式不需要先訓練模型。

### 8.3 評估流程

A、B、C 使用固定的訓練／驗證／測試節點切分（10%／10%／80%）。同一組配對使用相同種子與切分；每次定期線性評估只用訓練集標籤訓練分類器，依驗證集選擇分類器及編碼器檢查點，最後才使用測試集取得測試準確率。線性評估不將梯度傳回編碼器。新實驗與歷史 Q3 不共用統計口徑。

本儲存庫不提供任何 `.pt`、`.pth`、`.ckpt` 或 `.safetensors` 檔案，**包括固定資料切分張量**。原始切分保留在本機；新的執行需透過 `fixed_eval.py` 所用的切分流程重新建立索引。既有的種子、參數與部分切分雜湊值保留於文字紀錄，以供比對；不同版本、節點順序及 GPU 非確定性運算仍可能導致不同結果。相同種子不構成完全重現保證。

## 9. 儲存庫結構與研究資料

```text
EPAGCL/
├── README.md                          # 本研究說明
├── README_upstream.md                 # 原作者 README
├── Arguments.py / main.py / ...       # 原版程式及工程修正
├── EPAGCL_Q3_Cora_changes.patch
├── run_*_q3_all.sh                     # 歷史 Q3 執行腳本
├── extension_experiments/
│   ├── run.py                         # A
│   ├── fixed_eval.py
│   ├── vr_regularization.py
│   ├── tests/
│   ├── formal_A/                      # A 正式實驗腳本與雜湊資訊
│   │   ├── outputs/                  # A 原始輸出，僅本機
│   │   └── logs/                     # 訓練日誌，僅本機
│   ├── efficiency_B/                  # B
│   ├── reverse_drop_C/                # C
│   └── splits/                        # 切分檔案僅本機（不提交 .pt）
├── research_results/
│   ├── reproduction/
│   ├── q3_ablation/
│   ├── A_variance_regularization/
│   ├── B_computational_efficiency/
│   ├── C_reverse_edge_dropping/
│   │   ├── training/
│   │   └── epr_analysis/
│   └── manifest.csv
├── scripts/export_epagcl_results.py   # 研究結果匯出工具
├── figure3_output/
└── logs/                              # 全部僅本機，不提交
```

`research_results/` 的首次整理包含 689 個輕量實驗檔案。排除 log 目錄中的 2 個檔案後，目前 Git 版本保留 687 個實驗檔案及一份 [`manifest.csv`](research_results/manifest.csv)，共 688 個檔案。A 實驗原始輸出統一位於 `extension_experiments/formal_A/outputs/`；manifest 的 A 來源路徑同步更新，原始結果內容及 SHA-256 不變。原版重現三組、Q3 二十三組五次重複實驗全部保留；A、B、C 保留正式結果及前期探索、測速與診斷紀錄，不以準確率高低選擇結果。

`manifest.csv` 記錄來源路徑、匯出路徑、檔案大小及 SHA-256 內容雜湊值；匯出只複製資料，不修改原始紀錄，並將歷史資料夾名稱中的冒號轉為連字號以相容 Windows。完整 `trained_models/`、原始實驗 `outputs/`、所有 log 目錄及 .log 檔案、模型權重、切分張量、壓縮檔、虛擬環境與快取保留在本機，由 `.gitignore` 排除。ZIP 不作為重複的研究資料來源。

對保有完整原始輸出的本機專案，可使用下列指令檢查匯出計畫（僅預覽，不複製）：

```bash
python3 scripts/export_epagcl_results.py --root .
```

匯出工具使用 Python 標準函式庫；`--apply` 會建立新 `research_results/` 並核對檔案雜湊。已存在該目錄時會拒絕覆寫。一般 clone 不含完整原始輸出，無須執行匯出工具，即可閱讀已保留的研究結果。

## 10. 研究狀態與原始引用

目前的研究已取得三項主要觀察：A 在 Coauthor-Phy 的正式配對中緩解近似表示崩潰；B 在保留原損失數學目標的設計下改善主模型運算效率；C 顯示較小的 EPR 擾動與較少的孤立節點不必然轉化為更高的分類準確率。這三個方向尚未完成跨資料集驗證或整合式 A+B+C 訓練比較。現有數據、超參數與診斷紀錄可供後續分析，不宣稱為全面優於原論文的模型。

原版 EPAGCL 由 Yanchen Xu、Siqi Huang、Hongyuan Zhang 與 Xuelong Li 提出。使用原版方法時請引用原作者：

```bibtex
@inproceedings{xu2025epagcl,
  title = {Why Does Dropping Edges Usually Outperform Adding Edges in Graph Contrastive Learning?},
  author = {Xu, Yanchen and Huang, Siqi and Zhang, Hongyuan and Li, Xuelong},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {2025},
  volume = {39},
  number = {20},
  pages = {21824--21832},
  doi = {10.1609/aaai.v39i20.35488}
}
```

### 授權

原作者程式碼及第三方資料保留其原有權利。目前查閱的上游儲存庫未見明確的程式碼授權檔案，本專案不自行替原版程式碼設定開源授權。重製或散布原始碼及修改版本前，須確認原作者授權、GitHub 使用條款與各資料集條款。本文件與研究紀錄不表示取得任何額外散布授權。
