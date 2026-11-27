# EPAGCL 改進 B：訓練效率實驗設計（2026-11-20）

## 研究問題與方法界線

目標：在 VR-EPAGCL (E1) 防塌縮效果不受明顯損害的前提下，減少單一主模型 Epoch 與完整訓練的實際運行時間；不使用提前停止、不改變評估頻率來假裝核心演算法加速。

固定條件：Coauthor-Phy，VR λ=10、γ=0.002，GCN / feature dim=256，temperature=0.5，lr=0.01，weight_decay=0.001，edge add=0.1，edge drop=0.1/0.4，feature drop=0，batch_compute=True，batch size=256。沿用固定 split、驗證集選模，最後才用測試集。原始 Model.py/歷史實驗一律不改。

| 組別 | 修改 | 核心假設 | 狀態 |
|---|---|---|---|
| B0 | 原始 VR-EPAGCL | 現行運算成本的參照 | 已完成正式 E1 (1000 Epoch × 5)，仍需分項 profiling |
| B1 | 在 cal_loss 之前一次正規化兩個投影表示，批次重用 | 省下逐批 F.normalize 運算 | 待實作與測試 |
| B2 | B1 + 同時計算 2/4/8 批之相似度矩陣，保留每個方向原本隨機分組與候選樣本 | 減少 Python 迴圈和 GPU kernel launch 開銷 | 待實作與測試 |
| 其他 | 如果 profiler 證實 GCN/augmentation 更慢，改針對該部分設計 | 不應預設損失函式一定是瓶頸 | 依測量決定 |

**關鍵實作限制**：目前原始 `MyModel.loss` 調用兩次 `cal_loss`，兩次各自產生 `torch.randperm(num_nodes)`。B1/B2 應保持同樣的方向、分組、負樣本集、分母與 loss 公式，不可共用單一 randperm 造成樣本集合變動。最後一批 189 節點須單獨驗證。

## 第 1 階段：B0 短時效能診斷（現在執行）

運行 `profile_b0.py`。使用現有專案依賴；不訓練線性 probe、不讀取 test labels、不修改正式實驗輸出。預熱 5 steps，無 Profiler 計時 20 steps，另用 torch.profiler 量測 5 steps。先判斷時間耗在 (1) graph augmentation、(2) 兩視圖 GCN / projection、(3) contrastive、(4) VR penalty、(5) backward、(6) optimizer，並取得 CPU / CUDA top kernels。

**計時口徑**：訓練單步時間 `timing.json` 取無 Profiler 的 CUDA 同步牆鐘時間；Profiler 結果只用來定位瓶頸，不能直接拿 Profiler 開啟時的牆鐘時間計算加速倍率。`final.json.elapsed_seconds` 另包含定期線性 probe，不可視為純主模型訓練時間。

## 第 2 階段：B1/B2 損失與梯度等價性

使用固定的相同 z1/z2 輸入，先保存並恢復 CPU/CUDA 的 RNG 狀態，確保原版與候選版使用相同的隨機分批。測試 N=64、256、257、1025，並補 Coauthor-Phy N=34493 真實形狀。比較原始對比 loss、輸入梯度與模型參數梯度；要求 allclose(rtol=1e-4, atol=1e-5) 並報告最大絕對／相對差異。此容差是預先提出的試驗規格，超出時需分析，不能直接調大門檻掩蓋差異。VR 項完全沿用原式。

**不得以相同 Seed 但不同 GPU 執行的 Loss 曲線完全一致，取代同張量的等價性驗證。**

## 第 3 階段：速度與記憶體微型比較

在相同 GPU / batch size / 資料前處理 / 超參數下，對 B0、B1、B2 各做三輪重複計時；每輪預熱 5 steps、計時 20 steps，版本執行順序採 B0/B1/B2、B2/B1/B0、B0/B1/B2 交錯。記錄平均、median、P90、峰值 GPU allocated memory，以及是否 OOM。

實測加速倍率：`median_time(B0) / median_time(candidate)`。同時另外報告 `mean_time`，不要只挑最快一次。若候選不能穩定加速或導致顯存不足，則不進入長訓練。

建議暫訂放行門檻：每 Epoch 中位數至少減少 **10%** 且不會造成數值/梯度不一致或 GPU 顯存不足；這是研究設計門檻，非已觀察到的改善。

## 第 4 階段：完整訓練品質驗證

先用同一程式版本跑一個 1000-Epoch Fast VR 冒煙測試；若穩定，優先以**新的、事先固定的三個配對 Seed** 同時執行 B0 與最佳候選。若資源許可或用於正式學術報告，再補足五個配對 Seed。固定 1000 Epoch、每 50 Epoch probe、每次 probe 3000 Epoch、固定 split、驗證集選模、訓練期間不提前停止。保留所有結果。

核心指標：(i) 最終高 Loss 停滯比例（不得直接等同塌縮），(ii) 驗證選模後測試準確率的分布與標準差，(iii) 最佳驗證 encoder 的 mean_feature_std 與 mean_pairwise_cosine，(iv) 主模型 Epoch 時間、完整 wall-clock、峰值 GPU 記憶體，(v) 版本/參數雜湊。

VR-EPAGCL 現有五種子正式結果 (20261123–20261127)：Test mean **94.046%** (sample std **0.340 pp**)，最終高 Loss 停滯 **0/5**；1000 Epoch 單次 total run elapsed 平均 **3484.91s**（包含 20 次定期線性評估，排除預處理）。這些是歷史基準，不應與不同協定或不同硬體的測時混淆。

## 第 5 階段：研究結論

區分三個陳述：
1. 計算等價性：固定輸入、固定分批下 loss 和 gradient 一致（或誤差有報告）。
2. 效能改善：實測單 Epoch / 完整訓練時間下降，且顯存可接受。
3. 穩定性保持：新種子完整訓練中，分類、表示與停滯結果不因加速而惡化。

提前停止繼續僅為 Shadow Mode / 附屬研究；本計畫不會以減少 Epoch 數作為 B 的核心加速成果。
