EPAGCL 改進 B1：獨立實作、等價性與 B0/B1 速度對照
==================================================

用途
----
本套件建立 B1：在損失函式進入逐批運算之前，將 z1、z2 各正規化一次；
除此之外保留兩方向各自 torch.randperm、對比樣本、分母、VR 正則化與訓練設定。
不用修改 Model.py、profile_b0.py、正式 A 實驗輸出、原始模型檢查點。

需求
----
請沿用已能執行 B0 的 EPAGCL WSL Python 環境，不需要安裝新套件。
預設用 CUDA GPU；本套件不會進行線性評估或讀取測試標籤。

放置路徑
--------
把以下三個 .py 檔放入：
~/projects/EPAGCL/extension_experiments/efficiency_B/

  model_b1.py       獨立 MyModelB1；只覆寫 loss 的計算路徑
  verify_b1.py      精確比較 B0/B1 的 loss、z1/z2 梯度、參數梯度、RNG
  benchmark_b01.py  在相同 GPU 上三輪交錯計時、記錄顯存

README_B1_zh_TW.txt 可一併存入同資料夾，但不影響程式執行。

第一步：數值與梯度等價性驗證（必須先通過）
--------------------------------------
cd ~/projects/EPAGCL
python -u extension_experiments/efficiency_B/verify_b1.py \
    --device cuda:0 --seed 20261123 --batch-size 256

會執行：
  固定投影張量：N = 64、256、257、1025、34493
  各自比較：loss、對 z1/z2 的梯度、兩方向隨機分批一致性
  小型合成圖：N = 64、256、257、1025
  各自比較：模型參數梯度、原版 VR 與 total loss
  固定容差：rtol=1e-4、atol=1e-5，不得放寬
  34,493 節點的最後一批為 189 節點

輸出：
extension_experiments/efficiency_B/outputs/<時間>_B1_verify_seed_20261123/equivalence.json
只有所有檢查通過，JSON 的 "passed" 才會是 true。
若失敗，保留 JSON；先分析不一致，不要直接跑速度比較。

第二步：B0/B1 三輪交錯對照（第一步通過後）
----------------------------------------
cd ~/projects/EPAGCL
VERIFY=$(find extension_experiments/efficiency_B/outputs \
    -type f -name equivalence.json | sort | tail -n 1)
python -u extension_experiments/efficiency_B/benchmark_b01.py \
    --verification-json "$VERIFY" \
    --device cuda:0 --seed 20261123 \
    --rounds 3 --warmup 5 --timed 20

順序：Round 1 B0/B1、Round 2 B1/B0、Round 3 B0/B1。
每輪兩版本固定相同初始化種子與資料、GPU、優化器和訓練參數。
每次均測量 5 個 warmup + 20 個無 Profiler 的正式訓練步驟。

輸出：
extension_experiments/efficiency_B/outputs/<時間>_B01_compare/comparison.json
內含每輪 mean、median、P90、顯存峰值、每步原始計時、配對與合併加速率。
正式時間不包括預處理與線性評估，不能直接跟 1000 Epoch 的 elapsed 相比。
若驗證未通過、源碼在驗證後變更、GPU 不相同，計時程式會拒絕執行。

研究判讀
--------
研究計畫的暫定 B1 放行門檻為：單 Epoch 中位數至少降低 10%，
且等價性驗證通過，GPU 顯存可以接受。實驗數據未取得前不可宣稱改善。
圖增強可能是主要時間來源，因此 B1 也可能只帶來有限的整體加速。

驗證狀態
--------
作者端完成語法檢查與小規模 CPU 測試。由於執行環境缺乏原始 PyG/CUDA，
尚未在實際 RTX 2070 GPU 或 Coauthor-Phy 資料上驗證。
真正的通過狀態以你在 WSL 執行 verify_b1.py 產生的 JSON 為準。
