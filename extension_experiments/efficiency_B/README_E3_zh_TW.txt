EPAGCL E3：B0 VR-EPAGCL vs B2x8 Fast VR-EPAGCL（2026-11-20）
=================================================================

研究定位
---------
本階段將 B2x8（經 B1/B2 的固定輸入數值／梯度等價性驗證）
放入與原 run.py 相同的正式訓練協定，比較「完整品質」和
「包含線性評估的完整耗時」。B2x8 短時微型測試下降 77.27% 只是
本階段的研究動機，不是 1000 Epoch 的既有成果。

檔案安裝：只新增，不覆蓋
------------------------
將 train_e3.py、summarize_e3.py 放入
    ~/projects/EPAGCL/extension_experiments/efficiency_B/
並保留同資料夾中的 model_b1.py、model_b2.py。不改 Model.py、run.py、
fixed_eval.py、vr_regularization.py，也不修改既有 formal_A、outputs。

環境：已可跑 B0/B1/B2 的 WSL EPAGCL Python 環境、RTX 2070。
不需額外安裝套件。程式依賴你現有的 extension_experiments/fixed_eval.py。

固定實驗協定
------------
資料：Coauthor-Phy，34,493 節點；固定 10%／10%／80% 切分。
模型：GCN 輸出/投影 256，temperature=0.5。
優化器：Adam，lr=0.01，weight_decay=0.001。
圖增強：edge add 0.1，edge drop 0.1/0.4，feature drop 0/0。
對比損失：batch_compute=True；批大小 256（使用原始方法預設）。
VR：lambda=10、gamma=0.002；變異正則化函式原封不動。
1000 Epoch；每 50 Epoch 線性評估、probe 3000 Epoch，依驗證集選模；
訓練中不提前停止，最終選模後才計算測試準確率。

唯一的實驗因素：--variant b0 或 b2x8。
B0 使用原始 Model.MyModel；B2x8 使用 MyModelB2(groups_per_call=8)。
兩個版本都使用相同 run.py 的圖增強、固定切分、評估器。

重要：Phase、Seed 預先固定
-------------------------
1. 功能冒煙 sanity（20 Epoch，probe 30 Epoch）：Seed 20261128，B2x8。
2. 完整單次 smoke（1000 Epoch，probe 3000 Epoch）：Seed 20261128，B2x8。
3. 正式三個新 Seed（每個 Seed B0 與 B2x8 一對）：
   20261129、20261130、20261201；這是 Seed 編號，不代表實際日期。
先執行 1，核對結果，再執行 2；確認 2 完成後才執行 3。
不能因為某個 Seed 準確率偏低就移除或重抽。

步驟一：20 Epoch 功能冒煙（現在先做）
-----------------------------------
cd ~/projects/EPAGCL
python -u extension_experiments/efficiency_B/train_e3.py \
    --variant b2x8 --phase sanity --seed 20261128 --device cuda:0

正常完成會印 E3 COMPLETED，並產出獨立路徑，例如：
    extension_experiments/efficiency_B/outputs/<時間>_E3_sanity_b2x8_.../
所有輸出保存在新資料夾，不覆寫歷史成果。

步驟二：1000 Epoch 單次品質／流程冒煙
------------------------------------
python -u extension_experiments/efficiency_B/train_e3.py \
    --variant b2x8 --phase smoke --seed 20261128 --device cuda:0

檢查 probe_count=20、completed_epochs=1000、final.json 存在，
並核對測試準確率、最終對比 Loss、表示 Std/Cosine、耗時及 peak MiB。
這是單次品質測試，不是正式的多 Seed 統計。

步驟三：正式配對 Seed（只有 smoke 成功後才執行）
------------------------------------------------
按照以下固定順序各開一個獨立 Python 程序，避免 GPU 同時競爭：

Seed 20261129：先 B0、後 B2x8
Seed 20261130：先 B2x8、後 B0
Seed 20261201：先 B0、後 B2x8

每個指令格式（替換 VARIANT 和 SEED）：
python -u extension_experiments/efficiency_B/train_e3.py \
    --variant VARIANT --phase formal --seed SEED --device cuda:0

事先固定三組配對的順序；不挑選比較有利的 Run。
若未來增加五種子，先固定額外兩組 Seed 再執行。

步驟四：結果完整後唯讀彙整
-------------------------
python -u extension_experiments/efficiency_B/summarize_e3.py

這個彙整程式不讀測試標籤、不重跑訓練、不刪檔，僅讀各 Run 的
final.json、args.json、environment.json、source_sha256.json；
檢查配對 Seed 的固定切分雜湊、GPU、PyTorch 版本、程式雜湊及協定一致性。
若資料不完整或重複，不產生誤導性的整體估計。

輸出檔案
--------
    args.json                固定訓練設定、Seed、Phase、Variant
    source_sha256.json       原始、B1/B2、E3 執行碼 SHA256
    environment.json         GPU、PyTorch、split 路徑及雜湊
    run_01_seed_*/loss.csv   每 Epoch 對比/VR/總損失
    run_01_seed_*/validation.csv  每 50 Epoch 的最佳驗證結果
    run_01_seed_*/epoch_timing.csv 每 Epoch 純主模型與 probe 分開計時
    run_01_seed_*/best_by_validation.pt 最佳驗證選模權重
    run_01_seed_*/final.json 最終測試、選模資訊、表示診斷、時間及顯存
    summary.json            單次彙整

時間口徑：main_model_seconds = GPU 同步的每步主模型時間加總，排除 probe。
linear_probe_seconds = 20 次 Probe 總時間（含 Encoder 推論及模型儲存）。
elapsed_seconds = 整次訓練＋驗證＋最終測試，不含一次性資料預處理。
新增逐 Epoch GPU 同步計時，會影響相對 run.py 的執行條件：
請只在相同 E3 計時程式下比較 B0/B2x8；不得將短時 77.27% 延伸宣稱
完整訓練已節省 77.27% 時間。

來源和限制
----------
B0 Baseline 以已上傳 Model.py 的原始 MyModel 為準；
VR 函式以原 vr_regularization.py 為準；
圖增強、固定切分與線性 probe 由現有 run.py/fixed_eval.py 提供。
這次未上傳 fixed_eval.py，程式直接沿用本地現有版本並記錄 SHA256；
品質驗證仍會受非確定性 GPU 計算影響，不要求不同版本 1000 Epoch
Loss 曲線逐點完全一致。
