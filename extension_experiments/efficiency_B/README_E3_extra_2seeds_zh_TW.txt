EPAGCL E3 改進 B：追加兩組配對種子（2026-11-21）
================================================

用途
----
沿用原來的 train_e3.py（不修改程式），在原始三組 Seed 已完成的前提下，自動依序增加四次完整正式訓練。每次 1000 Epoch、每 50 Epoch 一次線性 Probe（3000 Epoch）、VR lambda=10、gamma=0.002、原始動態增強、沒有提前停止。

預先固定順序（Seed 是亂數整數，而不是日期）
--------------------------------------------
1. 20261202 / b2x8
2. 20261202 / b0
3. 20261203 / b0
4. 20261203 / b2x8

安裝與執行
----------
1. 將 run_e3_extra_2seeds.py 放進：
   ~/projects/EPAGCL/extension_experiments/efficiency_B/

2. 先預覽，不啟動訓練：
   cd ~/projects/EPAGCL
   python -u extension_experiments/efficiency_B/run_e3_extra_2seeds.py --dry-run

3. 確認順序與既有來源版本無誤後，正式執行：
   python -u extension_experiments/efficiency_B/run_e3_extra_2seeds.py

設計保護
--------
- 六次原始正式訓練（20261129/30、20261201）必須存在且可完成稽核。
- E3 Smoke（20261128、1000 Epoch）必須已成功。
- 新執行之程式 SHA256、參數、GPU 與原六次一致；若變更則停止。
- 每次只有一個訓練程序使用 GPU，執行完成才能進行下一次。
- 中斷後可重執行同命令，已完成且驗證通過的 Run 會跳過；失敗或存在不一致的輸出會停止，並不以低分為由重訓。
- 原始 Model.py、run.py、train_e3.py 不會更改，歷史資料不覆蓋。

輸出與核查
----------
- 正式輸出：extension_experiments/efficiency_B/outputs/*_E3_formal_*
- 本輪日誌：extension_experiments/efficiency_B/outputs/e3_extra_batch_logs/<session>/
- 四次成功後，呼叫既有 summarize_e3.py 比較全部五個 Seed，結果保存為 paired_summary.log。
- 將新的 paired_summary.log 上傳供下一輪模型品質與效率分析。保留 Seed 20261129 的原始低分樣本。

檢查狀態
--------
本腳本已通過 Python 語法檢查、模擬 dry-run、逐次執行、斷點續跑與五 Seed 彙整呼叫測試；尚未在使用者 WSL／GPU 上實際訓練。
