EPAGCL 改進 B：第 1 階段基準效能分析（B0）
=======================================

目的：只測量現有 VR-EPAGCL 的一個主模型訓練 Epoch，不變更任何
原始檔、模型權重、固定切分、正式實驗結果，也不進行線性評估。

環境：使用原本能執行 EPAGCL 的 WSL Python/uv 環境；不需安裝新套件。

1. 將 profile_b0.py 放入：
   ~/projects/EPAGCL/extension_experiments/efficiency_B/profile_b0.py

2. 在 WSL 執行：
   cd ~/projects/EPAGCL
   python -u extension_experiments/efficiency_B/profile_b0.py \
       --device cuda:0 --seed 20261123 --warmup 5 --timed 20 \
       --profile-active 5

輸出位於：
   extension_experiments/efficiency_B/outputs/<時間>_B0_seed_20261123/
   timing.json             主模型 Epoch 計時（無 Profiler 額外開銷）
   stage_diagnostic.csv   主模型各階段同步計時（僅用來粗估瓶頸）
   profiler_top_ops.txt    最耗時 PyTorch/CUDA 運算（診斷用，不能作加速倍率）

請將上述三個結果檔與終端輸出提供給我，再決定 B1/B2 的優化順序。

研究注意事項：
- timing.json 的每 Epoch 時間不含資料前處理與線性評估，
  不能直接與正式 1000 Epoch 的 final.json 總時間相除。
- stage_diagnostic.csv 每段都額外做 GPU 同步，僅適合診斷相對瓶頸。
- profiler 本身有額外測量成本，報告總時間要使用無 Profiler 計時。
- 本次沒有調整原始 MyModel.loss、正負樣本或 VR 公式。
- 現有 GPU 的運算若非確定性，同 Seed 的長訓練軌跡可能不同；
  因此速度分析和模型品質分析應分開設計。
