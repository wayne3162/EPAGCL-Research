# Experiment C — Reverse edge-dropping strategy

## 目的
比較 EPAGCL 原始刪邊正規化與反向正規化。唯一改動的訓練策略是刪邊權重轉換方向；保留原始加邊、刪邊抽樣、特徵遮蔽、模型及對比損失。

- `baseline`：使用根目錄 `Augmentation.drop_edge_weights` 的原始公式，並按邊端點對齊權重與原始抽樣順序。
- `reverse_drop`：呼叫 `augmentation_c.reverse_drop_edge_weights`，將 `(w - w.min()) / (w.mean() - w.min())` 改成 `(w.max() - w) / (w.max() - w.mean())`。
- `both`（預設）：同一份設定依 seed 配對執行上述兩組，不加入 VR，也不提前停止訓練。

## 實驗隔離

本版移除其他實驗留下的 variance_loss 欄位、零值 VR 紀錄、特徵變異與餘弦相似度診斷、以及特定 epoch 的額外 loss 紀錄。兩組直接以 `model.loss()` 反向傳播，每次完整執行指定的 epoch。只保留一般執行時間、固定資料切分、驗證集選模、Accuracy、實際刪邊率與邊索引對齊檢查；不匯入 A/B 的訓練模組。

## 放置位置
將整個 `reverse_drop_C/` 資料夾放到 `EPAGCL/extension_experiments/` 下。若已建立同名資料夾，僅將本包三個檔案放入其中。不要覆蓋根目錄的 `Augmentation.py`、`main.py`，或原有 A/B 程式。

保留既有 runner 的加邊行為：`--edge-add-rate` 傳入候選邊篩選函式，`add_edges()` 仍沿用原始預設抽樣參數，沒有把加邊率另外改傳給它。

共用現有 `Model.py`、`Augmentation.py`、`Load_data.py`、`Utils.py`、`extension_experiments/fixed_eval.py`、`extension_experiments/splits/`。不需額外套件；沿用原專案可正常執行的 Python 環境。

## 執行方式
從 EPAGCL 根目錄，在原有環境執行。先以 Cora 做小型測試（只確認可執行，不用此結果判斷效果）：

```bash
python extension_experiments/reverse_drop_C/run.py \
  --method both --dataset Cora --device cuda:0 \
  --epoch 20 --repeat 1 --seed 20261119 \
  --eval 10 --eval-epoch 30 \
  --lr 0.001 --weight-decay 0.0001 --temperature 0.3 \
  --edge-add-rate 0.1 --edge-drop-rate-1 0.2 --edge-drop-rate-2 0.3 \
  --feat-drop-rate-1 0.1 --feat-drop-rate-2 0.1
```

正式 Cora 配對比較（參考使用者提供的 Cora 筆記設定，沿用 extension runner 其他預設）：

```bash
python extension_experiments/reverse_drop_C/run.py \
  --method both --dataset Cora --device cuda:0 \
  --epoch 500 --repeat 5 --seed 20261119 \
  --eval 50 --eval-epoch 3000 \
  --optim Adam --lr 0.001 --weight-decay 0.0001 \
  --temperature 0.3 --edge-add-rate 0.1 \
  --edge-drop-rate-1 0.2 --edge-drop-rate-2 0.3 \
  --feat-drop-rate-1 0.1 --feat-drop-rate-2 0.1 \
  --feat-dim 256 --proj-hidden-dim 256 \
  --eval-optim Adam --eval-lr 0.01 --eval-weight-decay 0.0
```

如資料集位置非預設 `../datasets/Cora`，加上 `--dataset-path /完整/資料集/目錄`。`--batch-compute` 預設為 true（與原 extension runner 一致）。如已使用 uv 執行此專案，將上述 `python` 換成 `uv run python` 即可，不需重建環境。

## 輸出
`reverse_drop_C/outputs/<時間>_both_Cora/` 內：

- `args.json`：完整設定；`source_sha256.json`：原始依賴與 C 程式雜湊值。
- `baseline/`、`reverse_drop/`：每組的 `summary.json` 及每個 seed 的 `loss.csv`、`validation.csv`、`edge_drop.csv`、`final.json`、最佳驗證集 checkpoint。
- `comparison.csv`：相同 seed 的 Test Accuracy 與反轉組減原始組的百分點差。
- `summary.json`：各組 Test Accuracy 平均與母體標準差、配對差平均、截斷後預期刪邊率與實際平均刪邊率。
- `partial_summary.json`：已完成的 runs。
- `edge_alignment.json`：原始及無向化邊數、重複邊數、缺少反向邊的數量，以及是否需要重新對應權重。

`edge_drop.csv` 的刪除數是加邊前的 `edge_index` 欄數差，單位為有向 edge entries（不是無向邊對）。兩組使用同一個 p，但截斷可能使實際刪邊率不同。若差異明顯，後續可另做匹配刪邊率的控制實驗。此版本不計算 EPR，不以標籤參與圖增強；若要驗證 EPR 理論，另設計訓練後的離線 EPR 分析。

## 安全檢查與限制
原始 `drop_edge_weights()` 會呼叫 `to_undirected()`，但 `drop_edges()` 使用原始 `data.edge_index` 的欄順序。修正版透過有序端點 `(source, target)` 將兩組權重對回原始邊索引，保留原始圖及邊順序，不修改原始檔案。若存在重複邊或缺少反向邊，程式保留原始抽樣欄位並提示警告，兩組仍使用無向化後的節點度數；此時解讀結果須考慮理論的無向圖假設。若權重全相同或無法對應，則停止。

若原始程式或先前 A/B runner 存在權重與邊錯位，修正版的 baseline 不等於逐位元重現舊 baseline；應以 C 內部的配對 baseline 為對照。此修正不改變兩組之間唯一的實驗策略差異，即刪邊正規化方向。

注意：同 seed 不代表 CUDA 所有操作必然逐位元可重現；兩組使用相同初始化和切分設定，但後續訓練圖不同。未執行真實 Cora 或 GPU 訓練，請先跑小型測試。
