# B1：重用 L2 正規化結果

**繁體中文** · [English](README_B1.en.md)

[延伸實驗總覽](../README.md) · [B 總覽](README.md) · [B 結果](../../research_results/B_computational_efficiency/)

## 1. 計算改良

[model_b1.py](model_b1.py) 在原 EPAGCL 的對比損失計算前，分別對兩個視圖的投影表示做一次 L2 正規化（`F.normalize`，p=2、dim=1、eps=1e-12）。後續同視圖及跨視圖相似度重用這些表示，避免反覆正規化；不 detach 張量，因此保留梯度傳遞。令 $I$ 為目前一個獨立批次的節點索引，$\tau$ 為溫度參數，單方向損失可表示為：

$$
\widehat z_i^{(v)}=\frac{z_i^{(v)}}{\max(\lVert z_i^{(v)}\rVert_2,\epsilon)},\quad
s_{ij}^{vw}=\exp\left(\frac{\widehat z_i^{(v)}\cdot\widehat z_j^{(w)}}{\tau}\right).
$$

$$
\ell_i^{1\to2}=-\log\frac{s_{ii}^{12}}{\sum_{j\in I}s_{ij}^{11}+\sum_{j\in I}s_{ij}^{12}-s_{ii}^{11}}.
$$

反向損失交換視圖，兩方向仍分別抽取原本的隨機排列（`randperm`）。B1 不擴大負樣本集合，不變更批次大小、VR 損失或線性評估規格。

## 2. 固定輸入等價性驗證

[verify_b1.py](verify_b1.py) 使用 `rtol=1e-4`、`atol=1e-5`、batch size 256，檢查對比損失及輸入梯度。節點數 64、256、257、1025、34493 均通過；包含 VR 的模型參數梯度在前四個規模通過（未在 34493 規模做同項模型梯度測試）。檢查兩方向隨機排列與損失後 RNG 狀態一致。`passed=true` 表示在測試容許誤差內，並非浮點數逐位元一致。

來源：[equivalence.json](../../research_results/B_computational_efficiency/20261120_172257_B1_verify_seed_20261123/equivalence.json)。

## 3. B0/B1 短期測速

Coauthor-Phy，RTX 2070，無 profiler，不含線性評估與前處理。

| Record | Rounds | Warmup | Timed | B0 median (s) | B1 median (s) | Reduction |
| --- | --- | --- | --- | --- | --- | --- |
| 20261120_172552_B01_compare | 3 | 5 | 20 | 1.6266 | 1.1393 | 29.95% |
| 20261120_180316_B01_compare | 1 | 5 | 20 | 1.4464 | 1.0622 | 26.56% |

第一次測試第三輪 B0 中位數 4.7069 秒，B1 為 0.9882 秒，較其他輪有明顯波動；另一次重測仍觀察到 26.56% 中位數時間減少。B1 的結論以實際成對測速為限，不宣稱固定倍率加速。B0/B1/B2 的統一 10 次預熱測試見 [B2](README_B2.md)。

來源：[第一次比較](../../research_results/B_computational_efficiency/20261120_172552_B01_compare/comparison.json)、[重測](../../research_results/B_computational_efficiency/20261120_180316_B01_compare/comparison.json)。

## 4. 使用入口

```bash
python extension_experiments/efficiency_B/verify_b1.py --help
python extension_experiments/efficiency_B/benchmark_b01.py --help
```

完整參數與環境以各 JSON 為準；執行入口與原始輸出留在 B 目錄。Log 不上傳。B1 是 B2 的基礎；完整訓練比較見 [E3](README_E3.md)，不是 B1 單獨完整訓練的證據。
