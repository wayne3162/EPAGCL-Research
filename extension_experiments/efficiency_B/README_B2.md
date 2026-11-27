# B2：批次矩陣乘法

**繁體中文** · [English](README_B2.en.md)

[延伸實驗總覽](../README.md) · [B 總覽](README.md) · [B 結果](../../research_results/B_computational_efficiency/) · [B1](README_B1.md)

## 1. 計算改良

[model_b2.py](model_b2.py) 延續 B1 的正規化重用，將多個相互獨立的相似度矩陣乘法合併成 `torch.bmm`（Batched Matrix Multiplication）。令每批次大小為 $b=256$，特徵維度為 $D$，同一呼叫包含 $g$ 個批次（$g\in\{2,4,8\}$）。以 $[g,b,D]$ 與 $[g,D,b]$ 進行批次乘法，得到 $[g,b,b]$ 相似度，而不是建立 $[gb,gb]$ 的跨批次矩陣。

此方式保留每個批次自己的正負樣本、原分批大小與兩方向各自的 `randperm`；尾端不足 256 的批次獨立計算，不填補、不丟棄。Coauthor-Phy 34,493 個節點可分為 134 個完整批次及 189 個尾端節點。運算目標與 B1 相同，但浮點累加順序可能不同。

## 2. 數值驗證

[verify_b2.py](verify_b2.py) 驗證群組大小 2、4、8。固定輸入的節點規模 64、256、257、1025、34493 之損失、輸入梯度皆通過；包含 VR 的模型參數梯度檢查限於前四個規模。`rtol=1e-4`、`atol=1e-5`；檢查雙方向排列與 RNG 狀態一致。此驗證不能推出完整訓練軌跡完全一致。

來源：[equivalence_b2.json](../../research_results/B_computational_efficiency/20261120_181453_B2_verify_seed_20261123/equivalence_b2.json)。

## 3. 統一測速結果

Coauthor-Phy，RTX 2070，PyTorch 2.3.0+cu121；10 次預熱、20 次計時、三輪，無 profiler、不含線性評估及前處理。P90 是 90 百分位數時間，與平均數、中位數分開報告。

| Variant | Median (s) | Mean (s) | P90 (s) | Reduction vs B0 |
| --- | --- | --- | --- | --- |
| B0 | 1.6023 | 1.5497 | 1.8779 | 0.00% |
| B1 | 0.9525 | 1.1317 | 1.3768 | 40.56% |
| B2x2 | 0.5145 | 0.6998 | 0.9792 | 67.89% |
| B2x4 | 0.4324 | 0.6235 | 0.9026 | 73.01% |
| B2x8 | 0.3642 | 0.5540 | 0.8456 | 77.27% |

B2×8 合併中位數比 B0 低 77.27%（約 4.40 倍）。三輪中位數均比各自 B0 快；較早的 5 次預熱版本第一輪 B2×8 為 2.3194 s、B0 為 1.6247 s，反而較慢約 42.76%。因此保留早期與主要測速。峰值 allocated memory 約 5555 MiB，未顯示明顯下降。

來源：[comparison_b012.json](../../research_results/B_computational_efficiency/20261120_183748_B012_compare/comparison_b012.json)、[5 次預熱測試](../../research_results/B_computational_efficiency/20261120_181707_B012_compare/comparison_b012.json)。

## 4. 使用入口

```bash
python extension_experiments/efficiency_B/verify_b2.py --help
python extension_experiments/efficiency_B/benchmark_b012.py --help
```

B2×8 後續進入 [E3 完整訓練](README_E3.md)。短期測速的 4.40 倍不能替代完整訓練約 1.637 倍的結果。原始輸出與 Log 保留本機；精選驗證與比較 JSON 在研究結果目錄。
