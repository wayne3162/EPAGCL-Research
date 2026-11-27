# EPAGCL 延伸實驗

**繁體中文** · [English](README.en.md)

本目錄以原版 EPAGCL 為基礎，分別研究表示穩定性、計算效率，以及邊擾動與 EPR 的關係。A、B、C 是三個可獨立檢驗的研究方向；B 延續 A 的 VR 設定，C 則使用不含 VR 的比較組。目前尚未進行三者整合訓練。

## 實驗導覽

| 實驗 | 研究問題 | 正式觀察 | 文件 |
| --- | --- | --- | --- |
| A：變異數正規化 (VR) | 表示是否發生近似崩潰？ | Coauthor-Phy：59.148 ± 19.087% → 94.046 ± 0.340% | [中文](formal_A/README.md) / [EN](formal_A/README.en.md) |
| B：計算效率 | 能否保留原對比損失目標並降低計算時間？ | B2×8 短期步驟中位數減少 77.27%；完整訓練平均減少 38.89% | [中文](efficiency_B/README.md) / [EN](efficiency_B/README.en.md) |
| C：反向刪邊 | 改變刪邊排序能否保持原圖 EPR？ | Cora：EPR 更接近原圖、孤立節點減少；平均準確率未提升 | [中文](reverse_drop_C/README.md) / [EN](reverse_drop_C/README.en.md) |

## 共用評估協定

A、B、C 採固定種子與 10% / 10% / 80% 訓練、驗證、測試節點切分。凍結編碼器後訓練線性分類器，使用訓練集標籤更新分類器，驗證集選擇分類器及編碼器檢查點，最後評估測試集一次。比較時採各正式配對的測試準確率（百分比）與樣本標準差；pp 表示百分點，不等於相對百分比。歷史 Q3 評估口徑不同，不混入這些統計。

## 程式與資料位置

```text
extension_experiments/
├── run.py                      # A 的訓練入口
├── fixed_eval.py               # 共用固定切分、線性評估
├── vr_regularization.py        # A/B 使用的 VR 損失
├── tests/test_basic.py
├── formal_A/                   # A 說明、正式腳本、歷史來源雜湊
│   ├── outputs/                # 僅本機
│   └── logs/                   # 僅本機
├── efficiency_B/               # B1、B2、測速、E3 訓練與診斷
│   └── outputs/                # 僅本機
├── reverse_drop_C/             # C 訓練與離線 EPR 分析
│   ├── outputs/                # 僅本機
│   └── epr_outputs/            # 僅本機
└── splits/                     # 共用固定切分，.pt 僅本機
```

GitHub 只提供程式碼、文件及 [精選研究結果](../research_results/)。A 已改用 `formal_A/outputs/`；B、C 路徑不變。`logs/`、`log/`、`*_logs/`、`*_log/`、`.log`（含輪替日誌）、原始輸出、模型與切分張量不提交。`loss.csv`、`validation.csv` 等結構化研究數據仍保留。來源與 SHA-256 見 [manifest.csv](../research_results/manifest.csv)。目前匯出工具 [export_epagcl_results.py](../scripts/export_epagcl_results.py) 僅預覽或建立新目錄，不覆寫現有結果。

## 執行入口

在專案根目錄、已配置的 Python 環境執行。原始資料集通常位於專案外 `../datasets/`。參考 [原作者 README](../README_upstream.md) 與 [專案 README](../README.md) 的環境說明。B 測速紀錄為 RTX 2070 / PyTorch 2.3.0+cu121，不是其他硬體的速度保證。

```bash
python extension_experiments/run.py --help
python extension_experiments/efficiency_B/train_e3.py --help
python extension_experiments/reverse_drop_C/run.py --help
python extension_experiments/reverse_drop_C/epr_analysis.py --help
python -m unittest discover -s extension_experiments/tests -v
```

Clone 不含 `.pt`、`.pth`、`.ckpt` 或 `.safetensors`。重跑須重新產生固定切分及需要的權重；種子相同不保證跨版本、跨 GPU、跨節點排序的逐位元重現。詳細參數、正式種子與證據請見各實驗文件。歷史雜湊屬當時的程式版本，不因移動輸出目錄而改寫。
