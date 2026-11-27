"""EPAGCL Figure 3: t-SNE of original features and learned node embeddings.

Run from the EPAGCL project root, with its uv virtual environment activated.
Example:
  python plot_figure3.py \
    --run-dir 'trained_models/Cora/completed_1111_21:46:56' --run 2
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save images without requiring a graphical Linux desktop.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from sklearn.manifold import TSNE

from Load_data import get_dataset
from Model import MyModel


def read_run_settings(file: Path) -> dict[str, str]:
    if not file.is_file():
        raise FileNotFoundError(f"找不到訓練設定檔：{file}")
    settings: dict[str, str] = {}
    for line in file.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            settings[key.strip()] = value.strip()
    required = ("dataset", "dataset_path", "feat_dim", "proj_hidden_dim", "temperature")
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError(f"args.txt 缺少必要設定：{', '.join(missing)}")
    return settings


def reduce_tsne(values: np.ndarray, seed: int) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="繪製 EPAGCL Figure 3 的 t-SNE 比較")
    parser.add_argument("--run-dir", type=Path, required=True, help="completed_... 訓練目錄")
    parser.add_argument("--run", type=int, default=2, help="使用第幾次實驗的權重")
    parser.add_argument("--seed", type=int, default=42, help="t-SNE 隨機種子")
    parser.add_argument("--output-dir", type=Path, default=Path("figure3_output"))
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    settings = read_run_settings(run_dir / "args.txt")
    checkpoint = run_dir / str(args.run) / "best_ac_ckpt.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到模型權重：{checkpoint}")

    dataset_name = settings["dataset"]
    print(f"讀取資料集：{dataset_name}")
    # 與官方 main.py 相同的 get_dataset()，但以 CPU 推論即可。
    data = get_dataset(settings["dataset_path"], dataset_name)
    x = data.x.detach().cpu().numpy()
    labels = data.y.reshape(-1).detach().cpu().numpy().astype(int)

    model = MyModel(
        data.num_features,
        int(settings["feat_dim"]),
        int(settings["proj_hidden_dim"]),
        float(settings["temperature"]),
    )
    # 讀取自己訓練產生的 state_dict；不會修改 checkpoint。
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        # 官方 main.py 進行線性評估時也是使用模型輸出的 [0]。
        embeddings = model(data.x, data.edge_index)[0].detach().cpu().numpy()

    if len(x) != len(embeddings) or len(x) != len(labels):
        raise ValueError("原始特徵、模型表示與標籤的節點數不一致")
    print(f"節點數：{len(labels)}；原始維度：{x.shape[1]}；模型表示維度：{embeddings.shape[1]}")

    print("計算原始特徵的 t-SNE ...")
    original_2d = reduce_tsne(x, args.seed)
    print("計算 EPAGCL 節點表示的 t-SNE ...")
    learned_2d = reduce_tsne(embeddings, args.seed)

    classes = np.unique(labels)
    palette = plt.get_cmap("tab10").colors
    color_map = {int(label): palette[i % len(palette)] for i, label in enumerate(classes)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    first_panel = "c" if dataset_name.lower() == "citeseer" else "a"
    second_panel = "d" if dataset_name.lower() == "citeseer" else "b"
    panels = (
        (original_2d, f"({first_panel}) {dataset_name}: Original node features"),
        (learned_2d, f"({second_panel}) {dataset_name}: EPAGCL embeddings (Run {args.run})"),
    )
    for ax, (points, title) in zip(axes, panels):
        for label in classes:
            mask = labels == label
            ax.scatter(
                points[mask, 0], points[mask, 1],
                s=7, color=color_map[int(label)], alpha=0.75, linewidths=0,
            )
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6,
               color=color_map[int(label)], label=f"Class {int(label)}")
        for label in classes
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7),
               frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.03, right=0.98, top=0.89, bottom=0.14, wspace=0.10)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_name}_run{args.run}_tsne"
    png_path = args.output_dir / f"{stem}.png"
    svg_path = args.output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"PNG 圖片：{png_path.resolve()}")
    print(f"SVG 向量圖：{svg_path.resolve()}")


if __name__ == "__main__":
    main()
