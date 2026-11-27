from pathlib import Path

import torch
import torch.nn.functional as F

from Load_data import get_dataset
from Model import MyModel


BASE = Path(
    "trained_models/Q3_EPAGCL/Coauthor-Phy/"
    "completed_1114_22:11:30"
)

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


def read_arg(name, default):
    args_file = BASE / "args.txt"

    for line in args_file.read_text().splitlines():
        if line.startswith(name + ":"):
            return line.split(":", 1)[1].strip().strip("'\"")

    return default


def describe(name, x):
    x = x.detach().float()

    norms = torch.linalg.vector_norm(x, dim=1)
    zero_ratio = (norms < 1e-8).float().mean().item()
    feature_std = x.std(dim=0, unbiased=False).mean().item()

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(
        x.size(0), generator=generator
    )[:512].to(x.device)

    sampled = F.normalize(x[indices], dim=1)
    sim = sampled @ sampled.T

    n = sim.size(0)
    avg_cosine = (
        (sim.sum() - sim.diag().sum()) / (n * (n - 1))
    ).item()

    print(f"\n[{name}]")
    print(f"  Shape: {tuple(x.shape)}")
    print(f"  All finite: {torch.isfinite(x).all().item()}")
    print(f"  Mean feature std: {feature_std:.8f}")
    print(f"  Mean vector norm: {norms.mean().item():.8f}")
    print(f"  Zero-vector ratio: {zero_ratio:.4%}")
    print(f"  Average pairwise cosine: {avg_cosine:.8f}")


def analyze_run(run_id, data):
    path = BASE / str(run_id) / "ckpt_epoch1000.pth"

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    # 從權重形狀取得模型維度，避免手動猜測
    feat_dim = state["fc2.weight"].shape[0]
    proj_hidden_dim = state["fc1.weight"].shape[0]

    model = MyModel(
        in_channels=data.num_features,
        out_channels=feat_dim,
        num_proj_hidden=proj_hidden_dim,
        temperature=0.5
    ).to(DEVICE)

    model.load_state_dict(state)
    model.eval()

    with torch.inference_mode():
        out, proj = model(data.x, data.edge_index)
        hidden_pre_relu = model.fc1(out)
        hidden = F.relu(hidden_pre_relu)

        relu_zero_ratio = (
            (hidden == 0).float().mean().item()
        )

        print("\n" + "=" * 55)
        print(f"EPAGCL Run {run_id}")
        print(f"Checkpoint: {path}")
        print("=" * 55)

        describe("GCN encoder output (out)", out)
        describe("Projection output (proj)", proj)

        print("\n[Projection hidden layer]")
        print(f"  ReLU zero ratio: {relu_zero_ratio:.4%}")
        print(
            "  Hidden feature std: "
            f"{hidden.std(dim=0, unbiased=False).mean().item():.8f}"
        )

    del model, state, out, proj, hidden, hidden_pre_relu

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def main():
    dataset_path = read_arg("dataset_path", "./data")

    print(f"Device: {DEVICE}")
    print(f"Dataset path: {dataset_path}")

    data = get_dataset(
        dataset_path,
        "Coauthor-Phy"
    ).to(DEVICE)

    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.edge_index.size(1)}")
    print(f"Features: {data.num_features}")

    analyze_run(1, data)
    analyze_run(3, data)


if __name__ == "__main__":
    main()
