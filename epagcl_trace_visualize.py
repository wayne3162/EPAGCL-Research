#!/usr/bin/env python3
"""Visual exploration of Cora and trace of a trained EPAGCL node classifier.

Use in official EPAGCL repository, next to Model.py and Load_data.py.

Modes:
  - no --checkpoint: inspect the dataset without training or model weights.
  - with --checkpoint: inspect frozen encoder, selected GCN layers, a newly
    trained linear probe, and held-out node predictions.
  - optional --snapshot: draw the EXACT two graph views previously saved from
    one training epoch (does not fabricate past augmentations).

Images/CSV are not evidence that this machine has reproduced the AAAI paper.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def args_parser():
    p = argparse.ArgumentParser(description="Visualize dataset, optional training views and inference for EPAGCL")
    p.add_argument("--dataset", default="Cora")
    p.add_argument("--data-root", type=Path, default=Path("../datasets"))
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Optional official encoder .pth. Without it: data exploration only.")
    p.add_argument("--snapshot", type=Path, default=None,
                   help="Optional train_trace_epochN.pt captured during official training")
    p.add_argument("--out-dir", type=Path, default=Path("epagcl_trace_results"))
    p.add_argument("--node-id", type=int, default=None,
                   help="Focus node. Default: first TEST node if a checkpoint is supplied, otherwise 0")
    p.add_argument("--hops", type=int, default=2, choices=(1, 2))
    p.add_argument("--max-nodes", type=int, default=45,
                   help="Number of displayed vertices in focal subgraph (does not affect inference)")
    p.add_argument("--method", choices=("pca", "tsne"), default="pca",
                   help="2D display: PCA is fast; t-SNE requires scikit-learn and costs more")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--feat-dim", type=int, default=256)
    p.add_argument("--proj-hidden-dim", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--eval-epochs", type=int, default=3000)
    p.add_argument("--eval-lr", type=float, default=0.01)
    p.add_argument("--eval-weight-decay", type=float, default=0.0)
    return p.parse_args()


def safe_load_tensors(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def make_undirected_graph(num_nodes, edge_index):
    edges = edge_index.detach().cpu().numpy()
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from((int(a), int(b)) for a, b in zip(edges[0], edges[1]) if int(a) != int(b))
    return g


def focal_nodes(graph, center, hops, cap):
    dist = nx.single_source_shortest_path_length(graph, center, cutoff=hops)
    # Include center first, nearby nodes in ascending node-ID order for stability.
    return sorted(dist.keys(), key=lambda x: (dist[x], x))[:max(1, cap)]


def graph_plot(original_g, current_g, nodes, positions, center, labels, path, title):
    baseline = nx.Graph(original_g.subgraph(nodes))
    current = nx.Graph(current_g.subgraph(nodes))
    old_edges = {tuple(sorted(edge)) for edge in baseline.edges()}
    new_edges = {tuple(sorted(edge)) for edge in current.edges()}
    common = sorted(old_edges & new_edges)
    removed = sorted(old_edges - new_edges)
    added = sorted(new_edges - old_edges)
    fig, ax = plt.subplots(figsize=(9, 7.2))
    ax.set_title(title, fontsize=13)
    nx.draw_networkx_edges(baseline, positions, edgelist=common,
                           edge_color="#9ca3af", alpha=.55, width=1, ax=ax)
    if removed:
        nx.draw_networkx_edges(baseline, positions, edgelist=removed,
                               edge_color="#dc2626", alpha=.85,
                               width=1.6, style="dashed", ax=ax)
    if added:
        nx.draw_networkx_edges(current, positions, edgelist=added,
                               edge_color="#16a34a", alpha=.85,
                               width=1.9, ax=ax)
    values = [int(labels[i]) for i in nodes]
    nx.draw_networkx_nodes(baseline, positions, nodelist=nodes,
                           node_color=values, cmap=plt.get_cmap("tab10"),
                           vmin=0, vmax=max(9, int(np.max(labels))),
                           node_size=[320 if i == center else 125 for i in nodes],
                           edgecolors=["black" if i == center else "white" for i in nodes],
                           linewidths=[2.2 if i == center else .65 for i in nodes], ax=ax)
    if len(nodes) <= 45:
        nx.draw_networkx_labels(baseline, positions, labels={int(i):str(i) for i in nodes},
                                font_size=6.5, ax=ax)
    ax.text(.01, .01, f"Nodes shown: {len(nodes)} | same node positions across views\n"
                      f"Unchanged edges: {len(common)}  removed: {len(removed)}  added: {len(added)}\n"
                      "Colors = ground-truth class (for display ONLY); dashed red = removed; green = added",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=.88))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return dict(visible_nodes=len(nodes), visible_unchanged_edges=len(common),
                visible_removed_edges=len(removed), visible_added_edges=len(added))


def feature_plot(features, nodes, center, path):
    # Subsample nodes and dimensions so an actual 1433-dimensional bag-of-words
    # vector can be understood without fabricating words/abstracts.
    view_nodes = [center] + [n for n in nodes if n != center][:15]
    xx = features[view_nodes]
    top_features = np.argsort(np.abs(xx).sum(axis=0))[::-1][:min(42, xx.shape[1])]
    vals = xx[:, top_features]
    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    im = ax.imshow(vals, cmap="viridis", interpolation="nearest", aspect="auto")
    ax.set_title("Original Cora node features: selected node + neighborhood")
    ax.set_xlabel("Selected feature indices (numeric; no paper titles in Planetoid object)")
    ax.set_ylabel("Node ID")
    ax.set_yticks(np.arange(len(view_nodes)), labels=[str(i) for i in view_nodes], fontsize=8)
    ax.set_xticks(np.arange(0, len(top_features), 2),
                  labels=[str(top_features[i]) for i in range(0, len(top_features), 2)],
                  rotation=70, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=.8, label="Feature value")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def reduce_to_2d(features, method, seed):
    from sklearn.decomposition import PCA
    xx = np.asarray(features, dtype=np.float32)
    if xx.shape[0] < 4:
        raise ValueError("Need >=4 nodes for dimensionality reduction")
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(xx)
    from sklearn.manifold import TSNE
    k = min(50, xx.shape[0] - 1, xx.shape[1])
    xx = PCA(n_components=k, random_state=seed).fit_transform(xx) if k > 2 else xx
    return TSNE(n_components=2, random_state=seed, init="pca",
                learning_rate="auto", perplexity=min(30, max(2, (len(xx)-1)//3))).fit_transform(xx)


def scatter_plot(features, labels, center, title, path, method, seed):
    coords = reduce_to_2d(features, method, seed)
    fig, ax = plt.subplots(figsize=(8.7, 6.7))
    labels = np.asarray(labels)
    for cls in np.unique(labels):
        mask = (labels == cls)
        ax.scatter(coords[mask, 0], coords[mask, 1], label=f"Class {cls}",
                   s=11, alpha=.62, edgecolors="none")
    ax.scatter([coords[center, 0]], [coords[center, 1]],
               marker="*", s=190, color="black", label=f"Node {center}")
    ax.set_title(title + f" ({method.upper()})")
    ax.set_xlabel("Dimension 1 (relative only)")
    ax.set_ylabel("Dimension 2 (relative only)")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.text(.01, -.12, "Projection is for exploration; 2D distances and cluster separation do not prove accuracy.",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_split(n, seed, device):
    ids = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).to(device)
    a = int(n*.1)
    b = int(n*.1)
    if a < 1 or b < 1 or n-a-b < 1:
        raise ValueError("Dataset too small for 10/10/80 split")
    return ids[:a], ids[a:a+b], ids[a+b:]


def linear_probe(emb, truth, splits, epochs, lr, weight_decay):
    import copy
    train, valid, test = splits
    if epochs <= 0:
        raise ValueError("--eval-epochs must be >=1")
    clf = torch.nn.Linear(emb.shape[1], int(truth.max().item()) + 1).to(emb.device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_score = -1.
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        clf.train()
        optimizer.zero_grad()
        loss = criterion(clf(emb[train]), truth[train])
        loss.backward()
        optimizer.step()
        clf.eval()
        with torch.no_grad():
            p = clf(emb[valid]).argmax(dim=1)
            val_acc = (p == truth[valid]).float().mean().item()
        if val_acc > best_score:
            best_score, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(clf.state_dict())
    clf.load_state_dict(best_state)
    clf.eval()
    with torch.no_grad():
        logits = clf(emb)
        probs = logits.softmax(dim=-1)
        pred = probs.argmax(dim=-1)
        acc = (pred[test] == truth[test]).float().mean().item()
    return (logits.detach().cpu(), probs.detach().cpu(), pred.detach().cpu(),
            dict(test_accuracy=acc, best_validation_accuracy=best_score,
                 best_linear_epoch=best_epoch, n_test=len(test)))


def confusion_plot(truth, pred, test_ids, num_classes, path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(truth[test_ids], pred[test_ids], labels=np.arange(num_classes))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max()*.55 else "black")
    ax.set_xlabel("Predicted class ID")
    ax.set_ylabel("True class ID")
    ax.set_title("Held-out test-node classification confusion matrix")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    fig.colorbar(im, ax=ax, shrink=.75)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return cm.tolist()


def probability_plot(probs, center, truth, pred, path, split):
    p = probs[center].numpy()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.bar(np.arange(len(p)), p, color=["#16a34a" if i == int(pred[center]) else "#9ca3af"
                                             for i in range(len(p))])
    ax.set_xticks(np.arange(len(p)), labels=[f"Class {i}" for i in range(len(p))])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Linear-classifier softmax score")
    ax.set_title(f"Node {center} ({split}): TRUE={int(truth[center])}, PRED={int(pred[center])}, "
                 f"{'CORRECT' if int(truth[center]) == int(pred[center]) else 'WRONG'}")
    for rect, v in zip(bars, p):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + .025,
                f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def hidden_layer_plot(traces, raw, embedding, center, path):
    samples = [("Raw input features", raw[center])] + traces + [("Encoder output", embedding[center])]
    n_show = 32
    matrix = np.zeros((len(samples), n_show), dtype=np.float32)
    for i, (_, vec) in enumerate(samples):
        vv = np.asarray(vec)[:n_show]
        matrix[i, :len(vv)] = vv / (np.max(np.abs(vv)) + 1e-12)
    fig, ax = plt.subplots(figsize=(9.7, max(3.4, 1.1 + len(samples)*.65)))
    im = ax.imshow(matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax.set_yticks(range(len(samples)), labels=[f"{name} [dim {len(vec)}]" for name, vec in samples])
    ax.set_xlabel("First 32 features (each row normalized independently for DISPLAY)")
    ax.set_title(f"Node {center}: layer-by-layer numeric activation trace")
    fig.colorbar(im, ax=ax, shrink=.7)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_test_csv(test_ids, labels, pred, probs, path):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "true_class_id", "predicted_class_id", "correct", "max_softmax_score"])
        for i in test_ids:
            writer.writerow([int(i), int(labels[i]), int(pred[i]),
                             int(labels[i] == pred[i]), f"{float(probs[i].max()):.5f}"])


def main():
    args = args_parser()
    from Load_data import get_dataset  # official project
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Try --device cpu")
    device = torch.device(device_name)
    data = get_dataset(str(args.data_root / args.dataset), args.dataset)
    n = int(data.x.shape[0])
    if n > 12000:
        raise ValueError("This educational trace tool targets smaller graphs (e.g. Cora): N<=12000")
    classes = int(data.y.max().item()) + 1
    x = data.x.detach().cpu().numpy()
    labels = data.y.reshape(-1).detach().cpu().numpy().astype(int)
    original = make_undirected_graph(n, data.edge_index)
    splits = make_split(n, args.seed, device)
    train_ids, valid_ids, test_ids = [t.cpu().numpy() for t in splits]
    center = int(args.node_id) if args.node_id is not None else (int(test_ids[0]) if args.checkpoint else 0)
    if not 0 <= center < n:
        raise ValueError(f"Invalid --node-id={center}, valid range: 0..{n-1}")
    subset = focal_nodes(original, center, args.hops, args.max_nodes)
    positions = nx.spring_layout(original.subgraph(subset), seed=args.seed)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    path = out / "01_original_neighborhood.png"
    graph_plot(original, original, subset, positions, center, labels, path,
               f"Original {args.dataset} citation network (focus node {center}, <= {args.hops} hops)")
    saved.append(str(path))
    path = out / "02_local_feature_matrix.png"
    feature_plot(x, subset, center, path)
    saved.append(str(path))
    path = out / f"03_original_features_{args.method}.png"
    scatter_plot(x, labels, center, "Original node features", path, args.method, args.seed)
    saved.append(str(path))
    info = {
        "dataset": args.dataset, "graph_nodes": n, "unique_undirected_edges": original.number_of_edges(),
        "edge_index_columns": int(data.edge_index.shape[1]), "feature_matrix_shape": list(x.shape),
        "classes": classes, "center_node": center, "center_true_class_id_for_visualization_only": int(labels[center]),
        "center_degree": original.degree(center), "visible_neighborhood_nodes": list(map(int, subset)),
        "center_nonzero_feature_indices_first_30": np.flatnonzero(x[center]).astype(int).tolist()[:30],
        "class_ids_not_human_readable_titles": True,
        "note": "No labels are used in EPAGCL contrastive pretraining. Plot colors are for interpretation."
    }
    if args.snapshot:
        snapshot = safe_load_tensors(args.snapshot, "cpu")
        for key in ("original_edges", "view_drop_only_edges", "view_add_drop_edges"):
            if key not in snapshot:
                raise KeyError(f"Training snapshot lacks {key}. Follow the README instrumentation step.")
        saved_original = make_undirected_graph(n, snapshot["original_edges"])
        if {tuple(sorted(e)) for e in saved_original.edges()} != {tuple(sorted(e)) for e in original.edges()}:
            raise ValueError("Snapshot original graph does not match current dataset (check dataset and order)")
        for key, name, title in (("view_drop_only_edges", "07_view_drop_only.png", "TRAINING snapshot: DROP ONLY"),
                                 ("view_add_drop_edges", "08_view_add_drop.png", "TRAINING snapshot: ADD and DROP")):
            graph = make_undirected_graph(n, snapshot[key])
            filename = out/name
            local_stats = graph_plot(original, graph, subset, positions, center, labels, filename, title)
            saved.append(str(filename))
            info[key] = {"unique_undirected_edges": graph.number_of_edges(), **local_stats}
        info["training_snapshot_path"] = str(args.snapshot)
        info["training_snapshot_note"] = "Exact edges saved from one training epoch; only local neighborhood shown."
    if args.checkpoint:
        from Model import MyModel  # official model
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        model = MyModel(data.num_features, args.feat_dim, args.proj_hidden_dim, args.temperature).to(device)
        model.load_state_dict(safe_load_tensors(args.checkpoint, device), strict=True)
        model.eval()
        layer_traces = []
        hooks = []
        def make_hook(name):
            def hook(_, _inputs, output):
                v = output[0] if isinstance(output, tuple) else output
                if torch.is_tensor(v) and v.ndim == 2 and v.shape[0] == n:
                    layer_traces.append((name, v[center].detach().cpu().numpy()))
            return hook
        for name, mod in model.named_modules():
            if mod.__class__.__name__ in ("GCNConv", "GATConv", "GraphConv", "SAGEConv"):
                hooks.append(mod.register_forward_hook(make_hook(name)))
        with torch.no_grad():
            output = model(data.x.to(device), data.edge_index.to(device))
        for h in hooks:
            h.remove()
        h = output[0] if isinstance(output, (tuple, list)) else output
        h = h.detach()
        embeddings = h.cpu().numpy()
        path = out / f"04_trained_embeddings_{args.method}.png"
        scatter_plot(embeddings, labels, center, "EPAGCL trained encoder embeddings", path, args.method, args.seed)
        saved.append(str(path))
        raw_vectors = [(name, vec) for name, vec in layer_traces]
        path = out / "09_selected_node_layer_activations.png"
        hidden_layer_plot(raw_vectors, x, embeddings, center, path)
        saved.append(str(path))
        logits, probs, pred, result = linear_probe(
            h, data.y.reshape(-1).long().to(device), splits,
            args.eval_epochs, args.eval_lr, args.eval_weight_decay
        )
        labels_tensor = torch.as_tensor(labels)
        path = out / "05_test_confusion_matrix.png"
        cm = confusion_plot(labels, pred.numpy(), test_ids, classes, path)
        saved.append(str(path))
        split_name = ("test" if center in test_ids else "train" if center in train_ids else "validation")
        path = out / "06_focused_node_class_probabilities.png"
        probability_plot(probs, center, labels, pred, path, split_name)
        saved.append(str(path))
        save_test_csv(test_ids, labels, pred.numpy(), probs.numpy(), out/"test_predictions.csv")
        info.update({"checkpoint": str(args.checkpoint),
                     "prediction": {"split": split_name, "true_class_id": int(labels[center]),
                                    "predicted_class_id": int(pred[center]),
                                    "correct": bool(int(pred[center]) == int(labels[center])),
                                    "softmax_scores": [float(z) for z in probs[center]],
                                    "logits": [float(z) for z in logits[center]]},
                     "linear_probe": result, "confusion_matrix": cm,
                     "encoder_output_shape": list(embeddings.shape),
                     "center_embedding_first_16": embeddings[center,:16].tolist(),
                     "layer_trace": [{"layer": name, "vector_dim": int(len(vec)),
                                      "l2_norm": float(np.linalg.norm(vec)),
                                      "first_16": vec[:16].tolist()} for name,vec in layer_traces],
                     "holdout_eval_note": "Encoder uses the FULL original graph, but classifier uses only train node LABELS. "
                         "Validation labels pick the best classifier epoch. Test labels are inspected AFTER selection."})
        saved.append(str(out/"test_predictions.csv"))
        print(f"Held-out TEST accuracy: {result['test_accuracy']*100:.2f}%")
        print(f"Node {center}: true={labels[center]} predicted={int(pred[center])} ({split_name})")
    (out/"trace_summary.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Dataset:", args.dataset, "nodes:", n, "undirected edges:", original.number_of_edges(),
          "node features:", x.shape[1], "classes:", classes)
    print("Focused node:", center)
    print("Output files:")
    for f in saved + [str(out/"trace_summary.json")]:
        print(" -", f)
    if not args.checkpoint:
        print("Dataset exploration only. Use --checkpoint to visualize REAL encoder and classifier outputs.")


if __name__ == "__main__":
    main()
