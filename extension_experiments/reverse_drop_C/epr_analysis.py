"""Experiment C: compare augmentations using the original GitHub EPR formula.

Place in EPAGCL/extension_experiments/reverse_drop_C/.
No model training, no VR, no early stopping, no alternative EPR definition.
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch_geometric.utils import degree, to_undirected

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

METHODS = ("baseline", "reverse_drop")
STAGES = ("view1_drop", "view1_final", "view2_drop")


@torch.no_grad()
def calculate_epr(edge_index, label):
    """Original EPR.py loop, adapted to accept any graph.

    CPU evaluation avoids per-node CUDA synchronization. Explicit num_nodes
    retains trailing isolated nodes. Flattening handles [N, 1] labels.
    Returns NaN if the graph contains no nonisolated nodes (undefined EPR).
    """
    label = label.reshape(-1).cpu()
    edge_index = to_undirected(edge_index.cpu(), num_nodes=label.size(0))
    deg = degree(edge_index[0], num_nodes=label.size(0)).to(torch.int32)
    computed_deg = 0
    epr = []
    for i in range(label.size(0)):
        d = deg[i]
        if d == 0:
            continue
        neighbour = edge_index[1][computed_deg : computed_deg + d]
        computed_deg += d
        neighbour_deg = deg[neighbour]
        neighbour_y = label[neighbour]
        wrong_neighbour_mask = (neighbour_y != label[i])
        wrong_passing = 1 / neighbour_deg[wrong_neighbour_mask] ** 0.5
        all_passing = 1 / neighbour_deg ** 0.5
        epr.append((wrong_passing.sum()/all_passing.sum()).item())
    value = float(np.array(epr).mean()) if epr else float('nan')
    return value, int((deg == 0).sum().item()), int(edge_index.size(1))


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--dataset-path', default=None)
    parser.add_argument('--device', default='cpu', help='Device for graph augmentation; EPR is evaluated on CPU')
    parser.add_argument('--samples', type=int, default=100, help='Random augmentations per method per seed')
    parser.add_argument('--repeat', type=int, default=5, help='Number of seed blocks')
    parser.add_argument('--seed', type=int, default=20261119)
    parser.add_argument('--edge-add-rate', type=float, default=0.1)
    parser.add_argument('--edge-drop-rate-1', type=float, default=0.2)
    parser.add_argument('--edge-drop-rate-2', type=float, default=0.3)
    args = parser.parse_args()
    if args.samples < 1 or args.repeat < 1 or args.seed < 0:
        parser.error('samples and repeat must be positive; seed must be nonnegative')
    if not 0 < args.edge_add_rate <= 1 or not all(0 <= p <= 1 for p in (args.edge_drop_rate_1, args.edge_drop_rate_2)):
        parser.error('edge-add-rate must be in (0,1]; drop rates must be in [0,1]')
    return args


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def stat(values, op=np.mean):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(op(x)) if x.size else None


@torch.no_grad()
def main():
    args = get_args()
    from Augmentation import add_edges, add_edge_weights, drop_edges, drop_edge_weights
    from Load_data import get_dataset
    from augmentation_c import prepare_edge_alignment, reverse_drop_edge_weights, validate_weights

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    path = (Path(args.dataset_path) if args.dataset_path else REPO.parent / 'datasets' / args.dataset).expanduser().resolve()
    data = get_dataset(str(path), args.dataset).to(device)
    edges = data.edge_index
    label = data.y.reshape(-1).cpu()
    if label.numel() != data.num_nodes or (label < 0).any().item():
        raise ValueError('Expected one known class label per node')

    # Same weight-to-edge alignment as experiment C training (both groups).
    canonical, mapping, alignment = prepare_edge_alignment(edges)
    weights = {}
    for method, fn in (('baseline', drop_edge_weights), ('reverse_drop', reverse_drop_edge_weights)):
        w = fn(canonical)
        validate_weights(w, canonical, method)
        weights[method] = w[mapping]
        validate_weights(weights[method], edges, method)
    add_weights, add_index = add_edge_weights(edges, args.edge_add_rate)
    if add_weights.numel() == 0 or not torch.isfinite(add_weights).all().item() or (add_weights < 0).any().item() or add_weights.mean().item() <= 0:
        raise ValueError('Invalid original addition weights')

    original_epr, original_isolates, original_undirected = calculate_epr(edges, label)
    if not np.isfinite(original_epr):
        raise ValueError('Original EPR is undefined')
    n_edges = edges.size(1)
    out = HERE / 'epr_outputs' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.dataset + '_original_epr')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'args.json', dict(vars(args), dataset_path_resolved=str(path), epr_definition='original GitHub EPR.py: mean over nonisolated nodes after to_undirected', add_sampling_p=0.5, add_sampling_threshold=0.9))
    write_json(out / 'edge_alignment.json', alignment)
    write_json(out / 'original_epr.json', {'epr': original_epr, 'isolated_nodes': original_isolates, 'raw_edge_entries': n_edges, 'undirected_edge_entries': original_undirected})
    print(f'Output: {out}\nOriginal EPR: {original_epr:.8f}', flush=True)

    rows = []
    for method in METHODS:
        print(f'[{method}] Starting all {args.repeat * args.samples} samples', flush=True)
        for block in range(args.repeat):
            seed = args.seed + block
            print(f'[{method}] Seed {seed}: {args.samples} samples', flush=True)
            for sample in range(1, args.samples + 1):
                # Same seed for corresponding samples of each group. This is a
                # new structural experiment, not a replay of training RNG.
                sample_seed = (seed * 1000003 + sample) % (2**63 - 1)
                torch.manual_seed(sample_seed)
                e1 = drop_edges(edges, weights[method], args.edge_drop_rate_1)
                e2 = drop_edges(edges, weights[method], args.edge_drop_rate_2)
                final = add_edges(add_weights, add_index, e1)
                for stage, graph in (('view1_drop', e1), ('view1_final', final), ('view2_drop', e2)):
                    epr, isolates, undirected_count = calculate_epr(graph, label)
                    drop_only = stage != 'view1_final'
                    rows.append({'method': method, 'seed': seed, 'sample': sample, 'sample_seed': sample_seed,
                                 'stage': stage, 'epr': epr, 'delta_epr': epr - original_epr,
                                 'abs_delta_epr': abs(epr - original_epr), 'isolated_nodes': isolates,
                                 'raw_edge_entries': graph.size(1), 'undirected_edge_entries': undirected_count,
                                 'actual_drop_rate_raw': (1 - graph.size(1)/n_edges) if drop_only else '',
                                 'actual_drop_rate_undirected': (1 - undirected_count/original_undirected) if drop_only else ''})
                if sample % 25 == 0 or sample == args.samples:
                    print(f'[{method}] seed={seed} {sample}/{args.samples}', flush=True)
            write_csv(out / 'epr_samples.csv', rows)
        print(f'[{method}] Completed all {args.repeat * args.samples} samples', flush=True)

    summary = []
    comparison = []
    seed_summary = []
    for stage in STAGES:
        groups = {m: [r for r in rows if r['method'] == m and r['stage'] == stage] for m in METHODS}
        for m, group in groups.items():
            valid = [r for r in group if np.isfinite(r['epr'])]
            summary.append({'method': m, 'stage': stage, 'original_epr': original_epr,
                            'samples': len(group), 'valid_samples': len(valid),
                            'mean_epr': stat([r['epr'] for r in valid]),
                            'std_epr_population': stat([r['epr'] for r in valid], np.std),
                            'mean_delta_epr': stat([r['delta_epr'] for r in valid]),
                            'mean_abs_delta_epr': stat([r['abs_delta_epr'] for r in valid]),
                            'fraction_epr_increased': stat([r['delta_epr'] > 0 for r in valid]),
                            'mean_isolated_nodes': stat([r['isolated_nodes'] for r in group]),
                            'mean_actual_drop_rate_raw': stat([r['actual_drop_rate_raw'] for r in group if r['actual_drop_rate_raw'] != '']),
                            'mean_actual_drop_rate_undirected': stat([r['actual_drop_rate_undirected'] for r in group if r['actual_drop_rate_undirected'] != ''])})
        b = {(r['seed'], r['sample']): r for r in groups['baseline']}
        v = {(r['seed'], r['sample']): r for r in groups['reverse_drop']}
        if b.keys() != v.keys():
            raise RuntimeError('Paired seed/sample keys differ')
        paired = [(k, b[k], v[k]) for k in b if np.isfinite(b[k]['epr']) and np.isfinite(v[k]['epr'])]
        def paired_summary(pairs):
            return {'valid_pairs': len(pairs),
                    'baseline_mean_epr': stat([x['epr'] for _,x,y in pairs]),
                    'reverse_drop_mean_epr': stat([y['epr'] for _,x,y in pairs]),
                    'reverse_minus_baseline': stat([y['epr']-x['epr'] for _,x,y in pairs])}
        comparison.append({'stage': stage, **paired_summary(paired)})
        for seed in range(args.seed, args.seed + args.repeat):
            seed_summary.append({'seed': seed, 'stage': stage, **paired_summary([p for p in paired if p[0][0] == seed])})
    write_csv(out / 'summary.csv', summary)
    write_csv(out / 'comparison.csv', comparison)
    write_csv(out / 'seed_summary.csv', seed_summary)
    print('\nEPR (ratio 0-1), Reverse Drop minus Baseline:', flush=True)
    for row in comparison:
        print(row, flush=True)
    print(f'Results saved: {out}', flush=True)


if __name__ == '__main__':
    main()
