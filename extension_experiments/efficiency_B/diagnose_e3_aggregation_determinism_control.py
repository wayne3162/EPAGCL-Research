"""E3 diagnostic: same frozen GCNConv propagation with/without deterministic algorithms.

Does NOT train, tune or modify the E3/B0/B2x8 models. Requires the original
EPAGCL repository, torch 2.3.0+cu121 / PyG 2.5.3 and CUDA RTX 2070.

Run from repository root:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u \
    extension_experiments/efficiency_B/diagnose_e3_aggregation_determinism_control.py \
    --seed 20261129 --repeats 2 --device cuda:0

Results: outputs/<timestamp>_E3_aggregation_determinism_control_seed_.../control_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from train_e3 import (
    B_ROOT, FIXED, REPO_ROOT, add_edge_weights, drop_edge_weights,
    get_dataset, make_augmented_views, model_for, set_seed, source_hashes,
    sync,
)

EXPECTED_HASHES = {
    'Model.py': '66bf131decb3b6cc527c92209dcb4a55d38648ea55ca7e6ac4cf70b1f316c033',
    'Augmentation.py': 'a880882304d3fb6ea80781b3c116e669a4581562210e0ef756c463cff26ba787',
    'extension_experiments/efficiency_B/train_e3.py': 'b2a728e3039e5648ecd66ac56d9f9afa28ae329debf49179d1fe951fd6908161',
}


def compare(left: torch.Tensor, right: torch.Tensor) -> dict:
    if left.shape != right.shape or left.dtype != right.dtype:
        return {'exact_equal': False, 'different_elements': None, 'max_abs_diff': None}
    equal = bool(torch.equal(left, right))
    if equal:
        return {'exact_equal': True, 'different_elements': 0, 'max_abs_diff': 0.0}
    return {
        'exact_equal': False,
        'different_elements': int(torch.count_nonzero(left != right).item()),
        'max_abs_diff': float((left.double() - right.double()).abs().max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261129)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dataset-path', default=None)
    parser.add_argument('--output-root', default=None)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 4:
        parser.error('--repeats must be between 2 and 4')
    if args.device != 'cuda:0' or not torch.cuda.is_available():
        parser.error('An available cuda:0 GPU is required')
    device = torch.device(args.device)
    gpu = torch.cuda.get_device_name(device)
    if gpu != 'NVIDIA GeForce RTX 2070':
        parser.error(f'Expected NVIDIA GeForce RTX 2070, got {gpu}')

    hashes = source_hashes()
    wrong = [name for name, value in EXPECTED_HASHES.items() if hashes.get(name) != value]
    if wrong:
        raise RuntimeError(f'Original code changed; stopping to protect audit: {wrong}')

    root = (Path(args.output_root).expanduser().resolve() if args.output_root
            else B_ROOT / 'outputs')
    root.mkdir(parents=True, exist_ok=True)
    output_dir = root / (
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        + f'_E3_aggregation_determinism_control_seed_{args.seed}'
    )
    output_dir.mkdir()

    data_path = (Path(args.dataset_path).expanduser().resolve() if args.dataset_path
                 else (REPO_ROOT.parent / 'datasets' / 'Coauthor-Phy').resolve())
    if not data_path.parent.is_dir():
        raise FileNotFoundError(f'Missing dataset parent: {data_path.parent}')

    config = SimpleNamespace(**FIXED)
    print('Loading graph; performing a single frozen-input aggregation control only.', flush=True)
    set_seed(args.seed)
    data = get_dataset(str(data_path), 'Coauthor-Phy').to(device)
    add_weights, adding_edge = add_edge_weights(data.edge_index, config.edge_add_rate)
    drop_weights = drop_edge_weights(data.edge_index)
    sync(device)

    set_seed(args.seed)
    model = model_for('b0', data.num_features, config, device)
    model.train()
    (x1, edge1), (_, _) = make_augmented_views(
        data, add_weights, adding_edge, drop_weights, config
    )
    if model.conv1.cached:
        raise RuntimeError('GCNConv cached should be False in original E3')
    conv = model.conv1
    edge_index, edge_weight = gcn_norm(
        edge1, None, num_nodes=x1.size(0),
        improved=conv.improved, add_self_loops=conv.add_self_loops,
        flow=conv.flow, dtype=x1.dtype,
    )
    # Freeze the linear projection and normalized graph for all propagation calls.
    projected = conv.lin(x1).detach()
    edge_index = edge_index.detach()
    edge_weight = edge_weight.detach()
    sync(device)
    print(f'GPU={gpu}, seed={args.seed}, nodes={len(x1)}, edges={edge_index.size(1)}')

    groups = {}
    errors = {}
    previous_setting = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        for group, deterministic in (
            ('normal_before', False),
            ('strict_deterministic', True),
            ('normal_after', False),
        ):
            print(f'[{group}] deterministic={deterministic}', flush=True)
            torch.use_deterministic_algorithms(deterministic, warn_only=False)
            torch.backends.cudnn.deterministic = deterministic
            snaps = []
            try:
                for _ in range(args.repeats):
                    # No model parameter update or graph regeneration.
                    with torch.no_grad():
                        value = conv.propagate(
                            edge_index, x=projected, edge_weight=edge_weight, size=None
                        )
                    sync(device)
                    snaps.append(value.detach().cpu().contiguous().clone())
            except RuntimeError as exc:
                errors[group] = f'{type(exc).__name__}: {exc}'
                print(f'  UNSUPPORTED/ERROR: {errors[group]}', flush=True)
            if snaps and len(snaps) == args.repeats:
                groups[group] = snaps
                stat = compare(snaps[0], snaps[1])
                print(f"  first two exact_equal={stat['exact_equal']}, "
                      f"different_elements={stat['different_elements']}, "
                      f"max_abs_diff={stat['max_abs_diff']}", flush=True)
            del snaps
    finally:
        torch.use_deterministic_algorithms(previous_setting, warn_only=False)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic

    records = []
    for group, values in groups.items():
        for i in range(1, len(values)):
            records.append({'comparison': f'{group}: repeat_1_vs_{i+1}', **compare(values[0], values[i])})
    for ga, gb in (
        ('normal_before', 'normal_after'),
        ('normal_before', 'strict_deterministic'),
        ('strict_deterministic', 'normal_after'),
    ):
        if ga in groups and gb in groups:
            records.append({'comparison': f'{ga}_vs_{gb}', **compare(groups[ga][0], groups[gb][0])})

    filename = output_dir / 'control_comparisons.csv'
    with filename.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['comparison', 'exact_equal', 'different_elements', 'max_abs_diff']
        )
        writer.writeheader()
        writer.writerows(records)

    result = {
        'purpose': 'Single frozen-input GCNConv propagation control; not training/benchmark',
        'seed': args.seed, 'gpu': gpu, 'torch_version': torch.__version__,
        'repeats_per_condition': args.repeats,
        'source_sha256': hashes,
        'conditions': ['normal_before', 'strict_deterministic', 'normal_after'],
        'comparisons': records, 'errors': errors,
        'limitations': [
            'Exact output repeatability for this operation does not guarantee entire training determinism.',
            'The strict mode may trigger a deterministic fallback or fail for unsupported PyG/CUDA kernels.',
            'CUBLAS_WORKSPACE_CONFIG must be set before the Python process starts if applicable.',
            'GPU sync and CPU snapshots make timing incomparable to E3; no speed conclusions.',
            'No formal E3 results, model code, parameters, splits, labels, or checkpoints changed.',
        ],
    }
    summary_path = output_dir / 'control_summary.json'
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Summary:', summary_path)
    print('CSV:', filename)
    print('Finished (diagnostic only).')


if __name__ == '__main__':
    main()
