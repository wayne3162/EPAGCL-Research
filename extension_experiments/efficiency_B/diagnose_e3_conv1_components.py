"""E3: isolate first-layer GCNConv nondeterminism on identical augmented input.

Diagnostic ONLY. Does not change Model.py, E3 training, splits or checkpoints.
Uses the user's existing PyTorch/PyG environment and train_e3.py utilities.

Usage (from project root):
 python -u extension_experiments/efficiency_B/diagnose_e3_conv1_components.py \
   --seed 20261129 --restarts 2 --repeats 3 --device cuda:0

Tested with Python syntax checks; this authoring environment has no CUDA/PyG.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from train_e3 import (
    B_ROOT, FIXED, REPO_ROOT, add_edge_weights, drop_edge_weights,
    get_dataset, make_augmented_views, model_for, set_seed, source_hashes, sync,
)

# These are the source hashes from the verified first-step diagnostic. Fail fast
# rather than silently comparing a changed implementation against older results.
EXPECTED = {
    'Model.py': '66bf131decb3b6cc527c92209dcb4a55d38648ea55ca7e6ac4cf70b1f316c033',
    'Augmentation.py': 'a880882304d3fb6ea80781b3c116e669a4581562210e0ef756c463cff26ba787',
    'extension_experiments/efficiency_B/train_e3.py': 'b2a728e3039e5648ecd66ac56d9f9afa28ae329debf49179d1fe951fd6908161',
}


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261129)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--restarts', type=int, default=2,
                        help='Independent model/augmentation restarts with the same seed')
    parser.add_argument('--repeats', type=int, default=3,
                        help='Repeated aggregation/full-conv calls per restart')
    parser.add_argument('--dataset-path', default=None)
    parser.add_argument('--output-root', default=None)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    if not 2 <= args.restarts <= 4 or not 2 <= args.repeats <= 5:
        parser.error('--restarts must be 2-4; --repeats must be 2-5')
    if args.device != 'cuda:0':
        parser.error('Only the original cuda:0 GPU is valid for this comparison')
    return args


def snapshot(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    # A synchronization/copy is intentional; these are NOT speed measurements.
    sync(device)
    return value.detach().to('cpu').contiguous().clone()


def diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape or a.dtype != b.dtype:
        return {'equal': False, 'different_elements': None, 'max_abs_diff': None,
                'shape_equal': False}
    equal = bool(torch.equal(a, b))
    if equal:
        return {'equal': True, 'different_elements': 0, 'max_abs_diff': 0.0,
                'shape_equal': True}
    n = int(torch.count_nonzero(a != b).item())
    if a.dtype.is_floating_point:
        max_abs = float((a.to(torch.float64) - b.to(torch.float64)).abs().max().item())
    else:
        max_abs = float((a.to(torch.float64) - b.to(torch.float64)).abs().max().item())
    return {'equal': False, 'different_elements': n, 'max_abs_diff': max_abs,
            'shape_equal': True}


def measure_one_restart(restart: int, seed: int, args_model, data,
                        add_weights, adding_edge, drop_weights, device,
                        repeats: int) -> dict[str, torch.Tensor]:
    # Match E3: model initialization before the first on-the-fly augmentation.
    set_seed(seed)
    model = model_for('b0', data.num_features, args_model, device)
    model.train()
    (x1, edge1), (x2, edge2) = make_augmented_views(
        data, add_weights, adding_edge, drop_weights, args_model
    )
    if not torch.equal(x1, data.x) or not torch.equal(x2, data.x):
        raise RuntimeError('Feature-drop=0 input should equal the original features')
    del x2, edge2
    conv1 = model.conv1
    if conv1.cached:
        raise RuntimeError('This experiment requires uncached original GCNConv')

    observations = {
        'input/edge1': snapshot(edge1, device),
    }
    # Same gcn_norm call signature as torch_geometric GCNConv.forward.
    normalized_edge, normalized_weight = gcn_norm(
        edge1, None, num_nodes=x1.size(0),
        improved=conv1.improved, add_self_loops=conv1.add_self_loops,
        flow=conv1.flow, dtype=x1.dtype,
    )
    observations['norm/edge_index'] = snapshot(normalized_edge, device)
    observations['norm/edge_weight'] = snapshot(normalized_weight, device)

    # Independent linear operation on the identical raw input.
    linear_x = conv1.lin(x1)
    observations['linear/output'] = snapshot(linear_x, device)

    # Reuse EXACTLY the same precomputed normalized graph and linear projection.
    # Thus the variable portion is only message propagation/aggregation.
    for i in range(1, repeats + 1):
        aggregate = conv1.propagate(
            normalized_edge, x=linear_x, edge_weight=normalized_weight, size=None
        )
        observations[f'aggregate/repeat_{i:02d}'] = snapshot(aggregate, device)
        del aggregate

    # Full untouched GCNConv.forward, including its own gcn_norm and bias.
    for i in range(1, repeats + 1):
        full = conv1(x1, edge1)
        observations[f'full_conv/repeat_{i:02d}'] = snapshot(full, device)
        del full

    print(f'[restart {restart}] nodes={data.x.size(0)} edges={edge1.size(1)} '
          f'normalized_edges={normalized_edge.size(1)}', flush=True)
    del model, linear_x
    return observations


def main():
    args = args_parser()
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    gpu = torch.cuda.get_device_name(device)
    if gpu != 'NVIDIA GeForce RTX 2070':
        raise RuntimeError(f'Expected NVIDIA GeForce RTX 2070, got {gpu}')
    actual_hashes = source_hashes()
    changed = [name for name, expected in EXPECTED.items()
               if actual_hashes.get(name) != expected]
    if changed:
        raise RuntimeError(f'Training source hashes differ from E3 evidence: {changed}')

    dataset_path = Path(args.dataset_path).expanduser().resolve() if args.dataset_path else (
        REPO_ROOT.parent / 'datasets' / 'Coauthor-Phy'
    ).resolve()
    if not dataset_path.parent.is_dir():
        raise FileNotFoundError(f'Dataset parent missing: {dataset_path.parent}')
    output_root = (Path(args.output_root).expanduser().resolve() if args.output_root
                   else B_ROOT / 'outputs')
    output_root.mkdir(parents=True, exist_ok=True)
    folder = output_root / (
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        + f'_E3_conv1_components_seed_{args.seed}'
    )
    folder.mkdir(exist_ok=False)
    print('E3 conv1 isolation diagnostic (NOT performance or formal-quality data)')
    print(f'GPU={gpu} Seed={args.seed} Restarts={args.restarts} Repeats={args.repeats}')
    print('Results:', folder, flush=True)

    # Dataset and graph-add/drop scores are preprocessed once, as in E3.
    args_model = SimpleNamespace(**FIXED)
    set_seed(args.seed)
    data = get_dataset(str(dataset_path), 'Coauthor-Phy').to(device)
    add_weights, adding_edge = add_edge_weights(data.edge_index, args_model.edge_add_rate)
    drop_weights = drop_edge_weights(data.edge_index)
    sync(device)

    comparisons = []
    reference = None
    for restart in range(1, args.restarts + 1):
        current = measure_one_restart(
            restart, args.seed, args_model, data, add_weights,
            adding_edge, drop_weights, device, args.repeats
        )
        if reference is None:
            reference = current
        else:
            for name in reference:
                stat = diff(reference[name], current[name])
                comparisons.append({'scope': f'restart_1_vs_{restart}',
                                    'component': name, **stat})
        for prefix in ('aggregate', 'full_conv'):
            baseline = current[f'{prefix}/repeat_01']
            for rep in range(2, args.repeats + 1):
                name = f'{prefix}/repeat_{rep:02d}'
                stat = diff(baseline, current[name])
                comparisons.append({'scope': f'within_restart_{restart}',
                                    'component': name, **stat})
        print(f'  restart {restart}: captured {len(current)} stage outputs', flush=True)

    fields = ('scope', 'component', 'equal', 'shape_equal',
              'different_elements', 'max_abs_diff')
    with (folder / 'component_comparisons.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    mismatches = [r for r in comparisons if not r['equal']]
    result = {
        'purpose': 'Isolate forward GCNConv into graph norm / linear / propagation',
        'seed': args.seed, 'device': str(device), 'gpu': gpu,
        'torch_version': torch.__version__,
        'restarts': args.restarts, 'repeats_per_restart': args.repeats,
        'source_sha256': actual_hashes,
        'first_cross_restart_mismatch': next(
            (r['component'] for r in comparisons
             if r['scope'] == 'restart_1_vs_2' and not r['equal']), None),
        'mismatch_count': len(mismatches),
        'mismatches': mismatches,
        'warning': ('Captures synchronize CUDA and can change scheduling. '
                    'A repeated result does not establish deterministic training; '
                    'a propagation mismatch suggests (but does not prove) '
                    'nondeterministic graph aggregation. No models or splits changed.'),
    }
    (folder / 'component_summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print('Mismatch rows:', len(mismatches))
    for r in mismatches[:35]:
        print(' ',r['scope'],r['component'],
              'different_elements=',r['different_elements'],
              'max_abs=',r['max_abs_diff'])
    print('CSV:', folder / 'component_comparisons.csv')
    print('Summary:', folder / 'component_summary.json')
    print('This is NOT a speed benchmark.', flush=True)


if __name__ == '__main__':
    main()
