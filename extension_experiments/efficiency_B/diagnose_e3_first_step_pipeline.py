"""Locate the earliest observable numerical divergence WITHIN E3's first step.

Same-seed, same-variant, independent model/optimizer restarts are compared.
Training, graph augmentation, GCN, VR, and loss implementations are imported
unchanged from E3. Every tensor capture synchronizes and copies to CPU, so the
program is deliberately invasive and is NOT a speed or formal-quality test.

The tensor snapshots cover augmented inputs, conv1/conv2 outputs for both views,
encoder/projector outputs, losses, parameter gradients, Adam state and updated
parameters. RNG states are hashed at three locations.

Place this script beside train_e3.py, model_b1.py, model_b2.py. Run from the
repository root, e.g.:
    python -u extension_experiments/efficiency_B/diagnose_e3_first_step_pipeline.py \
      --seed 20261129 --variants b0,b2x8 --repeats 2 --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_e3 import (
    B_ROOT,
    FIXED,
    REPO_ROOT,
    add_edge_weights,
    drop_edge_weights,
    get_dataset,
    get_optim,
    make_augmented_views,
    model_for,
    set_seed,
    source_hashes,
    sync,
    variance_floor_loss,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=20261129)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--variants', default='b0,b2x8')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--reference', default=None,
                   help='Existing e3_same_seed_replay_20261129/replay_comparison.json')
    p.add_argument('--dataset-path', default=None)
    p.add_argument('--output-root', default=None)
    args = p.parse_args()
    args.variants = args.variants.split(',')
    if len(args.variants) != len(set(args.variants)) or not args.variants or any(
        x not in {'b0', 'b2x8'} for x in args.variants
    ):
        p.error('use --variants b0,b2x8, b0, or b2x8')
    if args.repeats < 2 or args.repeats > 5:
        p.error('--repeats must be from 2 through 5')
    if args.seed < 0:
        p.error('--seed must be nonnegative')
    if args.device != 'cuda:0':
        p.error('the diagnostic is intended for the original cuda:0 GPU')
    return args


def verify_history(args, digests):
    path = (Path(args.reference).expanduser().resolve() if args.reference else
            B_ROOT / 'outputs' / 'e3_same_seed_replay_20261129' /
            'replay_comparison.json')
    if not path.is_file():
        raise FileNotFoundError(f'Missing provenance reference: {path}')
    reference = json.loads(path.read_text(encoding='utf-8'))
    if int(reference['seed']) != args.seed:
        raise ValueError('Reference seed does not match argument')
    old = reference.get('shared_original_source_sha256', {})
    altered = [name for name, value in digests.items() if old.get(name) != value]
    if altered:
        raise RuntimeError(f'Source-version mismatch: {altered}')
    return str(path)


def capture(tensor):
    return tensor.detach().to('cpu').contiguous().clone()


def tensor_digest(tensor):
    x = tensor.contiguous()
    h = hashlib.sha256()
    h.update(str(x.dtype).encode('ascii'))
    h.update(str(tuple(x.shape)).encode('ascii'))
    h.update(x.numpy().tobytes())
    return h.hexdigest()


def rng_digest(device):
    def h(blob):
        return hashlib.sha256(blob).hexdigest()
    return {
        'torch_cpu': tensor_digest(torch.get_rng_state()),
        'torch_cuda': tensor_digest(torch.cuda.get_rng_state(device).cpu()),
        'python': h(pickle.dumps(random.getstate(), protocol=4)),
        'numpy': h(pickle.dumps(np.random.get_state(), protocol=4)),
    }


def snapshot_once(variant, replay, seed, args_model, data, add_weights,
                  adding_edge, drop_weights, device):
    """Exactly one independent optimizer update, with tensor stage snapshots."""
    set_seed(seed)
    model = model_for(variant, data.num_features, args_model, device)
    optimizer = get_optim(model.parameters(), args_model)
    model.train()
    optimizer.zero_grad()
    sync(device)

    snaps = {}
    rng = {}
    for name, param in sorted(model.named_parameters()):
        snaps['00_initial/' + name] = capture(param)
    rng['before_aug'] = rng_digest(device)

    (x1, edge1), (x2, edge2) = make_augmented_views(
        data, add_weights, adding_edge, drop_weights, args_model
    )
    sync(device)
    rng['after_aug'] = rng_digest(device)
    # Coauthor-Phy features can exceed 1 GiB per view.  With feature drop=0,
    # compare the input tensors directly against the ONE fixed dataset tensor
    # on device instead of materializing redundant CPU feature snapshots.
    # Abort on a changed feature tensor; this experiment intentionally freezes it.
    if not torch.equal(x1, data.x) or not torch.equal(x2, data.x):
        raise RuntimeError('Augmented features changed despite feature-drop rate zero')
    for name, t in [('01_input/edge1', edge1), ('01_input/edge2', edge2)]:
        snaps[name] = capture(t)

    # Model.forward invokes conv1, conv2 for view 1, then for view 2.
    # Forward hooks observe the actual original operators without replacing them.
    count = {'conv1': 0, 'conv2': 0}
    handles = []
    def make_hook(name):
        def save_output(_module, _inp, output):
            count[name] += 1
            key = f'02_forward_view{count[name]}/{name}'
            snaps[key] = capture(output)
        return save_output
    for name in ('conv1', 'conv2'):
        handles.append(getattr(model, name).register_forward_hook(make_hook(name)))
    try:
        out1, z1 = model(x1, edge1)
        snaps['03_view1/encoder_out'] = capture(out1)
        snaps['03_view1/projected_z'] = capture(z1)
        out2, z2 = model(x2, edge2)
        snaps['04_view2/encoder_out'] = capture(out2)
        snaps['04_view2/projected_z'] = capture(z2)
    finally:
        for handle in handles:
            handle.remove()
    if count != {'conv1': 2, 'conv2': 2}:
        raise AssertionError(f'Unexpected GCN call count: {count}')

    contrastive = model.loss(z1, z2, args_model.batch_compute)
    vr = variance_floor_loss(out1, out2, gamma=args_model.vr_gamma)
    total = contrastive + args_model.vr_lambda * vr
    if not torch.isfinite(total).item():
        raise FloatingPointError('First-step total loss is not finite')
    sync(device)
    rng['after_loss'] = rng_digest(device)
    for name, t in [('05_loss/contrastive', contrastive),
                    ('05_loss/variance', vr), ('05_loss/total', total)]:
        snaps[name] = capture(t)

    total.backward()
    sync(device)
    for name, p in sorted(model.named_parameters()):
        if p.grad is None:
            raise RuntimeError(f'Missing gradient for trainable parameter {name}')
        snaps['06_gradient/' + name] = capture(p.grad)

    optimizer.step()
    sync(device)
    for name, p in sorted(model.named_parameters()):
        snaps['08_updated/' + name] = capture(p)
        state = optimizer.state[p]
        for key, value in sorted(state.items()):
            if isinstance(value, torch.Tensor):
                snaps[f'07_optimizer/{name}.{key}'] = capture(value)

    scalars = {'variant': variant, 'repeat': replay,
               'contrastive_loss': float(contrastive.detach().item()),
               'variance_loss': float(vr.detach().item()),
               'total_loss': float(total.detach().item()),
               'edge1_count': int(edge1.shape[1]),
               'edge2_count': int(edge2.shape[1]),
               'both_features_equal_fixed_dataset': True}
    print(f'[{variant} repeat={replay}] '
          f'contrastive={scalars["contrastive_loss"]:.10f} '
          f'VR={scalars["variance_loss"]:.10f} '
          f'edges=({scalars["edge1_count"]},{scalars["edge2_count"]})',
          flush=True)
    return {'tensors': snaps, 'rng': rng, 'scalars': scalars}


def tensor_difference(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return {'exact_equal': False, 'shape_match': False,
                'different_elements': None, 'max_abs_diff': None,
                'max_abs_reference': None}
    exact = tensor_digest(a) == tensor_digest(b)
    if exact:
        return {'exact_equal': True, 'shape_match': True,
                'different_elements': 0, 'max_abs_diff': 0.0,
                'max_abs_reference': float(a.abs().max().item()) if a.numel() else 0.0}
    different = int(torch.count_nonzero(a != b).item())
    if a.dtype.is_floating_point or a.dtype.is_complex:
        max_abs_diff = float((a - b).abs().max().item())
        max_abs_ref = float(torch.maximum(a.abs().max(), b.abs().max()).item())
    elif a.dtype == torch.bool:
        max_abs_diff = 1.0 if different else 0.0
        max_abs_ref = 1.0
    else:
        # Difference of integer edge indices. Avoid integer subtraction overflow.
        max_abs_diff = float((a.double() - b.double()).abs().max().item())
        max_abs_ref = float(torch.maximum(a.double().abs().max(), b.double().abs().max()).item())
    return {'exact_equal': False, 'shape_match': True,
            'different_elements': different, 'max_abs_diff': max_abs_diff,
            'max_abs_reference': max_abs_ref}


def compare(variant, reference, new, replay):
    if set(reference['tensors']) != set(new['tensors']):
        raise RuntimeError(f'Captured tensor names differ in {variant} replay {replay}')
    rows = []
    for name in reference['tensors']:
        # Stage labels order by the lexical prefix (00,01,...,08).
        a, b = reference['tensors'][name], new['tensors'][name]
        stats = tensor_difference(a, b)
        rows.append({'variant': variant, 'compared_repeat': replay,
                     'stage': name.split('/')[0], 'tensor': name.split('/', 1)[-1],
                     'shape': str(tuple(a.shape)), 'dtype': str(a.dtype),
                     **stats})
    # Explicit execution order, since lexical sorting would incorrectly put
    # view 2's conv1/conv2 before view 1's final encoder/projector outputs.
    stage_order = {
        '00_initial': 0, '01_input': 1,
        '02_forward_view1': 2, '03_view1': 3,
        '02_forward_view2': 4, '04_view2': 5,
        '05_loss': 6, '06_gradient': 7,
        '07_optimizer': 8, '08_updated': 9,
    }
    rows.sort(key=lambda x: (stage_order[x['stage']], x['tensor']))
    first = next((r for r in rows if not r['exact_equal']), None)
    rng = {stage: {kind: reference['rng'][stage][kind] == new['rng'][stage][kind]
                   for kind in reference['rng'][stage]}
           for stage in reference['rng']}
    mismatches = [r for r in rows if not r['exact_equal']]
    print(f'[{variant} repeat 1 vs {replay}] '
          f'first differing tensor: '
          f'{(first["stage"] + "/" + first["tensor"]) if first else "NONE"}; '
          f'mismatched tensors={len(mismatches)}/{len(rows)}; '
          f'all RNG equal={all(all(s.values()) for s in rng.values())}',
          flush=True)
    if first:
        print(f'    first difference count={first["different_elements"]} '
              f'max_abs={first["max_abs_diff"]}', flush=True)
    return rows, {'compared_repeat': replay,
                  'first_different_tensor': ((first['stage'] + '/' + first['tensor']) if first else None),
                  'first_different_tensor_max_abs': first['max_abs_diff'] if first else None,
                  'mismatch_count': len(mismatches),
                  'recorded_rng_state_equal': rng,
                  'augmented_edge_tensors_equal': all(
                      r['exact_equal'] for r in rows
                      if r['stage'] == '01_input' and r['tensor'].startswith('edge')),
                  'gcns_and_loss_equal': {k: all(r['exact_equal'] for r in rows if r['stage']==k)
                                          for k in ('02_forward_view1', '02_forward_view2',
                                                    '03_view1', '04_view2', '05_loss',
                                                    '06_gradient', '07_optimizer', '08_updated')},
                  'note': 'These findings describe an invasive synchronized first-step diagnostic, not formal training.'}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; do not run the real diagnostic on CPU')
    device = torch.device(args.device)
    gpu = torch.cuda.get_device_name(device)
    if gpu != 'NVIDIA GeForce RTX 2070':
        raise RuntimeError(f'Expected RTX 2070, got {gpu!r}')
    digests = source_hashes()
    source = verify_history(args, digests)
    path = (Path(args.dataset_path).expanduser().resolve() if args.dataset_path else
            (REPO_ROOT.parent / 'datasets' / 'Coauthor-Phy').resolve())
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    out_root = (Path(args.output_root).expanduser().resolve() if args.output_root else
                B_ROOT / 'outputs')
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / (datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                          + f'_E3_first_step_pipeline_seed_{args.seed}')
    out_dir.mkdir(parents=False, exist_ok=False)
    print('E3 first-step stage comparison; diagnostic only, not speed/quality validation')
    print(f'GPU={gpu}; seed={args.seed}; versions={args.variants}; repeats={args.repeats}')
    print(f'Output: {out_dir}', flush=True)

    # Precompute input graph and edge weights once. The per-variant model and RNG
    # are completely restarted for every independent one-step replay.
    set_seed(args.seed)
    data = get_dataset(str(path), 'Coauthor-Phy').to(device)
    add_weights, adding_edge = add_edge_weights(data.edge_index, FIXED['edge_add_rate'])
    drop_weights = drop_edge_weights(data.edge_index)
    sync(device)
    args_model = SimpleNamespace(**FIXED)
    all_rows = []
    summary = {}
    for variant in args.variants:
        baseline = snapshot_once(variant, 1, args.seed, args_model, data,
                                 add_weights, adding_edge, drop_weights, device)
        compare_results = []
        for rep in range(2, args.repeats + 1):
            candidate = snapshot_once(variant, rep, args.seed, args_model, data,
                                      add_weights, adding_edge, drop_weights, device)
            rows, info = compare(variant, baseline, candidate, rep)
            all_rows.extend(rows)
            compare_results.append(info)
            del candidate
        summary[variant] = compare_results
        del baseline

    csv_path = out_dir / 'stage_tensor_comparison.csv'
    fields = ['variant', 'compared_repeat', 'stage', 'tensor', 'shape', 'dtype',
              'exact_equal', 'shape_match', 'different_elements', 'max_abs_diff',
              'max_abs_reference']
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    result = {
        'purpose': 'Within-variant, same-seed, first-step source of numerical divergence',
        'gpu': gpu, 'torch_version': torch.__version__, 'seed': args.seed,
        'variants': args.variants, 'repeats_per_variant': args.repeats,
        'expected_original_source_sha256': digests,
        'reference': source, 'conditions': {
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
        },
        'summary': summary,
        'limitations': [
            'Only within-variant comparisons are scientifically meaningful here.',
            'Forward hooks and CPU copies are intrusive: absence of divergence does not prove determinism.',
            'Matching RNG/graph fingerprints cannot establish determinism of all GPU kernels.',
            'A mismatch in conv1 output supports a forward-path numerical discrepancy but not a specific kernel-level cause.',
            'Do not use timing or merge this diagnostic with five-seed formal accuracy statistics.',
            'No validation/test data, fixed splits, historical checkpoints, or original source are modified.'
        ],
    }
    json_path = out_dir / 'first_step_summary.json'
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Finished. Summary: {json_path}')
    print(f'Full tensor diffs: {csv_path}', flush=True)


if __name__ == '__main__':
    main()
