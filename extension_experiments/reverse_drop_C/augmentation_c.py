"""Experiment C: reverse only the EPAGCL edge-drop score normalization."""
import torch
from torch_geometric.utils import degree, to_undirected


def reverse_drop_edge_weights(edge_index):
    edge_index = to_undirected(edge_index)
    deg = degree(edge_index[1])
    w = 1 / (deg[edge_index[0]] * deg[edge_index[1]]) ** 0.5
    # The only experimental change relative to Augmentation.drop_edge_weights.
    weights = (w.max() - w) / (w.max() - w.mean())
    return weights


def prepare_edge_alignment(edge_index):
    """Map canonical undirected weights to the original sampling columns.

    Neither the original edge_index nor the input graph is modified.
    Every original edge is matched by its ordered (source, target) pair.
    """
    if edge_index.ndim != 2 or edge_index.size(0) != 2 or edge_index.size(1) == 0:
        raise ValueError('Expected nonempty edge_index with shape [2, E]')
    if edge_index.dtype != torch.long or (edge_index < 0).any().item():
        raise ValueError('Expected nonnegative torch.long edge indices')
    canonical = to_undirected(edge_index)
    num_nodes = int(edge_index.max().item()) + 1
    if num_nodes > 3037000499:
        raise ValueError('Node index range is too large for int64 edge keys')
    # Encode ordered endpoint pairs as unique integer keys. Sort explicitly
    # rather than relying on a particular PyG version's ordering convention.
    original_keys = edge_index[0] * num_nodes + edge_index[1]
    canonical_keys = canonical[0] * num_nodes + canonical[1]
    sorted_keys, order = canonical_keys.sort()
    positions = torch.searchsorted(sorted_keys, original_keys.contiguous())
    if (positions >= sorted_keys.numel()).any().item():
        raise ValueError('An original edge is missing from to_undirected(edge_index)')
    if not torch.equal(sorted_keys[positions], original_keys):
        raise ValueError('Edge endpoint matching failed')
    original_to_canonical = order[positions]
    unique_original = int(torch.unique(original_keys).numel())
    same_order = torch.equal(edge_index, canonical)
    report = {
        'original_edge_entries': int(edge_index.size(1)),
        'unique_original_edge_entries': unique_original,
        'canonical_undirected_edge_entries': int(canonical.size(1)),
        'duplicate_original_entries': int(edge_index.size(1)) - unique_original,
        'reverse_entries_added_by_to_undirected': int(canonical.size(1)) - unique_original,
        'original_order_equals_canonical': same_order,
        'sampling_edge_index_changed': False,
        'weight_alignment': 'identity' if same_order else 'ordered_endpoint_pair_mapping',
    }
    return canonical, original_to_canonical, report


def validate_weights(weights, edge_index, method):
    if weights.ndim != 1 or weights.numel() != edge_index.size(1):
        raise ValueError(f'{method}: weight count does not match edge count')
    if not torch.isfinite(weights).all().item() or (weights < 0).any().item() or weights.mean().item() <= 0:
        raise ValueError(
            f'{method}: invalid weights (possibly all edge weights are equal). '
            'Both original and reversed formulas require nonzero normalization denominators.'
        )


def expected_drop_rate(weights, p, threshold=1.0):
    """Same probability calculation as original drop_edges, without sampling."""
    prob = weights / weights.mean() * p
    prob = prob.clamp(max=threshold)
    return float(prob.mean().item())
