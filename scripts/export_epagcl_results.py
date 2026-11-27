#!/usr/bin/env python3
"""Export lightweight EPAGCL research evidence. Dry run is the default.

Run from any directory with Python 3.9+; no third-party packages are needed.
Original experiments are never moved, edited, or deleted.
"""
import argparse
import csv
import hashlib
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

TEXT_AND_FIGURES = {
    '.json', '.csv', '.tsv', '.txt', '.md', '.patch', '.diff',
    '.png', '.jpg', '.jpeg', '.svg',
}
WEIGHTS = {'.pt', '.pth', '.ckpt', '.safetensors', '.pkl', '.pickle'}
LEGACY_NAMES = {'args.txt', 'result.txt', 'loss.txt', 'eval.txt'}


def regular_files(directory):
    for base, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {'.git', '.venv', '__pycache__'}
                         and not re.fullmatch(r'(?:logs?|.*_logs?)', d, re.I)
                         and not (Path(base) / d).is_symlink())
        for name in sorted(files):
            p = Path(base) / name
            if (p.is_file() and not p.is_symlink()
                    and not re.search(r'\.log(?:\.[^/]*)?$', name, re.I)):
                yield p


def portable_path(path):
    # Preserve hierarchy, changing Windows-incompatible characters in the copy only.
    return Path(*(re.sub(r'[<>:"\\|?*]', '-', part) for part in path.parts))


def raw_trace(path):
    name = path.name.lower()
    return (name in {'trace.json', 'profiler.json', 'profiler_trace.json'}
            or name.endswith('.trace.json') or 'chrome_trace' in name
            or ('trace' in name and 'profiler' in name and 'summary' not in name))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home() / 'projects/EPAGCL')
    parser.add_argument('--apply', action='store_true', help='Create research_results; default is preview only')
    parser.add_argument('--max-mib', type=int, default=10, help='Maximum individual evidence file size (default 10 MiB)')
    args = parser.parse_args()
    if args.max_mib < 1:
        parser.error('--max-mib must be positive')
    root = args.root.expanduser().resolve()
    dest = root / 'research_results'
    if not all((root / n).is_file() for n in ('main.py', 'Model.py', 'Augmentation.py')):
        parser.error('Not an EPAGCL project root: ' + str(root))
    sources = {
        'A_variance_regularization': root / 'extension_experiments/formal_A/outputs',
        'B_computational_efficiency': root / 'extension_experiments/efficiency_B/outputs',
        'C_reverse_edge_dropping/training': root / 'extension_experiments/reverse_drop_C/outputs',
        'C_reverse_edge_dropping/epr_analysis': root / 'extension_experiments/reverse_drop_C/epr_outputs',
    }
    legacy = root / 'trained_models'
    for directory in [legacy, *sources.values()]:
        if not directory.is_dir() or directory.is_symlink():
            parser.error('Missing or symbolic-link source directory: ' + str(directory))
        if not directory.resolve().is_relative_to(root):
            parser.error('Source is outside project: ' + str(directory))
    if args.apply and (dest.exists() or dest.is_symlink()):
        parser.error('research_results already exists; refusing to overwrite it')
    rows, skipped, excluded = [], [], Counter()
    used = set()
    max_bytes = args.max_mib * 1024 * 1024

    def select(category, source, relative):
        rel = portable_path(Path(category) / relative)
        size = source.stat().st_size
        if size > max_bytes:
            skipped.append(('above_size_limit', str(source.relative_to(root)), size))
            return
        key = str(rel).casefold()
        if key in used:
            raise RuntimeError('Destination collision after filename normalization: ' + str(rel))
        used.add(key)
        rows.append((category, source, rel, size))

    groups = Counter()
    for p in regular_files(legacy):
        if p.name != 'result.txt':
            continue
        folder = p.parent
        rel = folder.relative_to(legacy)
        if (not folder.name.startswith('completed_') or
                any(x.endswith('_Test') or x.startswith('in-progress_') for x in rel.parts)):
            excluded['legacy_test_or_incomplete_group'] += 1
            continue
        config = folder / 'args.txt'
        if not config.is_file() or config.is_symlink():
            skipped.append(('missing_legacy_args', str(p.relative_to(root)), p.stat().st_size))
            continue
        config_text = config.read_text(encoding='utf-8', errors='replace')
        if not re.search(r'^\s*repeat\s*:\s*5\s*$', config_text, re.MULTILINE):
            excluded['legacy_non_five_repeat_group'] += 1
            continue
        category = 'q3_ablation' if rel.parts[0].startswith('Q3_') else 'reproduction'
        groups[category] += 1
        for file in regular_files(folder):
            if file.name in LEGACY_NAMES:
                select(category, file, file.relative_to(legacy))

    for category, directory in sources.items():
        for p in regular_files(directory):
            ext = p.suffix.lower()
            if ext in WEIGHTS:
                excluded[category + ': weights'] += 1
            elif ext not in TEXT_AND_FIGURES:
                excluded[category + ': other file types'] += 1
                # List unfamiliar outputs for review, without copying them.
                skipped.append(('not_in_allowlist', str(p.relative_to(root)), p.stat().st_size))
            elif raw_trace(p):
                skipped.append(('raw_profiler_trace', str(p.relative_to(root)), p.stat().st_size))
            else:
                select(category, p, p.relative_to(directory))

    rows.sort(key=lambda row: str(row[2]))
    totals = defaultdict(lambda: [0, 0])
    for category, source, rel, size in rows:
        totals[category][0] += 1
        totals[category][1] += size
    print('MODE:', 'COPY' if args.apply else 'DRY RUN (no project changes)')
    print('ROOT:', root)
    print('DESTINATION:', dest)
    print('\nLEGACY FIVE-REPEAT GROUP COUNTS:', dict(groups))
    print('\nCATEGORY\tFILES\tMIB')
    for category, (count, size) in sorted(totals.items()):
        print(f'{category}\t{count}\t{size/1048576:.3f}')
    print(f'TOTAL\t{len(rows)}\t{sum(r[3] for r in rows)/1048576:.3f}')
    print('\nEXCLUDED COUNTS:')
    for key, count in sorted(excluded.items()):
        print(f'{key}\t{count}')
    print('\nSKIPPED FILES FOR REVIEW (reason, bytes, original path):')
    for reason, path, size in sorted(skipped):
        print(f'{reason}\t{size}\t{path}')
    print('\nSELECTED FILES (bytes, original path, destination relative to research_results):')
    for category, source, rel, size in rows:
        print(f'{size}\t{source.relative_to(root)}\t{rel}')
    if not args.apply:
        print('\nPreview only. Run with --apply after reviewing the plan.')
        return
    if not rows:
        parser.error('No files selected; refusing to create an empty export')
    # Build in a fresh temporary directory. Copy metadata and verify the bytes.
    # If copying fails, TemporaryDirectory only cleans up its own new files.
    with tempfile.TemporaryDirectory(prefix='.epagcl-export-', dir=root) as temp:
        stage = Path(temp) / 'research_results'
        stage.mkdir()
        with (stage / 'manifest.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['category', 'source_path', 'export_path', 'size_bytes', 'sha256'])
            for category, source, rel, size in rows:
                target = stage / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                checksum = digest(target)
                if target.stat().st_size != size or digest(source) != checksum:
                    raise RuntimeError('Source changed during export: ' + str(source))
                writer.writerow([category, source.relative_to(root), rel, size, checksum])
        if dest.exists() or dest.is_symlink():
            raise FileExistsError('Destination appeared during export: ' + str(dest))
        stage.rename(dest)
    print('\nExport complete:', dest)
    print('Original data unchanged. Review manifest.csv and exported contents before Git staging.')


if __name__ == '__main__':
    main()
