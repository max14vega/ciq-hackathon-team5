#!/usr/bin/env python3
"""Project Prometheus E6 offline analysis, following DATA_SPEC.md v1.

Run from any directory:
  python scripts/compare_predictions.py --input-root results/E6-bulk --output-root results/E6-bulk/analysis_local

Expected inputs: <input-root>/<segment>/<segment>__<ckpt>__<layers>.tif
Output is organized per segment/group; cluster originals are never overwritten.
"""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CHECKPOINTS = ('ck010k', 'ck040k', 'ck075k')
LAYERS = ('L04-20', 'L06-22', 'L08-24')
NAME_RE = re.compile(r'^(w\d+)__(ck\d+k)__(L\d+-\d+)(?:__.*)?\.tif$', re.I)


def discover(root):
    grouped = {}
    for path in root.rglob('*.tif'):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        segment, ckpt, layers = match.groups()
        if segment != path.parent.name:
            raise ValueError(f'File segment and directory disagree: {path}')
        key = (ckpt, layers)
        grouped.setdefault(segment, {})
        if key in grouped[segment]:
            raise ValueError(f'Duplicate configuration {segment} {key}: {path}')
        grouped[segment][key] = path
    return grouped


def read_uint8(path, shape=None):
    arr = tifffile.imread(path)
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise ValueError(f'{path}: expected 2D uint8 TIFF; got {arr.shape} {arr.dtype}')
    if shape is not None and arr.shape != shape:
        raise ValueError(f'{path}: shape {arr.shape} differs from {shape}; spatial alignment required')
    return arr


def preview(image, path, cmap, title, vmax=None):
    h, w = image.shape
    scale = max(1, int(np.ceil(max(h, w) / 2000)))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image[::scale, ::scale], cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def analyze_group(paths, outdir, threshold=0.5, chunk_rows=256, write_maps=True):
    """Compute exact DATA_SPEC metrics using row chunks, avoiding huge float stacks."""
    outdir.mkdir(parents=True, exist_ok=True)
    labels = [p.stem for p in paths]
    images = [read_uint8(p) for p in paths]
    shape = images[0].shape
    for path, img in zip(paths, images):
        if img.shape != shape:
            raise ValueError(f'{path}: shape {img.shape} differs from {shape}')
    n = len(images)
    if n < 2:
        raise ValueError('Need at least two configurations')
    h, w = shape
    # Pixel score is raw uint8 / 255. Threshold must be strictly greater.
    threshold_byte = threshold * 255
    valid_count = disagree_count = strong_count = 0
    sum_std = 0.0
    ink_counts = np.zeros(n, dtype=np.int64)
    pair_counts = {(i, j): dict(agree=0, intersection=0, union=0,
                                sx=0.0, sy=0.0, sxx=0.0, syy=0.0, sxy=0.0)
                   for i in range(n) for j in range(i+1, n)}
    if write_maps:
        disagreement = np.zeros(shape, dtype=np.uint8)
        std_map = np.zeros(shape, dtype=np.uint8)
        mean_map = np.zeros(shape, dtype=np.uint8)
    for start in range(0, h, chunk_rows):
        stop = min(h, start + chunk_rows)
        raw = np.stack([img[start:stop] for img in images])
        valid = raw.max(axis=0) > 0
        vcount = int(valid.sum())
        if vcount == 0:
            continue
        scores = raw.astype(np.float32) / 255.0
        binary = raw > threshold_byte
        votes = binary.sum(axis=0)
        is_disagree = (votes > 0) & (votes < n)
        std = scores.std(axis=0, ddof=0)
        mean = scores.mean(axis=0)
        valid_count += vcount
        disagree_count += int(np.count_nonzero(is_disagree & valid))
        strong_count += int(np.count_nonzero((std > 0.2) & valid))
        sum_std += float(std[valid].sum(dtype=np.float64))
        ink_counts += np.count_nonzero(binary & valid[None, :, :], axis=(1, 2))
        if write_maps:
            disagreement[start:stop] = np.where(valid & is_disagree, 255, 0).astype(np.uint8)
            std_map[start:stop] = np.where(valid, np.clip(2 * std, 0, 1) * 255, 0).astype(np.uint8)
            mean_map[start:stop] = np.where(valid, np.clip(mean, 0, 1) * 255, 0).astype(np.uint8)
        for (i, j), stats in pair_counts.items():
            a = binary[i][valid]
            b = binary[j][valid]
            stats['agree'] += int(np.count_nonzero(a == b))
            stats['intersection'] += int(np.count_nonzero(a & b))
            stats['union'] += int(np.count_nonzero(a | b))
            # Correlation is unchanged by /255, so use uint8 scores here.
            x = raw[i][valid].astype(np.float64)
            y = raw[j][valid].astype(np.float64)
            stats['sx'] += float(x.sum())
            stats['sy'] += float(y.sum())
            stats['sxx'] += float(np.dot(x, x))
            stats['syy'] += float(np.dot(y, y))
            stats['sxy'] += float(np.dot(x, y))
    if valid_count == 0:
        raise ValueError('No valid pixels found (all TIFFs are zero)')
    pairs = {}
    for (i, j), s in pair_counts.items():
        m = valid_count
        cov = s['sxy'] - s['sx'] * s['sy'] / m
        vx = s['sxx'] - s['sx'] ** 2 / m
        vy = s['syy'] - s['sy'] ** 2 / m
        corr = float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else None
        pairs[f'{labels[i]} vs {labels[j]}'] = {
            'agree_pct': 100 * s['agree'] / m,
            'ink_iou': s['intersection'] / s['union'] if s['union'] else None,
            'corr': corr,
        }
    metrics = {
        'configs': labels,
        'valid_pixels': valid_count,
        'pct_pixels_disagree': 100 * disagree_count / valid_count,
        'mean_score_std': sum_std / valid_count,
        'pct_pixels_high_uncertainty_std_gt_0.2': 100 * strong_count / valid_count,
        'per_config_pct_ink': {label: 100 * int(c) / valid_count for label, c in zip(labels, ink_counts)},
        'pairwise': pairs,
    }
    if write_maps:
        for name, arr in [('disagreement_map', disagreement),
                          ('uncertainty_std', std_map), ('ensemble_mean', mean_map)]:
            tifffile.imwrite(outdir / f'{name}.tif', arr, compression='zlib')
        preview(disagreement, outdir / 'disagreement_map.png', 'gray', 'Ink-call disagreement', vmax=255)
        preview(std_map, outdir / 'uncertainty_std.png', 'inferno', 'Score standard deviation (2× scaled)', vmax=255)
        preview(mean_map, outdir / 'ensemble_mean.png', 'gray', 'Mean ink score', vmax=255)
    (outdir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(f'{outdir}: {n} configs; {valid_count:,} valid pixels; '
          f'{metrics["pct_pixels_disagree"]:.2f}% disagree; '
          f'{metrics["pct_pixels_high_uncertainty_std_gt_0.2"]:.3f}% std>0.2')
    return metrics


def main():
    ap = argparse.ArgumentParser(description='Analyze Project Prometheus TIFF predictions per DATA_SPEC v1')
    ap.add_argument('--input-root', type=Path, required=True, help='E6-bulk directory with segment subdirectories')
    ap.add_argument('--output-root', type=Path, required=True, help='Separate analysis directory (never input root)')
    ap.add_argument('--chunk-rows', type=int, default=256)
    ap.add_argument('--no-maps', action='store_true', help='Compute JSON metrics only')
    args = ap.parse_args()
    if args.chunk_rows < 1:
        ap.error('--chunk-rows must be positive')
    root = args.input_root.resolve()
    out = args.output_root.resolve()
    if not root.is_dir():
        ap.error(f'Input directory does not exist: {root}')
    if out == root or root in out.parents and out.name in {s for s in discover(root)}:
        ap.error('Choose a separate output directory')
    files = discover(root)
    if not files:
        ap.error(f'No matching TIFFs in {root}')
    result = {'threshold': 0.5, 'segments': {}, 'sensitivity': {}}
    for segment, index in sorted(files.items()):
        expected = {(c, l) for c in CHECKPOINTS for l in LAYERS}
        missing = expected - set(index)
        if missing:
            print(f'WARNING {segment}: {len(missing)} of 9 E6 configurations missing: {sorted(missing)}')
        groups = {
            'all_9': [(c, l) for c in CHECKPOINTS for l in LAYERS],
            'vary_checkpoint': [(c, 'L06-22') for c in CHECKPOINTS],
            'vary_depth': [('ck075k', l) for l in LAYERS],
        }
        result['segments'][segment] = {}
        for group, keys in groups.items():
            absent = [k for k in keys if k not in index]
            if absent:
                print(f'SKIP {segment}/{group}: missing {absent}')
                continue
            paths = [index[k] for k in keys]
            result['segments'][segment][group] = analyze_group(
                paths, out / segment / group, chunk_rows=args.chunk_rows,
                write_maps=not args.no_maps)
        # Recalculate threshold sensitivity for all 9, using same valid-mask definition.
        if not missing:
            images = [read_uint8(index[key]) for key in sorted(expected)]
            counts = {str(t): 0 for t in (0.4, 0.5, 0.6)}
            total = 0
            for start in range(0, images[0].shape[0], args.chunk_rows):
                raw = np.stack([im[start:start+args.chunk_rows] for im in images])
                valid = raw.max(axis=0) > 0
                total += int(valid.sum())
                for t in (0.4, 0.5, 0.6):
                    votes = (raw > t * 255).sum(axis=0)
                    counts[str(t)] += int(np.count_nonzero(valid & (votes > 0) & (votes < len(images))))
            result['sensitivity'][segment] = {'all_9': {'pct_pixels_disagree_at': {
                t: 100 * count / total for t, count in counts.items()}}}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'\nSaved combined metrics: {out / "metrics.json"}')


if __name__ == '__main__':
    main()
