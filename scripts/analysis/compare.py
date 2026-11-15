#!/usr/bin/env python3
"""
compare.py — compare ink predictions from several model configurations.

What it does (plain English):
  * Loads 2+ ink-prediction images (.tif) of the SAME scroll segment, each made
    by a different checkpoint / depth window.
  * Measures how much they agree (pixel agreement, IoU, correlation).
  * Builds a per-pixel "uncertainty" map: where the configurations disagree.
  * Optionally scores each prediction against hand labels (AUC), if given.
  * Writes metrics.json + PNG images you can put straight on the one-sheet.

Usage:
  python compare.py --out /data/results/sweep/analysis \
      /data/results/sweep/w043_step-010000.tif /data/results/sweep/w043_step-040000.tif ...
  Optional: --labels path/to/label.png_or_tif   --threshold 0.5   --max-png 2000
"""
import argparse, json, itertools, os, time
import numpy as np
import tifffile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load_pred(path):
    """Load a prediction and scale it to 0..1 floats, whatever dtype it was saved as."""
    a = tifffile.imread(path)
    a = np.squeeze(a)
    if a.ndim == 3:                       # (C,H,W) or (H,W,C): take the first channel
        a = a[0] if a.shape[0] < a.shape[-1] else a[..., 0]
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float32) / np.iinfo(a.dtype).max
    else:
        a = a.astype(np.float32)
        if a.max() > 1.0:                 # e.g. float 0..255
            a = a / a.max()
    return a


def load_labels(path, shape):
    if path.endswith((".tif", ".tiff")):
        l = np.squeeze(tifffile.imread(path))
    else:
        l = np.array(Image.open(path).convert("L"))
    if l.shape != shape:                  # resize labels to the prediction grid
        l = np.array(Image.fromarray((l > 0).astype(np.uint8) * 255).resize(
            (shape[1], shape[0]), Image.NEAREST))
    return l > 0


def auc(scores, labels):
    """ROC AUC via ranks (Mann-Whitney U). No sklearn needed."""
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return None
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def to_png(arr, path, max_side, cmap=None):
    """Save a 0..1 array as a (downscaled) PNG. cmap='heat' gives a black→red→yellow map."""
    a = np.clip(np.nan_to_num(arr), 0, 1)
    if cmap == "heat":
        rgb = np.stack([np.clip(a * 2, 0, 1), np.clip(a * 2 - 1, 0, 1), np.zeros_like(a)], -1)
        img = Image.fromarray((rgb * 255).astype(np.uint8), "RGB")
    else:
        img = Image.fromarray((a * 255).astype(np.uint8), "L")
    s = max_side / max(img.size)
    if s < 1:
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.BILINEAR)
    img.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("preds", nargs="+", help="prediction .tif files (same segment)")
    p.add_argument("--out", required=True)
    p.add_argument("--labels", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-png", type=int, default=2000)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    names = [os.path.splitext(os.path.basename(f))[0] for f in a.preds]
    preds = [load_pred(f) for f in a.preds]
    shapes = {x.shape for x in preds}
    if len(shapes) != 1:                  # crop to the common area if sizes differ slightly
        h = min(s[0] for s in shapes); w = min(s[1] for s in shapes)
        print(f"!! shapes differ {shapes}; cropping all to {(h, w)}")
        preds = [x[:h, :w] for x in preds]
    stack = np.stack(preds)               # (N, H, W)
    valid = stack.max(0) > 0              # ignore pure-background pixels (outside the page)
    print(f"{len(preds)} predictions, size {stack.shape[1:]}, valid pixels {valid.sum():,}")

    binar = stack > a.threshold
    vote = binar.mean(0)                  # fraction of configs that say "ink"
    disagree = (vote > 0) & (vote < 1)    # at least one says ink and one says not
    std = stack.std(0)

    m = {
        "configs": names,
        "threshold": a.threshold,
        "image_shape": list(stack.shape[1:]),
        "valid_pixels": int(valid.sum()),
        "per_config": {},
        "pairwise": {},
        "overall": {
            "pct_pixels_all_agree": float(100 * (1 - disagree[valid].mean())),
            "pct_pixels_disagree": float(100 * disagree[valid].mean()),
            "mean_score_std": float(std[valid].mean()),
            "pct_pixels_high_uncertainty_std_gt_0.2": float(100 * (std[valid] > 0.2).mean()),
        },
    }
    for n, x, b in zip(names, stack, binar):
        m["per_config"][n] = {"mean_score": float(x[valid].mean()),
                              "pct_pixels_ink": float(100 * b[valid].mean())}
    for (i, ni), (j, nj) in itertools.combinations(enumerate(names), 2):
        bi, bj = binar[i][valid], binar[j][valid]
        inter, union = (bi & bj).sum(), (bi | bj).sum()
        m["pairwise"][f"{ni} vs {nj}"] = {
            "pixel_agreement_pct": float(100 * (bi == bj).mean()),
            "ink_iou": float(inter / union) if union else None,
            "score_correlation": float(np.corrcoef(stack[i][valid], stack[j][valid])[0, 1]),
        }

    if a.labels and os.path.exists(a.labels):
        lab = load_labels(a.labels, stack.shape[1:])
        lv = lab[valid]
        m["vs_labels"] = {n: {"auc": auc(x[valid], lv)} for n, x in zip(names, stack)}
        m["vs_labels"]["ensemble_mean"] = {"auc": auc(stack.mean(0)[valid], lv)}

    # Images for the one-sheet
    for n, x in zip(names, stack):
        to_png(x, os.path.join(a.out, f"pred_{n}.png"), a.max_png)
    to_png(stack.mean(0), os.path.join(a.out, "ensemble_mean.png"), a.max_png)
    to_png(std / max(std.max(), 1e-6), os.path.join(a.out, "uncertainty_std.png"), a.max_png, "heat")
    to_png(disagree.astype(np.float32), os.path.join(a.out, "disagreement_map.png"), a.max_png, "heat")

    m["analysis_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(json.dumps(m["overall"], indent=2))
    for k, v in m["pairwise"].items():
        print(f"  {k}: agree {v['pixel_agreement_pct']:.1f}%  IoU {v['ink_iou']}  r={v['score_correlation']:.3f}")
    if "vs_labels" in m:
        print("  AUC vs labels:", {k: v["auc"] for k, v in m["vs_labels"].items()})
    print("wrote", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
