# Project Prometheus — Data & Metrics Spec (v1, 2026-11-14 12:55 CT)

Shared contract between the cluster pipeline (Fuzzball workflows) and offline analysis scripts.
If you compute a metric differently, give it a different name.

---

## 1. Prediction files (`.tif`)

| Property | Value |
|---|---|
| Format | single-channel, **uint8**, tiled TIFF (128×128 tiles) |
| Value meaning | `p = value / 255` = model ink score in [0, 1] (uncalibrated sigmoid output) |
| `0` | outside the papyrus (never predicted) |
| Typical background | ~0.37 (≈ 94/255) on papyrus with no ink |
| Pixel size | 9.362 µm (same grid as the input surface volume) |
| Shapes | w043: 6120 × 8120 · w024: 7100 × 6620 · w035: see file |
| Model | `scrollprize/ink_9um`, `hybrid_3d2d-seed42`, checkpoints `step-010000 / 040000 / 075000` |
| Inference | patch 128, stride 64 (50% overlap), Hann blending, fp16 autocast, `--no-compile`, no TTA |
| Read in Python | `p = tifffile.imread(path).astype("float32") / 255` |

Runs with different `--num-workers/--batch-size` give the same map (E5 vs E1: r = 0.99999, max diff 1/255).

## 2. File naming

```
results/<EXPERIMENT>/<segment>/<segment>__<ckpt>__<layers>.tif
```

| Token | Values | Meaning |
|---|---|---|
| `<EXPERIMENT>` | `E1-baseline`, `E4-depth`, `E5-workers`, `E6-bulk`, `E7-adaptive` | experiment ID |
| `<segment>` | `w024` (held out), `w043`, `w035` (both in training data) | PHerc. 0139 segment |
| `<ckpt>` | `ck010k`, `ck040k`, `ck075k` | training step of the checkpoint |
| `<layers>` | `L04-20`, `L06-22` (default), `L08-24` | 17 consecutive surface-volume layers fed to the model (inclusive) |

Split a name with `stem.split("__")` → `[segment, ckpt, layers]` (+ optional suffix, e.g. `w8b128`).

**E6-bulk will produce** (18 files): `w024` and `w043` × `ck010k/ck040k/ck075k` × `L04-20/L06-22/L08-24`.

Legacy names from earlier runs (same content, old naming):

| Old path | Equivalent new name |
|---|---|
| `results/baseline/w043_step-075000.tif` | `E1-baseline/w043/w043__ck075k__L06-22.tif` |
| `results/sweep/w024/par2/w024_step-0X0000.tif` | `E2-2node/w024/w024__ck0X0k__L06-22.tif` |

Other files per experiment folder:

| File | Content |
|---|---|
| `timing_*.json` | one per task: `{experiment, task, segment?, checkpoint?, layers?, start, end, wall_clock_s}` (unix seconds) |
| `gpu_*.csv` | `nvidia-smi` every 5 s: `timestamp, utilization.gpu [%], memory.used [MiB], power.draw [W]` |
| `run_summary.json` | `{experiment, tasks, sum_task_seconds, experiment_wall_clock_s}` |
| `analysis/metrics.json` | see §3 |
| `analysis/disagreement_map.tif` | uint8, 255 = pixel in disagreement (§4.3), 0 otherwise |
| `analysis/uncertainty_std.tif` | uint8, `min(2·std, 1) · 255` (so 255 ⇔ std ≥ 0.5) |
| `analysis/ensemble_mean.tif` | uint8, mean score across configs · 255 |
| `analysis/*.png` | downscaled previews (max 2000 px) for slides only — do not analyse PNGs |

## 3. `metrics.json` format

E2 / E4 (one group of configs):

```json
{
  "configs": ["w024_step-010000", "w024_step-040000", "w024_step-075000"],
  "valid_pixels": 40869376,
  "pct_pixels_disagree": 31.99,
  "mean_score_std": 0.0672,
  "pct_pixels_high_uncertainty_std_gt_0.2": 0.446,
  "per_config_pct_ink": {"w024_step-010000": 39.91, "...": 0},
  "pairwise": {"A vs B": {"agree_pct": 77.33, "ink_iou": 0.508, "corr": 0.696}}
}
```

E6 (planned): one block per segment, each with the same fields computed over three **groups**:

```json
{
  "threshold": 0.5,
  "segments": {
    "w024": {
      "all_9":            { ...same fields as above... },
      "vary_checkpoint":  { ...configs = ck010k/040k/075k at L06-22... },
      "vary_depth":       { ...configs = L04-20/L06-22/L08-24 at ck075k... }
    },
    "w043": { "...": "same" }
  },
  "sensitivity": {"w024": {"all_9": {"pct_pixels_disagree_at": {"0.4": 0, "0.5": 0, "0.6": 0}}}}
}
```

## 4. Metric definitions

Let `p_i(x)` be config *i*'s score at pixel *x*, for *N* configs in a group.

1. **Valid pixels**: `valid(x) = max_i p_i(x) > 0`. Excludes area outside the papyrus. Every percentage below is over valid pixels.
2. **Ink call**: `ink_i(x) = p_i(x) > 0.5` (strictly greater).
3. **Disagreement** (the "32%"): `vote(x) = mean_i ink_i(x)`; a pixel **disagrees** if `0 < vote(x) < 1`, i.e. at least one config calls ink and at least one does not.
   `pct_pixels_disagree = 100 · mean_valid(disagree)`.
4. **Score spread**: `std(x)` = population standard deviation (**ddof = 0**) of `p_1(x)…p_N(x)`.
   `mean_score_std = mean_valid(std)`.
5. **Strongly contested** (the "0.45%"): `std(x) > 0.2`.
   `pct_pixels_high_uncertainty_std_gt_0.2 = 100 · mean_valid(std > 0.2)`.
6. **Per-config ink fraction**: `100 · mean_valid(ink_i)`.
7. **Pairwise**, over valid pixels: `agree_pct = 100 · mean(ink_i == ink_j)`; `ink_iou = |ink_i ∧ ink_j| / |ink_i ∨ ink_j|`; `corr` = Pearson correlation of `p_i`, `p_j`.

Reference implementation (matches the cluster code):

```python
import numpy as np, tifffile
S = np.stack([tifffile.imread(f).astype(np.float32) / 255 for f in files])   # (N, H, W)
valid = S.max(0) > 0
B = S > 0.5
vote = B.mean(0)
disagree = (vote > 0) & (vote < 1)
std = S.std(0)                                                                # ddof=0
pct_disagree = 100 * disagree[valid].mean()
pct_strong = 100 * (std[valid] > 0.2).mean()
```

## 5. Caveats (state these when presenting)

- **Disagreement is not error.** No ground truth on w024. Labels on w043/w035 exist but those segments were in training: use them only as a sanity check.
- **Scores are uncalibrated.** 0.5 is a convention, not a probability boundary; background sits near 0.37. E6 reports disagreement at 0.4 / 0.5 / 0.6 to show sensitivity.
- **Depends on N.** With more configs, the "any disagreement" rate can only rise; compare groups with the same N (all E6 single-factor groups have N = 3).
- **std scale.** For N = 3 scores in [0, 1], std > 0.2 needs a large spread (e.g. 0.2 / 0.5 / 0.7); it is a deliberately strict "strongly contested" bar.
- **Pixel-level, not area-weighted by papyrus layer or text line.** Region-level summaries (tiles, lines) should be named separately, e.g. `pct_tiles_disagree_128px`.
