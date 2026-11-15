# E7 / E7b / E7-fair — Uncertainty-guided compute allocation: methods and validity

> **Update (E7-fair, 3:00 pm):** the original E7 uniform baseline chose extra configurations at random. A teammate review found that a sensibly chosen uniform (covering all three checkpoints) beat the 2-pilot adaptive. E7-fair (`analysis/e7_fair.py`) re-tests against **every possible uniform combination** and uses a stronger adaptive with a 3-checkpoint base. The E7-fair result (end of this file) supersedes the original E7 claim.

Script: `analysis/e7_adaptive.py` (E7, offline), `analysis/e7b_mask.py` + `analysis/e7b_compare.py` (E7b, live GPU runs), `analysis/e7_figure.py` (figure).
Inputs: the 18 E6 prediction TIFFs (`results/E6-bulk/<seg>/<seg>__<ckpt>__<layers>.tif`), uint8, value/255 = ink probability, 0 = outside papyrus.

## Question
For a fixed GPU budget, can a cheap uncertainty signal decide where extra inference is spent, so the result is closer to the expensive 9-configuration ensemble than spending the same budget evenly?

## Definitions
- **Configurations (9):** checkpoints {10k, 40k, 75k} × depth windows {L04-20, L06-22, L08-24}, all 17 layers.
- **Reference:** per-pixel mean of all 9 configurations; ink call = mean > 0.5. This is the expensive answer we try to approximate. **It is not ground truth** (w024 has no labels).
- **valid:** pixel is non-zero in any configuration (papyrus footprint; identical across runs).
- **Contested pixel:** std across the 9 configurations > 0.2.
- **Wrong call:** strategy's ink call (≥0.5 threshold on its mean) differs from the reference call.
- **Budget B:** measured in full-segment passes (1 pass = one configuration over the whole segment). B ∈ {3, 4, 5, 6}.

## Strategies (all at budget B)
| Strategy | What runs | Cost (passes) |
|---|---|---|
| Uniform | 2 pilot configs + (B−2) extra configs, everywhere. Extra configs drawn at random (3 seeds, averaged). | B |
| Adaptive | 2 pilot configs everywhere; the remaining 7 configs on the top-k tiles ranked by **pilot disagreement**. | 2 + 7·f, with f = (B−2)/7 → B |
| Random tiles (control) | same as adaptive, but the k tiles chosen at random (3 seeds). | B |

- **Pilots:** `ck075k__L06-22` and `ck010k__L06-22` (final vs earliest checkpoint at default depth).
- **Tiles:** 256 × 256 px; a tile is usable if ≥10% of it is papyrus. k = round(f × usable tiles).
- **Tile score (adaptive):** mean |pilot₁ − pilot₂| over valid pixels in the tile.

## Is any future information used? (leakage check)
- **Tile selection uses only the two pilot runs**, which in a real deployment are run first. It does **not** use the 9-config std, the reference, or labels. ✅
- `valid` / usable tiles come from the papyrus footprint, which is identical in every run (available from the pilots alone). ✅
- **Design-level caveat:** the pilot pair was chosen because E2/E6 showed checkpoints matter more than depth — a choice informed by earlier runs on the same segment. A deployment would fix the pilot pair in advance.
- **By construction**, refined tiles reproduce the reference exactly (they run all 9 configs). This advantage is shared by the random-tiles control, so **adaptive vs random isolates the value of choosing tiles by uncertainty.**
- An earlier "oracle" column ranked tiles by tile-average 9-config std; it is **not** a true upper bound (averaging dilutes sparse contested pixels) and is not used in claims.

## Results (offline, E7) — wrong calls vs reference, w024 · w043
| B | Area refined | Uniform | Adaptive | Contested covered: random → adaptive |
|---|---|---|---|---|
| 3 | 14% | 5.7 · 4.7% | 5.6 · 4.8% (tie) | 14 → 34% · 16 → 34% |
| 4 | 29% | 5.3 · 4.4% | 4.4 · 3.8% | 30 → 44% · 27 → 57% |
| 5 | 43% | 5.2 · 4.4% | 3.3 · 2.9% | 45 → 57% · 41 → 65% |
| 6 | 57% | 3.5 · 2.9% | 2.3 · 2.0% | 61 → 73% · 58 → 75% |

## Live validation (E7b, w024, B = 4)
- Mask = the exact E7 selection (29.7% of papyrus), written as a TIFF and passed to `infer.py --mask-path`; `infer.py` skips blocks with no mask foreground.
- The 7 non-pilot configs were re-run on GPUs with the mask (both nodes).
- **Quality reproduced exactly from real outputs:** 4.38% wrong calls overall, 29.6% on contested pixels (identical to the E7 simulation). Masked outputs match the full runs inside the mask (r ≥ 0.99993; small differences only at mask edges).
- **Cost:** masked runs ~60 s vs ~79 s full → **23.5% GPU time saved** for 70% less area. ~52 s per run is fixed overhead (model load, scan open, full-size output write).
- Measured totals: adaptive 580 GPU-s vs uniform-4 ≈ 316 GPU-s vs all-9 710 GPU-s.

## What we can and cannot claim
- ✅ **Validated:** uncertainty-guided tile selection, using only information available at inference time, gets closer to the full ensemble than uniform or random allocation at equal *area* budget — and this holds on real GPU outputs, not just in simulation.
- ✅ **Validated:** the GPU work inside `infer.py` scales with the masked area.
- ❌ **Not validated:** wall-clock/GPU-time savings at segment scale. With today's per-run overhead, adaptive costs more GPU-seconds than uniform-4. The equal-budget comparison is in *area-passes*, a cost model that holds only once per-run overhead is amortized (E8 packing; larger regions; loading the scan once per job).
- ❌ Not accuracy: the reference is the 9-config ensemble, not ground truth.

**Strongest honest one-liner:** *Uncertainty-guided allocation is a validated way to approximate the full ensemble with less computed area; turning that into wall-clock savings requires amortizing per-run overhead, which E8 shows is feasible.*

## E7-fair: fair baseline and stronger adaptive (supersedes the E7 claim)
Script: `analysis/e7_fair.py`. Same reference, tiles, and wrong-call metric as above.

- **Uniform, all subsets:** every combination of B configurations out of 9 (84–126 subsets), reporting mean / median / best / worst. "Best" is chosen with hindsight against the reference, so it is a strong (unfair-to-adaptive) baseline.
- **Uniform, stratified:** only subsets that include all three checkpoints.
- **Adaptive (3-base):** run `ck010k`, `ck040k`, `ck075k` at L06-22 everywhere (3 passes). Rank 256-px tiles by the mean per-pixel std of those three runs. Refine the top fraction f = (B−3)/6 of tiles with all 9 configurations. Uses only the 3 base runs → no future information.
- **Random tiles (control):** same 3-run base, same number of refined tiles, chosen at random (3 seeds).

| B | Uniform best (hindsight) | Uniform mean | Uniform stratified mean | Adaptive 3-base (area refined) | Random tiles, 3-base | Original E7 (2-pilot) |
|---|---|---|---|---|---|---|
| 3 | 2.83 · 2.48 | 6.35 · 5.50 | 3.58 · 3.12 | 2.83 · 2.52 (0%) | 2.83 · 2.52 | 5.57 · 4.75 |
| 4 | 2.70 · 2.30 | 5.12 · 4.38 | 4.15 · 3.54 | **2.27 · 1.96** (17%) | 2.33 · 2.12 | 4.38 · 3.78 |
| 5 | 2.35 · 1.98 | 4.09 · 3.48 | 3.73 · 3.17 | **1.70 · 1.42** (33%) | 1.87 · 1.71 | 3.31 · 2.93 |
| 6 | 1.50 · 1.26 | 3.21 · 2.71 | 3.12 · 2.63 | **1.15 · 0.93** (50%) | 1.39 · 1.28 | 2.33 · 1.97 |

Wrong calls vs the 9-config reference (%), w024 · w043.

Wrong calls on strongly contested pixels, adaptive 3-base vs random tiles: B4 2.1 vs 2.5 · 1.9 vs 4.1; B5 1.6 vs 2.1 · 1.1 vs 3.4; B6 0.9 vs 1.5 · 0.7 vs 2.6. Contested pixels covered by refined tiles at B4: 25 vs 16% · 38 vs 20%.

**Findings**
1. At B ≥ 4, on both segments, adaptive 3-base beats every uniform choice, including the best one picked with hindsight: 15–28% fewer wrong calls.
2. Most of the gain over average uniform comes from a **stratified base** (one run per checkpoint). Uncertainty-guided tile choice adds a smaller overall gain over random tiles (3–27%), but a large one on contested pixels (16–73%). Those are the pixels a scholar would care about.
3. The original 2-pilot E7 does **not** beat a sensibly chosen uniform; its earlier "beats uniform" claim held only against random configuration choice. It is retained only as the configuration used in the E7b live test.
4. E7-fair is an offline replay on real E6 outputs. E7b showed the replay matches live masked GPU runs exactly for the 2-pilot version. The 3-base version has not been run live. The cost caveat (per-run overhead) applies unchanged.
