# Experiment log

Short history of every experiment, including the ones we replaced, so the repo only carries final versions.

| ID | Question | Result | Workflow |
|---|---|---|---|
| E1 | Does the public model run on Fuzzball + GB10? | Baseline ink map for w043; matches official preview visually | `01-infer-baseline.fz` |
| E2/E3 | Do checkpoints differ, and does a second node help? | 3 checkpoints; 1.48× on 3 jobs (max possible 1.5×) | `02-sweep-checkpoints.fz` |
| E4 | Does CT depth matter? | **Superseded:** used 16 layers by mistake (`--layer-end` is exclusive). Fixed in E6 (17 layers, checked per run). | — |
| E5 | Why is each run slow? | Data-loader workers: 5.9× faster per run (86 s) | inline in early workflows |
| E6 | 3 checkpoints × 3 depths on w024 + w043 | Checkpoint flips 32.0/29.8% of pixels, depth 14.7/12.4% | `03-E6-nine-way-study.fz` |
| Scaling | Same 18 jobs, 1 GPU vs 2 nodes | 1,478 s vs 750 s: 1.97×, 98.5% efficiency, +0.5%/job | `04-scaling-benchmark.fz` |
| E8 | Several jobs per GPU? | K=2: 2.0–2.1× throughput, −13–22% energy/run, identical outputs | `05-E8-gpu-packing.fz` |
| Consensus | Which pixels do all runs agree on? | Unanimous ink 12.3/5.9%; deep conflict ~0.2% (edges excluded; v1 used votes, v2 had edge artifacts) | `06-consensus-map.fz` |
| Labels | Does agreement match annotators (w043)? | Unanimous 92.1% precision vs 80.2% single run, base rate 44.3% | `07-labels-check.fz` |
| Cross-seed | Does a second, independently trained model agree? | Retraining flips 16–20%; 62–64% of unanimous ink kept; 18/18 precision 94.2% | `08-seed43-cross-seed.fz` |
| E7 | Refine only disputed tiles? | **Superseded:** 2-pilot version compared against an unfair baseline (caught in review). | — |
| E7-fair | Same, vs every uniform subset and random tiles | 15–28% fewer errors than best uniform; 3–27% vs random tiles | `09-E7-fair.fz` |
| E7b | Does the replay hold on live GPUs? | Matched exactly; masked runs 60 s vs 79 s; per-run startup limits GPU-time savings | `10-E7b-live.fz` |
| Best map | Official vs single vs averaged runs | Averaging 9 runs: correlation with official map 0.385→0.473 (w043), 0.473→0.553 (w024) | `11-best-map.fz` |
| Host OS | What runs on the nodes? | CIQ kernel 6.18.43-2.1.el9_clk, NVIDIA GB10, driver 580.173.02, CUDA 13.0, on both nodes | `12-host-os.fz` |
| Readability | Can we rank tiles by legibility? | **Experimental:** GPU scoring 16× faster than CPU, but ranking not validated (0.22–0.54) | `experimental/` |

Bugs we caught and fixed: 16-layer depth window (E4), layer-count check matching a timestamp (E6), unfair E7 baseline, edge artifacts in the conflict map, labels stored as sharded Zarr v3 (wrote our own reader).

## Where things live

- `workflows/`: final Fuzzfiles, numbered in run order. Use these to reproduce.
- `configs/`: the early workflows as first run (baseline, E3, E4, E5, E6). Kept for history. `configs/E4-*` is the 16-layer version and `configs/E6-bulk.yaml` predates the layer-check fix; the corrected E6 is `workflows/03-E6-nine-way-study.fz`.
- `experiment_exports/fuzzball/`: outputs exported from the cluster, one folder per experiment.
- `scripts/`: local analysis (`compare_predictions.py` reproduces E6 off-cluster); `scripts/analysis/` holds the code the workflows run.
