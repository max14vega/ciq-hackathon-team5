import json, os, glob
import numpy as np, tifffile
R = os.environ.get("R", "/data/results/E6-bulk")
O = os.environ.get("O", "/data/results/E7b-live")
seg = "w024"
CK = ["ck010k", "ck040k", "ck075k"]
LY = ["L04-20", "L06-22", "L08-24"]
PILOT = ["ck075k__L06-22", "ck010k__L06-22"]
names = [c + "__" + l for c in CK for l in LY]
REST = [n for n in names if n not in PILOT]
d = O + "/" + seg
m = tifffile.imread(d + "/mask_B4.tif").squeeze() > 0
full = {n: tifffile.imread(R + "/" + seg + "/" + seg + "__" + n + ".tif").squeeze().astype(np.float32) / 255 for n in names}
S = np.stack([full[n] for n in names])
valid = S.max(0) > 0
ref = S.mean(0)
refB = ref > 0.5
hard = valid & (S.std(0) > 0.2)
masked = {}
chk = {}
for n in REST:
    a = tifffile.imread(d + "/" + seg + "__" + n + "__masked.tif").squeeze().astype(np.float32) / 255
    masked[n] = a
    inside = m & valid
    chk[n] = {"corr_inside_mask": float(np.corrcoef(a[inside], full[n][inside])[0, 1]),
              "max_abs_diff_inside_0_255": float(255 * np.abs(a[inside] - full[n][inside]).max()),
              "pct_nonzero_outside_mask": float(100 * (a[~m] > 0).mean())}
pilot = (full[PILOT[0]] + full[PILOT[1]]) / 2
nine = (full[PILOT[0]] + full[PILOT[1]] + sum(masked[n] for n in REST)) / 9
est = np.where(m, nine, pilot)
wrong = (est > 0.5) != refB
quality = {"call_err_pct": float(100 * wrong[valid].mean()),
           "call_err_on_contested_pct": float(100 * wrong[hard].mean())}
e6 = {}
for f in glob.glob(R + "/" + seg + "/timing_*.json"):
    x = json.load(open(f))
    e6[x["name"].split("__", 1)[1]] = x["wall_clock_s"]
tm = [json.load(open(f)) for f in glob.glob(d + "/timing_*.json")]
mk = {x["config"]: x["wall_clock_s"] for x in tm}
full_rest = sum(e6[n] for n in REST)
masked_rest = sum(mk[n] for n in REST)
pilot_s = sum(e6[n] for n in PILOT)
mean_full = sum(e6.values()) / len(e6)
adaptive_s = pilot_s + masked_rest
uniform4_s = 4 * mean_full
all9_s = sum(e6[n] for n in names)
wall = max(x["end"] for x in tm) - min(x["start"] for x in tm)
cost = {"masked_runs_gpu_s": masked_rest, "same_7_runs_full_gpu_s": full_rest,
        "gpu_time_saved_on_7_runs_pct": round(100 * (1 - masked_rest / full_rest), 1),
        "per_run_masked_s": mk, "per_run_full_s": {n: e6[n] for n in REST},
        "adaptive_total_gpu_s": adaptive_s, "uniform_4_pass_gpu_s": round(uniform4_s, 1),
        "full_9_config_gpu_s": all9_s,
        "adaptive_vs_full9_saved_pct": round(100 * (1 - adaptive_s / all9_s), 1),
        "adaptive_vs_uniform4_ratio": round(adaptive_s / uniform4_s, 3),
        "masked_phase_wall_clock_s_2_nodes": wall}
info = json.load(open(d + "/mask_info.json"))
out = {"experiment": "E7b-live", "mask": info, "cost_measured": cost,
       "quality_vs_full_9": quality, "masked_output_check": chk}
json.dump(out, open(d + "/metrics.json", "w"), indent=2)
print(json.dumps(out, indent=2))
print("\nSUMMARY")
print(" mask covers %.1f%% of the papyrus" % info["mask_pct_of_valid"])
print(" 7 masked runs: %d GPU-s vs %d GPU-s full  -> %.1f%% saved" % (masked_rest, full_rest, cost["gpu_time_saved_on_7_runs_pct"]))
print(" adaptive total %d GPU-s | uniform 4-pass %.0f | full 9-config %d" % (adaptive_s, uniform4_s, all9_s))
print(" wrong calls vs full result: overall %.2f%%, contested %.2f%%" % (quality["call_err_pct"], quality["call_err_on_contested_pct"]))
print(" masked output matches full run inside mask: min corr %.5f" % min(v["corr_inside_mask"] for v in chk.values()))
print("DONE")
