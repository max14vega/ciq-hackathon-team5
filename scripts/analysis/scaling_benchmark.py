import glob, json, os
import numpy as np, tifffile
B = "/data/results/scaling"
E6 = "/data/results/E6-bulk"
def load(phase):
    return [json.load(open(f)) for f in glob.glob(B + "/" + phase + "/*/timing_*.json")]
def gpu(phase):
    u, p = [], []
    for f in glob.glob(B + "/" + phase + "/*/gpu_*.csv"):
        for line in open(f):
            x = [q.strip() for q in line.split(",")]
            try:
                u.append(float(x[1])); p.append(float(x[2]))
            except Exception:
                pass
    return (round(float(np.mean(u)), 1) if u else None, round(float(np.mean(p)), 1) if p else None)
out = {"experiment": "scaling benchmark: same 18 E6 jobs, 1 GPU at a time vs 2 nodes", "phases": {}}
for ph, gpus in [("one", 1), ("two", 2)]:
    t = load(ph)
    if not t:
        continue
    wall = max(x["end"] for x in t) - min(x["start"] for x in t)
    s = sum(x["wall_clock_s"] for x in t)
    ok = sum(1 for x in t if x["rc"] == 0)
    u, p = gpu(ph)
    out["phases"][ph] = {"gpus_used_at_once": gpus, "jobs_completed": ok, "jobs_total": len(t),
                         "wall_clock_s": wall, "sum_job_s": s,
                         "mean_job_s": round(s / len(t), 1), "min_job_s": min(x["wall_clock_s"] for x in t),
                         "max_job_s": max(x["wall_clock_s"] for x in t),
                         "throughput_jobs_per_min": round(60 * ok / wall, 3) if wall else None,
                         "scheduling_overhead_s": round(wall - s / gpus, 1),
                         "gpu_util_mean_pct": u, "gpu_power_mean_w": p,
                         "per_job_s": {x["name"]: x["wall_clock_s"] for x in sorted(t, key=lambda z: z["name"])}}
if "one" in out["phases"] and "two" in out["phases"]:
    a, b = out["phases"]["one"], out["phases"]["two"]
    sp = a["wall_clock_s"] / b["wall_clock_s"]
    out["measured_speedup"] = round(sp, 3)
    out["throughput_improvement"] = round(b["throughput_jobs_per_min"] / a["throughput_jobs_per_min"], 3)
    out["scaling_efficiency_pct"] = round(100 * sp / 2, 1)
    out["mean_job_slowdown_on_two_nodes_pct"] = round(100 * (b["mean_job_s"] / a["mean_job_s"] - 1), 1)
chk = {}
for f in sorted(glob.glob(B + "/two/*/w0*__*.tif"))[:2]:
    n = os.path.basename(f)[:-4]
    seg = n[:4]
    x = tifffile.imread(f).astype(np.float32).ravel()
    y = tifffile.imread(E6 + "/" + seg + "/" + n + ".tif").astype(np.float32).ravel()
    chk[n] = float(np.corrcoef(x, y)[0, 1])
out["outputs_match_E6_corr"] = chk
json.dump(out, open(B + "/scaling_benchmark.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "phases"}, indent=2))
for ph, v in out["phases"].items():
    print(ph, {k: v[k] for k in v if k != "per_job_s"})
if "measured_speedup" in out:
    print("\nSUMMARY: 1 GPU at a time %ds | 2 nodes %ds | speedup %.2fx | efficiency %.1f%% | throughput %.2f -> %.2f jobs/min" % (
        out["phases"]["one"]["wall_clock_s"], out["phases"]["two"]["wall_clock_s"], out["measured_speedup"],
        out["scaling_efficiency_pct"], out["phases"]["one"]["throughput_jobs_per_min"], out["phases"]["two"]["throughput_jobs_per_min"]))
print("DONE")
