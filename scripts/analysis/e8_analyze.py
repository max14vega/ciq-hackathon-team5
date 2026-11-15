import json, os, glob
import numpy as np, tifffile
R = os.environ.get("R", "/data/results/E8-pack")
E6 = os.environ.get("E6", "/data/results/E6-bulk")
out = {"experiment": "E8-pack", "workers_per_process": 4, "segments": {}}
for seg in ["w024", "w043"]:
    rows = {}
    for K in [1, 2, 3, 4]:
        d = R + "/" + seg + "/K" + str(K)
        if not os.path.exists(d + "/batch.txt"):
            continue
        _, t0, t1 = open(d + "/batch.txt").read().split()
        wall = float(t1) - float(t0)
        runs = []
        for f in sorted(glob.glob(d + "/done_*.txt")):
            cfg, rc, s, e = open(f).read().split()
            r = {"config": cfg, "rc": int(rc), "run_s": round(float(e) - float(s), 1)}
            p = d + "/" + seg + "__" + cfg + ".tif"
            q = E6 + "/" + seg + "/" + seg + "__" + cfg + ".tif"
            if os.path.exists(p) and os.path.exists(q):
                a = tifffile.imread(p).astype(np.float32).ravel()
                b = tifffile.imread(q).astype(np.float32).ravel()
                r["corr_vs_E6"] = float(np.corrcoef(a, b)[0, 1])
            runs.append(r)
        util, pw = [], []
        for line in open(d + "/gpu.csv"):
            p = [x.strip() for x in line.split(",")]
            try:
                util.append(float(p[1]))
                pw.append(float(p[2]))
            except Exception:
                pass
        rows[K] = {"concurrent_runs": K, "wall_s": round(wall, 1),
                   "runs_per_hour": round(3600 * K / wall, 1),
                   "gpu_util_mean_pct": round(float(np.mean(util)), 1) if util else None,
                   "gpu_power_mean_w": round(float(np.mean(pw)), 1) if pw else None,
                   "runs": runs}
    if 1 in rows:
        for K in rows:
            rows[K]["throughput_vs_K1"] = round(rows[K]["runs_per_hour"] / rows[1]["runs_per_hour"], 2)
    out["segments"][seg] = {str(k): v for k, v in rows.items()}
json.dump(out, open(R + "/metrics.json", "w"), indent=2)
print(json.dumps(out, indent=2))
print("\nSUMMARY  (each process: 4 data workers, batch 128; all runs on one GB10 at once)")
for seg, rows in out["segments"].items():
    print("==", seg, "==")
    print("  K  wall_s  runs/hour  vs_K1  gpu_util%  power_W  min_corr_vs_E6  failed")
    for K, r in rows.items():
        cs = [x.get("corr_vs_E6") for x in r["runs"] if x.get("corr_vs_E6") is not None]
        fl = sum(1 for x in r["runs"] if x["rc"] != 0)
        print("  %s  %6.1f  %9.1f  %5.2f  %9s  %7s  %14s  %d" % (
            K, r["wall_s"], r["runs_per_hour"], r.get("throughput_vs_K1", 0),
            r["gpu_util_mean_pct"], r["gpu_power_mean_w"],
            ("%.5f" % min(cs)) if cs else "-", fl))
print("DONE")
