import csv, glob, json, os
import numpy as np, tifffile
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
O = os.environ.get("O", "/data/results/readability")
T = 512
W = {"contrast": 0.25, "decisiveness": 0.25, "edge": 0.2, "stroke": 0.15, "ink_band": 0.15, "agreement": 0.2}
BAND = (0.03, 0.40)
MINPAP = 0.9
CTX = 256
SIG = ["papyrus_frac", "ink_frac", "decisiveness", "contrast", "edge", "stroke", "agreement"]

def load(pattern):
    rows = []
    for f in sorted(glob.glob(O + "/" + pattern)):
        for r in csv.DictReader(open(f)):
            for k in SIG:
                r[k] = float(r[k]) if r[k] != "" else None
            for k in ["tile_row", "tile_col", "x0", "y0"]:
                r[k] = int(r[k])
            rows.append(r)
    return rows

def key(r):
    return (r["segment"], r["source"], r["tile_row"], r["tile_col"])

def pct(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.ones_like(x)
    return np.argsort(np.argsort(x, kind="stable"), kind="stable") / (len(x) - 1)

def readability(rows, use_agreement):
    s = {k: pct([r[k] for r in rows]) for k in ["contrast", "decisiveness", "edge", "stroke"]}
    s["ink_band"] = np.array([1.0 if BAND[0] <= r["ink_frac"] <= BAND[1] else 0.0 for r in rows])
    ws = {k: W[k] for k in ["contrast", "decisiveness", "edge", "stroke", "ink_band"]}
    if use_agreement:
        s["agreement"] = pct([r["agreement"] for r in rows])
        ws["agreement"] = W["agreement"]
    tot = sum(ws.values())
    return sum(ws[k] * s[k] for k in ws) / tot

def spearman(a, b):
    if len(a) < 3:
        return None
    return float(np.corrcoef(pct(a), pct(b))[0, 1])

gpu = load("scores_gpu_*.csv")
cpu = load("scores_cpu_*.csv")
cmap = {key(r): r for r in cpu}
diff = {k: 0.0 for k in SIG}
matched = 0
for r in gpu:
    c = cmap.get(key(r))
    if c is None:
        continue
    matched += 1
    for k in SIG:
        if r[k] is not None and c[k] is not None:
            diff[k] = max(diff[k], abs(r[k] - c[k]))

main = [r for r in gpu if r["source"] == "mean9" and r["papyrus_frac"] >= MINPAP]
score = readability(main, True)
for r, s in zip(main, score):
    r["readability"] = float(s)
main.sort(key=lambda r: -r["readability"])
with open(O + "/ranking_tiles.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["rank", "readability", "segment", "source", "tile_row", "tile_col", "x0", "y0"] + SIG)
    for i, r in enumerate(main, 1):
        wr.writerow([i, round(r["readability"], 4), r["segment"], "mean9 (9 E6 configs)", r["tile_row"], r["tile_col"], r["x0"], r["y0"]] + [round(r[k], 5) for k in SIG])

segs = sorted({r["segment"] for r in main})
pos = {(r["segment"], r["tile_row"], r["tile_col"]): r for r in main}
cons, files, val = {}, [], {}
sources = sorted({r["source"] for r in gpu if r["source"] not in ("mean9", "official")})
for src in sources + ["mean9"]:
    rs = [r for r in gpu if r["source"] == src and (r["segment"], r["tile_row"], r["tile_col"]) in pos]
    if not rs:
        continue
    sc = readability(rs, False)
    for seg in segs:
        sel = [(r, s) for r, s in zip(rs, sc) if r["segment"] == seg]
        if not sel:
            continue
        v = sorted((s for _, s in sel), reverse=True)
        top = v[:max(1, len(v) // 4)]
        files.append([seg, src, float(np.mean(top)), len(v)])
        if src != "mean9":
            a = [s for _, s in sel]
            b = [pos[(seg, r["tile_row"], r["tile_col"])]["readability"] for r, _ in sel]
            cons.setdefault(seg, {})[src] = spearman(a, b)
off = [r for r in gpu if r["source"] == "official" and (r["segment"], r["tile_row"], r["tile_col"]) in pos]
if off:
    so = readability(off, False)
    for seg in segs:
        sel = [(r, s) for r, s in zip(off, so) if r["segment"] == seg]
        a = [s for _, s in sel]
        b = [pos[(seg, r["tile_row"], r["tile_col"])]["readability"] for r, _ in sel]
        val[seg] = {"tiles": len(sel), "spearman_ours_vs_official_readability": spearman(a, b)}
files.sort(key=lambda x: -x[2])
with open(O + "/ranking_files.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["rank", "file_score_top25pct_tiles", "segment", "source", "n_tiles"])
    for i, f in enumerate(files, 1):
        wr.writerow([i, round(f[2], 4), f[0], f[1], f[3]])

lab = [r for r in main if BAND[0] <= r["ink_frac"] <= BAND[1]]
lab.sort(key=lambda r: r["agreement"])
with open(O + "/ranking_label_first.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["rank", "agreement", "segment", "tile_row", "tile_col", "x0", "y0", "readability"])
    for i, r in enumerate(lab, 1):
        wr.writerow([i, round(r["agreement"], 4), r["segment"], r["tile_row"], r["tile_col"], r["x0"], r["y0"], round(r["readability"], 4)])

imgs, offs, lims = {}, {}, {}
for seg in segs:
    p = O + "/" + seg + "_mean9_q.tif"
    if os.path.exists(p):
        a = tifffile.imread(p).astype(np.float32) / 255
        v = a[a > 0]
        lims[seg] = np.percentile(v, [50, 99.5]) if v.size else (0, 1)
        imgs[seg] = a
    p = O + "/" + seg + "_official_on_our_grid.tif"
    if os.path.exists(p):
        offs[seg] = tifffile.imread(p).astype(np.float32) / 255

def crop(seg, r, which):
    src = imgs if which == "ours" else offs
    if seg not in src:
        return Image.new("RGB", (240, 240), (40, 40, 40))
    full = src[seg]
    H, Wd = full.shape
    y0, x0 = max(0, r["y0"] - CTX), max(0, r["x0"] - CTX)
    y1, x1 = min(H, r["y0"] + T + CTX), min(Wd, r["x0"] + T + CTX)
    a = np.zeros((T + 2 * CTX, T + 2 * CTX), np.float32)
    oy, ox = y0 - (r["y0"] - CTX), x0 - (r["x0"] - CTX)
    a[oy:oy + (y1 - y0), ox:ox + (x1 - x0)] = full[y0:y1, x0:x1]
    if which == "ours":
        lo, hi = lims[seg]
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8)).resize((240, 240), Image.LANCZOS).convert("RGB")
    k = 240 / (T + 2 * CTX)
    ImageDraw.Draw(im).rectangle([CTX * k, CTX * k, (CTX + T) * k, (CTX + T) * k], outline=(0, 200, 90), width=1)
    return im

def gallery(rs, path, title, label):
    cols = 4
    cw, chh = 490, 270
    rows_n = -(-len(rs) // cols) if rs else 1
    cv = Image.new("RGB", (cols * cw, rows_n * chh + 40), (0, 0, 0))
    d = ImageDraw.Draw(cv)
    d.text((10, 12), title, fill=(255, 255, 255))
    for i, r in enumerate(rs):
        x, y = (i % cols) * cw, 40 + (i // cols) * chh
        cv.paste(crop(r["segment"], r, "ours"), (x + 5, y + 26))
        cv.paste(crop(r["segment"], r, "official"), (x + 248, y + 26))
        d.text((x + 6, y + 6), label(i, r), fill=(255, 255, 255))
    cv.save(path)

lbl = lambda i, r: "#%d %s x=%d y=%d R=%.2f  (ours | official)" % (i + 1, r["segment"], r["x0"], r["y0"], r["readability"])
gallery(main[:20], O + "/gallery_top20.png", "Most readable tiles (green box = 512 px tile, 4.8 mm; shown with 256 px context). Ours = average of 9 configs", lbl)
gallery(main[-5:][::-1], O + "/gallery_bottom5.png", "Least readable tiles", lambda i, r: "#%d %s x=%d y=%d R=%.2f" % (len(main) - i, r["segment"], r["x0"], r["y0"], r["readability"]))
gallery(lab[:12], O + "/gallery_label_first.png", "Label first: ink present but the 9 configs disagree most",
        lambda i, r: "#%d %s x=%d y=%d agree=%.2f" % (i + 1, r["segment"], r["x0"], r["y0"], r["agreement"]))

for seg in segs:
    if seg not in imgs:
        continue
    lo, hi = lims[seg]
    a = np.clip((imgs[seg] - lo) / max(hi - lo, 1e-6), 0, 1)
    sc = 1600 / a.shape[1]
    im = Image.fromarray((a * 255).astype(np.uint8)).resize((1600, int(a.shape[0] * sc)), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(im)
    for i, r in enumerate([r for r in lab if r["segment"] == seg][:10]):
        d.rectangle([r["x0"] * sc, r["y0"] * sc, (r["x0"] + T) * sc, (r["y0"] + T) * sc], outline=(255, 140, 0), width=3)
    for i, r in enumerate(main[:20]):
        if r["segment"] != seg:
            continue
        d.rectangle([r["x0"] * sc, r["y0"] * sc, (r["x0"] + T) * sc, (r["y0"] + T) * sc], outline=(0, 230, 90), width=4)
        d.text((r["x0"] * sc + 4, r["y0"] * sc + 4), "#%d" % (i + 1), fill=(0, 255, 100))
    im.save(O + "/readability_map_" + seg + ".png")

tg = [json.load(open(f)) for f in glob.glob(O + "/timing_gpu_*.json")]
tc = [json.load(open(f)) for f in glob.glob(O + "/timing_cpu_*.json")]
def tot(ts, k):
    return sum(s[k] for t in ts for s in t["segments"].values())
util, pw = [], []
for f in glob.glob(O + "/gpu_*.csv"):
    for line in open(f):
        p = [x.strip() for x in line.split(",")]
        try:
            util.append(float(p[1]))
            pw.append(float(p[2]))
        except Exception:
            pass
g_c, c_c = tot(tg, "compute_s"), tot(tc, "compute_s")
g_l, c_l = tot(tg, "load_s"), tot(tc, "load_s")
per_task = [sum(s["load_s"] + s["compute_s"] for s in t["segments"].values()) for t in tg]
bench = {"segments": segs, "files_scored": 9 * len(segs), "sources_per_segment": "9 configs + mean9",
         "tiles_ranked": len(main), "tile_px": T,
         "gpu_device": [t.get("device") for t in tg],
         "gpu_compute_s": round(g_c, 3), "cpu_compute_s": round(c_c, 3),
         "compute_speedup": round(c_c / g_c, 1) if g_c else None,
         "gpu_load_s": round(g_l, 3), "cpu_load_s": round(c_l, 3),
         "end_to_end_gpu_s": round(g_l + g_c, 3), "end_to_end_cpu_s": round(c_l + c_c, 3),
         "end_to_end_speedup": round((c_l + c_c) / (g_l + g_c), 2) if (g_l + g_c) else None,
         "two_node_wall_s": round(max(per_task), 3) if per_task else None,
         "gpu_util_mean_pct": round(float(np.mean(util)), 1) if util else None,
         "gpu_power_mean_w": round(float(np.mean(pw)), 1) if pw else None,
         "gpu_vs_cpu_rows_matched": matched,
         "max_abs_diff_gpu_vs_cpu": {k: float(v) for k, v in diff.items()}}
json.dump(bench, open(O + "/benchmark.json", "w"), indent=2)
cons_sum = {seg: {"mean": float(np.mean([v for v in d.values() if v is not None])),
                  "min": float(np.min([v for v in d.values() if v is not None]))} for seg, d in cons.items()}
summ = {"tile_px": T, "min_papyrus_frac": MINPAP, "weights": W, "ink_band": BAND,
        "rescale": {seg: t["segments"][seg]["rescale_applied"] for t in tg for seg in t["segments"]},
        "raw_percentiles_mean9": {seg: t["segments"][seg]["raw_p01_p50_p99_of_mean9"] for t in tg for seg in t["segments"]},
        "tiles_ranked": len(main), "consistency_config_vs_mean9_spearman": cons_sum,
        "consistency_per_config": cons, "validation_vs_official": val,
        "top10": [{"rank": i + 1, "segment": r["segment"], "x0": r["x0"], "y0": r["y0"], "readability": round(r["readability"], 3)} for i, r in enumerate(main[:10])],
        "label_first_top5": [{"segment": r["segment"], "x0": r["x0"], "y0": r["y0"], "agreement": round(r["agreement"], 3)} for r in lab[:5]]}
json.dump(summ, open(O + "/rank_summary.json", "w"), indent=2)
print(json.dumps(bench, indent=2))
print(json.dumps(summ, indent=2))
print("\nSUMMARY")
print(" ranked %d tiles (%d px) across %s" % (len(main), T, ", ".join(segs)))
print(" GPU compute %.2f s vs CPU %.2f s -> %sx; end-to-end %sx; two-node wall %s s" % (g_c, c_c, bench["compute_speedup"], bench["end_to_end_speedup"], bench["two_node_wall_s"]))
print(" GPU vs CPU max diff:", {k: round(v, 6) for k, v in diff.items()})
print(" consistency (configs vs mean9, Spearman):", {s: round(v["mean"], 3) for s, v in cons_sum.items()})
print(" validation vs official (Spearman):", {s: (round(v["spearman_ours_vs_official_readability"], 3) if v["spearman_ours_vs_official_readability"] is not None else None) for s, v in val.items()})
print(" top 5:", [(r["segment"], r["x0"], r["y0"], round(r["readability"], 2)) for r in main[:5]])
print("wrote", sorted(f for f in os.listdir(O) if f.endswith((".csv", ".png", ".json")) and not f.startswith(("scores_", "timing_", "gpu_"))))
print("DONE")
