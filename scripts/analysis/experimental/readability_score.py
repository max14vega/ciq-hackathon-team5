import argparse, csv, json, os, time
import numpy as np, tifffile
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["gpu", "cpu"], required=True)
ap.add_argument("--segs", nargs="+", required=True)
ap.add_argument("--official", action="store_true")
args = ap.parse_args()
R = os.environ.get("R", "/data/results/E6-bulk")
SG = os.environ.get("SG", "/data/segments")
O = os.environ.get("O", "/data/results/readability")
os.makedirs(O, exist_ok=True)
T = 512
CK = ["ck010k", "ck040k", "ck075k"]
LY = ["L04-20", "L06-22", "L08-24"]
names = [c + "__" + l for c in CK for l in LY]
FIELDS = ["segment", "source", "tile_row", "tile_col", "x0", "y0", "papyrus_frac", "ink_frac",
          "decisiveness", "contrast", "edge", "stroke", "agreement"]

if args.backend == "gpu":
    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    KX = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=dev)[None, None]
    KY = KX.transpose(2, 3).contiguous()
    _w = torch.rand(1, 1, 1024, 1024, device=dev)
    F.conv2d(_w, KX, padding=1); F.max_pool2d(_w, 3, stride=1, padding=1); torch.quantile(_w.flatten()[:1000], 0.9)
    if dev == "cuda":
        torch.cuda.synchronize()
else:
    from scipy import ndimage

def sync():
    if args.backend == "gpu" and dev == "cuda":
        torch.cuda.synchronize()

def tsum(a, Hp, Wp):
    if args.backend == "gpu":
        a = F.pad(a, (0, Wp - a.shape[1], 0, Hp - a.shape[0]))
        return a.reshape(Hp // T, T, Wp // T, T).sum((1, 3)).double().cpu().numpy()
    a = np.pad(a, ((0, Hp - a.shape[0]), (0, Wp - a.shape[1])))
    return a.reshape(Hp // T, T, Wp // T, T).sum((1, 3), dtype=np.float64)

def signals(q, v, disagree=None):
    H, W = q.shape
    Hp, Wp = -(-H // T) * T, -(-W // T) * T
    if args.backend == "gpu":
        vf = v.float()
        ink = (q > 0.5).float() * vf
        dec = ((q < 0.4) | (q > 0.6)).float() * vf
        gx = F.conv2d(q[None, None], KX, padding=1)[0, 0]
        gy = F.conv2d(q[None, None], KY, padding=1)[0, 0]
        mag = torch.sqrt(gx * gx + gy * gy)
        ve = (-F.max_pool2d(-vf[None, None], 3, stride=1, padding=1))[0, 0]
        sample = mag[ve > 0][::max(1, int((ve > 0).sum().item()) // 1000000)]
        thr = float(torch.quantile(sample.float(), 0.9).item()) if sample.numel() else 0.0
        edgep = ((mag > thr).float() * ve)
    else:
        vf = v.astype(np.float32)
        ink = (q > 0.5).astype(np.float32) * vf
        dec = ((q < 0.4) | (q > 0.6)).astype(np.float32) * vf
        gx = ndimage.correlate(q, np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], np.float32), mode="constant")
        gy = ndimage.correlate(q, np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], np.float32), mode="constant")
        mag = np.sqrt(gx * gx + gy * gy)
        ve = ndimage.minimum_filter(vf, size=3, mode="nearest")
        sel = mag[ve > 0]
        sample = sel[::max(1, sel.size // 1000000)]
        thr = float(np.quantile(sample, 0.9)) if sample.size else 0.0
        edgep = (mag > thr).astype(np.float32) * ve
    s_v = tsum(vf, Hp, Wp)
    s_ink = tsum(ink, Hp, Wp)
    s_dec = tsum(dec, Hp, Wp)
    s_qi = tsum(q * ink, Hp, Wp)
    s_qn = tsum(q * (vf - ink), Hp, Wp)
    s_mag = tsum(mag * ve, Hp, Wp)
    s_ve = tsum(ve, Hp, Wp)
    s_edge = tsum(edgep, Hp, Wp)
    s_dis = tsum(disagree, Hp, Wp) if disagree is not None else None
    sync()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {"papyrus_frac": s_v / (T * T),
               "ink_frac": np.where(s_v > 0, s_ink / s_v, 0),
               "decisiveness": np.where(s_v > 0, s_dec / s_v, 0),
               "contrast": np.where((s_ink > 0) & (s_v - s_ink > 0), s_qi / np.maximum(s_ink, 1) - s_qn / np.maximum(s_v - s_ink, 1), 0),
               "edge": np.where(s_ve > 0, s_mag / s_ve, 0),
               "stroke": np.where(s_ink >= 0.01 * s_v, s_edge / np.maximum(s_ink, 1), 0)}
        if s_dis is not None:
            out["agreement"] = np.where(s_v > 0, 1 - s_dis / s_v, 0)
    return out, thr

def core_mask(valid_np, f=8, it=8):
    from scipy import ndimage as ndi
    h, w = valid_np.shape
    hh, ww = h // f, w // f
    sm = valid_np[:hh * f, :ww * f].reshape(hh, f, ww, f).all((1, 3))
    sm = ndi.binary_erosion(sm, iterations=it)
    out = np.zeros_like(valid_np)
    out[:hh * f, :ww * f] = np.repeat(np.repeat(sm, f, 0), f, 1)
    return out

def to_dev(a):
    if args.backend == "gpu":
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    return a

def rescale(p, do):
    if args.backend == "gpu":
        return torch.clamp((p - 0.25) / 0.5, 0, 1) * (p > 0).float() if do else p
    return np.clip((p - 0.25) / 0.5, 0, 1) * (p > 0) if do else p

rows, timing = [], {"backend": args.backend, "tile_px": T, "segments": {}}
for seg in args.segs:
    tl = time.time()
    raw = [tifffile.imread(R + "/" + seg + "/" + seg + "__" + n + ".tif").squeeze() for n in names]
    load_s = time.time() - tl
    m = (sum(a.astype(np.float32) for a in raw) / 9.0)
    core_np = core_mask(m > 0)
    pv = m[m > 0][::97] / 255.0
    p01, p50, p99 = [float(x) for x in np.percentile(pv, [1, 50, 99])]
    do_rescale = p01 >= 0.15
    info = {"files": [seg + "__" + n + ".tif" for n in names], "load_s": round(load_s, 3),
            "raw_p01_p50_p99_of_mean9": [round(p01, 3), round(p50, 3), round(p99, 3)],
            "rescale_applied": bool(do_rescale), "rescale": "(p-0.25)/0.5 clipped to [0,1]" if do_rescale else "none",
            "edge_threshold_p90": {}}
    info["core_mask"] = "papyrus eroded by 64 px (8 x 8-px blocks); pixels outside are ignored"
    del m
    core = to_dev(core_np)
    sync()
    tc = time.time()
    acc = None
    vote = None
    valid = None
    for n, a in zip(names, raw):
        p = (to_dev(a).float() / 255.0) if args.backend == "gpu" else a.astype(np.float32) / 255.0
        v = p > 0
        valid = v if valid is None else (valid | v)
        acc = (p.clone() if args.backend == "gpu" else p.copy()) if acc is None else acc + p
        b = (p > 0.5)
        if args.backend == "gpu":
            vote = b.to(torch.uint8) if vote is None else vote + b.to(torch.uint8)
        else:
            vote = b.astype(np.uint8) if vote is None else vote + b.astype(np.uint8)
        s, thr = signals(rescale(p, do_rescale), v & core)
        info["edge_threshold_p90"][n] = thr
        for (r, c), _ in np.ndenumerate(s["papyrus_frac"]):
            rows.append([seg, n, r, c, c * T, r * T] + [float(s[k][r, c]) for k in FIELDS[6:12]] + [""])
    mean9 = acc / 9.0
    if args.backend == "gpu":
        dis = ((vote > 0) & (vote < 9)).float() * valid.float()
    else:
        dis = ((vote > 0) & (vote < 9)).astype(np.float32) * valid
    q9 = rescale(mean9, do_rescale)
    s, thr = signals(q9, valid & core, dis * (core.float() if args.backend == "gpu" else core))
    sync()
    info["compute_s"] = round(time.time() - tc, 3)
    info["edge_threshold_p90"]["mean9"] = thr
    for (r, c), _ in np.ndenumerate(s["papyrus_frac"]):
        rows.append([seg, "mean9", r, c, c * T, r * T] + [float(s[k][r, c]) for k in FIELDS[6:13]])
    q9n = q9.cpu().numpy() if args.backend == "gpu" else q9
    if args.official:
        tifffile.imwrite(O + "/" + seg + "_mean9_q.tif", (np.clip(q9n, 0, 1) * 255).astype(np.uint8))
        import glob
        cands = [f for f in glob.glob(SG + "/" + seg + "/ink-detection/**/*", recursive=True)
                 if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))]
        if cands:
            f = max(cands, key=os.path.getsize)
            o = np.squeeze(tifffile.imread(f)) if f.lower().endswith((".tif", ".tiff")) else np.array(Image.open(f).convert("L"))
            if o.ndim == 3:
                o = o[0] if o.shape[0] < o.shape[-1] else o[..., 0]
            o = o.astype(np.float32)
            o = o / max(float(o.max()), 1e-6)
            h, w = q9n.shape
            og = np.array(Image.fromarray((o * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR))
            del o
            tifffile.imwrite(O + "/" + seg + "_official_on_our_grid.tif", og)
            oq = to_dev(og).float() / 255.0 if args.backend == "gpu" else og.astype(np.float32) / 255.0
            so, _ = signals(oq, valid & core)
            for (r, c), _ in np.ndenumerate(so["papyrus_frac"]):
                rows.append([seg, "official", r, c, c * T, r * T] + [float(so[k][r, c]) for k in FIELDS[6:12]] + [""])
            info["official_file"] = f
    timing["segments"][seg] = info
    del raw
tag = args.backend + "_" + "_".join(args.segs)
with open(O + "/scores_" + tag + ".csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(FIELDS)
    wr.writerows(rows)
timing["device"] = (dev if args.backend == "gpu" else "cpu")
json.dump(timing, open(O + "/timing_" + tag + ".json", "w"), indent=2)
print(json.dumps(timing, indent=2))
print("rows", len(rows))
