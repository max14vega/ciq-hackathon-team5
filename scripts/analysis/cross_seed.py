import glob, json, os, itertools
import numpy as np, tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
Image.MAX_IMAGE_PIXELS = None
R42 = os.environ.get("R42", "/data/results/E6-bulk")
R43 = os.environ.get("R43", "/data/results/seed43")
SG = os.environ.get("SG", "/data/segments")
O = os.environ.get("O", "/data/results/cross-seed")
os.makedirs(O, exist_ok=True)
CK = ["ck010k", "ck040k", "ck075k"]
LY = ["L04-20", "L06-22", "L08-24"]
CFG = [c + "__" + l for c in CK for l in LY]
CROPS = {"w024": [2800, 600, 4500, 4800], "w043": [3300, 1200, 5000, 5400]}
import zarr
import numcodecs

def _decode(buf, codecs, cs, dt):
    endian = "<"
    transpose = None
    for c in reversed(codecs):
        n = c["name"]
        cfg = c.get("configuration", {})
        if n == "blosc":
            buf = numcodecs.Blosc().decode(buf)
        elif n == "zstd":
            buf = numcodecs.Zstd().decode(buf)
        elif n in ("gzip", "zlib"):
            buf = numcodecs.GZip().decode(buf) if n == "gzip" else numcodecs.Zlib().decode(buf)
        elif n == "crc32c":
            buf = bytes(buf)[:-4]
        elif n == "bytes":
            endian = ">" if cfg.get("endian") == "big" else "<"
        elif n == "transpose":
            transpose = cfg.get("order")
    arr = np.frombuffer(bytes(buf), dtype=dt.newbyteorder(endian)).reshape(cs)
    if transpose:
        arr = arr.transpose(np.argsort(transpose))
    return arr

def v3_read(apath):
    meta = json.load(open(os.path.join(apath, "zarr.json")))
    shape = meta["shape"]
    cs = meta["chunk_grid"]["configuration"]["chunk_shape"]
    dt = np.dtype(meta["data_type"])
    codecs = meta.get("codecs", [])
    enc = meta.get("chunk_key_encoding", {"name": "default", "configuration": {"separator": "/"}})
    sep = enc.get("configuration", {}).get("separator", "/")
    prefix = "c" + sep if enc.get("name", "default") == "default" else ""
    fill = meta.get("fill_value", 0) or 0
    out = np.full(shape, fill, dtype=dt)
    shard = codecs[0] if codecs and codecs[0]["name"] == "sharding_indexed" else None
    nchunks = [-(-shape[i] // cs[i]) for i in range(len(shape))]
    for idx in np.ndindex(*nchunks):
        key = prefix + sep.join(str(i) for i in idx)
        fp = os.path.join(apath, *key.split("/"))
        if not os.path.exists(fp):
            continue
        raw = open(fp, "rb").read()
        base = [i * c for i, c in zip(idx, cs)]
        if shard is None:
            pieces = [(base, cs, _decode(raw, codecs, cs, dt))]
        else:
            cfg = shard["configuration"]
            ics = cfg["chunk_shape"]
            per = [cs[k] // ics[k] for k in range(len(cs))]
            n = int(np.prod(per))
            ic = cfg.get("index_codecs", [{"name": "bytes"}])
            extra = 4 if any(c["name"] == "crc32c" for c in ic) else 0
            isz = n * 16 + extra
            ib = raw[-isz:] if cfg.get("index_location", "end") == "end" else raw[:isz]
            index = np.frombuffer(ib[:n * 16], dtype="<u8").reshape(per + [2])
            pieces = []
            for j in np.ndindex(*per):
                off, nb = index[j]
                if off == 2 ** 64 - 1 or nb == 2 ** 64 - 1:
                    continue
                arr = _decode(raw[int(off):int(off) + int(nb)], cfg["codecs"], ics, dt)
                pieces.append(([base[k] + j[k] * ics[k] for k in range(len(cs))], ics, arr))
        for b0, shp, arr in pieces:
            sl = tuple(slice(b0[k], min(b0[k] + shp[k], shape[k])) for k in range(len(shape)))
            if any(x.start >= x.stop for x in sl):
                continue
            out[sl] = arr[tuple(slice(0, x.stop - x.start) for x in sl)]
    return out

def list_nodes(path):
    nodes = []
    for root, dirs, files in os.walk(path):
        if "zarr.json" in files:
            m = json.load(open(os.path.join(root, "zarr.json")))
            if m.get("node_type") == "array":
                nodes.append(("v3", os.path.relpath(root, path), tuple(m["shape"]), m["data_type"], root))
        if ".zarray" in files:
            m = json.load(open(os.path.join(root, ".zarray")))
            nodes.append(("v2", os.path.relpath(root, path), tuple(m["shape"]), m["dtype"], root))
    return nodes

def load_zarr_2d(path, target_hw):
    nodes = list_nodes(path)
    desc = [(n[0], n[1], n[2], n[3]) for n in nodes]
    print("arrays in", path, ":", desc)
    if not nodes:
        raise RuntimeError("no zarr arrays found under " + path)
    def hw(sh):
        sh = [d for d in sh if d > 1]
        return (sh[-2], sh[-1]) if len(sh) >= 2 else (0, 0)
    big = [n for n in nodes if hw(n[2])[0] >= 0.9 * target_hw[0]]
    pick = min(big, key=lambda n: hw(n[2])[0] * hw(n[2])[1]) if big else max(nodes, key=lambda n: hw(n[2])[0] * hw(n[2])[1])
    x = v3_read(pick[4]) if pick[0] == "v3" else np.asarray(zarr.open(pick[4], mode="r")[...])
    x = np.squeeze(x)
    while x.ndim > 2:
        x = x[0] if x.shape[0] < x.shape[-1] else x[..., 0]
    return x, pick[0] + ":" + pick[1], desc


def load(root, seg):
    return [tifffile.imread(root + "/" + seg + "/" + seg + "__" + c + ".tif").squeeze() for c in CFG]

def pct_dis(a, b, valid):
    return float(100 * (((a > 127) != (b > 127)) & valid).sum() / valid.sum())

def core_of(valid):
    h, w = valid.shape
    f = 8
    sm = valid[:h // f * f, :w // f * f].reshape(h // f, f, w // f, f).all((1, 3))
    sm = ndi.binary_erosion(sm, iterations=16)
    c = np.zeros_like(valid)
    c[:h // f * f, :w // f * f] = np.repeat(np.repeat(sm, f, 0), f, 1)
    return c

out = {"experiment": "cross-seed consensus: seed42 x 9 configs + seed43 x 9 configs (independent training runs)", "segments": {}}
for seg in ["w024", "w043", "w035"]:
    if not os.path.exists(R43 + "/" + seg) or not os.path.exists(R42 + "/" + seg):
        print("skip", seg)
        continue
    A = load(R42, seg)
    B = load(R43, seg)
    valid = np.zeros(A[0].shape, bool)
    for x in A + B:
        valid |= x > 0
    v42 = sum((x > 127).astype(np.uint8) for x in A)
    v43 = sum((x > 127).astype(np.uint8) for x in B)
    m42 = sum(x.astype(np.float32) for x in A) / 9 / 255
    m43 = sum(x.astype(np.float32) for x in B) / 9 / 255
    s1 = sum(x.astype(np.float32) / 255 for x in A + B)
    s2 = sum((x.astype(np.float32) / 255) ** 2 for x in A + B)
    m18 = s1 / 18
    std18 = np.sqrt(np.maximum(s2 / 18 - m18 ** 2, 0))
    del s1, s2
    core = core_of(valid)
    nv = valid.sum()
    u42, u43 = valid & (v42 == 9), valid & (v43 == 9)
    u18 = u42 & u43
    r = {"papyrus_pixels": int(nv),
         "unanimous_ink_seed42_pct": round(float(100 * u42.sum() / nv), 2),
         "unanimous_ink_seed43_pct": round(float(100 * u43.sum() / nv), 2),
         "unanimous_ink_both_18_of_18_pct": round(float(100 * u18.sum() / nv), 2),
         "seed42_unanimous_also_unanimous_in_seed43_pct": round(float(100 * u18.sum() / max(u42.sum(), 1)), 1),
         "ink_any_single_run_avg_pct": round(float(np.mean([100 * ((x > 127) & valid).sum() / nv for x in A + B])), 2),
         "deep_conflict_interior_18_pct": round(float(100 * (valid & core & (std18 > 0.2)).sum() / nv), 3)}
    same = [pct_dis(a, b, valid) for a, b in zip(A, B)]
    ckp = [pct_dis(A[CFG.index(c1 + "__" + l)], A[CFG.index(c2 + "__" + l)], valid) for l in LY for c1, c2 in itertools.combinations(CK, 2)]
    dep = [pct_dis(A[CFG.index(c + "__" + l1)], A[CFG.index(c + "__" + l2)], valid) for c in CK for l1, l2 in itertools.combinations(LY, 2)]
    r["pairwise_disagreement_pct"] = {
        "different_seed_same_ckpt_and_depth": round(float(np.mean(same)), 2),
        "different_checkpoint_same_seed": round(float(np.mean(ckp)), 2),
        "different_depth_same_seed": round(float(np.mean(dep)), 2),
        "different_seed_final_ckpt_default_depth": round(same[CFG.index("ck075k__L06-22")], 2)}
    lab_dirs = sorted(glob.glob(SG + "/" + seg + "/ink-labels/*/*/"))
    if lab_dirs:
        try:
            base = lab_dirs[-1]
            h, w = valid.shape
            lab, _, _ = load_zarr_2d(base + "inklabels.zarr", (h, w))
            sup, _, _ = load_zarr_2d(base + "supervision.zarr", (h, w))
            def grid(x):
                x8 = ((x.astype(np.float32) / max(float(x.max()), 1)) * 255).astype(np.uint8)
                return np.array(Image.fromarray(x8).resize((w, h), Image.NEAREST)) > 127
            L, S = grid(lab), grid(sup)
            sv = S & valid
            Lv = L & sv
            def pr(pred):
                p = pred & sv
                return {"precision_pct": round(float(100 * (p & Lv).sum() / max(p.sum(), 1)), 1),
                        "recall_pct": round(float(100 * (p & Lv).sum() / max(Lv.sum(), 1)), 1)}
            r["vs_labels"] = {"supervised_pct_of_papyrus": round(float(100 * sv.sum() / nv), 2),
                              "single_run_seed42_75k": pr(A[CFG.index("ck075k__L06-22")] > 127),
                              "single_run_seed43_75k": pr(B[CFG.index("ck075k__L06-22")] > 127),
                              "average_of_18": pr(m18 > 0.5),
                              "unanimous_seed42_9of9": pr(u42),
                              "unanimous_seed43_9of9": pr(u43),
                              "unanimous_both_18of18": pr(u18),
                              "agreed_background_18of18_contains_labeled_ink_pct": round(float(100 * (Lv & (v42 == 0) & (v43 == 0)).sum() / max(Lv.sum(), 1)), 1)}
            ys, xs = np.nonzero(S)
            if len(ys):
                CROPS.setdefault(seg, [max(0, ys.min() - 150), max(0, xs.min() - 150), min(h, ys.max() + 150), min(w, xs.max() + 150)])
        except Exception as e:
            r["vs_labels_error"] = str(e)
    rgb = np.zeros(valid.shape + (3,), np.uint8)
    rgb[valid & ~u18 & ~((v42 == 0) & (v43 == 0))] = (90, 60, 20)
    rgb[u18] = (255, 255, 255)
    rgb[ndi.binary_dilation(valid & core & (std18 > 0.2), iterations=4) & core] = (255, 30, 30)
    def save(arr, path, width, title):
        im = Image.fromarray(arr)
        im = im.resize((width, max(1, int(im.size[1] * width / im.size[0]))), Image.LANCZOS)
        cv = Image.new("RGB", (im.size[0], im.size[1] + 40), (0, 0, 0))
        cv.paste(im, (0, 40))
        ImageDraw.Draw(cv).text((10, 14), title, fill=(255, 255, 255))
        cv.save(path)
    t = seg + ": white = all 18 agree ink (2 independent models x 9 settings); brown = not unanimous; red = deep conflict (interior)"
    save(rgb, O + "/cross_seed_consensus_" + seg + ".png", 1800, t)
    if seg in CROPS:
        y0, x0, y1, x1 = CROPS[seg]
        save(rgb[y0:y1, x0:x1], O + "/cross_seed_consensus_crop_" + seg + ".png", 1800, t)
    out["segments"][seg] = r
    del A, B
json.dump(out, open(O + "/cross_seed_metrics.json", "w"), indent=2)
print(json.dumps(out, indent=2))
print("\nSUMMARY")
for seg, r in out["segments"].items():
    p = r["pairwise_disagreement_pct"]
    print("%s: unanimous ink seed42 %.1f%% | seed43 %.1f%% | both %.1f%% (%.0f%% of seed42-unanimous kept) | disagree: seed %.1f%% ckpt %.1f%% depth %.1f%%" % (
        seg, r["unanimous_ink_seed42_pct"], r["unanimous_ink_seed43_pct"], r["unanimous_ink_both_18_of_18_pct"],
        r["seed42_unanimous_also_unanimous_in_seed43_pct"], p["different_seed_same_ckpt_and_depth"],
        p["different_checkpoint_same_seed"], p["different_depth_same_seed"]))
    if "vs_labels" in r:
        L = r["vs_labels"]
        print("   labels: single42 %s | unanimous42 %s | unanimous BOTH %s" % (L["single_run_seed42_75k"], L["unanimous_seed42_9of9"], L["unanimous_both_18of18"]))
print("DONE")
