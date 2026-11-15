import glob, json, os
import numpy as np, tifffile
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
R = os.environ.get("R", "/data/results")
SG = os.environ.get("SG", "/data/segments")
O = R + "/best"
CK = ["ck010k", "ck040k", "ck075k"]
LY = ["L04-20", "L06-22", "L08-24"]
P = 1200
out = {}

def stretch(a, valid):
    lo, hi = np.percentile(a[valid], [50, 99.5])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * valid

def rd(p):
    return tifffile.imread(p).squeeze().astype(np.float32) / 255

for seg in ["w043", "w024"]:
    std_run = rd(R + "/E6-bulk/" + seg + "/" + seg + "__ck075k__L06-22.tif")
    best = rd(O + "/" + seg + "/" + seg + "__ck075k__L06-22__tta.tif")
    S = np.stack([rd(R + "/E6-bulk/" + seg + "/" + seg + "__" + c + "__" + l + ".tif") for c in CK for l in LY])
    mean9 = S.mean(0)
    del S
    valid = (std_run > 0) | (best > 0)
    h, w = std_run.shape
    off = np.zeros_like(std_run)
    cands = [p for p in glob.glob(SG + "/" + seg + "/ink-detection/**/*", recursive=True)
             if p.lower().endswith((".tif", ".tiff", ".png", ".jpg"))]
    info = {}
    if cands:
        f = max(cands, key=os.path.getsize)
        a = np.squeeze(tifffile.imread(f)) if f.lower().endswith((".tif", ".tiff")) else np.array(Image.open(f).convert("L"))
        if a.ndim == 3:
            a = a[0] if a.shape[0] < a.shape[-1] else a[..., 0]
        a = a.astype(np.float32)
        a = a / a.max() if a.max() > 0 else a
        off = np.array(Image.fromarray((a * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255
        m = valid & (off > 0)
        for k, v in [("standard", std_run), ("tta_mirror", best), ("mean9", mean9)]:
            info["corr_" + k + "_vs_official"] = float(np.corrcoef(v[m], off[m])[0, 1])
    sj = R + "/showcase/showcase_summary.json"
    crop = None
    if os.path.exists(sj):
        crop = json.load(open(sj)).get(seg, {}).get("crop_px")
    if not crop:
        crop = [h // 3, w // 4, h // 3 + 1000, w // 4 + 1600]
    y0, x0, y1, x1 = crop
    info["crop_px"] = crop
    panels = [("official reference", off),
              ("ours: standard run (75k, layers 6-22)", stretch(std_run, valid)),
              ("ours: best quality (same + mirror TTA)", stretch(best, valid)),
              ("ours: average of 9 configurations", stretch(mean9, valid))]
    for tag, sl in [("full", (slice(None), slice(None))), ("crop", (slice(y0, y1), slice(x0, x1)))]:
        ims = []
        for t, a in panels:
            im = Image.fromarray((np.clip(a[sl], 0, 1) * 255).astype(np.uint8))
            im = im.resize((P, max(1, int(im.size[1] * P / im.size[0]))), Image.LANCZOS)
            ims.append((t, im))
        hh = max(i.size[1] for _, i in ims)
        cv = Image.new("L", (2 * P, 2 * (hh + 36)), 0)
        d = ImageDraw.Draw(cv)
        for k, (t, im) in enumerate(ims):
            x, y = (k % 2) * P, (k // 2) * (hh + 36)
            cv.paste(im, (x, y + 36))
            d.text((x + 10, y + 10), t, fill=255)
        cv.save(O + "/best_" + tag + "_" + seg + ".png")
    out[seg] = info
json.dump(out, open(O + "/best_summary.json", "w"), indent=2)
print(json.dumps(out, indent=2))
print("wrote", sorted(f for f in os.listdir(O) if f.endswith((".png", ".json"))))
