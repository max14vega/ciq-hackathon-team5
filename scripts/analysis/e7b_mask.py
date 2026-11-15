import json, os
import numpy as np, tifffile
R = os.environ.get("R", "/data/results/E6-bulk")
O = os.environ.get("O", "/data/results/E7b-live")
T = 256
B = 4
seg = "w024"
CK = ["ck010k", "ck040k", "ck075k"]
LY = ["L04-20", "L06-22", "L08-24"]
PILOT = ["ck075k__L06-22", "ck010k__L06-22"]
names = [c + "__" + l for c in CK for l in LY]
S = [tifffile.imread(R + "/" + seg + "/" + seg + "__" + n + ".tif").squeeze() for n in names]
h, w = S[0].shape
H, W = -(-h // T) * T, -(-w // T) * T
S = np.stack([np.pad(a.astype(np.float32) / 255, ((0, H - h), (0, W - w))) for a in S])
valid = S.max(0) > 0
pi = [names.index(p) for p in PILOT]
d = np.abs(S[pi[0]] - S[pi[1]]) * valid
n = valid.reshape(H // T, T, W // T, T).sum((1, 3))
sc = d.reshape(H // T, T, W // T, T).sum((1, 3)) / np.maximum(n, 1)
usable = n >= 0.1 * T * T
k = int(round((B - 2) / 7.0 * usable.sum()))
idx = np.argsort(-np.where(usable, sc, -1).ravel())[:k]
m = np.zeros(usable.size, bool)
m[idx] = True
m = np.repeat(np.repeat(m.reshape(usable.shape), T, 0), T, 1)[:h, :w]
os.makedirs(O + "/" + seg, exist_ok=True)
tifffile.imwrite(O + "/" + seg + "/mask_B4.tif", (m * 255).astype(np.uint8))
info = {"segment": seg, "budget_full_passes": B, "tile_px": T, "tiles_selected": k,
        "usable_tiles": int(usable.sum()), "mask_pct_of_image": float(100 * m.mean()),
        "mask_pct_of_valid": float(100 * (m & valid[:h, :w]).sum() / valid.sum())}
json.dump(info, open(O + "/" + seg + "/mask_info.json", "w"), indent=2)
print(json.dumps(info, indent=2))
