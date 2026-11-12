"""Upload your own seismic + ask the AI about what's in it (traps, bright spots...)."""
import io, os, re, json, glob, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

from plots import list_seismic

# Where uploaded volumes are saved (stays on the HPC, never goes to GitHub)
UPLOAD_DIR = "/data/epic20/APGCE/apgce_team_01/uploads"


# ================= 1. UPLOADED VOLUMES =================
def save_upload(uploaded_file):
    """Save a file from st.file_uploader into UPLOAD_DIR. Returns its path."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, os.path.basename(uploaded_file.name))
    uploaded_file.seek(0)
    with open(path, "wb") as out:
        shutil.copyfileobj(uploaded_file, out, length=16 * 1024 * 1024)
    return path


def list_all_seismic(extra_paths=()):
    """Organisers' volumes + uploaded volumes + any file paths typed in by the user."""
    vols = dict(list_seismic())
    if os.path.isdir(UPLOAD_DIR):
        for ext in ("*.sgy", "*.segy", "*.SGY", "*.SEGY"):
            for f in sorted(glob.glob(os.path.join(UPLOAD_DIR, ext))):
                vols["📤 " + os.path.basename(f)] = f
    for p in extra_paths:
        if os.path.isfile(p):
            vols["📁 " + os.path.basename(p)] = p
    return vols


# ================= 2. AUTO-SCAN FOR BRIGHT SPOTS =================
def scan_bright_spots(slice_pack, sensitivity=2.0, min_cells=12, top=8):
    """Find patches on the time slice where the amplitude is much stronger than normal reflections.
    sensitivity = how many times stronger than a typical strong reflection (lower = finds more).
    Bright spots can be a sign of gas, but can also be other things: always check them."""
    grid, uil, uxl, t_used = slice_pack
    valid = ~np.isnan(grid)
    if valid.sum() < 50:
        return []
    a = np.abs(np.nan_to_num(grid))
    typical = float(np.percentile(a[valid], 90)) or 1e-9      # a typical strong reflection
    lab, n = ndimage.label((a >= sensitivity * typical) & valid)
    spots = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        ii, jj = np.where(lab[sl] == k)
        if len(ii) < min_cells:
            continue
        ii, jj = ii + sl[0].start, jj + sl[1].start
        spots.append({
            "inline": int(uil[int(round(ii.mean()))]), "crossline": int(uxl[int(round(jj.mean()))]),
            "il_from": int(uil[ii.min()]), "il_to": int(uil[ii.max()]),
            "xl_from": int(uxl[jj.min()]), "xl_to": int(uxl[jj.max()]),
            "cells": int(len(ii)), "strength": round(float(a[ii, jj].mean() / typical), 1),
            "time": t_used,
        })
    spots.sort(key=lambda s: s["cells"] * s["strength"], reverse=True)
    spots = spots[:top]
    for i, s in enumerate(spots, start=1):
        s["id"] = f"B{i}"
    return spots


# ================= 3. ONE PICTURE FOR GEMMA =================
def views_png(il_pack, xl_pack, slice_pack, spots=()):
    """Draw map + inline + crossline side by side as ONE labelled picture for the AI."""
    grid, uil, uxl, t_used = slice_pack
    il_data, il_pos, ts, il_used = il_pack
    xl_data, xl_pos, _, xl_used = xl_pack

    fig, ax = plt.subplots(1, 3, figsize=(18, 6.5), gridspec_kw={"width_ratios": [1, 1.4, 1.4]})

    c = np.nanpercentile(np.abs(grid), 98) or 1.0
    ax[0].imshow(grid, cmap="gray", vmin=-c, vmax=c, aspect="auto", origin="lower",
                 extent=[uxl[0], uxl[-1], uil[0], uil[-1]])
    ax[0].axhline(il_used, color="deepskyblue", lw=1.5)
    ax[0].axvline(xl_used, color="cyan", lw=1.5)
    for s in spots:
        ax[0].plot(s["crossline"], s["inline"], "o", mfc="none", mec="magenta", ms=12, mew=2)
        ax[0].text(s["crossline"], s["inline"], " " + s["id"], color="magenta", fontsize=10)
    ax[0].set_title(f"A: MAP (time slice at {t_used:.0f} ms)")
    ax[0].set_xlabel("Crossline"); ax[0].set_ylabel("Inline")

    for k, (data, pos, title, xlab) in enumerate([
            (il_data, il_pos, f"B: INLINE {il_used}", "Crossline"),
            (xl_data, xl_pos, f"C: CROSSLINE {xl_used}", "Inline")], start=1):
        c = np.percentile(np.abs(data), 98) or 1.0
        ax[k].imshow(data, cmap="gray", vmin=-c, vmax=c, aspect="auto",
                     extent=[pos[0], pos[-1], ts[-1], ts[0]])
        ax[k].axhline(t_used, color="yellow", lw=1)
        ax[k].set_title(title); ax[k].set_xlabel(xlab); ax[k].set_ylabel("TWT (ms)")
        ax[k].grid(alpha=0.25)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return buf.getvalue()


# ================= 4. WHAT WE ASK GEMMA =================
SYSTEM = (
    "You are an experienced seismic interpreter helping a geoscience team. "
    "You have no tools: never output function calls. Be honest about uncertainty: "
    "you are looking at a picture, so your interpretation is a first pass that a geoscientist must confirm."
)


def build_prompt(question, il_used, xl_used, t_used, spots, history_text=""):
    spot_lines = "\n".join(
        f"- {s['id']}: inline {s['inline']}, crossline {s['crossline']} "
        f"(inlines {s['il_from']}-{s['il_to']}, crosslines {s['xl_from']}-{s['xl_to']}), "
        f"{s['strength']}x stronger than typical reflections"
        for s in spots) or "- none found on this time slice"
    return f"""The image has three panels of the same seismic volume:
A = MAP: time slice at {t_used:.0f} ms (x axis = crossline, y axis = inline). Blue line = inline {il_used}, cyan line = crossline {xl_used}. Magenta circles = bright spots.
B = INLINE {il_used}: vertical section (x axis = crossline, y axis = two-way time in ms, deeper = lower).
C = CROSSLINE {xl_used}: vertical section (x axis = inline, y axis = two-way time in ms).
The yellow line on B and C marks {t_used:.0f} ms.

An automatic amplitude scan of the time slice found these bright spots (possible gas indicators, but could also be lithology or noise):
{spot_lines}
{history_text}
QUESTION: {question}

Answer in clear plain text. Possible trap types to consider: anticline / four-way closure, fault trap, stratigraphic pinch-out, unconformity trap, channel, reef / mound, salt or shale diapir.
Use the axis numbers to say WHERE things are (e.g. "on panel B around crosslines 2150-2250, 1700-1850 ms").
Only describe features you can actually see. If nothing is clear, say so.
Keep the answer short: at most about 200 words.

At the very end, add one line that starts with CANDIDATES: followed by a JSON list of the features you identified on panel B or C, using this format:
CANDIDATES: [{{"panel": "inline", "type": "anticline", "from": 2150, "to": 2250, "t_from": 1700, "t_to": 1850, "confidence": "medium", "reason": "reflectors dome upward"}}]
"panel" is "inline" for panel B or "crossline" for panel C; "from"/"to" are x-axis numbers, "t_from"/"t_to" are times in ms; "confidence" is low, medium or high.
If you identified nothing, write: CANDIDATES: []"""


def split_candidates(text, xr_inline=None, xr_crossline=None, t_range=None):
    """Separate Gemma's answer from its CANDIDATES list. Bad or impossible entries are dropped."""
    m = re.search(r"CANDIDATES:\s*(\[.*\])", text, re.S)
    answer = text[:m.start()].strip() if m else text.replace("CANDIDATES:", "").strip()
    cands = []
    if m:
        try:
            raw = json.loads(m.group(1))
        except Exception:
            raw = []
        for c in raw if isinstance(raw, list) else []:
            try:
                panel = str(c.get("panel", "")).lower()
                panel = "inline" if panel.startswith(("inline", "b")) else "crossline" if panel.startswith(("cross", "c")) else None
                a, b = sorted([float(c["from"]), float(c["to"])])
                t1, t2 = sorted([float(c["t_from"]), float(c["t_to"])])
            except Exception:
                continue
            if panel is None:
                continue
            xr = xr_inline if panel == "inline" else xr_crossline
            if xr is not None and (b < xr[0] or a > xr[1]):
                continue                                   # outside the section: ignore
            if t_range is not None and (t2 < t_range[0] or t1 > t_range[1]):
                continue
            cands.append({"panel": panel, "type": str(c.get("type", "feature"))[:40],
                          "from": a, "to": b, "t_from": t1, "t_to": t2,
                          "confidence": str(c.get("confidence", "low")).lower(),
                          "reason": str(c.get("reason", ""))[:200]})
    for i, c in enumerate(cands, start=1):
        c["id"] = f"T{i}"
    return answer, cands

    # ================= 5. CROPPED READING (faster + focused) =================
import segyio
from plots import _geometry, _trace_xy, _ibm2ieee


def crop_slice(path, time_ms, il_rng, xl_rng, max_points=40000):
    """Time slice for ONLY the cropped area. Reads just the traces inside the crop.
    Returns (slice_pack for the grid / 3D / AI, points for the top view or None)."""
    with segyio.open(path, "r", ignore_geometry=True) as f:
        t = np.asarray(f.samples)
        n, ns = f.tracecount, len(t)
        fmt = int(f.format)
        ext = int(f.bin[segyio.BinField.ExtendedHeaders])
    if fmt not in (1, 5):
        raise ValueError(f"Unsupported sample format {fmt}")
    k = int(np.abs(t - time_ms).argmin())

    il, xl = _geometry(path)
    sel = np.where((il >= il_rng[0]) & (il <= il_rng[1]) & (xl >= xl_rng[0]) & (xl <= xl_rng[1]))[0]
    if len(sel) == 0:
        raise ValueError("The crop area has no traces in it")

    start = 3600 + 3200 * max(ext, 0)
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=start, shape=(n, 240 + ns * 4))
    raw = np.ascontiguousarray(mm[sel, 240 + 4 * k: 244 + 4 * k]).view(">u4").ravel()
    vals = raw.view(">f4").astype(np.float32) if fmt == 5 else _ibm2ieee(raw.astype(np.uint32)).astype(np.float32)

    il_s, xl_s = il[sel], xl[sel]
    uil, uxl = np.unique(il_s), np.unique(xl_s)
    grid = np.full((len(uil), len(uxl)), np.nan, np.float32)
    grid[np.searchsorted(uil, il_s), np.searchsorted(uxl, xl_s)] = vals
    slice_pack = (grid, uil, uxl, float(t[k]))

    try:
        x, y = _trace_xy(path)
        x, y = x[sel], y[sel]
    except Exception:
        return slice_pack, None                      # no real-world X/Y in this volume

    keep = np.ones(len(sel), bool)
    if len(sel) > max_points:                        # thin out so the browser stays smooth
        f = int(np.ceil(np.sqrt(len(sel) / max_points)))
        s_il = int(np.min(np.diff(uil))) if len(uil) > 1 else 1
        s_xl = int(np.min(np.diff(uxl))) if len(uxl) > 1 else 1
        keep = ((((il_s - uil[0]) // s_il) % f == 0) & (((xl_s - uxl[0]) // s_xl) % f == 0))
    pts = {"x": x[keep], "y": y[keep], "v": vals[keep], "il": il_s[keep], "xl": xl_s[keep], "t": float(t[k])}
    return slice_pack, pts


def crop_section(pack, rng, t_rng):
    """Cut a section down to the crop: positions inside rng, times inside t_rng."""
    data, pos, ts, num = pack
    pm = (pos >= rng[0]) & (pos <= rng[1])
    tm = (ts >= t_rng[0]) & (ts <= t_rng[1])
    if pm.sum() < 2 or tm.sum() < 2:
        return pack
    return data[np.ix_(tm, pm)], pos[pm], ts[tm], num