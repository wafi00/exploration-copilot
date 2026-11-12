import glob, os, io, re
import lasio
import matplotlib
matplotlib.use("Agg")          # draw without opening a window
import matplotlib.pyplot as plt

DATA_DIR = "/data/epic20/APGCE/input_data/challenge1_challenge2"

# Common log names, in the order we like to show them
PREFERRED = ["GR", "RHOB", "NPHI", "DT", "RT", "RESD", "ILD", "PHIE", "SW", "VSH"]


def list_wells():
    """Find every .las file and return {well name: file path}."""
    files = glob.glob(os.path.join(DATA_DIR, "**", "*.las"), recursive=True)
    return {os.path.basename(f)[:-4]: f for f in sorted(files)}


def plot_well(path, max_tracks=5):
    """Draw up to 5 log curves side by side. Returns (png bytes, curve names)."""
    df = lasio.read(path).df()
    cols = [c for c in df.columns if c.upper() in PREFERRED] or list(df.columns)
    cols = cols[:max_tracks]

    fig, axes = plt.subplots(1, len(cols), figsize=(2.3 * len(cols), 8), sharey=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, c in zip(axes, cols):
        ax.plot(df[c], df.index, linewidth=0.6)
        ax.set_title(c)
        ax.grid(alpha=0.3)
    axes[0].invert_yaxis()                 # depth goes down
    axes[0].set_ylabel("Depth")
    fig.suptitle(os.path.basename(path))
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue(), cols

# ================= SEISMIC (SEG-Y) =================
import numpy as np
import segyio
from functools import lru_cache

# Common places (byte numbers) where files store inline / crossline numbers
BYTE_PAIRS = [(189, 193), (9, 21), (5, 21), (9, 17), (17, 13), (221, 225)]


def list_seismic():
    """Find every SEG-Y file and return {file name: path}."""
    files = []
    for ext in ("*.sgy", "*.segy", "*.SGY", "*.SEGY"):
        files += glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True)
    return {os.path.basename(f): f for f in sorted(set(files))}


@lru_cache(maxsize=8)
def _geometry(path):
    """Work out the inline and crossline number of every trace."""
    best, best_score = None, -1
    with segyio.open(path, "r", ignore_geometry=True) as f:
        n = f.tracecount
        for ib, xb in BYTE_PAIRS:
            il = np.asarray(f.attributes(ib)[:])
            xl = np.asarray(f.attributes(xb)[:])
            if len(np.unique(il)) < 2 or len(np.unique(xl)) < 2:
                continue                      # this spot is empty, try the next
            pairs = len(np.unique(il.astype(np.int64) * 10_000_000 + xl))
            if pairs == n:
                return il, xl                 # every trace has a unique (inline, crossline): perfect
            if pairs > best_score:
                best, best_score = (il, xl), pairs
    if best is None:
        raise ValueError("Couldn't find inline/crossline numbers in the trace headers")
    return best


def seismic_inline_range(path):
    """Return (first inline, last inline, step)."""
    il = np.unique(_geometry(path)[0])
    step = int(np.min(np.diff(il))) if len(il) > 1 else 1
    return int(il[0]), int(il[-1]), step


def plot_seismic_inline(path, inline):
    """Draw one inline slice. Returns (png bytes, the inline actually used)."""
    il, xl = _geometry(path)
    all_il = np.unique(il)
    inline = int(all_il[np.abs(all_il - inline).argmin()])     # snap to nearest real inline

    idx = np.where(il == inline)[0]                             # traces on this inline
    idx = idx[np.argsort(xl[idx], kind="stable")]               # order them left to right
    xls, keep = np.unique(xl[idx], return_index=True)           # drop any duplicates
    idx = idx[keep]

    with segyio.open(path, "r", ignore_geometry=True) as f:
        section = np.stack([f.trace[int(i)] for i in idx]).T    # rows = time, columns = crosslines
        t = f.samples

    clip = np.percentile(np.abs(section), 98)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(section, cmap="seismic", vmin=-clip, vmax=clip, aspect="auto",
              extent=[xls[0], xls[-1], t[-1], t[0]])
    ax.set_xlabel("Crossline")
    ax.set_ylabel("Time / depth (as stored in file)")
    ax.set_title(f"{os.path.basename(path)}: inline {inline}")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue(), inline

    # ================= WELLS ON THE SEISMIC =================
import pandas as pd

WELL_LIST = os.path.join(DATA_DIR, "well_data", "Challenge1_Challenge2_WellList_Information.xlsx")


def _find_col(cols, keys):
    """Find a column whose first word matches one of the keys (case-insensitive)."""
    for c in cols:
        first = re.split(r"[\s_(\[]", str(c).strip().lower())[0]
        if first in keys:
            return c
    return None


def read_well_list():
    """Read well name + X + Y from the organisers' Excel file. Returns a table: well, x, y."""
    for h in range(6):                                   # the real header may not be the first row
        df = pd.read_excel(WELL_LIST, header=h)
        name = _find_col(df.columns, {"well", "name", "wellname"})
        x = _find_col(df.columns, {"x", "easting"})
        y = _find_col(df.columns, {"y", "northing"})
        if name is not None and x is not None and y is not None:
            out = df[[name, x, y]].copy()
            out.columns = ["well", "x", "y"]
            out["x"] = pd.to_numeric(out["x"], errors="coerce")
            out["y"] = pd.to_numeric(out["y"], errors="coerce")
            return out.dropna()
    raise ValueError("Couldn't find Well / X / Y columns in the well list")


@lru_cache(maxsize=8)
def _trace_xy(path):
    """Real-world X/Y of every trace, read from the trace headers."""
    with segyio.open(path, "r", ignore_geometry=True) as f:
        scal = np.asarray(f.attributes(71)[:], dtype=float)
        scal = np.where(scal < 0, -1.0 / scal, np.where(scal > 0, scal, 1.0))
        for xb, yb in [(181, 185), (73, 77)]:          # two common places for X/Y
            x = np.asarray(f.attributes(xb)[:], dtype=float)
            y = np.asarray(f.attributes(yb)[:], dtype=float)
            if np.ptp(x) > 0 and np.ptp(y) > 0:
                return x * scal, y * scal
    raise ValueError("No X/Y coordinates found in the trace headers")


def wells_on_seismic(seis_path):
    """For each well, find the nearest trace: returns well, inline, crossline, distance (m)."""
    wl = read_well_list()
    il, xl = _geometry(seis_path)
    tx, ty = _trace_xy(seis_path)
    rows = []
    for _, w in wl.iterrows():
        d = np.hypot(tx - w["x"], ty - w["y"])
        i = int(d.argmin())
        rows.append({"well": str(w["well"]), "inline": int(il[i]),
                     "crossline": int(xl[i]), "dist_m": round(float(d[i]), 1)})
    return pd.DataFrame(rows)

# ================= SECTIONS FOR THE INTERACTIVE VIEWER =================
def crossline_range(path):
    """Return (first crossline, last crossline, step)."""
    xl = np.unique(_geometry(path)[1])
    step = int(np.min(np.diff(xl))) if len(xl) > 1 else 1
    return int(xl[0]), int(xl[-1]), step


def get_section(path, which, number):
    """Get one inline or crossline as numbers.
    Returns (data[time, position], positions along the section, times, number actually used)."""
    il, xl = _geometry(path)
    key, other = (il, xl) if which == "inline" else (xl, il)
    vals = np.unique(key)
    number = int(vals[np.abs(vals - number).argmin()])          # snap to a real line
    idx = np.where(key == number)[0]
    idx = idx[np.argsort(other[idx], kind="stable")]
    pos, keep = np.unique(other[idx], return_index=True)
    idx = idx[keep]
    with segyio.open(path, "r", ignore_geometry=True) as f:
        data = np.stack([f.trace[int(i)] for i in idx]).T
        t = np.asarray(f.samples)
    return data, pos, t, number

    # ================= TIME SLICE (MAP VIEW) =================
@lru_cache(maxsize=8)
def sample_times(path):
    """The list of times (ms) in each trace."""
    with segyio.open(path, "r", ignore_geometry=True) as f:
        return np.asarray(f.samples)


def _ibm2ieee(b):
    """Convert old IBM-format numbers (used by many SEG-Y files) to normal numbers."""
    sign = np.where(b >> 31, -1.0, 1.0)
    expo = ((b >> 24) & 0x7F).astype(np.int32) - 64
    mant = (b & 0x00FFFFFF).astype(np.float64) / 2**24
    return sign * mant * 16.0 ** expo


def get_time_slice(path, time_ms):
    """Read ONE value from EVERY trace at the given time, arranged as a map.
    Returns (grid[inline, crossline], inlines, crosslines, time actually used)."""
    with segyio.open(path, "r", ignore_geometry=True) as f:
        t = np.asarray(f.samples)
        n, ns = f.tracecount, len(t)
        fmt = int(f.format)
        ext = int(f.bin[segyio.BinField.ExtendedHeaders])
    if fmt not in (1, 5):
        raise ValueError(f"Unsupported sample format {fmt}")

    k = int(np.abs(t - time_ms).argmin())                    # nearest time sample
    start = 3600 + 3200 * max(ext, 0)                        # where the traces begin in the file
    trace_len = 240 + ns * 4                                 # header + samples, in bytes
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=start, shape=(n, trace_len))
    raw = np.ascontiguousarray(mm[:, 240 + 4 * k: 244 + 4 * k]).view(">u4").ravel()
    vals = raw.view(">f4").astype(np.float32) if fmt == 5 else _ibm2ieee(raw.astype(np.uint32)).astype(np.float32)

    il, xl = _geometry(path)
    uil, uxl = np.unique(il), np.unique(xl)
    grid = np.full((len(uil), len(uxl)), np.nan, np.float32)   # empty map
    grid[np.searchsorted(uil, il), np.searchsorted(uxl, xl)] = vals
    return grid, uil, uxl, float(t[k])

    # ================= TOP-VIEW MAP (REAL-WORLD X/Y) =================
def _slice_values(path, time_ms):
    """One value per trace at the given time (fast method)."""
    with segyio.open(path, "r", ignore_geometry=True) as f:
        t = np.asarray(f.samples)
        n, ns = f.tracecount, len(t)
        fmt = int(f.format)
        ext = int(f.bin[segyio.BinField.ExtendedHeaders])
    if fmt not in (1, 5):
        raise ValueError(f"Unsupported sample format {fmt}")
    k = int(np.abs(t - time_ms).argmin())
    start = 3600 + 3200 * max(ext, 0)
    trace_len = 240 + ns * 4
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=start, shape=(n, trace_len))
    raw = np.ascontiguousarray(mm[:, 240 + 4 * k: 244 + 4 * k]).view(">u4").ravel()
    vals = raw.view(">f4").astype(np.float32) if fmt == 5 else _ibm2ieee(raw.astype(np.uint32)).astype(np.float32)
    return vals, float(t[k])


def _line_step(v):
    u = np.unique(v)
    return int(np.min(np.diff(u))) if len(u) > 1 else 1


def get_map_xy(path, time_ms, max_points=60000):
    """Every trace's real X/Y + its value, thinned so the browser stays fast."""
    vals, t_used = _slice_values(path, time_ms)
    il, xl = _geometry(path)
    x, y = _trace_xy(path)
    keep = np.ones(len(vals), bool)
    if len(vals) > max_points:
        f = int(np.ceil(np.sqrt(len(vals) / max_points)))
        keep = ((((il - il.min()) // _line_step(il)) % f == 0) &
                (((xl - xl.min()) // _line_step(xl)) % f == 0))
    return {"x": x[keep], "y": y[keep], "v": vals[keep],
            "il": il[keep], "xl": xl[keep], "t": t_used}


def line_xy(path, which, number):
    """Real X/Y points along one inline or crossline (to draw it on the map)."""
    il, xl = _geometry(path)
    x, y = _trace_xy(path)
    key, other = (il, xl) if which == "inline" else (xl, il)
    idx = np.where(key == number)[0]
    idx = idx[np.argsort(other[idx], kind="stable")]
    idx = idx[::max(1, len(idx) // 300)]
    return x[idx], y[idx]


def point_xy(path, inline, crossline):
    """Real X/Y where an inline and crossline cross ('you are here')."""
    il, xl = _geometry(path)
    x, y = _trace_xy(path)
    i = np.where((il == inline) & (xl == crossline))[0]
    j = int(i[0]) if len(i) else int(np.argmin(np.abs(il - inline) + np.abs(xl - crossline)))
    return float(x[j]), float(y[j])


def nearest_lines(path, px, py):
    """Turn a clicked X/Y on the map into the nearest (inline, crossline)."""
    il, xl = _geometry(path)
    x, y = _trace_xy(path)
    i = int(np.argmin((x - px) ** 2 + (y - py) ** 2))
    return int(il[i]), int(xl[i])


# ================= INLINE/CROSSLINE -> REAL X/Y (for the 3D view) =================
@lru_cache(maxsize=8)
def il_xl_to_xy(path):
    """Find the rule: X = a + b*inline + c*crossline (and the same for Y), from all traces."""
    il, xl = _geometry(path)
    x, y = _trace_xy(path)
    A = np.column_stack([np.ones(len(il)), il.astype(float), xl.astype(float)])
    cx = np.linalg.lstsq(A, x, rcond=None)[0]
    cy = np.linalg.lstsq(A, y, rcond=None)[0]
    return tuple(cx), tuple(cy)