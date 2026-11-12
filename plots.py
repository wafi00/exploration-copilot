import glob, os, io
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