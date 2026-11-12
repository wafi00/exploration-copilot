"""
Exploration Copilot: data pipeline
APGCE 2026 GeoHackathon, Challenge 1

What this file does, in order (each step is one section below):

  STEP 1  Seismic geometry   read inline/crossline headers of every volume, with a
                             separate lookup per volume (volumes do not hold the same traces)
  STEP 2  Wells              well list, checkshots, deviation surveys, log zones (gas/oil flags)
  STEP 3  Well positions     where each well actually is at its reservoir (deviated wells)
  STEP 4  Depth to time      checkshots + a pattern-based well tie (log sand vs seismic sand)
  STEP 5  Horizon            auto-track the top of the reservoir around each well
  STEP 6  Trap tests         structural closure, hydrocarbon anomaly, edge-versus-contour test
  STEP 7  QC and trust       how much to trust each verdict, and what a geoscientist should check
  STEP 8  Evidence + AI      numbered evidence for Gemma, answers with images, debate mode
  STEP 9  Leads              untested look-alikes at the same level as each discovery

Usage in a notebook on the HPC:

    from copilot_pipeline import Copilot
    cp = Copilot()
    cp.prepare()            # steps 1-4  (about 2-3 minutes)
    cp.analyse_all()        # steps 5-7  (about 5-10 minutes)
    cp.summary()            # one table: verdict + trust for every well
    print(cp.ask("s-1", "Is the trap at s-1 structural or stratigraphic? Why?"))
    print(cp.debate("m-1"))
    cp.find_all_leads()     # step 9     (about 15-20 minutes)

Nothing in this file contains data. It reads the competition data on the HPC only.
"""

import os, base64
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save figures to files; show them with cp.show()
import matplotlib.pyplot as plt
import segyio, lasio, openai
from segyio import TraceField as TF
from scipy import ndimage
from scipy.interpolate import interp1d

# ----------------------------------------------------------------------------- settings
BASE = "/data/epic20/APGCE/input_data/challenge1_challenge2"
OUT = "/data/epic20/APGCE/apgce_team_01/copilot"
VOL_FILES = {
    "full": "stack/Challenge1_Challenge2_Full_Stack_cropped.sgy",
    "near": "stack/Challenge1_Challenge2_Near_Stack_cropped.sgy",
    "far":  "stack/Challenge1_Challenge2_Far_Stack_cropped.sgy",
    "sand": "inversion/Challenge1_Challenge2_Inversion_TotalSandProbability_cropped.sgy",
    "pimp": "inversion/Challenge1_Challenge2_Inversion_P_Impedance_cropped.sgy",
    "vpvs": "inversion/Challenge1_Challenge2_Inversion_VpVs_cropped.sgy",
}
GEMMA_URL = os.environ.get("GEMMA_URL", "http://<vllm-host>:8000/v1")
GEMMA_MODEL = "google/gemma-4-26B-A4B-it"

# Rules of thumb, set and reviewed by our geoscientists
SAND_CUT = 0.5          # sand probability above this counts as sand
CLOSE_MS = 10           # at least this much closure (ms) counts as a structural closure
FLAT_STD_MS = 6         # anomaly edge varying less than this (ms) counts as a flat level (fluid contact)
MIN_TIE_CORR = 0.3      # a well tie needs at least this log-seismic correlation to be applied
HALF_LINES = 120        # analysis box half-size in lines (120 x 25 m = 3 km each way)
HALF_T = 120            # analysis window half-size in ms
MIN_LEAD_KM2 = 0.10     # smallest lead worth reporting
WELL_BUFFER_M = 500     # a lead must be at least this far from any well


# ============================================================================= helpers
def zones_from_flag(d, flag, min_thick=1.0):
    """Continuous depth intervals where a gas/oil flag is on, with average log values."""
    if flag not in d:
        return []
    on = d[flag].fillna(0) > 0.5
    found = []
    for _, run in on[on].groupby((~on).cumsum()):
        top, bot = float(run.index.min()), float(run.index.max())
        if bot - top < min_thick:
            continue
        z = d.loc[top:bot]
        found.append({"top_m": round(top, 1), "base_m": round(bot, 1), "thick_m": round(bot - top, 1),
                      "VSH": round(float(z["VSH"].mean()), 2) if "VSH" in z else np.nan,
                      "PHIT": round(float(z["PHIT"].mean()), 2) if "PHIT" in z else np.nan,
                      "SWT": round(float(z["SWT"].mean()), 2) if "SWT" in z else np.nan})
    return found


def window_mean(v, t, a, b):
    m = (t >= a) & (t <= b)
    return float(np.nanmean(v[m])) if m.any() else np.nan


def autotrack(cube, wi, wj, t_axis, t_seed, search=2, min_frac=0.05):
    """Follow one reflector outward from the well, trace by trace (like DUG's autotracker).
    Seed: the strongest event within 12 ms of the gas top at the well.
    Each step: look 8 ms up/down from the neighbour's pick, keep the same polarity,
    stop if the event becomes weaker than min_frac of the seed amplitude."""
    n_i, n_j, n_t = cube.shape
    k0 = int(np.argmin(np.abs(t_axis - t_seed)))
    lo = max(k0 - 3, 0)
    k_seed = lo + int(np.nanargmax(np.abs(cube[wi, wj, lo:k0 + 4])))
    sign = np.sign(cube[wi, wj, k_seed])
    amp0 = abs(cube[wi, wj, k_seed])
    pick = np.full((n_i, n_j), -1, int)
    pick[wi, wj] = k_seed
    q = deque([(wi, wj)])
    while q:
        i, j = q.popleft()
        k = pick[i, j]
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < n_i and 0 <= b < n_j and pick[a, b] < 0:
                lo2, hi2 = max(k - search, 0), min(k + search + 1, n_t)
                seg = sign * cube[a, b, lo2:hi2]
                if np.isnan(seg).all():
                    continue
                kk = lo2 + int(np.nanargmax(seg))
                if sign * cube[a, b, kk] >= min_frac * amp0:
                    pick[a, b] = kk
                    q.append((a, b))
    H = np.where(pick >= 0, t_axis[np.clip(pick, 0, n_t - 1)], np.nan)
    return H, pick, sign


def cycle_skip_pct(H, step_ms=12):
    """Share of neighbouring traces whose horizon time jumps by more than step_ms (a cycle-skip sign)."""
    a = np.abs(np.diff(H, axis=0))
    b = np.abs(np.diff(H, axis=1))
    jumps = np.concatenate([a[np.isfinite(a)].ravel(), b[np.isfinite(b)].ravel()])
    return round(100 * float(np.mean(jumps > step_ms)), 2) if jumps.size else np.nan


def window_along(cube, pick, off, n):
    """Average a volume over n samples starting `off` samples below the horizon, carried sideways."""
    vals = [np.take_along_axis(cube, np.clip(pick + off + d, 0, cube.shape[2] - 1)[..., None], 2)[..., 0]
            for d in range(max(n, 1))]
    m = np.nanmean(vals, axis=0)
    m[pick < 0] = np.nan
    return m


def closure_at(H, i, j):
    """Spill-point test: deepest contour (ms below point i,j) that still closes around it."""
    valid = np.isfinite(H)
    t0 = H[i, j]
    best, reg_best = 0, np.zeros_like(valid)
    for c in range(2, 152, 2):
        lab, _ = ndimage.label(valid & (H <= t0 + c))
        reg = lab == lab[i, j]
        if reg[0].any() or reg[-1].any() or reg[:, 0].any() or reg[:, -1].any():
            break
        best, reg_best = c, reg
    return best, reg_best


def classify(closure_ms, inside_pct, edge_std, flat=FLAT_STD_MS, has_anomaly=True):
    if not has_anomaly:
        return "undetermined"
    if closure_ms >= CLOSE_MS and inside_pct >= 70 and edge_std <= flat:
        return "structural"
    if closure_ms >= CLOSE_MS or edge_std <= flat:
        return "combination"
    return "stratigraphic"


VERDICT_TEXT = {
    "structural": "Structural: the anomaly fills a closed high and stops at a constant level",
    "combination": "Combination: the anomaly stops at a constant level (fluid contact) but is not held by a closed high, "
                   "so the updip seal is stratigraphic or a fault",
    "stratigraphic": "Stratigraphic: no closed high, and the anomaly edge cuts across contours, following the sand body",
    "undetermined": "Undetermined: no seismic hydrocarbon anomaly is connected to the well",
}

SYSTEM = """You are an experienced exploration geoscientist acting as a copilot for a junior geoscientist.
Rules:
(1) Use only the EVIDENCE given, and cite evidence IDs in [brackets] for every claim. Never cite rule numbers.
(2) Work through, with headings: Reservoir, Charge, Seal, Trap geometry, Structural case, Stratigraphic case,
    Verdict with confidence, Key risks, What a geoscientist should check, Next steps.
(3) Under Charge, assess the AVO indicator and the Near and Far values explicitly; coal can mimic a gas
    response, so AVO is supporting evidence, never proof.
(4) Always argue both trap types before deciding.
(5) There are three trap outcomes:
    - Structural: the anomaly sits inside a closed high and its edge is at a constant level.
    - Stratigraphic: no closure, and the anomaly edge cuts across contours, following the sand body.
    - Combination: no closure, but the anomaly edge sits at a near-constant level (a fluid contact downdip),
      so the updip seal must be stratigraphic (pinch-out, clay plug) or a fault.
    If no seismic anomaly is connected to the well, the verdict is 'Undetermined'.
(6) The structure is in two-way time, not depth, so a small time closure can be a velocity effect: list it as a risk.
(7) Your confidence must not exceed the TRUST level given in the QC evidence. Explain the trust level.
(8) Under 'What a geoscientist should check', turn the QC warnings into concrete checks (which line, which time).
(9) Caveats and validity conditions limit evidence; they are never evidence for a trap type.
(10) If the evidence cannot settle something, say so and name the data that would.
(11) Never invent wells, numbers or field names. Be concise."""


# ============================================================================= the pipeline
class Copilot:
    def __init__(self, base=BASE, out=OUT):
        self.base, self.out = base, out
        os.makedirs(out, exist_ok=True)
        self.vol = {k: f"{base}/{v}" for k, v in VOL_FILES.items()}
        self.client = openai.OpenAI(base_url=GEMMA_URL, api_key="token-not-needed")
        self.results, self.leads = {}, None

    # ------------------------------------------------------------------ STEP 1: seismic geometry
    def load_seismic(self):
        """Read the inline/crossline of every trace in every volume, and build one lookup grid per volume."""
        self.geom = {}
        for name, path in self.vol.items():
            with segyio.open(path, ignore_geometry=True) as f:
                g = {"il": f.attributes(TF.INLINE_3D)[:].astype(int),
                     "xl": f.attributes(TF.CROSSLINE_3D)[:].astype(int),
                     "t": np.asarray(f.samples, float)}
                if name == "sand":
                    cx = f.attributes(TF.CDP_X)[:].astype(float)
                    cy = f.attributes(TF.CDP_Y)[:].astype(float)
                    scal = f.header[0][TF.SourceGroupScalar]
                    if scal < 0:
                        cx, cy = cx / -scal, cy / -scal
                    elif scal > 0:
                        cx, cy = cx * scal, cy * scal
                    self.il, self.xl, self.xy = g["il"], g["xl"], np.c_[cx, cy]
            self.geom[name] = g
        self.il_step = int(np.diff(np.unique(self.il)).min())
        self.xl_step = int(np.diff(np.unique(self.xl)).min())
        self.il0 = min(g["il"].min() for g in self.geom.values())
        self.xl0 = min(g["xl"].min() for g in self.geom.values())
        ni = (max(g["il"].max() for g in self.geom.values()) - self.il0) // self.il_step + 1
        nj = (max(g["xl"].max() for g in self.geom.values()) - self.xl0) // self.xl_step + 1
        rows = []
        for name, g in self.geom.items():
            grid = np.full((ni, nj), -1, np.int64)
            grid[(g["il"] - self.il0) // self.il_step, (g["xl"] - self.xl0) // self.xl_step] = np.arange(len(g["il"]))
            g["grid"] = grid
            rows.append({"volume": name, "traces": len(g["il"]), "inlines": f"{g['il'].min()}-{g['il'].max()}",
                         "crosslines": f"{g['xl'].min()}-{g['xl'].max()}", "samples": len(g["t"]),
                         "time_ms": f"{g['t'][0]:.0f}-{g['t'][-1]:.0f}"})
        d = self.xy[1:200] - self.xy[:199]
        self.bin_m = float(np.median(np.hypot(d[:, 0], d[:, 1])[np.hypot(d[:, 0], d[:, 1]) > 0]))
        self.cell_km2 = self.bin_m ** 2 / 1e6
        self.volumes = pd.DataFrame(rows)
        print(self.volumes.to_string(index=False))
        print(f"line step {self.il_step}, trace spacing about {self.bin_m:.1f} m")

    def tindex(self, name, il, xl):
        """Trace number(s) in a given volume for inline/crossline (array-friendly); -1 if not present."""
        il, xl = np.atleast_1d(np.asarray(il)), np.atleast_1d(np.asarray(xl))
        i, j = (il - self.il0) // self.il_step, (xl - self.xl0) // self.xl_step
        g = self.geom[name]["grid"]
        ok = (i >= 0) & (i < g.shape[0]) & (j >= 0) & (j < g.shape[1])
        out = np.full(i.shape, -1, np.int64)
        out[ok] = g[i[ok], j[ok]]
        return out

    def trace(self, name, il, xl):
        k = int(self.tindex(name, il, xl)[0])
        t = self.geom[name]["t"]
        if k < 0:
            return np.full(len(t), np.nan), t
        with segyio.open(self.vol[name], ignore_geometry=True) as f:
            return np.asarray(f.trace[k], float), t

    def nearest_line(self, x, y):
        i = int(np.hypot(self.xy[:, 0] - x, self.xy[:, 1] - y).argmin())
        return int(self.il[i]), int(self.xl[i]), float(np.hypot(self.xy[i, 0] - x, self.xy[i, 1] - y))

    # ------------------------------------------------------------------ STEP 2: wells
    def load_wells(self):
        wl = pd.read_excel(f"{self.base}/well_data/Challenge1_Challenge2_WellList_Information.xlsx", header=2)
        wl = wl.dropna(how="all").dropna(axis=1, how="all")
        wl["RT"] = pd.to_numeric(wl["RT"], errors="coerce").fillna(0.0)
        wl["TD"] = pd.to_numeric(wl["TD"], errors="coerce")
        self.wl = wl.set_index("Well Name")
        self.wells = list(self.wl.index)

        self.cks = {}
        for w in self.wells:
            p = f"{self.base}/well_data/checkshots/{w}_Checkshot.txt"
            if os.path.exists(p):
                d = pd.read_csv(p, sep=r"\s+", comment="#", header=None, names=["tvdss", "twt"]).dropna()
                d = d.sort_values("tvdss").drop_duplicates("tvdss")
                self.cks[w] = interp1d(d.tvdss, d.twt, bounds_error=False, fill_value="extrapolate")

        def nearest_ck(w):
            if w in self.cks:
                return w
            X, Y = self.wl.loc[w, "X"], self.wl.loc[w, "Y"]
            return min(self.cks, key=lambda c: np.hypot(X - self.wl.loc[c, "X"], Y - self.wl.loc[c, "Y"]))
        self.ck_for = {w: nearest_ck(w) for w in self.wells}

        self.surv = {}
        for w in self.wells:
            d = pd.read_csv(f"{self.base}/well_data/deviations/{w}_Deviation.txt", sep=r"\s+", comment="#",
                            header=None, names=["md", "azim", "incl"]).dropna().sort_values("md")
            md, I, A = d.md.values, np.radians(d.incl.values), np.radians(d.azim.values)
            N, E, V = [0.0], [0.0], [0.0]
            for k in range(1, len(md)):                         # minimum-curvature method
                dmd = md[k] - md[k - 1]
                cdl = np.cos(I[k] - I[k - 1]) - np.sin(I[k - 1]) * np.sin(I[k]) * (1 - np.cos(A[k] - A[k - 1]))
                dl = np.arccos(np.clip(cdl, -1, 1))
                rf = 1.0 if dl < 1e-9 else 2 / dl * np.tan(dl / 2)
                N.append(N[-1] + dmd / 2 * (np.sin(I[k - 1]) * np.cos(A[k - 1]) + np.sin(I[k]) * np.cos(A[k])) * rf)
                E.append(E[-1] + dmd / 2 * (np.sin(I[k - 1]) * np.sin(A[k - 1]) + np.sin(I[k]) * np.sin(A[k])) * rf)
                V.append(V[-1] + dmd / 2 * (np.cos(I[k - 1]) + np.cos(I[k])) * rf)
            self.surv[w] = {"md": md, "N": np.array(N), "E": np.array(E), "V": np.array(V), "I": I, "A": A}

        self.las, rows = {}, []
        for w in self.wells:
            d = lasio.read(f"{self.base}/well_data/las/{w}.las").df()
            self.las[w] = d
            for flag, fluid in [("PS_GIP", "gas"), ("PS_OIP", "oil")]:
                for z in zones_from_flag(d, flag):
                    rows.append({"well": w, "fluid": fluid, **z})
        self.pay = pd.DataFrame(rows)
        print(f"{len(self.wells)} wells, checkshots at {sorted(self.cks)}, {len(self.pay)} gas/oil zones")

    # ------------------------------------------------------------------ STEP 3: well positions at depth
    def path(self, w, md):
        """North, east offset and true vertical depth (from the rig floor) at measured depth(s)."""
        s = self.surv[w]
        md = np.atleast_1d(np.asarray(md, float))
        n, e, v = np.interp(md, s["md"], s["N"]), np.interp(md, s["md"], s["E"]), np.interp(md, s["md"], s["V"])
        b = md > s["md"][-1]
        dd = md[b] - s["md"][-1]
        n[b] = s["N"][-1] + dd * np.sin(s["I"][-1]) * np.cos(s["A"][-1])
        e[b] = s["E"][-1] + dd * np.sin(s["I"][-1]) * np.sin(s["A"][-1])
        v[b] = s["V"][-1] + dd * np.cos(s["I"][-1])
        return n, e, v

    def tvdss(self, w, md):
        return self.path(w, md)[2] - self.wl.loc[w, "RT"]

    def twt_raw(self, w, md):
        return self.cks[self.ck_for[w]](self.tvdss(w, md))

    def place_wells(self):
        self.main, self.pos, rows = {}, {}, []
        for w in self.wells:
            zz = self.pay[self.pay.well == w]
            if not len(zz):
                continue
            zz = zz[zz.fluid == "gas"] if (zz.fluid == "gas").any() else zz
            z = zz.sort_values("thick_m", ascending=False).iloc[0].to_dict()
            z["twt_top_raw"] = float(self.twt_raw(w, z["top_m"])[0])
            z["twt_base_raw"] = float(self.twt_raw(w, z["base_m"])[0])
            self.main[w] = z
            n, e, v = self.path(w, z["top_m"])
            il_s, xl_s, _ = self.nearest_line(self.wl.loc[w, "X"], self.wl.loc[w, "Y"])
            il_z, xl_z, _ = self.nearest_line(self.wl.loc[w, "X"] + e[0], self.wl.loc[w, "Y"] + n[0])
            self.pos[w] = {"il": il_z, "xl": xl_z, "surface_il": il_s, "surface_xl": xl_s,
                           "moved_m": round(float(np.hypot(n[0], e[0])))}
            rows.append({"well": w, "fluid": z["fluid"], "zone_md": z["top_m"], "thick_m": z["thick_m"],
                         "surface_il/xl": f"{il_s}/{xl_s}", "zone_il/xl": f"{il_z}/{xl_z}",
                         "moved_m": self.pos[w]["moved_m"], "checkshot": self.ck_for[w]})
        self.positions = pd.DataFrame(rows)

    # ------------------------------------------------------------------ STEP 4: well tie
    def tie_wells(self, half=100):
        """Shift each well's times so its log sand/shale pattern best matches seismic sand probability
        over +-100 ms around the zone (standard cross-correlation tie, limited to +-30 ms)."""
        self.tie = {}
        for w, z in self.main.items():
            d = self.las[w]
            if "VSH" not in d:
                self.tie[w] = {"shift_ms": 0.0, "corr": np.nan, "corr_untied": np.nan, "applied": False}
                continue
            md = d.index.values.astype(float)
            lg = pd.Series(1 - d["VSH"].values).rolling(51, center=True, min_periods=10).mean().values
            ok = np.isfinite(lg)
            t_log = self.twt_raw(w, md)
            v, t = self.trace("sand", self.pos[w]["il"], self.pos[w]["xl"])
            mid = (z["twt_top_raw"] + z["twt_base_raw"]) / 2
            m = (t >= mid - half) & (t <= mid + half)

            def corr(s):
                x = np.interp(t[m], t_log[ok] + s, lg[ok])
                return float(np.corrcoef(x, v[m])[0, 1]) if np.std(x) > 0 and np.std(v[m]) > 0 else np.nan
            shifts = np.arange(-30, 31, 2)
            cs = np.array([corr(s) for s in shifts])
            if np.all(np.isnan(cs)):
                self.tie[w] = {"shift_ms": 0.0, "corr": np.nan, "corr_untied": np.nan, "applied": False}
                continue
            k = int(np.nanargmax(cs))
            applied = cs[k] >= MIN_TIE_CORR
            self.tie[w] = {"shift_ms": float(shifts[k]) if applied else 0.0, "corr": round(float(cs[k]), 2),
                           "corr_untied": round(corr(0), 2), "applied": bool(applied)}
        self.ties = pd.DataFrame([{"well": w, **v, "checkshot": self.ck_for[w],
                                   "own_checkshot": w in self.cks} for w, v in self.tie.items()])

    def prepare(self):
        """Steps 1-4."""
        self.load_seismic()
        self.load_wells()
        self.place_wells()
        self.tie_wells()
        print("\nWell positions at the reservoir:\n", self.positions.to_string(index=False))
        print("\nWell ties:\n", self.ties.to_string(index=False))

    def zone_times(self, w):
        z, s = self.main[w], self.tie[w]["shift_ms"]
        return z["twt_top_raw"] + s, z["twt_base_raw"] + s

    # ------------------------------------------------------------------ STEP 5: data around a well
    def load_box(self, il_c, xl_c, t_center, half_lines=HALF_LINES, half_t=HALF_T, names=("full", "sand", "vpvs")):
        """Read a block of traces centred on (il_c, xl_c) from each named volume, using each volume's own lookup."""
        ils = il_c + self.il_step * np.arange(-half_lines, half_lines + 1)
        xls = xl_c + self.xl_step * np.arange(-half_lines, half_lines + 1)
        I, J = np.meshgrid(ils, xls, indexing="ij")
        box = {"ils": ils, "xls": xls, "wi": half_lines, "wj": half_lines}
        for name in names:
            t = self.geom[name]["t"]
            s0 = max(int(np.searchsorted(t, t_center - half_t)), 0)
            s1 = min(int(np.searchsorted(t, t_center + half_t)), len(t))
            idx = self.tindex(name, I.ravel(), J.ravel()).reshape(I.shape)
            cube = np.full(I.shape + (s1 - s0,), np.nan, np.float32)
            where = np.argwhere(idx >= 0)
            trs = idx[idx >= 0]
            order = np.argsort(trs)                              # read in file order (faster)
            with segyio.open(self.vol[name], ignore_geometry=True) as f:
                for (a, b), tr in zip(where[order], trs[order]):
                    cube[a, b] = f.trace[int(tr)][s0:s1]
            box[name] = cube
            box["t"] = t[s0:s1]
        return box

    # ------------------------------------------------------------------ STEP 6: trap tests
    def trap_test(self, box, H, pick, top_ms, base_ms):
        wi, wj = box["wi"], box["wj"]
        valid = np.isfinite(H)
        dt = box["t"][1] - box["t"][0]
        close_ms, clos = closure_at(H, wi, wj)
        ring = ndimage.binary_dilation(clos) & ~clos
        gap = float((~valid[ring]).mean()) if ring.any() else 0.0
        # sample the gas zone interval, carried sideways along the horizon; allow +-8 ms local snap at the well
        off = int(np.argmin(np.abs(box["t"] - top_ms))) - pick[wi, wj]
        n = max(1, int(round((base_ms - top_ms) / dt)) + 1)
        best_d, best_s = 0, -1.0
        for d in range(-2, 3):
            k = max(pick[wi, wj] + off + d, 0)
            s_ = float(np.nanmean(box["sand"][wi, wj, k:k + n]))
            if s_ > best_s:
                best_d, best_s = d, s_
        off += best_d
        sand = ndimage.median_filter(np.nan_to_num(window_along(box["sand"], pick, off, n), nan=0.0), 3)
        vp = window_along(box["vpvs"], pick, off, n)
        vpvs = ndimage.median_filter(np.nan_to_num(vp, nan=np.nanmax(vp)), 3)
        vw, bg = float(vpvs[wi, wj]), float(np.nanmedian(vpvs))
        thr = vw + 0.75 * (bg - vw) if vw < bg else vw + 0.05
        mask = (sand > SAND_CUT) & (vpvs < thr) & valid
        lab, _ = ndimage.label(mask, structure=np.ones((3, 3)))
        anom = (lab == lab[wi, wj]) if mask[wi, wj] else np.zeros_like(mask)
        r = {"horizon_ms": round(float(H[wi, wj]), 1), "snap_ms": round(float(best_d * dt), 1),
             "tracked_pct": round(100 * float(valid.mean())), "closure_ms": close_ms,
             "closure_km2": round(clos.sum() * self.cell_km2, 3), "closure_edge_on_gaps_pct": round(100 * gap),
             "sand_at_well": round(float(sand[wi, wj]), 2), "vpvs_at_well": round(vw, 2),
             "vpvs_background": round(bg, 2), "anomaly_km2": round(anom.sum() * self.cell_km2, 3),
             "anomaly_cells": int(anom.sum()), "anomaly_inside_closure_pct": 0, "anomaly_edge_std_ms": np.nan,
             "anomaly_hits_box_edge": bool(anom[0].any() or anom[-1].any() or anom[:, 0].any() or anom[:, -1].any())}
        if anom.any():
            edge = anom & ~ndimage.binary_erosion(anom)
            edge[[0, -1], :] = False
            edge[:, [0, -1]] = False
            et = H[edge & valid]
            r["anomaly_edge_std_ms"] = round(float(np.std(et)), 1) if et.size else np.nan
            r["anomaly_inside_closure_pct"] = round(100 * float((anom & clos).sum() / anom.sum()))
        r["verdict"] = classify(r["closure_ms"], r["anomaly_inside_closure_pct"],
                                r["anomaly_edge_std_ms"] if anom.any() else np.inf, has_anomaly=anom.any())
        return r, anom, clos

    def analyse(self, w):
        """Steps 5-6 for one well, plus its pictures."""
        top, bot = self.zone_times(w)
        p = self.pos[w]
        box = self.load_box(p["il"], p["xl"], (top + bot) / 2)
        H, pick, sign = autotrack(box["full"], box["wi"], box["wj"], box["t"], top)
        r, anom, clos = self.trap_test(box, H, pick, top, bot)
        r.update({"well": w, "fluid": self.main[w]["fluid"], "zone_md": self.main[w]["top_m"],
                  "thick_m": self.main[w]["thick_m"], "zone_twt_ms": round(top, 1), "il": p["il"], "xl": p["xl"],
                  "polarity": "peak" if sign > 0 else "trough", "cycle_skip_pct": cycle_skip_pct(H)})
        np.savez_compressed(f"{self.out}/horizon_{w}.npz", H=H, anom=anom, clos=clos, ils=box["ils"], xls=box["xls"])
        self._pictures(w, box, H, anom, clos)
        self.results[w] = r
        return r

    def _pictures(self, w, box, H, anom, clos):
        ils, xls, t, wi, wj = box["ils"], box["xls"], box["t"], box["wi"], box["wj"]
        Hm = np.ma.masked_invalid(H)
        # section through the well with the picked horizon; red where the anomaly is on this line
        fig, ax = plt.subplots(figsize=(11, 5))
        sec = box["full"][wi].T
        c = np.nanpercentile(np.abs(sec), 98)
        ax.imshow(sec, cmap="gray", vmin=-c, vmax=c, aspect="auto", extent=[xls[0], xls[-1], t[-1], t[0]])
        ax.plot(xls, H[wi], "y-", lw=1.5, label="auto-picked top reservoir")
        on = anom[wi]
        ax.plot(np.where(on, xls, np.nan), np.where(on, H[wi], np.nan), "r-", lw=4, label="hydrocarbon anomaly")
        ax.axvline(xls[wj], color="cyan", lw=1, label=f"well {w}")
        ax.set(title=f"{w}: inline {ils[wi]}", xlabel="crossline", ylabel="two-way time (ms)")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{self.out}/img_section_{w}.png", dpi=100)
        plt.close(fig)
        # map: horizon time contours, anomaly (red), closure (blue dashed), well (star)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(Hm, origin="lower", aspect="auto", cmap="viridis_r", extent=[xls[0], xls[-1], ils[0], ils[-1]])
        fig.colorbar(im, ax=ax, label="top reservoir time (ms), small = shallow")
        if np.isfinite(H).any():
            ax.contour(xls, ils, Hm, levels=np.arange(np.nanmin(H) // 10 * 10, np.nanmax(H) + 10, 10),
                       colors="k", linewidths=0.4)
        if clos.any():
            ax.contour(xls, ils, clos.astype(float), [0.5], colors="blue", linestyles="--", linewidths=2)
        if anom.any():
            ax.contour(xls, ils, anom.astype(float), [0.5], colors="red", linewidths=2)
        ax.plot(xls[wj], ils[wi], "w*", ms=16, mec="k")
        ax.set(title=f"{w}: red = hydrocarbon anomaly, blue dashed = structural closure",
               xlabel="crossline", ylabel="inline")
        fig.tight_layout()
        fig.savefig(f"{self.out}/img_map_{w}.png", dpi=100)
        plt.close(fig)

    # ------------------------------------------------------------------ STEP 7: QC and trust
    def qc(self, w):
        r, tie = self.results[w], self.tie[w]
        checks, scores = [], {}
        c = tie["corr"]
        scores["well_tie"] = 1.0 if c >= 0.5 else 0.5 if c >= MIN_TIE_CORR else 0.0
        if scores["well_tie"] < 1:
            checks.append(f"Well tie is weak (log-seismic correlation {c}). Check the tie at {w} on inline {r['il']} "
                          f"around {r['zone_twt_ms']:.0f} ms.")
        own = w in self.cks
        scores["checkshot"] = 1.0 if own else 0.5
        if not own:
            checks.append(f"No checkshot at {w}; time conversion borrowed from {self.ck_for[w]}.")
        sk = r["cycle_skip_pct"]
        scores["horizon"] = 1.0 if sk < 1 else 0.5 if sk < 3 else 0.0
        if scores["horizon"] < 1:
            checks.append(f"Auto-picked horizon may cycle-skip ({sk}% of steps jump >12 ms). Check inline {r['il']} and "
                          f"crossline {r['xl']} between {r['zone_twt_ms'] - 40:.0f} and {r['zone_twt_ms'] + 40:.0f} ms.")
        s = r["sand_at_well"]
        scores["log_vs_seismic"] = 1.0 if s >= 0.7 else 0.5 if s >= SAND_CUT else 0.0
        if scores["log_vs_seismic"] < 1:
            checks.append(f"Seismic sand probability at the logged zone is only {s}; the zone may be thinner than "
                          f"seismic resolution.")
        has = r["anomaly_cells"] > 0
        vs = {classify(r["closure_ms"], r["anomaly_inside_closure_pct"],
                       r["anomaly_edge_std_ms"] if has else np.inf, flat=f, has_anomaly=has) for f in (4, 6, 8)}
        scores["threshold_stability"] = 1.0 if len(vs) == 1 else 0.5
        if len(vs) > 1:
            checks.append(f"Verdict changes with the flat-edge threshold (edge spread {r['anomaly_edge_std_ms']} ms): "
                          f"a geoscientist should judge on the map whether the anomaly edge is flat.")
        trust = float(np.mean(list(scores.values())))
        level = "high" if trust >= 0.75 else "medium" if trust >= 0.5 else "low"
        r.update({"trust": round(trust, 2), "trust_level": level, "qc_scores": scores, "qc_checks": checks})
        return r

    def analyse_all(self):
        """Steps 5-7 for every well with a gas/oil zone."""
        for w in self.main:
            print("analysing", w, "...", end=" ")
            try:
                self.analyse(w)
                self.qc(w)
                print(self.results[w]["verdict"], "| trust", self.results[w]["trust_level"])
            except Exception as ex:
                print("failed:", ex)
        self.summary().to_csv(f"{self.out}/final_trap_table.csv", index=False)

    def summary(self):
        cols = ["well", "fluid", "thick_m", "sand_at_well", "closure_ms", "anomaly_km2", "anomaly_edge_std_ms",
                "cycle_skip_pct", "verdict", "trust", "trust_level"]
        df = pd.DataFrame(self.results.values())
        df = df.merge(self.ties[["well", "corr", "own_checkshot"]], on="well", how="left")
        return df[cols[:-2] + ["corr", "own_checkshot"] + cols[-2:]]

    # ------------------------------------------------------------------ STEP 8: evidence and AI
    def evidence(self, w):
        z, r, p = self.main[w], self.results[w], self.pos[w]
        top, bot = self.zone_times(w)
        val = {}
        for name in ("sand", "pimp", "vpvs", "near", "far"):
            v, t = self.trace(name, p["il"], p["xl"])
            val[name] = (round(window_mean(v, t, top, bot), 3), round(window_mean(v, t, top - 60, top), 3))
        nz, fz = val["near"][0], val["far"][0]
        avo = round(0.0 if fz > 0 else (fz - nz) * fz, 3)
        tie = self.tie[w]
        E = [
            f"E1 Well {w}: {z['fluid']} flagged by petrophysics from {z['top_m']} to {z['base_m']} m measured depth "
            f"({z['thick_m']} m thick).",
            f"E2 Log averages in the zone: shale volume {z['VSH']}, porosity {z['PHIT']}, water saturation {z['SWT']}.",
            f"E3 Position: the well reaches the zone at inline {p['il']}, crossline {p['xl']} ({p['moved_m']} m from its "
            f"surface location); zone at {top:.0f}-{bot:.0f} ms two-way time.",
            f"E4 Sand probability: {val['sand'][0]} in the zone vs {val['sand'][1]} in the 60 ms above.",
            f"E5 P-impedance: {val['pimp'][0]} in the zone vs {val['pimp'][1]} above.",
            f"E6 Vp/Vs: {val['vpvs'][0]} in the zone vs {val['vpvs'][1]} above.",
            f"E7 AVO indicator ((Far-Near)*Far where Far<0): {avo}. Near {nz}, Far {fz}. Valid for clastics, gentle "
            f"structure, roughly 1000-2500 m; coal can mimic a gas response.",
        ]
        if r["closure_ms"] > 0:
            E.append(f"E8 Auto-tracked top-reservoir horizon (two-way time): the well is inside a closed high with "
                     f"{r['closure_ms']} ms of closure over {r['closure_km2']} km2; {r['closure_edge_on_gaps_pct']}% of the "
                     f"closing contour lies against untracked areas (possible faults or poor data).")
        else:
            E.append("E8 Auto-tracked top-reservoir horizon (two-way time): the well is NOT inside a closed high.")
        if r["anomaly_cells"] > 0:
            E.append(f"E9 Hydrocarbon anomaly (sand probability > {SAND_CUT} and low Vp/Vs over the zone interval, "
                     f"connected to the well): {r['anomaly_km2']} km2, {r['anomaly_inside_closure_pct']}% inside the "
                     f"closure{'; it reaches the edge of the analysed area' if r['anomaly_hits_box_edge'] else ''}.")
            E.append(f"E10 Horizon time along the anomaly edge varies by {r['anomaly_edge_std_ms']} ms (standard "
                     f"deviation). Up to {FLAT_STD_MS} ms counts as a near-constant level (a fluid contact).")
        else:
            E.append(f"E9 No seismic hydrocarbon anomaly is connected to the well (sand probability "
                     f"{r['sand_at_well']} at the zone).")
        qc_text = "; ".join(r["qc_checks"]) if r["qc_checks"] else "all checks passed"
        E.append(f"E{len(E) + 1} QC and TRUST: trust level {r['trust_level'].upper()} ({r['trust']}). Well tie "
                 f"correlation {tie['corr']} (shift {tie['shift_ms']:+.0f} ms), checkshot "
                 f"{'own' if w in self.cks else 'borrowed from ' + self.ck_for[w]}, horizon cycle-skips "
                 f"{r['cycle_skip_pct']}%. Warnings: {qc_text}")
        E.append(f"E{len(E) + 1} Not available: fault picks, a depth-converted structure map, a regional report.")
        return "\n".join(E)

    def _image_part(self, path):
        with open(path, "rb") as fh:
            return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(fh.read()).decode()}}

    def ask(self, w, question, images=True, system=SYSTEM, temperature=0.3):
        """Ask the copilot about a well. With images=True, Gemma also sees the section and the map."""
        content = [{"type": "text", "text": f"EVIDENCE:\n{self.evidence(w)}\n\nQUESTION: {question}"}]
        if images:
            for kind in ("section", "map"):
                p = f"{self.out}/img_{kind}_{w}.png"
                if os.path.exists(p):
                    content.append(self._image_part(p))
        r = self.client.chat.completions.create(model=GEMMA_MODEL, temperature=temperature,
                                                messages=[{"role": "system", "content": system},
                                                          {"role": "user", "content": content}])
        return r.choices[0].message.content

    def debate(self, w):
        """Two advocates argue, then a chief geoscientist judges, using the same evidence."""
        ev = self.evidence(w)

        def call(sys, user):
            r = self.client.chat.completions.create(model=GEMMA_MODEL, temperature=0.4,
                                                    messages=[{"role": "system", "content": sys},
                                                              {"role": "user", "content": user}])
            return r.choices[0].message.content
        adv = ("You are a geoscientist arguing that the trap is {side}. Make the strongest honest case using only the "
               "EVIDENCE, citing IDs in [brackets], and admit the weakest point of your own case. Max 150 words.")
        s = call(adv.format(side="STRUCTURAL"), f"EVIDENCE:\n{ev}")
        g = call(adv.format(side="STRATIGRAPHIC"), f"EVIDENCE:\n{ev}")
        j = call("You are a chief geoscientist. Weigh both arguments against the EVIDENCE. Decide: structural, "
                 "stratigraphic, combination or undetermined. Your confidence must not exceed the TRUST level in the "
                 "evidence. Cite IDs. Max 150 words.",
                 f"EVIDENCE:\n{ev}\n\nSTRUCTURAL CASE:\n{s}\n\nSTRATIGRAPHIC CASE:\n{g}")
        text = f"## Structural advocate\n{s}\n\n## Stratigraphic advocate\n{g}\n\n## Chief geoscientist\n{j}"
        with open(f"{self.out}/debate_{w}.md", "w") as fh:
            fh.write(text)
        return text

    # ------------------------------------------------------------------ STEP 9: untested leads
    def find_leads(self, w, half_lines=200):
        """Track the discovery's level over a wider area and find patches with the same fingerprint, away from wells."""
        top, bot = self.zone_times(w)
        p = self.pos[w]
        box = self.load_box(p["il"], p["xl"], (top + bot) / 2, half_lines=half_lines)
        H, pick, _ = autotrack(box["full"], box["wi"], box["wj"], box["t"], top)
        wi, wj, dt = box["wi"], box["wj"], box["t"][1] - box["t"][0]
        off = int(np.argmin(np.abs(box["t"] - top))) - pick[wi, wj]
        n = max(1, int(round((bot - top) / dt)) + 1)
        sand = ndimage.median_filter(np.nan_to_num(window_along(box["sand"], pick, off, n), nan=0.0), 3)
        vp = window_along(box["vpvs"], pick, off, n)
        vpvs = ndimage.median_filter(np.nan_to_num(vp, nan=np.nanmax(vp)), 3)
        vw, bg = float(vpvs[wi, wj]), float(np.nanmedian(vpvs))
        thr = vw + 0.75 * (bg - vw) if vw < bg else bg - 0.05
        mask = (sand > SAND_CUT) & (vpvs < thr) & np.isfinite(H)
        lab, nlab = ndimage.label(mask, structure=np.ones((3, 3)))
        near = np.zeros_like(mask)
        rad = int(WELL_BUFFER_M / self.bin_m)
        for q in self.pos.values():
            for a_il, a_xl in ((q["il"], q["xl"]), (q["surface_il"], q["surface_xl"])):
                a, c = (a_il - box["ils"][0]) // self.il_step, (a_xl - box["xls"][0]) // self.xl_step
                if 0 <= a < mask.shape[0] and 0 <= c < mask.shape[1]:
                    near[max(a - rad, 0):a + rad + 1, max(c - rad, 0):c + rad + 1] = True
        above = window_along(box["sand"], pick, off - int(30 / dt), int(30 / dt))
        out, outline = [], np.zeros(mask.shape, bool)
        for k in range(1, nlab + 1):
            comp = lab == k
            if comp.sum() * self.cell_km2 < MIN_LEAD_KM2 or (comp & near).any():
                continue
            ii, jj = np.where(comp)
            t_ = int(np.argmin(H[ii, jj]))
            cl_ms, cl_reg = closure_at(H, ii[t_], jj[t_])
            edge = comp & ~ndimage.binary_erosion(comp)
            edge[[0, -1], :] = False
            edge[:, [0, -1]] = False
            est = float(np.nanstd(H[edge])) if edge.any() else np.nan
            inside = 100 * float((comp & cl_reg).sum() / comp.sum())
            outline |= comp
            out.append({"analogue_well": w, "il": int(box["ils"][int(ii.mean())]), "xl": int(box["xls"][int(jj.mean())]),
                        "twt_ms": round(float(np.nanmean(H[ii, jj]) + off * dt)),
                        "area_km2": round(comp.sum() * self.cell_km2, 2), "sand": round(float(sand[comp].mean()), 2),
                        "vpvs": round(float(vpvs[comp].mean()), 2), "vpvs_background": round(bg, 2),
                        "sand_above": round(float(np.nanmean(above[comp])), 2), "closure_ms": cl_ms,
                        "edge_std_ms": round(est, 1), "trap": classify(cl_ms, inside, est),
                        "touches_box_edge": bool(ii.min() == 0 or jj.min() == 0 or ii.max() == mask.shape[0] - 1
                                                 or jj.max() == mask.shape[1] - 1)})
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(sand, origin="lower", aspect="auto", cmap="YlOrBr", vmin=0, vmax=1,
                       extent=[box["xls"][0], box["xls"][-1], box["ils"][0], box["ils"][-1]])
        fig.colorbar(im, ax=ax, label="sand probability at the discovery's level")
        if outline.any():
            ax.contour(box["xls"], box["ils"], outline.astype(float), [0.5], colors="cyan", linewidths=1.5)
        own = (lab == lab[wi, wj]) if mask[wi, wj] else np.zeros_like(mask)
        if own.any():
            ax.contour(box["xls"], box["ils"], own.astype(float), [0.5], colors="red", linewidths=2)
        for q in self.pos.values():
            ax.plot(q["xl"], q["il"], "k^", ms=6)
        ax.set(xlim=(box["xls"][0], box["xls"][-1]), ylim=(box["ils"][0], box["ils"][-1]),
               title=f"Leads at the level of {w}: red = discovery, cyan = untested look-alikes",
               xlabel="crossline", ylabel="inline")
        fig.tight_layout()
        fig.savefig(f"{self.out}/img_leads_{w}.png", dpi=100)
        plt.close(fig)
        return out

    def find_all_leads(self):
        found = []
        for w in self.results:
            if self.results[w]["anomaly_cells"] > 0:
                print("searching around", w, "...")
                try:
                    found += self.find_leads(w)
                except Exception as ex:
                    print(w, "failed:", ex)
        if not found:
            self.leads = pd.DataFrame()
            return self.leads
        L = pd.DataFrame(found).sort_values("area_km2", ascending=False)
        keep = []
        for _, r in L.iterrows():                               # merge duplicates seen from neighbouring wells
            if all(np.hypot((r.il - k.il) / self.il_step, (r.xl - k.xl) / self.xl_step) * self.bin_m > 750
                   or abs(r.twt_ms - k.twt_ms) > 30 for k in keep):
                keep.append(r)
        self.leads = pd.DataFrame(keep).reset_index(drop=True)
        self.leads.insert(0, "lead", [f"L{i + 1:02d}" for i in range(len(self.leads))])
        self.leads.to_csv(f"{self.out}/leads.csv", index=False)
        return self.leads

    # ------------------------------------------------------------------ viewing
    def show(self, w, kinds=("section", "map")):
        from IPython.display import Image, display
        for k in kinds:
            p = f"{self.out}/img_{k}_{w}.png"
            if os.path.exists(p):
                display(Image(p))
