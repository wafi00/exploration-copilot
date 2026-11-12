"""
Exploration Copilot: extras (use together with copilot_pipeline.py)

  A. WELL STUDY      sand layers, fluids and contacts from the well logs, without markers
                     (what the mentor suggested: water saturation + sand layers)
  B. MANUAL HORIZON  use Team B's DUG Insight pick instead of the auto-pick, and score the
                     auto-pick against it (how far off is the computer, in ms?)
  C. SEISMIC VIEWER  inline / crossline / time slice of any volume, values at a point,
                     and "ask the copilot about what I am looking at"
  D. SAVE / LOAD     store results so the chat app does not need to re-run everything

Notebook usage (after cp.prepare() and cp.analyse_all()):

    import importlib, copilot_extras as cx; importlib.reload(cx)
    cx.well_study(cp)                 # A: sand/fluid/contact tables + well panel picture
    cx.install(cp)                    # adds the well study to every copilot answer
    cx.save_state(cp)                 # D: so the app can start quickly
    cx.build_time_cache(cp)           # C: once, about 10-20 min; enables fast time slices
"""

import os, json, pickle, base64, types

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import segyio
from scipy import ndimage
from scipy.interpolate import griddata

import copilot_pipeline as cpl

VSH_SAND = 0.40          # shale volume below this = sand
MIN_SAND_M = 1.0         # thinnest sand counted
BRIDGE_M = 0.5           # shale breaks thinner than this do not split a sand
WATER_SWT = 0.70         # water saturation above this = water-bearing
CONTACT_GROUP_M = 5.0    # contacts within this many metres are treated as the same level


# =============================================================================== A. WELL STUDY
def _runs(mask):
    """Start/end (exclusive) of each run of True values."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def well_study(cp, vsh_cut=VSH_SAND):
    """Find every sand in every well, its fluid, and whether a contact is seen inside it."""
    sands = []
    for w in cp.wells:
        d = cp.las[w]
        if "VSH" not in d:
            continue
        md = d.index.values.astype(float)
        step = float(np.median(np.diff(md)))
        vsh = d["VSH"].values
        sw = d["SWT"].values if "SWT" in d else np.full(len(md), np.nan)
        phi = d["PHIT"].values if "PHIT" in d else np.full(len(md), np.nan)
        gas = (d["PS_GIP"].fillna(0).values > 0.5) if "PS_GIP" in d else np.zeros(len(md), bool)
        oil = (d["PS_OIP"].fillna(0).values > 0.5) if "PS_OIP" in d else np.zeros(len(md), bool)
        sand = np.isfinite(vsh) & (vsh < vsh_cut)
        k = max(int(round(BRIDGE_M / step)), 1)
        sand = ndimage.binary_closing(sand, structure=np.ones(k + 1)) & np.isfinite(vsh)
        for a, b in _runs(sand):
            if (b - a) * step < MIN_SAND_M:
                continue
            g, o = gas[a:b], oil[a:b]
            hc = g | o
            water = ~hc & (np.nan_to_num(sw[a:b], nan=0) > WATER_SWT)
            top, base = md[a], md[b - 1]
            tv = cp.tvdss(w, np.array([top, base]))
            rec = {"well": w, "top_md": round(top, 1), "base_md": round(base, 1),
                   "top_tvdss": round(float(tv[0]), 1), "base_tvdss": round(float(tv[1]), 1),
                   "thick_m": round((b - a) * step, 1), "phit": round(float(np.nanmean(phi[a:b])), 3),
                   "swt": round(float(np.nanmean(sw[a:b])), 2), "hc_m": round(float(hc.sum() * step), 1),
                   "fluid": "water", "contact": "", "contact_tvdss": np.nan, "goc_tvdss": np.nan, "note": ""}
            if hc.any():
                rec["fluid"] = "gas+oil" if g.any() and o.any() else "gas" if g.any() else "oil"
                hi = np.where(hc)[0]
                first, last = hi[0], hi[-1]
                below = water[last + 1:]
                if below.sum() * step >= 1.0:                     # water under the hydrocarbon, same sand
                    wtop = last + 1 + int(np.argmax(below))
                    c_md = (md[a + last] + md[a + wtop]) / 2
                    rec["contact"] = ("GWC" if g[last] else "OWC") + " seen"
                    rec["contact_tvdss"] = round(float(cp.tvdss(w, np.array([c_md]))[0]), 1)
                elif (len(water) - 1 - last) * step <= 1.0:
                    rec["contact"] = "HC down to base of sand"
                else:
                    rec["contact"] = "HC base inside sand, unclear below"
                if g.any() and o.any():
                    gi, oi = np.where(g)[0], np.where(o)[0]
                    if gi.max() < oi.min():
                        c_md = (md[a + gi.max()] + md[a + oi.min()]) / 2
                        rec["goc_tvdss"] = round(float(cp.tvdss(w, np.array([c_md]))[0]), 1)
                if water[:first].sum() * step >= 1.0:
                    rec["note"] = "water above hydrocarbon in the same sand (possible baffle or separate lobes)"
            sands.append(rec)
    S = pd.DataFrame(sands)
    cp.sands = S
    hcS = S[S.fluid != "water"]
    rows = []
    for w in cp.wells:
        s = S[S.well == w]
        if not len(s):
            continue
        h = s[s.fluid != "water"]
        rows.append({"well": w, "sands": len(s), "net_sand_m": round(s.thick_m.sum(), 1),
                     "hc_sands": len(h), "net_pay_m": round(h.hc_m.sum(), 1),
                     "contacts_seen": int(h.contact.str.contains("seen").sum()),
                     "hc_to_base_of_sand": int((h.contact == "HC down to base of sand").sum()),
                     "pay_tvdss_range": f"{h.top_tvdss.min():.0f}-{h.base_tvdss.max():.0f}" if len(h) else ""})
    cp.well_summary = pd.DataFrame(rows)
    c = np.sort(hcS.contact_tvdss.dropna().values)
    groups = 0 if not len(c) else 1 + int((np.diff(c) > CONTACT_GROUP_M).sum())
    cp.field_contacts = {"contacts_seen": int(len(c)), "distinct_levels": groups,
                         "range_tvdss": f"{c.min():.0f}-{c.max():.0f}" if len(c) else "",
                         "hc_sands": int(len(hcS)), "hc_to_base": int((hcS.contact == "HC down to base of sand").sum())}
    S.to_csv(f"{cp.out}/well_sands.csv", index=False)
    cp.well_summary.to_csv(f"{cp.out}/well_summary.csv", index=False)
    _well_panel(cp)
    print(cp.well_summary.to_string(index=False))
    print("\nfield:", cp.field_contacts)
    return cp.well_summary


def _well_panel(cp):
    """A correlation-style panel: every sand in every well, coloured by fluid, in true vertical depth."""
    S = cp.sands
    order = sorted(S.well.unique(), key=lambda w: (cp.wl.loc[w, "X"], cp.wl.loc[w, "Y"]))
    col = {"water": "#9ecae1", "gas": "#d62728", "oil": "#2ca02c", "gas+oil": "#ff7f0e"}
    hc = S[S.fluid != "water"]
    lo = (hc.top_tvdss.min() if len(hc) else S.top_tvdss.min()) - 150
    hi = (hc.base_tvdss.max() if len(hc) else S.base_tvdss.max()) + 150
    fig, ax = plt.subplots(figsize=(1.1 * len(order) + 2, 9))
    for x, w in enumerate(order):
        ax.plot([x, x], [lo, hi], color="0.85", lw=8, zorder=0)
        for _, r in S[S.well == w].iterrows():
            if r.base_tvdss < lo or r.top_tvdss > hi:
                continue
            ax.add_patch(plt.Rectangle((x - 0.3, r.top_tvdss), 0.6, max(r.base_tvdss - r.top_tvdss, 0.5),
                                       color=col[r.fluid], zorder=2))
            if np.isfinite(r.contact_tvdss):
                ax.plot([x - 0.45, x + 0.45], [r.contact_tvdss] * 2, color="blue", lw=2, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylim(hi, lo)
    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_ylabel("true vertical depth below sea level (m)")
    ax.set_title("Sands from logs (no markers): blue = water, red = gas, green = oil, orange = gas+oil; "
                 "blue line = contact seen in well", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{cp.out}/img_well_panel.png", dpi=110)
    plt.close(fig)


def well_evidence_lines(cp, w):
    s = cp.well_summary.set_index("well").loc[w] if hasattr(cp, "well_summary") and w in set(cp.well_summary.well) else None
    if s is None:
        return []
    z = cp.main[w]
    S = cp.sands[cp.sands.well == w]
    m = S[(S.top_md <= z["base_m"]) & (S.base_md >= z["top_m"])]
    lines = [f"Well-log sand study (no markers; sand = shale volume < {VSH_SAND}): {s.sands} sands, {s.net_sand_m} m net "
             f"sand; {s.hc_sands} hydrocarbon-bearing sands with {s.net_pay_m} m net pay, stacked over "
             f"{s.pay_tvdss_range} m TVDSS; contacts seen in {s.contacts_seen} sands, hydrocarbon down to the base of "
             f"the sand in {s.hc_to_base_of_sand}."]
    if len(m):
        r = m.iloc[0]
        c = r.contact + (f" at {r.contact_tvdss:.0f} m TVDSS" if np.isfinite(r.contact_tvdss) else "")
        g = f"; gas-oil contact at {r.goc_tvdss:.0f} m TVDSS" if np.isfinite(r.goc_tvdss) else ""
        lines.append(f"The main zone lies in a {r.thick_m} m sand ({r.top_tvdss:.0f}-{r.base_tvdss:.0f} m TVDSS), fluid "
                     f"{r.fluid}: {c}{g}." + (f" Note: {r.note}." if r.note else ""))
    f = cp.field_contacts
    lines.append(f"Field-wide well study: {f['hc_sands']} hydrocarbon sands in the wells; {f['contacts_seen']} contacts "
                 f"seen at {f['distinct_levels']} different levels ({f['range_tvdss']} m TVDSS); {f['hc_to_base']} sands "
                 f"hold hydrocarbon down to their base. (Interpretation guide: many separate contact levels point to "
                 f"separate accumulations, i.e. compartments; one shared contact points to one connected structure.)")
    return lines


SYSTEM_PLUS = cpl.SYSTEM + """
(12) Stacked sands with separate contacts at different depths indicate separate accumulations (compartments), which
     favours stratigraphic traps (or fault compartments). One common contact shared by several wells favours a single
     connected structural accumulation. Hydrocarbon down to the base of a sand means the trap's lower limit is not seen
     at the well.
(13) Regional context, to be confirmed: the setting is believed to be a Malay Basin-style fluvial to estuarine and
     deltaic system with interbedded coals. Coals can mimic gas on AVO; isolated channel sands favour stratigraphic traps,
     while later compression in such basins commonly forms anticlines (structural traps)."""


def install(cp):
    """Add the well study to the evidence and the regional rules to the copilot's instructions."""
    orig_evidence, orig_ask = cpl.Copilot.evidence, cpl.Copilot.ask

    def evidence(self, w):
        text = orig_evidence(self, w)
        n = text.count("\n") + 1
        extra = [f"E{n + 1 + i} {line}" for i, line in enumerate(well_evidence_lines(self, w))]
        return text + ("\n" + "\n".join(extra) if extra else "")

    def ask(self, w, question, images=True, system=None, temperature=0.3):
        return orig_ask(self, w, question, images=images, system=system or SYSTEM_PLUS, temperature=temperature)
    cp.evidence = types.MethodType(evidence, cp)
    cp.ask = types.MethodType(ask, cp)
    print("installed: well study evidence + regional rules")


# =============================================================================== B. MANUAL HORIZON
def load_manual_horizon(path, il_col=None, xl_col=None, t_col=None):
    """Read a DUG Insight horizon export (text). Guesses the inline, crossline and time columns if not given:
    inline and crossline are integer-like columns in the survey ranges, time is in 500-3500."""
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.replace(",", " ").split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    n = max(len(r) for r in rows)
    A = np.array([r for r in rows if len(r) == n])
    if il_col is None:
        cand = [c for c in range(n) if np.allclose(A[:, c], np.round(A[:, c])) and 700 <= A[:, c].min() and A[:, c].max() <= 5000]
        t_c = [c for c in range(n) if 500 <= A[:, c].min() and A[:, c].max() <= 3600 and c not in cand]
        il_col, xl_col = cand[0], cand[1]
        t_col = t_c[-1]
        print(f"guessed columns: inline={il_col}, crossline={xl_col}, time={t_col} (pass them explicitly if wrong)")
    return pd.DataFrame({"il": A[:, il_col].astype(int), "xl": A[:, xl_col].astype(int), "t": A[:, t_col]})


def compare_horizons(cp, w, manual):
    """How far is the auto-pick from the geoscientist's pick, where both exist? Uses the saved auto horizon."""
    f = np.load(f"{cp.out}/horizon_{w}.npz")
    H, ils, xls = f["H"], f["ils"], f["xls"]
    i = (manual.il.values - ils[0]) // cp.il_step
    j = (manual.xl.values - xls[0]) // cp.xl_step
    ok = (i >= 0) & (i < len(ils)) & (j >= 0) & (j < len(xls))
    a, m = H[i[ok], j[ok]], manual.t.values[ok]
    both = np.isfinite(a)
    d = np.abs(a[both] - m[both])
    if not d.size:
        print("no overlap between the manual pick and the auto-pick box")
        return None
    res = {"well": w, "traces_compared": int(d.size), "median_diff_ms": round(float(np.median(d)), 1),
           "within_4ms_pct": round(100 * float((d <= 4).mean())), "within_8ms_pct": round(100 * float((d <= 8).mean())),
           "off_by_a_cycle_pct": round(100 * float((d > 12).mean()))}
    print(res)
    return res


def analyse_with_manual(cp, w, manual):
    """Re-run the trap test for one well using the geoscientist's horizon (interpolated between picked lines)."""
    top, bot = cp.zone_times(w)
    p = cp.pos[w]
    box = cp.load_box(p["il"], p["xl"], (top + bot) / 2)
    I, J = np.meshgrid(box["ils"], box["xls"], indexing="ij")
    sel = manual[(manual.il >= box["ils"][0]) & (manual.il <= box["ils"][-1]) &
                 (manual.xl >= box["xls"][0]) & (manual.xl <= box["xls"][-1])]
    if len(sel) < 10:
        print("too few manual picks inside this well's box")
        return None
    H = griddata((sel.il.values, sel.xl.values), sel.t.values, (I, J), method="linear")
    t = box["t"]
    pick = np.where(np.isfinite(H), np.argmin(np.abs(t[None, None, :] - np.nan_to_num(H)[..., None]), axis=2), -1)
    H = np.where(pick >= 0, t[np.clip(pick, 0, len(t) - 1)], np.nan)
    r, anom, clos = cp.trap_test(box, H, pick, top, bot)
    r.update({"well": w, "horizon_source": "manual (Team B)", "cycle_skip_pct": cpl.cycle_skip_pct(H)})
    cp._pictures(w + "_manual", box, H, anom, clos)
    print({k: r[k] for k in ("closure_ms", "anomaly_km2", "anomaly_edge_std_ms", "verdict")})
    return r


# =============================================================================== C. SEISMIC VIEWER
VOL_LABEL = {"full": "Full stack amplitude", "near": "Near stack amplitude", "far": "Far stack amplitude",
             "sand": "Total sand probability (0-1)", "vpvs": "Vp/Vs", "pimp": "P-impedance",
             "avo": "AVO indicator (Far-Near)*Far where Far<0"}
STYLE = {"full": ("gray", True), "near": ("gray", True), "far": ("gray", True), "sand": ("YlOrBr", False),
         "vpvs": ("viridis_r", False), "pimp": ("viridis", False), "avo": ("hot_r", False)}


def section(cp, vol, kind, number, t0=1500, t1=2500):
    """An inline (kind='inline') or crossline (kind='crossline') of a volume, between t0 and t1 ms.
    Returns (data [time x trace], along-axis numbers, time axis)."""
    if vol == "avo":
        n, along, t = section(cp, "near", kind, number, t0, t1)
        f, _, _ = section(cp, "far", kind, number, t0, t1)
        return np.where(f > 0, 0.0, (f - n) * f), along, t
    g = cp.geom[vol]
    grid = g["grid"]
    if kind == "inline":
        idx = grid[(number - cp.il0) // cp.il_step, :]
        along = cp.xl0 + cp.xl_step * np.arange(grid.shape[1])
    else:
        idx = grid[:, (number - cp.xl0) // cp.xl_step]
        along = cp.il0 + cp.il_step * np.arange(grid.shape[0])
    keep = np.where(idx >= 0)[0]
    if not len(keep):
        raise ValueError(f"no traces on {kind} {number}")
    keep = np.arange(keep.min(), keep.max() + 1)
    t = g["t"]
    m = (t >= t0) & (t <= t1)
    data = np.full((m.sum(), len(keep)), np.nan, np.float32)
    with segyio.open(cp.vol[vol], ignore_geometry=True) as f:
        for c, k in enumerate(keep):
            if idx[k] >= 0:
                data[:, c] = np.asarray(f.trace[int(idx[k])])[m]
    return data, along[keep], t[m]


def build_time_cache(cp, names=("full", "sand", "vpvs", "near", "far"), t0=1000, t1=3000, chunk=20000):
    """Re-sort volumes by time so a time slice loads instantly. About 2 GB per volume (float16).
    Check free space first (df -h in a terminal)."""
    for name in names:
        g = cp.geom[name]
        t = g["t"]
        k = (t >= t0) & (t <= t1)
        ni, nj = g["grid"].shape
        path = f"{cp.out}/tcache_{name}.npy"
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(int(k.sum()), ni, nj))
        mm[:] = np.nan
        ii = (g["il"] - cp.il0) // cp.il_step
        jj = (g["xl"] - cp.xl0) // cp.xl_step
        with segyio.open(cp.vol[name], ignore_geometry=True) as f:
            n = f.tracecount
            for a in range(0, n, chunk):
                b = min(a + chunk, n)
                block = np.asarray(f.trace.raw[a:b])[:, k]
                mm[:, ii[a:b], jj[a:b]] = block.T.astype(np.float16)
                print(f"\r{name}: {b / n:.0%}", end="")
        mm.flush()
        del mm
        with open(f"{cp.out}/tcache_{name}.json", "w") as fh:
            json.dump({"t": t[k].tolist()}, fh)
        print(f"\r{name}: done ({path})")


def has_cache(cp, vol):
    need = ("near", "far") if vol == "avo" else (vol,)
    return all(os.path.exists(f"{cp.out}/tcache_{v}.npy") for v in need)


def time_slice(cp, vol, t_ms):
    """A map of a volume at one time (needs build_time_cache). Returns (data [inline x crossline], ils, xls, t)."""
    if vol == "avo":
        n, ils, xls, tt = time_slice(cp, "near", t_ms)
        f, _, _, _ = time_slice(cp, "far", t_ms)
        return np.where(f > 0, 0.0, (f - n) * f), ils, xls, tt
    mm = np.load(f"{cp.out}/tcache_{vol}.npy", mmap_mode="r")
    t = np.array(json.load(open(f"{cp.out}/tcache_{vol}.json"))["t"])
    k = int(np.argmin(np.abs(t - t_ms)))
    data = np.asarray(mm[k], dtype=np.float32)
    ils = cp.il0 + cp.il_step * np.arange(data.shape[0])
    xls = cp.xl0 + cp.xl_step * np.arange(data.shape[1])
    return data, ils, xls, float(t[k])


def colour_range(vol, data, clip=98):
    cmap, sym = STYLE[vol]
    if vol == "sand":
        return cmap, 0.0, 1.0
    if sym:
        c = float(np.nanpercentile(np.abs(data), clip))
        return cmap, -c, c
    return cmap, float(np.nanpercentile(data, 100 - clip)), float(np.nanpercentile(data, clip))


def wells_near(cp, kind, number, tol_lines=4):
    """Wells (at reservoir position) within a few lines of this section, as {well: along-axis number}."""
    out = {}
    for w, p in cp.pos.items():
        if kind == "inline" and abs(p["il"] - number) <= tol_lines * cp.il_step:
            out[w] = p["xl"]
        if kind == "crossline" and abs(p["xl"] - number) <= tol_lines * cp.xl_step:
            out[w] = p["il"]
    return out


def render(cp, vol, view, number, data, along_or_ils, axis2, point=None, clip=98, path=None):
    """Save the current view as a PNG (used for the screen and sent to the AI)."""
    cmap, vmin, vmax = colour_range(vol, data, clip)
    fig, ax = plt.subplots(figsize=(12, 6))
    if view == "time slice":
        ils, xls = along_or_ils, axis2
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=[xls[0], xls[-1], ils[0], ils[-1]])
        for w, p in cp.pos.items():
            ax.plot(p["xl"], p["il"], "c^", ms=7, mec="k")
            ax.text(p["xl"], p["il"], " " + w, color="k", fontsize=8)
        if point:
            ax.plot(point[1], point[0], "m+", ms=22, mew=2)
        ax.set(xlabel="crossline", ylabel="inline", title=f"{VOL_LABEL[vol]}: time slice {number:.0f} ms")
    else:
        along, t = along_or_ils, axis2
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, extent=[along[0], along[-1], t[-1], t[0]])
        for w, x in wells_near(cp, view, number).items():
            ax.axvline(x, color="cyan", lw=1)
            ax.text(x, t[0], " " + w, color="cyan", va="top", fontsize=9)
        if point:
            ax.plot(point[1] if view == "inline" else point[0], point[2], "m+", ms=22, mew=2)
        ax.set(xlabel="crossline" if view == "inline" else "inline", ylabel="two-way time (ms)",
               title=f"{VOL_LABEL[vol]}: {view} {number}")
    fig.colorbar(im, ax=ax, label=VOL_LABEL[vol])
    fig.tight_layout()
    path = path or f"{cp.out}/view_{vol}_{view.replace(' ', '')}_{int(number)}.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def point_evidence(cp, il, xl, t_ms, half=8):
    """Numbered evidence at a chosen point: attributes vs 60 ms above, nearest wells and their results, nearest lead."""
    val = {}
    for v in ("sand", "vpvs", "pimp", "near", "far"):
        tr, t = cp.trace(v, il, xl)
        val[v] = (cpl.window_mean(tr, t, t_ms - half, t_ms + half), cpl.window_mean(tr, t, t_ms - 60, t_ms - half))
    nz, fz = val["near"][0], val["far"][0]
    avo = 0.0 if fz > 0 else (fz - nz) * fz
    V = [f"V1 Selected point: inline {il}, crossline {xl}, {t_ms:.0f} ms two-way time (values averaged over +-{half} ms).",
         f"V2 Sand probability {val['sand'][0]:.2f} at the point vs {val['sand'][1]:.2f} in the 60 ms above.",
         f"V3 Vp/Vs {val['vpvs'][0]:.2f} vs {val['vpvs'][1]:.2f} above; P-impedance {val['pimp'][0]:.0f} vs {val['pimp'][1]:.0f} above.",
         f"V4 AVO indicator {avo:.2f} (Near {nz:.2f}, Far {fz:.2f}); coal can mimic a gas response."]
    d = []
    for w, p in cp.pos.items():
        dist = np.hypot((p["il"] - il) / cp.il_step, (p["xl"] - xl) / cp.xl_step) * cp.bin_m
        r = cp.results.get(w, {})
        d.append((dist, w, r.get("verdict", "not analysed"), r.get("trust_level", "-"), cp.main[w]["fluid"],
                  r.get("zone_twt_ms", np.nan)))
    d.sort()
    near = "; ".join(f"{w} {dist / 1000:.1f} km away ({fl} at {zt:.0f} ms, trap {v}, trust {tl})"
                     for dist, w, v, tl, fl, zt in d[:3])
    V.append(f"V5 Nearest wells: {near}.")
    if getattr(cp, "leads", None) is not None and len(cp.leads):
        L = cp.leads.copy()
        L["dist"] = np.hypot((L.il - il) / cp.il_step, (L.xl - xl) / cp.xl_step) * cp.bin_m
        r = L.sort_values("dist").iloc[0]
        V.append(f"V6 Nearest untested lead: {r.lead} {r.dist / 1000:.1f} km away ({r.area_km2} km2, {r.trap}, at "
                 f"{r.twt_ms} ms, look-alike of {r.analogue_well}).")
    return "\n".join(V)


SYSTEM_VIEW = """You are an experienced exploration geoscientist acting as a copilot. The user is looking at a seismic
display (the image) and has selected a point (magenta cross). Rules:
(1) Separate what you SEE in the image from what the EVIDENCE says; cite evidence IDs like [V2] and write
    "(image)" for visual observations.
(2) Look for trap-related features: faults (reflectors offset along a line), channels (sinuous or lens-shaped bodies;
    in map view, winding high sand-probability belts), domes/anticlines (reflectors arching up), pinch-outs,
    flat spots (flat reflectors cutting across dipping ones), bright amplitudes.
(3) For the selected point, say whether it looks like a structural, stratigraphic, combination or no trap, and argue
    both cases briefly.
(4) Give confidence (high/medium/low); a single image justifies at most medium.
(5) End with 'What a geoscientist should check', naming specific lines, times or volumes.
(6) Never invent wells, numbers or field names. Be concise."""


def ask_view(cp, png_path, desc, il, xl, t_ms, question, temperature=0.3):
    ev = point_evidence(cp, il, xl, t_ms)
    img = {"type": "image_url",
           "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(png_path, "rb").read()).decode()}}
    content = [{"type": "text", "text": f"DISPLAY: {desc}\n\nEVIDENCE:\n{ev}\n\nQUESTION: {question}"}, img]
    r = cp.client.chat.completions.create(model=cpl.GEMMA_MODEL, temperature=temperature,
                                          messages=[{"role": "system", "content": SYSTEM_VIEW},
                                                    {"role": "user", "content": content}])
    return r.choices[0].message.content, ev


# =============================================================================== D. SAVE / LOAD
STATE_KEYS = ("wells", "wl", "cks", "ck_for", "surv", "las", "pay", "main", "pos", "positions", "tie", "ties",
              "results", "leads", "sands", "well_summary", "field_contacts")


def save_state(cp, path=None):
    path = path or f"{cp.out}/copilot_state.pkl"
    with open(path, "wb") as fh:
        pickle.dump({k: getattr(cp, k) for k in STATE_KEYS if hasattr(cp, k)}, fh)
    print("saved", path)


def load_state(cp, path=None):
    path = path or f"{cp.out}/copilot_state.pkl"
    with open(path, "rb") as fh:
        for k, v in pickle.load(fh).items():
            setattr(cp, k, v)
    print("loaded", path)
