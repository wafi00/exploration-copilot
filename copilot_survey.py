"""
Exploration Copilot: SURVEY-WIDE search for traps like the ones the wells found (geobody screening + analogue match)

  1. Read sand probability and Vp/Vs for the WHOLE survey over the pay interval (default 1,600-2,350 ms).
  2. Gas-like voxels = sand probability > 0.55 and Vp/Vs < 1.84 (Team B's calibration from the tied wells).
  3. Connect them into 3D bodies; keep bodies larger than 0.25 km2 in map view.
  4. Measure every body: area, thickness, shape (elongation, compactness), edge flatness, sand, Vp/Vs contrast, depth.
  5. Mark the bodies the wells drilled (the discoveries), and the undrilled ones (more than 500 m from any well).
  6. For each undrilled body: the MOST SIMILAR discovery, a similarity score and the three reasons; AVO at its crest.
  7. Rank them, draw the survey map, and write everything into master/ for the app.

    import importlib, copilot_survey as sv; importlib.reload(sv)
    B = sv.run(cp)               # about 5-15 minutes (reads two volumes once); needs about 8 GB of memory
"""

import os, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import segyio
from scipy import ndimage

import copilot_pipeline as cpl

SAND_CUT, VPVS_CUT = 0.55, 1.84
MIN_KM2 = 0.25
WELL_BUFFER_M = 500
FEATS = ["log10_area_km2", "thick_ms", "elongation", "compactness", "edge_std_ms", "sand", "vpvs_contrast", "top_ms"]
FEAT_WORDS = {"log10_area_km2": "size", "thick_ms": "thickness", "elongation": "elongation (channel-likeness)",
              "compactness": "outline shape", "edge_std_ms": "edge flatness", "sand": "sand probability",
              "vpvs_contrast": "Vp/Vs drop (gas sign)", "top_ms": "depth (time)"}


def _read_window(cp, name, t0, t1, chunk=50000):
    g = cp.geom[name]
    t = g["t"]
    k = (t >= t0) & (t <= t1)
    ni, nj = g["grid"].shape
    out = np.full((ni, nj, int(k.sum())), np.nan, np.float32)
    ii = (g["il"] - cp.il0) // cp.il_step
    jj = (g["xl"] - cp.xl0) // cp.xl_step
    with segyio.open(cp.vol[name], ignore_geometry=True) as f:
        n = f.tracecount
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            out[ii[a:b], jj[a:b]] = np.asarray(f.trace.raw[a:b])[:, k]
            print(f"\rreading {name}: {b / n:.0%}", end="")
    print()
    return out, t[k]


def _rel_avo(cp, il, xl, top, bot):
    """Team B's AVO: far/near in the zone compared with far/near 100-200 ms above."""
    def rms(v, t, a, b):
        m = (t >= a) & (t <= b)
        return float(np.sqrt(np.nanmean(v[m] ** 2))) if m.any() else np.nan
    if bot - top < 8:
        c = (top + bot) / 2; top, bot = c - 4, c + 4
    n, t = cp.trace("near", il, xl); f, _ = cp.trace("far", il, xl)
    z = rms(f, t, top, bot) / max(rms(n, t, top, bot), 1e-9)
    b = rms(f, t, top - 200, top - 100) / max(rms(n, t, top - 200, top - 100), 1e-9)
    return round(z / max(b, 1e-9), 2)


def run(cp, t0=1600, t1=2350, top_n=15):
    S, t = _read_window(cp, "sand", t0, t1)
    V, _ = _read_window(cp, "vpvs", t0, t1)
    dt = float(t[1] - t[0])
    gas = (np.nan_to_num(S) > SAND_CUT) & (np.nan_to_num(V, nan=9) < VPVS_CUT)
    gas = ndimage.binary_opening(gas, structure=np.ones((3, 3, 1), bool))       # remove single-trace speckle
    lab, n = ndimage.label(gas)
    print(f"{n} connected gas-like bodies before size filter")
    sizes = np.bincount(lab.ravel())
    objs = ndimage.find_objects(lab)
    cell = cp.cell_km2
    min_cells = int(MIN_KM2 / cell)
    ni, nj = lab.shape[:2]
    owner = np.zeros((ni, nj), np.int32)                                          # map of body numbers (for drawing)
    rows, feet = [], {}
    for lid in np.where(sizes >= min_cells)[0]:
        if lid == 0:
            continue
        sl = objs[lid - 1]
        sub = lab[sl] == lid
        foot = sub.any(2)
        if foot.sum() < min_cells:
            continue
        thick = sub.sum(2) * dt
        top = np.where(foot, t[sl[2]][np.argmax(sub, 2)], np.nan)
        ii, jj = np.where(foot)
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([ii, jj])))) [::-1]
        edge = foot & ~ndimage.binary_erosion(foot)
        A, P = foot.sum(), max(edge.sum(), 1)
        Vs, Ss = V[sl], S[sl]
        v_in, v_out = float(np.nanmean(Vs[sub])), float(np.nanmean(Vs[~sub & np.isfinite(Vs)]))
        ci, cj = np.unravel_index(np.nanargmin(top), top.shape)
        rid = len(rows) + 1
        owner[sl[0], sl[1]][foot] = rid
        feet[rid] = (sl[0], sl[1], foot)
        rows.append({"rid": rid, "label": int(lid),
                     "il": int(cp.il0 + cp.il_step * (sl[0].start + ii.mean())),
                     "xl": int(cp.xl0 + cp.xl_step * (sl[1].start + jj.mean())),
                     "crest_il": int(cp.il0 + cp.il_step * (sl[0].start + ci)),
                     "crest_xl": int(cp.xl0 + cp.xl_step * (sl[1].start + cj)),
                     "top_ms": round(float(np.nanmin(top))), "area_km2": round(A * cell, 2),
                     "log10_area_km2": float(np.log10(A * cell)), "thick_ms": round(float(thick[foot].mean()), 1),
                     "elongation": round(float(np.sqrt(ev[0] / max(ev[1], 1e-6))), 2),
                     "compactness": round(float(4 * np.pi * A / P ** 2), 2),
                     "edge_std_ms": round(float(np.nanstd(top[edge])), 1), "sand": round(float(np.nanmean(Ss[sub])), 2),
                     "vpvs": round(v_in, 3), "vpvs_contrast": round(v_out - v_in, 3)})
    B = pd.DataFrame(rows)
    print(f"{len(B)} bodies larger than {MIN_KM2} km2")

    # --- which bodies did the wells drill?
    B["drilled_by"] = ""
    for w, p in cp.pos.items():
        if w not in cp.main:
            continue
        top_, bot_ = cp.zone_times(w)
        i, j = (p["il"] - cp.il0) // cp.il_step, (p["xl"] - cp.xl0) // cp.xl_step
        kt = np.where((t >= top_ - 8) & (t <= bot_ + 8))[0]
        if not len(kt):
            continue
        v = lab[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2, kt.min():kt.max() + 1].ravel()
        v = v[v > 0]
        if len(v):
            lid = int(np.bincount(v).argmax())
            for idx in B.index[B.label == lid]:
                B.at[idx, "drilled_by"] = (B.at[idx, "drilled_by"] + " " + w).strip()
    # --- undrilled = no well (surface or pay position) within the buffer
    wellmask = np.zeros((ni, nj), bool)
    for p in cp.pos.values():
        for a, b in ((p["il"], p["xl"]), (p.get("surface_il", p["il"]), p.get("surface_xl", p["xl"]))):
            wellmask[(a - cp.il0) // cp.il_step, (b - cp.xl0) // cp.xl_step] = True
    D = ndimage.distance_transform_edt(~wellmask) * cp.bin_m
    fp = lambda rid, arr: arr[feet[rid][0], feet[rid][1]][feet[rid][2]]
    B["untested"] = [bool(fp(r.rid, D).min() >= WELL_BUFFER_M) and not r.drilled_by for r in B.itertuples()]
    B["dist_to_well_km"] = [round(float(fp(r.rid, D).min()) / 1000, 1) for r in B.itertuples()]

    # --- analogue match: most similar discovery for every undrilled body
    Z = (B[FEATS] - B[FEATS].mean()) / (B[FEATS].std() + 1e-9)
    disc = B[B.drilled_by != ""]
    out = []
    for r in B[B.untested].itertuples():
        best = None
        for d in disc.itertuples():
            diff = np.abs(Z.loc[r.Index, FEATS].values - Z.loc[d.Index, FEATS].values)
            dist = float(np.sqrt((diff ** 2).sum()))
            if best is None or dist < best[0]:
                best = (dist, d, diff)
        if best is None:
            continue
        dist, d, diff = best
        why = ", ".join(FEAT_WORDS[FEATS[i]] for i in np.argsort(diff)[:3])
        out.append({"rid": r.rid, "analogue": d.drilled_by, "similarity_pct": round(100 * float(np.exp(-dist / np.sqrt(len(FEATS)))))
                    , "similar_in": why})
    if not out:
        print("no drilled bodies or no undrilled bodies found; try a wider time window or lower cut-offs")
        return pd.DataFrame()
    A_ = pd.DataFrame(out)
    L = B[B.untested].merge(A_, on="rid")
    # --- style and AVO for the best candidates
    L["style"] = np.where(L.edge_std_ms <= 6, "flat edge (possible contact: combination-like)",
                          np.where(L.elongation >= 2.5, "elongated sand body (channel-like, stratigraphic)",
                                   "isolated sand body (lobe or bar, stratigraphic)"))
    L["pre_score"] = (0.4 * L.similarity_pct / 100 + 0.15 * L.sand + 0.15 * (L.vpvs_contrast / 0.25).clip(0, 1)
                      + 0.15 * (L.log10_area_km2.clip(-0.6, 1) + 0.6) / 1.6)
    L = L.sort_values("pre_score", ascending=False).reset_index(drop=True)
    head = L.head(max(top_n * 2, 30)).copy()
    head["avo_relative"] = [_rel_avo(cp, r.crest_il, r.crest_xl, r.top_ms, r.top_ms + max(r.thick_ms, 8))
                            for r in head.itertuples()]
    head["score"] = (head.pre_score + 0.15 * ((head.avo_relative - 1) / 0.5).clip(0, 1)).round(2)
    head = head.sort_values("score", ascending=False).reset_index(drop=True)
    head.insert(0, "id", [f"S{i + 1:02d}" for i in range(len(head))])
    cols = ["id", "il", "xl", "crest_il", "crest_xl", "top_ms", "area_km2", "thick_ms", "elongation", "edge_std_ms", "sand",
            "vpvs", "vpvs_contrast", "avo_relative", "style", "analogue", "similarity_pct", "similar_in", "dist_to_well_km", "score"]
    SL = head[cols]
    rid_of = dict(zip(head.id, head.rid))
    SL.to_csv(f"{cp.out}/survey_leads.csv", index=False)
    B.to_csv(f"{cp.out}/survey_bodies.csv", index=False)
    cp.survey = {"bodies": int(len(B)), "drilled": int((B.drilled_by != "").sum()), "untested": int(B.untested.sum()),
                 "window_ms": f"{t0}-{t1}", "area_screened_km2": round(float(np.isfinite(S[..., 0]).sum() * cell))}
    cp.survey_leads = SL

    # --- survey map
    thickmap = gas.sum(2) * dt
    ext = [cp.xl0, cp.xl0 + cp.xl_step * (nj - 1), cp.il0, cp.il0 + cp.il_step * (ni - 1)]
    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(np.where(np.isfinite(S[..., 0]), thickmap, np.nan), origin="lower", aspect="auto", cmap="Greys",
                   extent=ext, vmax=np.nanpercentile(thickmap, 99))
    fig.colorbar(im, ax=ax, label=f"gas-like sand thickness, {t0}-{t1} ms (ms)")
    X, Y = np.arange(ext[0], ext[1] + 1, cp.xl_step)[:nj], np.arange(ext[2], ext[3] + 1, cp.il_step)[:ni]
    drilled_ids = B.loc[B.drilled_by != "", "rid"].values
    if len(drilled_ids):
        ax.contour(X, Y, np.isin(owner, drilled_ids).astype(float), [0.5], colors="limegreen", linewidths=1.5)
    for r in SL.head(top_n).itertuples():
        m_ = np.zeros((ni, nj), bool); a0, a1, f0 = feet[rid_of[r.id]]; m_[a0, a1] = f0
        ax.contour(X, Y, m_.astype(float), [0.5], colors="red", linewidths=1.5)
        ax.text(r.xl, r.il, f" {r.id}", color="red", fontsize=10, weight="bold")
    for w, p in cp.pos.items():
        ax.plot(p["xl"], p["il"], "^", color="cyan", mec="k", ms=8)
        ax.text(p["xl"], p["il"], " " + w, fontsize=8)
    ax.set(title=f"Survey-wide screen: {len(B)} gas-like sand bodies; green = drilled by wells, red = top {top_n} undrilled "
                 f"look-alikes", xlabel="crossline", ylabel="inline")
    fig.tight_layout()
    fig.savefig(f"{cp.out}/img_survey_bodies.png", dpi=90)
    plt.close(fig)

    # --- master folder for the app and the AI
    mdir = f"{cp.out}/master"
    if os.path.isdir(mdir):
        SL.to_csv(f"{mdir}/survey_leads.csv", index=False)
        shutil.copy2(f"{cp.out}/img_survey_bodies.png", f"{mdir}/images/survey_bodies.png")
        for w in cp.wells:
            p = f"{mdir}/evidence/{w}.txt"
            if not os.path.exists(p):
                continue
            body = B[B.drilled_by.str.contains(rf"\b{w}\b", regex=True)]
            mine = SL[SL.analogue.str.contains(rf"\b{w}\b", regex=True)].head(3)
            txt = open(p).read().split("\nSurvey-wide screening")[0]
            line = (f"Survey-wide screening ({cp.survey['area_screened_km2']} km2, {t0}-{t1} ms; gas-like = sand > {SAND_CUT} "
                    f"and Vp/Vs < {VPVS_CUT}): {cp.survey['bodies']} sand bodies, {cp.survey['drilled']} drilled by wells, "
                    f"{cp.survey['untested']} undrilled.")
            if len(body):
                b = body.iloc[0]
                line += (f" This well's pay body: {b.area_km2} km2, {b.thick_ms} ms thick, elongation {b.elongation}, "
                         f"edge variation {b.edge_std_ms} ms.")
            if len(mine):
                line += " Most similar undrilled bodies: " + "; ".join(
                    f"{r.id} at inline {r.crest_il}/crossline {r.crest_xl}, {r.top_ms} ms, {r.area_km2} km2, {r.style}, "
                    f"similarity {r.similarity_pct}% (similar in {r.similar_in}), AVO relative {r.avo_relative}"
                    for r in mine.itertuples()) + "."
            open(p, "w").write(txt + "\n" + line)
        print(f"master updated ({mdir})")
    try:
        import copilot_extras as cx
        cx.save_state(cp)
    except Exception:
        pass
    print(cp.survey)
    return SL
