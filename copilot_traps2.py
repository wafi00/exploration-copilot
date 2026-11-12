"""
Exploration Copilot: MORE ways to tell structural from stratigraphic (each one checkable by a geoscientist)

  1. Fault detection     coherence along the reservoir: low similarity to neighbouring traces = a break
  2. Fault vs channel    Team B's rule: faults stay STRAIGHT and IN PLACE 40 ms above and below; channel edges wiggle
  3. Fault-bounded gas   share of the gas outline that lies against a fault
  4. Contact conformance does the gas outline follow the contour of the contact seen in the well logs?
                         (yes = fluid contact in a structure; no = the sand body controls the trap)
  5. Nearest fault       distance and direction from the well (compare with Team B's DUG observations)

    import importlib, copilot_traps2 as t2; importlib.reload(t2)
    t2.run_all(cp)                       # about 5-8 minutes; pictures img_faults_<well>.png; updates master/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

import copilot_pipeline as cpl

HALF = 5            # +-20 ms coherence window
SHIFT = 10          # 40 ms above / below for the persistence test
NEAR_TRACES = 2     # "lies against a fault" = within 2 traces (50 m)


def _corr_pair(cube, k, axis, lag, half=HALF):
    d = np.arange(-half, half + 1)
    if axis == 0:
        ca, cb, ka = cube[:-1], cube[1:], k[:-1]
    else:
        ca, cb, ka = cube[:, :-1], cube[:, 1:], k[:, :-1]
    n = cube.shape[2] - 1
    a = np.take_along_axis(ca, np.clip(ka[..., None] + d, 0, n), 2).astype(float)
    b = np.take_along_axis(cb, np.clip(ka[..., None] + d + lag, 0, n), 2).astype(float)
    a -= a.mean(-1, keepdims=True); b -= b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1) + 1e-12)


def coherence_map(cube, pick, shift=0):
    """Similarity of each trace to its 4 neighbours in a +-20 ms window around the horizon (+shift samples).
    1 = continuous layers, low = a break (fault or channel edge). Small dips allowed (+-1 sample lag)."""
    k = np.where(pick >= 0, pick + shift, 0)
    ni, nj = pick.shape
    c0 = np.max([_corr_pair(cube, k, 0, l) for l in (-1, 0, 1)], axis=0)
    c1 = np.max([_corr_pair(cube, k, 1, l) for l in (-1, 0, 1)], axis=0)
    acc, cnt = np.zeros((ni, nj)), np.zeros((ni, nj))
    acc[:-1] += c0; cnt[:-1] += 1; acc[1:] += c0; cnt[1:] += 1
    acc[:, :-1] += c1; cnt[:, :-1] += 1; acc[:, 1:] += c1; cnt[:, 1:] += 1
    C = acc / np.maximum(cnt, 1)
    C[pick < 0] = np.nan
    return C


def classify_breaks(C, C_up, C_dn, H, q=10):
    """Low-coherence lines; 'fault-like' if straight and present 40 ms above or below, else 'channel-edge-like'."""
    def lowmask(X, z):                                           # robust: well below the typical coherence
        med = np.nanmedian(X); mad = 1.4826 * np.nanmedian(np.abs(X - med)) + 1e-3
        return np.nan_to_num(X, nan=1) < min(med - z * mad, np.nanpercentile(X, q))
    low = lowmask(C, 2.5)
    step = np.zeros_like(low)
    if np.isfinite(H).any():
        g = np.nan_to_num(np.maximum(np.abs(np.diff(H, axis=0, append=np.nan)), np.abs(np.diff(H, axis=1, append=np.nan))))
        step = g > 12                                            # horizon jumps across a fault
    low |= step & np.isfinite(H)
    grow = np.ones((7, 7), bool)                                  # a dipping fault moves sideways with time
    up = ndimage.binary_dilation(lowmask(C_up, 2.0), grow)
    dn = ndimage.binary_dilation(lowmask(C_dn, 2.0), grow)
    lab, n = ndimage.label(low, structure=np.ones((3, 3)))
    fault, edge = np.zeros_like(low), np.zeros_like(low)
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < 12:
            continue
        ii, jj = np.where(comp)
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([ii, jj]))))[::-1]
        straight = np.sqrt(ev[0] / max(ev[1], 1e-6)) >= 3          # long and narrow
        persist = (comp & (up | dn)).sum() / comp.sum() >= 0.5      # also there 40 ms above or below
        if straight and persist:
            fault |= comp
        else:
            edge |= comp
    return fault, edge


def contact_time(cp, w):
    """Contact (or 'hydrocarbon down to') at the well's main pay sand, converted to tied two-way time."""
    if not hasattr(cp, "sands"):
        return None, None
    z = cp.main[w]
    S = cp.sands[(cp.sands.well == w) & (cp.sands.top_md <= z["base_m"]) & (cp.sands.base_md >= z["top_m"])]
    S = S[S.fluid != "water"]
    if not len(S):
        return None, None
    r = S.iloc[0]
    sh = cp.tie.get(w, {}).get("shift_ms", 0.0)
    ck = cp.cks[cp.ck_for[w]]
    if np.isfinite(r.contact_tvdss):
        return float(ck(r.contact_tvdss)) + sh, f"{r.contact.split()[0]} seen at {r.contact_tvdss:.0f} m TVDSS"
    return float(ck(r.base_tvdss)) + sh, f"hydrocarbon down to base of sand at {r.base_tvdss:.0f} m TVDSS (contact deeper)"


def analyse_faults(cp, w):
    top, bot = cp.zone_times(w)
    p = cp.pos[w]
    box = cp.load_box(p["il"], p["xl"], (top + bot) / 2, half_t=160)
    ils, xls, t, wi, wj = box["ils"], box["xls"], box["t"], box["wi"], box["wj"]
    H, pick, _ = cpl.autotrack(box["full"], wi, wj, t, top)
    r, anom, clos = cp.trap_test(box, H, pick, top, bot)
    C = coherence_map(box["full"], pick)
    C_up, C_dn = coherence_map(box["full"], pick, -SHIFT), coherence_map(box["full"], pick, SHIFT)
    fault, edge = classify_breaks(C, C_up, C_dn, H)
    bm = cp.bin_m
    out = {"faults_km": round(fault.sum() * bm / 1000, 1), "channel_edges_km": round(edge.sum() * bm / 1000, 1)}
    # nearest fault to the well
    if fault.any():
        dist = ndimage.distance_transform_edt(~fault)
        out["nearest_fault_m"] = round(float(dist[wi, wj]) * bm)
        fi, fj = np.unravel_index(np.argmin(np.where(fault, np.hypot(*np.meshgrid(np.arange(fault.shape[0]) - wi,
                                                                                     np.arange(fault.shape[1]) - wj, indexing="ij")), np.inf)), fault.shape)
        ang = np.degrees(np.arctan2(fj - wj, fi - wi))           # +inline vs +crossline direction
        out["nearest_fault_dir"] = ("toward higher crosslines" if abs(fj - wj) >= abs(fi - wi) and fj > wj else
                                    "toward lower crosslines" if abs(fj - wj) >= abs(fi - wi) else
                                    "toward higher inlines" if fi > wi else "toward lower inlines")
    else:
        out["nearest_fault_m"], out["nearest_fault_dir"] = None, "none found"
    # gas outline against faults / channel edges
    if anom.any():
        ring = anom & ~ndimage.binary_erosion(anom)
        near_f = ndimage.binary_dilation(fault, np.ones((2 * NEAR_TRACES + 1,) * 2))
        near_e = ndimage.binary_dilation(edge, np.ones((2 * NEAR_TRACES + 1,) * 2))
        out["gas_edge_on_fault_pct"] = round(100 * float((ring & near_f).sum() / ring.sum()))
        out["gas_edge_on_channel_edge_pct"] = round(100 * float((ring & near_e).sum() / ring.sum()))
    else:
        out["gas_edge_on_fault_pct"] = out["gas_edge_on_channel_edge_pct"] = None
    # contact conformance
    tc, ctxt = contact_time(cp, w)
    out["contact"] = ctxt or "no hydrocarbon sand matched at the main pay"
    region = np.zeros_like(anom)
    if tc is not None and np.isfinite(H[wi, wj]):
        lab, _ = ndimage.label(np.isfinite(H) & (H <= tc))
        if lab[wi, wj]:
            region = lab == lab[wi, wj]
        if anom.any() and region.any():
            inter, union = (anom & region).sum(), (anom | region).sum()
            out["contact_conformance_iou"] = round(float(inter / union), 2)
            out["gas_area_vs_contact_area"] = round(float(anom.sum() / region.sum()), 2)
            out["contact_region_hits_box_edge"] = bool(region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any())
    out.setdefault("contact_conformance_iou", None)
    # reading for the AI and for Team B to QC
    elems = []
    if (out["gas_edge_on_fault_pct"] or 0) >= 30:
        elems.append(f"gas outline lies against faults along {out['gas_edge_on_fault_pct']}% of its edge (fault seal likely)")
    if (out["gas_edge_on_channel_edge_pct"] or 0) >= 30:
        elems.append(f"gas outline follows channel-edge-like breaks along {out['gas_edge_on_channel_edge_pct']}% of its edge")
    iou = out["contact_conformance_iou"]
    if iou is not None:
        if iou >= 0.5:
            elems.append(f"gas outline conforms to the contact contour (overlap {iou}): fluid-contact controlled, structural element")
        elif out.get("contact_region_hits_box_edge"):
            elems.append(f"the area above the contact is open (not closed) and much larger than the gas outline "
                         f"(overlap {iou}): structure does not hold the gas, the sand body does")
        else:
            elems.append(f"gas outline does not follow the contact contour (overlap {iou}): sand-body control")
    out["trap_elements"] = "; ".join(elems) if elems else "no fault or contact control detected"
    cp.results[w].update({f"t2_{k}": v for k, v in out.items()})
    _picture(cp, w, box, H, C, fault, edge, anom, region, tc, out)
    return out


def _picture(cp, w, box, H, C, fault, edge, anom, region, tc, out):
    ils, xls, wi, wj = box["ils"], box["xls"], box["wi"], box["wj"]
    ext = [xls[0], xls[-1], ils[0], ils[-1]]
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    im = ax[0].imshow(C, origin="lower", aspect="auto", cmap="gray", vmin=np.nanpercentile(C, 2), vmax=1, extent=ext)
    fig.colorbar(im, ax=ax[0], label="coherence (dark = break)")
    if fault.any(): ax[0].contour(xls, ils, fault.astype(float), [0.5], colors="red", linewidths=1.5)
    if edge.any(): ax[0].contour(xls, ils, edge.astype(float), [0.5], colors="deepskyblue", linewidths=1)
    if anom.any(): ax[0].contour(xls, ils, anom.astype(float), [0.5], colors="yellow", linewidths=2)
    ax[0].plot(xls[wj], ils[wi], "w*", ms=16, mec="k")
    ax[0].set(title=f"{w}: breaks along the reservoir\nred = fault-like (straight, repeats 40 ms above/below), "
                    f"blue = channel-edge-like, yellow = gas outline", xlabel="crossline", ylabel="inline")
    Hm = np.ma.masked_invalid(H)
    im = ax[1].imshow(Hm, origin="lower", aspect="auto", cmap="viridis_r", extent=ext)
    fig.colorbar(im, ax=ax[1], label="top reservoir time (ms)")
    ax[1].contour(xls, ils, Hm, levels=np.arange(np.nanmin(H) // 5 * 5, np.nanmax(H) + 5, 5), colors="k", linewidths=0.3)
    if tc is not None:
        ax[1].contour(xls, ils, Hm, levels=[tc], colors="cyan", linewidths=2.5, linestyles="--")
    if region.any(): ax[1].contourf(xls, ils, region.astype(float), [0.5, 1.5], colors=["cyan"], alpha=0.15)
    if anom.any(): ax[1].contour(xls, ils, anom.astype(float), [0.5], colors="red", linewidths=2)
    if fault.any(): ax[1].contour(xls, ils, fault.astype(float), [0.5], colors="magenta", linewidths=1)
    ax[1].plot(xls[wj], ils[wi], "w*", ms=16, mec="k")
    ax[1].set(title=f"Contact test: cyan dashed = {out['contact']}\nred = gas outline; overlap {out['contact_conformance_iou']}",
              xlabel="crossline", ylabel="inline")
    fig.suptitle(f"{w}: {out['trap_elements']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{cp.out}/img_faults_{w}.png", dpi=90)
    plt.close(fig)


def run_all(cp, wells=None):
    rows = []
    for w in (wells or [w for w in cp.results if cp.results[w].get("anomaly_cells", 0) > 0]):
        print("faults and contacts:", w, "...", end=" ")
        try:
            o = analyse_faults(cp, w)
            rows.append({"well": w, **o})
            print(o["trap_elements"])
        except Exception as ex:
            print("failed:", ex)
    T = pd.DataFrame(rows)
    T.to_csv(f"{cp.out}/trap_tests_2.csv", index=False)
    mdir = f"{cp.out}/master"
    if os.path.isdir(mdir) and len(T):
        M = pd.read_csv(f"{mdir}/master_wells.csv")
        M = M.drop(columns=[c for c in M if c.startswith("t2_")]).merge(T.add_prefix("t2_").rename(columns={"t2_well": "well"}),
                                                                         on="well", how="left")
        M.to_csv(f"{mdir}/master_wells.csv", index=False)
        for r in T.itertuples():
            p = f"{mdir}/evidence/{r.well}.txt"
            if os.path.exists(p):
                txt = open(p).read().split("\nFault and contact tests")[0]
                line = (f"Fault and contact tests (coherence along the reservoir, faults = straight breaks repeating 40 ms "
                        f"above and below): nearest fault {r.nearest_fault_m} m {r.nearest_fault_dir}; fault-like breaks "
                        f"{r.faults_km} km, channel-edge-like breaks {r.channel_edges_km} km in the 6 x 6 km area; gas outline "
                        f"on faults {r.gas_edge_on_fault_pct}%, on channel edges {r.gas_edge_on_channel_edge_pct}%; contact: "
                        f"{r.contact}; gas outline vs contact contour overlap {r.contact_conformance_iou}. Reading: "
                        f"{r.trap_elements}.")
                open(p, "w").write(txt + "\n" + line)
            src = f"{cp.out}/img_faults_{r.well}.png"
            if os.path.exists(src):
                import shutil; shutil.copy2(src, f"{mdir}/images/faults_{r.well}.png")
        print(f"master updated ({mdir})")
    try:
        import copilot_extras as cx; cx.save_state(cp)
    except Exception:
        pass
    return T
