"""Tab 6 of app.py: Seismic viewer and 'Ask the AI about this point'.

Viewer code (crop, map view, 3D view, upload, bright spots, AI trap boxes) comes from the
Seismic AI Lab in copilot/exploration-copilot. The AI call goes through CopilotAPI.ask_section, with
measured [V] evidence from copilot_extras.point_evidence for the competition volumes.
"""
import io, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

LAB_DIR = "/data/epic20/APGCE/apgce_team_01/copilot/exploration-copilot"
TEAMC_DIR = "/data/epic20/APGCE/apgce_team_01/teamC_traps"
if LAB_DIR not in sys.path:
    sys.path.insert(0, LAB_DIR)

import plots
from plots import (seismic_inline_range, crossline_range, get_section, sample_times,
                   wells_on_seismic, nearest_lines, read_well_list, il_xl_to_xy)
import plotly.graph_objects as go
from viewer import _thin
from viewer import (section_figure, map_figure, birdseye_figure, aerial_figure,
                    add_candidate_boxes, add_findings_to_map)
from seismic_ai import (save_upload, list_all_seismic, scan_bright_spots, build_prompt,
                        split_candidates, crop_slice, crop_section)


# ---------- Cached readers ----------
@st.cache_data(max_entries=4)
def _ranges(path):
    return seismic_inline_range(path), crossline_range(path)


@st.cache_data(max_entries=40)
def _section(path, which, number):
    return get_section(path, which, number)


@st.cache_data(max_entries=4)
def _times(path):
    return sample_times(path)


MAP_POINTS = 20000          # top-view points sent to the browser (was 40,000)


@st.cache_data(max_entries=20)
def _cropped_slice(path, time_ms, il_rng, xl_rng):
    pack, pts = crop_slice(path, time_ms, il_rng, xl_rng, max_points=MAP_POINTS)
    if pts is not None:                                     # smaller numbers = smaller message to the browser
        pts = {k: (np.asarray(v, np.float32) if k in ("x", "y", "v") else
                   np.asarray(v, np.int32) if k in ("il", "xl") else v) for k, v in pts.items()}
    return pack, pts


@st.cache_data(max_entries=4)
def _coef(path):
    return il_xl_to_xy(path)


@st.cache_data(max_entries=4)
def _wells_table(path):
    df = wells_on_seismic(path)
    return df[df.dist_m < 50].reset_index(drop=True)


@st.cache_data(max_entries=1)
def _well_xy():
    return read_well_list()


@st.cache_resource(show_spinner=False)
def _evidence_engine():
    """The notebook pipeline with saved well results, for measured evidence at a point (loads once)."""
    import copilot_pipeline as cpl
    import copilot_extras as cx
    cp = cpl.Copilot()
    cp.load_seismic()
    cx.load_state(cp)
    try:                                   # same set-up as copilot_app.py: the field knowledge adds its well table
        if hasattr(cp, "sands"):           # and notes to the evidence at a point
            cx.install(cp)
        import copilot_knowledge as ck
        ck.install(cp)
    except Exception:
        pass
    return cp, cx


def _knowledge(competition):
    """The field knowledge pack (from the 'training'). Field calibration only fits the competition survey."""
    try:
        import copilot_knowledge as ck
        if competition:
            return ck.KNOWLEDGE_BLOCK
        return ck.TRAP_CATALOGUE + "\n\n" + ck.TRAP_CAUTIONS
    except Exception:
        return ""


def _snap(v, lo, hi, step):
    v = lo + round((v - lo) / step) * step
    return int(min(max(v, lo), hi))


def _snap_range(r, lo, hi, step):
    a, b = _snap(min(r), lo, hi, step), _snap(max(r), lo, hi, step)
    if b - a < 2 * step:
        a, b = max(lo, a - 2 * step), min(hi, b + 2 * step)
    return (a, b)


def _is_competition(path):
    return os.path.abspath(path).startswith(os.path.abspath(plots.DATA_DIR))


def _views_png(il_pack, xl_pack, slice_pack, point, spots=()):
    """Map + inline + crossline as ONE picture for the AI, with a magenta cross at the point."""
    grid, uil, uxl, t_used = slice_pack
    il_data, il_pos, ts, il_used = il_pack
    xl_data, xl_pos, _, xl_used = xl_pack
    p_il, p_xl, p_t = point

    fig, ax = plt.subplots(1, 3, figsize=(18, 6.5), gridspec_kw={"width_ratios": [1, 1.4, 1.4]})
    c = np.nanpercentile(np.abs(grid), 98) or 1.0
    ax[0].imshow(grid, cmap="gray", vmin=-c, vmax=c, aspect="auto", origin="lower",
                 extent=[uxl[0], uxl[-1], uil[0], uil[-1]])
    ax[0].axhline(il_used, color="deepskyblue", lw=1.2)
    ax[0].axvline(xl_used, color="cyan", lw=1.2)
    for s in spots:
        ax[0].plot(s["crossline"], s["inline"], "o", mfc="none", mec="orange", ms=12, mew=2)
        ax[0].text(s["crossline"], s["inline"], " " + s["id"], color="orange", fontsize=10)
    ax[0].plot(p_xl, p_il, "m+", ms=24, mew=3)
    ax[0].set_title(f"A: MAP (time slice at {t_used:.0f} ms)")
    ax[0].set_xlabel("Crossline"); ax[0].set_ylabel("Inline")

    for k, (data, pos, title, xlab, px) in enumerate([
            (il_data, il_pos, f"B: INLINE {il_used}", "Crossline", p_xl),
            (xl_data, xl_pos, f"C: CROSSLINE {xl_used}", "Inline", p_il)], start=1):
        c = np.percentile(np.abs(data), 98) or 1.0
        ax[k].imshow(data, cmap="gray", vmin=-c, vmax=c, aspect="auto", extent=[pos[0], pos[-1], ts[-1], ts[0]])
        ax[k].axhline(t_used, color="yellow", lw=1)
        ax[k].plot(px, p_t, "m+", ms=24, mew=3)
        ax[k].set_title(title); ax[k].set_xlabel(xlab); ax[k].set_ylabel("TWT (ms)")
        ax[k].grid(alpha=0.25)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return buf.getvalue()


def _add_picture(msgs, part):
    """The user's own picture goes with the question, after the seismic view."""
    if part is None:
        return msgs
    msgs = [dict(m) for m in msgs]
    c = msgs[-1]["content"]
    msgs[-1]["content"] = (list(c) if isinstance(c, list) else [{"type": "text", "text": str(c)}]) + [
        {"type": "text", "text": "The user also attached this picture:"}, part]
    return msgs


def _section_messages(api, image, question, desc, evidence, history, examples):
    """Same messages as CopilotAPI.ask_section (so the answer can stream)."""
    history = [h for h in (history or []) if h.get("content")]
    first_q = history[0]["content"] if history else question
    content = api._example_parts(examples)
    content.append({"type": "text", "text": f"DISPLAY: {desc or 'seismic section'}\n\nEVIDENCE:\n"
                                            f"{evidence or 'none (image only)'}\n\nQUESTION: {first_q}"})
    content.append(api._img_any(image))
    msgs = [{"role": "system", "content": api.SYSTEM_SECTION}, {"role": "user", "content": content}]
    if history:
        msgs += [{"role": h["role"], "content": h["content"]} for h in history[1:]]
        msgs.append({"role": "user", "content": question})
    return msgs


BRIGHT_SCALE = [[0, "#FFE066"], [0.5, "#FF8C1A"], [1, "#E8112D"]]    # yellow -> orange -> red
SAND_SCALE = [[0, "#7BE495"], [1, "#0B7A3E"]]                       # light -> dark green


def _sand_volume():
    """The competition's sand-probability inversion volume, if there is one."""
    try:
        for name, p in plots.list_seismic().items():
            if "sandprob" in name.lower().replace("_", "").replace(" ", ""):
                return p
    except Exception:
        pass
    return None


def _bright_cut(values, pct):
    a = np.abs(np.asarray(values, float))
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 100 - pct)) if a.size else np.inf


def _overlay_section(fig, data, pos, ts, keep, colorscale, zmin, zmax, name):
    """Transparent everywhere except where keep is True (NaN cells are not drawn)."""
    z = np.where(keep, data, np.nan)
    z, x, y = _thin(z, pos, ts)
    fig.add_trace(go.Heatmap(z=z, x=x, y=y, colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=False,
                             opacity=0.8, hoverinfo="skip", name=name))
    return fig


SEC_W, SEC_H = 1000, 700     # picture size of a section (pixels); plenty for a half-screen panel


def _resample(a, ys_src, xs_src, ys_dst, xs_dst):
    """Nearest-neighbour: put array a (rows = ys_src, cols = xs_src) onto the ys_dst / xs_dst grid."""
    ri = np.clip(np.searchsorted(ys_src, ys_dst), 0, len(ys_src) - 1)
    ci = np.clip(np.searchsorted(xs_src, xs_dst), 0, len(xs_src) - 1)
    out = a[np.ix_(ri, ci)].astype(float)
    off = (xs_dst < xs_src[0]) | (xs_dst > xs_src[-1])
    out[:, off] = np.nan
    offr = (ys_dst < ys_src[0]) | (ys_dst > ys_src[-1])
    out[offr, :] = np.nan
    return out


def _paint(rgb, value, keep, scale, vmin, vmax, alpha=0.8):
    """Blend a colour scale into rgb where keep is True."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("s", [c for _, c in scale])
    keep = keep & np.isfinite(value)
    if not keep.any():
        return rgb
    t = np.clip((value - vmin) / ((vmax - vmin) or 1.0), 0, 1)
    col = cmap(np.nan_to_num(t))[..., :3]
    rgb[keep] = (1 - alpha) * rgb[keep] + alpha * col[keep]
    return rgb


def fast_section(data, pos, ts, title, xlabel, colours="Grey", wells=(), cross_at=None, cross_label="",
                 time_line=None, sand_pack=None, bright_pct=None, sand_cut=0.5):
    """A section sent as ONE compressed picture instead of hundreds of thousands of numbers.
    Same look as viewer.section_figure; highlights are painted into the picture."""
    import base64
    from matplotlib import colormaps
    from PIL import Image
    rs, cs = max(1, int(np.ceil(data.shape[0] / SEC_H))), max(1, int(np.ceil(data.shape[1] / SEC_W)))
    d, x, y = data[::rs, ::cs].astype(float), pos[::cs], ts[::rs]
    clip = float(np.nanpercentile(np.abs(data), 98)) or 1.0
    norm = np.clip((np.nan_to_num(d) + clip) / (2 * clip), 0, 1)
    rgb = colormaps["Greys" if colours == "Grey" else "RdBu"](norm)[..., :3]
    if sand_pack is not None:
        sd, sp, st_, _ = sand_pack
        sv = _resample(np.asarray(sd, float), np.asarray(st_, float), np.asarray(sp, float), y.astype(float), x.astype(float))
        rgb = _paint(rgb, sv, sv >= sand_cut, SAND_SCALE, sand_cut, 1.0)
    if bright_pct:
        a = np.abs(d)
        cut = _bright_cut(data, bright_pct)
        rgb = _paint(rgb, a, a >= cut, BRIGHT_SCALE, cut, float(np.nanmax(np.abs(data))), alpha=0.85)
    # small crops: enlarge block by block (not smoothed) so the browser shows crisp traces
    ky, kx = max(1, 500 // rgb.shape[0]), max(1, 700 // rgb.shape[1])
    if ky > 1 or kx > 1:
        rgb = np.repeat(np.repeat(rgb, ky, axis=0), kx, axis=1)
    buf = io.BytesIO()
    Image.fromarray((rgb * 255).astype(np.uint8)).save(buf, format="JPEG", quality=88)
    src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    # light invisible grid so hovering still shows position, time and amplitude
    hs, hc = max(1, data.shape[0] // 120), max(1, data.shape[1] // 160)
    fig = go.Figure(go.Heatmap(z=np.round(data[::hs, ::hc], 3), x=pos[::hc], y=ts[::hs], opacity=0, showscale=False,
                               hovertemplate=f"{xlabel}: %{{x}}<br>TWT: %{{y}} ms<br>Amplitude: %{{z:.3g}}<extra></extra>"))
    # the seismic picture, stretched to fill the axes (a layout image keeps the panel's normal shape)
    fig.add_layout_image(dict(source=src, xref="x", yref="y", x=float(x[0]) - dx / 2, y=float(y[0]) - dy / 2,
                              sizex=float(x[-1] - x[0]) + dx, sizey=float(y[-1] - y[0]) + dy, sizing="stretch",
                              xanchor="left", yanchor="top", layer="below"))
    for name, p in wells:
        fig.add_vline(x=p, line=dict(color="orange", width=1.5, dash="dot"), annotation_text=name,
                      annotation_position="top", annotation_font_color="orange")
    if cross_at is not None:
        fig.add_vline(x=cross_at, line=dict(color="#4DA3FF", width=1.5), annotation_text=cross_label,
                      annotation_position="bottom")
    if time_line is not None:
        fig.add_hline(y=time_line, line=dict(color="yellow", width=1.5))
    fig.update_xaxes(title=xlabel, range=[float(x[0]) - dx / 2, float(x[-1]) + dx / 2], showgrid=False)
    fig.update_yaxes(title="TWT (ms)", range=[float(y[-1]) + dy / 2, float(y[0]) - dy / 2], showgrid=False)
    fig.update_layout(title=title, height=620, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _highlight_section(fig, data, pos, ts, sand_pack, bright_pct, sand_cut):
    """Sand (green) then bright amplitudes (yellow-red) on a section; the base seismic stays underneath.
    Bright = the strongest bright_pct % of THIS section (each view uses its own cut-off)."""
    cut_b = _bright_cut(data, bright_pct) if bright_pct else None
    if sand_pack is not None:
        sd, sp, st_, _ = sand_pack
        fig = _overlay_section(fig, sd, sp, st_, sd >= sand_cut, SAND_SCALE, sand_cut, 1.0, "Sand probability")
    if cut_b is not None:
        a = np.abs(data)
        fig = _overlay_section(fig, a, pos, ts, a >= cut_b, BRIGHT_SCALE, cut_b, float(np.nanmax(a)),
                               "Bright amplitudes")
    n_new = (sand_pack is not None) + (cut_b is not None)
    if n_new:                                          # keep wells, lines and the cross on top
        d = fig.data
        fig.data = d[:1] + d[-n_new:] + d[1:-n_new]
    return fig


def _overlay_points(fig, pts, keep, values, colorscale, zmin, zmax, size, name):
    """Extra coloured squares on the map view, just above the seismic (under lines, wells and the star)."""
    if not np.any(keep):
        return fig
    fig.add_trace(go.Scattergl(x=pts["x"][keep], y=pts["y"][keep], mode="markers", name=name, hoverinfo="skip",
                               marker=dict(color=values[keep], colorscale=colorscale, cmin=zmin, cmax=zmax,
                                           size=size, symbol="square")))
    fig.data = (fig.data[0], fig.data[-1]) + tuple(fig.data[1:-1])
    return fig


def _overlay_grid(fig, grid, uil, uxl, keep, colorscale, zmin, zmax, name):
    rs, cs = max(1, grid.shape[0] // 500), max(1, grid.shape[1] // 500)
    z = np.where(keep, grid, np.nan)[::rs, ::cs]
    fig.add_trace(go.Heatmap(z=z, x=uxl[::cs], y=uil[::rs], colorscale=colorscale, zmin=zmin, zmax=zmax,
                             showscale=False, opacity=0.9, hoverinfo="skip", name=name))
    fig.data = (fig.data[0], fig.data[-1]) + tuple(fig.data[1:-1])
    return fig


CAND_RULE = ("\n\nAt the very end, add one line that starts with CANDIDATES: followed by a JSON list of features "
             "you identified on panel B or C, in the same format as before, or CANDIDATES: []")


def render(api, feedback=None, live=None, clean=None, image_part=None):
    clean = clean or (lambda t: t)
    ss = st.session_state
    ss.setdefault("extra_paths", [])
    ss.setdefault("lab_chat", [])      # {"role", "content" (shown), "sent" (sent to the AI), "cands"}
    ss.setdefault("cands", [])
    ss.setdefault("spots", [])

    ss.setdefault("lab_on", True)
    if not st.toggle("Open the seismic viewer", key="lab_on",
                     help="Switch off to stop reading seismic."):
        st.caption("Switch on to explore the seismic volumes and ask the AI about any point. "
                   "The first view can take up to a minute to load.")
        return

    ctl, main = st.columns([1, 3])

    # ---------------- Volume ----------------
    with ctl:
        with st.expander("Add your own seismic volume", expanded=False):
            up = st.file_uploader("Upload a SEG-Y file (.sgy / .segy)", type=["sgy", "segy"])
            if up is not None and ss.get("last_upload") != up.name:
                with st.spinner("Saving the file on the HPC..."):
                    save_upload(up)
                ss.last_upload = up.name
                ss.pick_name = "📤 " + up.name
                st.success(f"Saved: {up.name}")
            st.caption("For big files (and the demo), paste the path of a file already on the HPC:")
            typed = st.text_input("File path on the HPC", placeholder="/data/epic20/APGCE/.../my_volume.sgy")
            if st.button("Add this file", use_container_width=True) and typed.strip():
                if typed.strip() not in ss.extra_paths:
                    ss.extra_paths.append(typed.strip())
                ss.pick_name = "📁 " + typed.strip().split("/")[-1]

        vols = list_all_seismic(tuple(ss.extra_paths))
        if not vols:
            st.error("No seismic volumes found.")
            return
        names = list(vols.keys())
        if ss.get("pick_name") in names:
            ss.vol_pick = ss.pop("pick_name")
        if ss.get("vol_pick") not in names:
            ss.vol_pick = next((n for n in names if "Full_Stack" in n), names[0])
        vol = st.selectbox("Seismic volume", names, key="vol_pick")
        path = vols[vol]

    try:
        (il_lo, il_hi, il_step), (xl_lo, xl_hi, xl_step) = _ranges(path)
        t = _times(path)
    except Exception as e:
        st.error(f"Couldn't read this volume: {e}")
        return
    t0, t1 = int(t[0]), int(t[-1])
    dt = max(1, int(round(t[1] - t[0]))) if len(t) > 1 else 4

    if ss.get("vol_path") != path:                       # new volume: start fresh
        ss.vol_path = path
        ss.cands, ss.spots, ss.lab_chat = [], [], []
        for k in ("il", "xl", "map_t", "crop_il", "crop_xl", "crop_t"):
            ss.pop(k, None)

    ss.setdefault("crop_il", (il_lo, il_hi))
    ss.setdefault("crop_xl", (xl_lo, xl_hi))
    ss.setdefault("crop_t", (t0, t1))
    ss.setdefault("il", il_lo + ((il_hi - il_lo) // 2 // il_step) * il_step)
    ss.setdefault("xl", xl_lo + ((xl_hi - xl_lo) // 2 // xl_step) * xl_step)
    ss.setdefault("map_t", int(t[len(t) // 2]))

    if "map_jump" in ss:
        ss.il, ss.xl = ss.pop("map_jump")
    if "time_jump" in ss:
        ss.map_t = int(round(ss.pop("time_jump")))

    ss.crop_il = _snap_range(ss.crop_il, il_lo, il_hi, il_step)
    ss.crop_xl = _snap_range(ss.crop_xl, xl_lo, xl_hi, xl_step)
    ss.crop_t = _snap_range(ss.crop_t, t0, t1, dt)
    (cil0, cil1), (cxl0, cxl1), (ct0, ct1) = ss.crop_il, ss.crop_xl, ss.crop_t
    ss.il = _snap(ss.il, cil0, cil1, il_step)
    ss.xl = _snap(ss.xl, cxl0, cxl1, xl_step)
    ss.map_t = _snap(ss.map_t, ct0, ct1, dt)
    ss.il_box, ss.xl_box = ss.il, ss.xl

    try:
        wells = _wells_table(path)
    except Exception:
        wells = None

    # ---------------- Controls ----------------
    with ctl:
        def il_from_box():
            ss.il = _snap(ss.il_box, cil0, cil1, il_step)

        def xl_from_box():
            ss.xl = _snap(ss.xl_box, cxl0, cxl1, xl_step)

        def reset_crop():
            ss.crop_il, ss.crop_xl, ss.crop_t = (il_lo, il_hi), (xl_lo, xl_hi), (t0, t1)

        def crop_to_wells():
            if wells is not None and len(wells):
                m = ss.get("crop_margin", 40)
                ss.crop_il = (int(wells.inline.min()) - m * il_step, int(wells.inline.max()) + m * il_step)
                ss.crop_xl = (int(wells.crossline.min()) - m * xl_step, int(wells.crossline.max()) + m * xl_step)

        if wells is not None and len(wells):
            def go_to_well():
                w = ss.lab_well_pick
                if w != "-":
                    row = wells[wells.well == w].iloc[0]
                    if not (cil0 <= row.inline <= cil1 and cxl0 <= row.crossline <= cxl1):
                        reset_crop()
                    ss.il, ss.xl = int(row.inline), int(row.crossline)
            st.selectbox("Jump to well", ["-"] + wells.well.tolist(), key="lab_well_pick", on_change=go_to_well)

        cropped = (ss.crop_il != (il_lo, il_hi) or ss.crop_xl != (xl_lo, xl_hi) or ss.crop_t != (t0, t1))
        with st.expander("Crop area" + ("  (on)" if cropped else ""), expanded=False):
            st.slider("Inlines", il_lo, il_hi, step=il_step, key="crop_il")
            st.slider("Crosslines", xl_lo, xl_hi, step=xl_step, key="crop_xl")
            st.slider("Time window (ms)", t0, t1, step=dt, key="crop_t")
            if wells is not None and len(wells):
                ss.setdefault("crop_margin", 40)
                st.number_input("Margin around wells (lines)", 5, 400, step=5, key="crop_margin")
                st.button("Crop to the wells", on_click=crop_to_wells, use_container_width=True)
            st.button("Show the whole survey", on_click=reset_crop, use_container_width=True)

        st.number_input(f"Inline ({cil0} to {cil1})", min_value=cil0, max_value=cil1,
                        step=il_step, key="il_box", on_change=il_from_box)
        st.slider("Inline slider", cil0, cil1, step=il_step, key="il", label_visibility="collapsed")
        st.number_input(f"Crossline ({cxl0} to {cxl1})", min_value=cxl0, max_value=cxl1,
                        step=xl_step, key="xl_box", on_change=xl_from_box)
        st.slider("Crossline slider", cxl0, cxl1, step=xl_step, key="xl", label_visibility="collapsed")
        st.number_input(f"Map time (ms, {ct0} to {ct1})", min_value=ct0, max_value=ct1, step=dt, key="map_t")

        with st.expander("Display settings", expanded=False):
            dot = st.slider("Map view: dot size", 2, 12, 5)
            z_stretch = st.slider("3D view: wall height", 0.2, 1.5, 0.5, 0.1)
            sensitivity = st.slider("Bright-spot sensitivity (lower = finds more)", 1.2, 5.0, 2.0, 0.1)

    il, xl, map_t = ss.il, ss.xl, ss.map_t
    crop_il, crop_xl, crop_t = ss.crop_il, ss.crop_xl, ss.crop_t

    # ---------------- Read only the cropped area ----------------
    with main:
        with st.spinner("Reading seismic (the first time can take up to a minute)..."):
            try:
                il_pack = crop_section(_section(path, "inline", il), crop_xl, crop_t)
                xl_pack = crop_section(_section(path, "crossline", xl), crop_il, crop_t)
                slice_pack, pts = _cropped_slice(path, map_t, crop_il, crop_xl)
            except Exception as e:
                st.error(f"Couldn't read this part of the volume: {e}")
                return
    il_data, il_pos, ts, il_used = il_pack
    xl_data, xl_pos, _, xl_used = xl_pack
    t_used = slice_pack[3]

    try:
        cf = _coef(path)
    except Exception:
        cf = None

    wells_in = None
    if wells is not None and len(wells):
        wells_in = wells[(wells.inline.between(*crop_il)) & (wells.crossline.between(*crop_xl))]
    wxy = None
    if cf is not None and wells_in is not None and len(wells_in):
        try:
            wxy = _well_xy()
            wxy = wxy[wxy.well.astype(str).isin(wells_in.well)]
        except Exception:
            wxy = None

    spots_now = [s for s in ss.spots if s["time"] == t_used]
    cands_il = [c for c in ss.cands if c["panel"] == "inline" and c["line"] == il_used]
    cands_xl = [c for c in ss.cands if c["panel"] == "crossline" and c["line"] == xl_used]

    # ---------------- Map and sections ----------------
    with main:
        if cropped:
            st.info(f"Cropped to inlines {crop_il[0]}-{crop_il[1]}, crosslines {crop_xl[0]}-{crop_xl[1]}, "
                    f"{crop_t[0]}-{crop_t[1]} ms. Change it under Crop area.")
        # ----- Colour highlights -----
        sand_path = _sand_volume() if _is_competition(path) else None
        with st.container(border=True):
            h1, h2, h3 = st.columns([1.2, 1.2, 1])
            with h1:
                hl_bright = st.toggle("Highlight bright amplitudes", key="hl_bright",
                                      help="Strongest reflections in yellow-red: possible gas, must be checked.")
                if hl_bright:
                    ss.setdefault("hl_pct", 3)
                    st.slider("Strongest % of amplitudes", 1, 10, key="hl_pct")
            with h2:
                hl_sand = st.toggle("Show sand probability", key="hl_sand", disabled=sand_path is None,
                                    help=("Green where the inversion says sand." if sand_path else
                                          "Only for the competition volumes (needs the sand-probability inversion)."))
                if hl_sand and sand_path:
                    ss.setdefault("hl_sand_cut", 0.5)
                    st.slider("Sand probability above", 0.3, 0.9, step=0.05, key="hl_sand_cut")
            with h3:
                ss.setdefault("lab_colours", "Grey")
                colours = st.radio("Seismic colours", ["Grey", "Red-Blue"], horizontal=True, key="lab_colours")
            legend = []
            if hl_bright:
                legend.append(f"**Yellow-red** = strongest {ss.hl_pct}% of amplitudes in each view (possible gas, must be checked)")
            if hl_sand and sand_path:
                legend.append(f"**Green** = sand probability above {ss.hl_sand_cut:.2f}")
            if legend:
                st.caption(" · ".join(legend) + ". Shown on the map view and both sections.")

        sand_il = sand_xl = sand_pts = sand_slice = None
        if hl_sand and sand_path:
            try:
                sand_il = crop_section(_section(sand_path, "inline", il), crop_xl, crop_t)
                sand_xl = crop_section(_section(sand_path, "crossline", xl), crop_il, crop_t)
                sand_slice, sand_pts = _cropped_slice(sand_path, map_t, crop_il, crop_xl)
            except Exception:
                st.caption("Sand probability isn't available at this position.")
        cut_b = _bright_cut(pts["v"] if pts is not None else slice_pack[0], ss.hl_pct) if hl_bright else None

        if ss.get("lab_view") not in ("Map view", "3D view"):
            ss.lab_view = "Map view"
        view = st.radio("View", ["Map view", "3D view"], horizontal=True, label_visibility="collapsed", key="lab_view")
        top = view == "Map view"

        def to_xy(IL, XL):
            (a0, a1, a2), (b0, b1, b2) = cf
            IL, XL = np.asarray(IL, float), np.asarray(XL, float)
            return a0 + a1 * IL + a2 * XL, b0 + b1 * IL + b2 * XL

        if top and cf is not None and pts is not None:
            fig = birdseye_figure(pts, to_xy([il, il], crop_xl), to_xy(crop_il, [xl, xl]),
                                  tuple(float(v) for v in to_xy(il, xl)), wxy, il, xl, colours, dot)
            # each overlay is slotted in just above the seismic, so add bright first to keep it on top of sand
            if cut_b is not None:
                av = np.abs(np.asarray(pts["v"], float))
                fig = _overlay_points(fig, pts, av >= cut_b, av, BRIGHT_SCALE, cut_b, float(np.nanmax(av)),
                                      dot, "Bright amplitudes")
            if sand_pts is not None:
                sv = np.asarray(sand_pts["v"], float)
                fig = _overlay_points(fig, sand_pts, sv >= ss.hl_sand_cut, sv, SAND_SCALE, ss.hl_sand_cut, 1.0,
                                      dot, "Sand probability")
            fig = add_findings_to_map(fig, spots_now, ss.cands, cf)
            fig.update_layout(title=f"Map view: time slice at {t_used:.0f} ms")
            st.caption("Click anywhere on the map to move there. The red star is the point the AI is asked about.")
            event = st.plotly_chart(fig, use_container_width=True, key="lab_top",
                                    on_select="rerun", selection_mode="points")
            sel = event.selection.points if event and event.selection else []
            if sel:
                click = nearest_lines(path, sel[0]["x"], sel[0]["y"])
                if click != ss.get("last_click"):
                    ss.last_click = click
                    ss.map_jump = click
                    st.rerun()
        elif top:
            st.caption("This volume has no real-world X/Y in its headers, so the map is in inline/crossline numbers.")
            grid, uil, uxl, _ = slice_pack
            gfig = map_figure(grid, uil, uxl, t_used, il, xl, colours, wells_in)
            if cut_b is not None:
                ag = np.abs(grid)
                gfig = _overlay_grid(gfig, ag, uil, uxl, ag >= cut_b, BRIGHT_SCALE, cut_b, float(np.nanmax(ag)),
                                     "Bright amplitudes")
            if sand_slice is not None:
                sg, su, sx, _ = sand_slice
                gfig = _overlay_grid(gfig, sg, su, sx, sg >= ss.hl_sand_cut, SAND_SCALE, ss.hl_sand_cut, 1.0,
                                     "Sand probability")
            event = st.plotly_chart(gfig, use_container_width=True, key="lab_grid", on_select="rerun",
                                    selection_mode="points")
            sel = event.selection.points if event and event.selection else []
            if sel:
                click = (round(sel[0]["y"]), round(sel[0]["x"]))
                if click != ss.get("last_click"):
                    ss.last_click = click
                    ss.map_jump = click
                    st.rerun()
        elif cf is not None:
            st.caption("Drag to rotate, scroll to zoom. Move the slices with the controls on the left."
                       + (" Colour highlights show on the map view and the sections." if (hl_bright or hl_sand) else ""))
            f3d = aerial_figure(slice_pack, il_pack, xl_pack, cf, wxy, crop_t, colours, z_stretch)
            f3d.update_layout(title=f"3D view: inline {il_used}, crossline {xl_used}, time slice {t_used:.0f} ms")
            st.plotly_chart(f3d, use_container_width=True, key="lab_3d")
        else:
            st.info("The 3D view needs real-world X/Y in the trace headers, and this volume has none.")

    il_wells, xl_wells = [], []
    if wells_in is not None:
        il_wells = [(r.well, r.crossline) for r in wells_in.itertuples() if abs(r.inline - il_used) <= 10]
        xl_wells = [(r.well, r.inline) for r in wells_in.itertuples() if abs(r.crossline - xl_used) <= 10]

    left, right = st.columns(2)
    with left:
        f1 = fast_section(il_data, il_pos, ts, f"Inline {il_used}", "Crossline", colours, il_wells, xl_used,
                          f"CL {xl_used}", t_used, sand_il, ss.hl_pct if hl_bright else None,
                          ss.get("hl_sand_cut", 0.5))
        f1.add_scatter(x=[xl_used], y=[t_used], mode="markers", showlegend=False, hoverinfo="skip",
                       marker=dict(symbol="cross-thin", size=22, line=dict(color="magenta", width=3)))
        st.plotly_chart(add_candidate_boxes(f1, cands_il, "inline"), use_container_width=True)
    with right:
        f2 = fast_section(xl_data, xl_pos, ts, f"Crossline {xl_used}", "Inline", colours, xl_wells, il_used,
                          f"IL {il_used}", t_used, sand_xl, ss.hl_pct if hl_bright else None,
                          ss.get("hl_sand_cut", 0.5))
        f2.add_scatter(x=[il_used], y=[t_used], mode="markers", showlegend=False, hoverinfo="skip",
                       marker=dict(symbol="cross-thin", size=22, line=dict(color="magenta", width=3)))
        st.plotly_chart(add_candidate_boxes(f2, cands_xl, "crossline"), use_container_width=True)

    # ================= ASK THE AI ABOUT THIS POINT =================
    st.subheader("Ask the AI about this point")
    competition = _is_competition(path)
    st.markdown(f"Point: **inline {il_used}, crossline {xl_used}, {t_used:.0f} ms** (magenta cross). "
                "Move it with the inline, crossline and map time controls, or click the map.")
    if not competition:
        st.caption("Image-only interpretation: no well data for this volume.")

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Scan this time slice for bright spots", use_container_width=True):
            ss.spots = scan_bright_spots(slice_pack, sensitivity)
            spots_now = ss.spots
            if not spots_now:
                st.info("No bright spots at this time. Try another map time or a lower sensitivity.")
    with c2:
        if spots_now:
            st.markdown(f"**Bright spots at {t_used:.0f} ms** (unusually strong amplitude, possible gas, must be checked):")
            for s in spots_now:
                a, b = st.columns([5, 1])
                a.markdown(f"**{s['id']}**: inline {s['inline']}, crossline {s['crossline']} "
                           f"(about {s['cells']} traces), **{s['strength']}x** stronger than typical reflections")
                if b.button("Go", key=f"go_{s['id']}"):
                    ss.map_jump = (s["inline"], s["crossline"])
                    st.rerun()

    examples = None
    if os.path.isfile(os.path.join(TEAMC_DIR, "labels.csv")):
        if st.checkbox("Learn from labelled trap examples", key="lab_examples",
                       help="Shows the AI three labelled pictures of real traps before it answers."):
            try:
                examples = type(api).trap_examples(TEAMC_DIR)[:3]
            except Exception:
                st.caption("The labelled examples could not be loaded.")

    for i, msg in enumerate(ss.lab_chat):
        with st.chat_message(msg["role"]):
            st.markdown(clean(msg["content"]))
            if msg.get("img"):
                st.image(msg["img"], width=260)
            if msg.get("evidence"):
                with st.expander("Evidence at the point"):
                    st.code(clean(msg["evidence"]), language=None)
            if msg["role"] == "assistant" and msg.get("cands"):
                st.markdown("**Features marked on the sections** (red dashed boxes, red lines on the map):")
                for c in msg["cands"]:
                    a, b = st.columns([5, 1])
                    where = (f"inline {c['line']}, crosslines {c['from']:.0f}-{c['to']:.0f}" if c["panel"] == "inline"
                             else f"crossline {c['line']}, inlines {c['from']:.0f}-{c['to']:.0f}")
                    a.markdown(f"**{c['id']}: {c['type']}** ({c['confidence']} confidence): {where}, "
                               f"{c['t_from']:.0f}-{c['t_to']:.0f} ms. {clean(c['reason'])}")
                    if b.button("Go", key=f"go_{i}_{c['id']}"):
                        mid_x, mid_t = (c["from"] + c["to"]) / 2, (c["t_from"] + c["t_to"]) / 2
                        ss.map_jump = (c["line"], mid_x) if c["panel"] == "inline" else (mid_x, c["line"])
                        ss.time_jump = mid_t
                        st.rerun()

    if feedback and len(ss.lab_chat) >= 2 and ss.lab_chat[-1]["role"] == "assistant":
        feedback("Seismic viewer", f"{vol} IL {il_used} XL {xl_used} {t_used:.0f} ms",
                 ss.lab_chat[-2]["content"], ss.lab_chat[-1]["content"])

    if ss.lab_chat and st.button("New conversation", key="lab_new"):
        ss.lab_chat, ss.cands, ss.spots = [], [], []
        st.rerun()

    q = None
    qc = st.columns(3)
    for col, text in zip(qc, ["Is there a trap at this point? Structural or stratigraphic?",
                              "Are there any other possible traps in this view?",
                              "Explain the bright spots and the risks."]):
        if col.button(text, use_container_width=True, key=f"lab_q_{text}"):
            q = text
    user_pic = None
    with st.form("lab_form", clear_on_submit=True):
        typed_q = st.text_input("Your question", placeholder="e.g. Is there a fault trap on this inline?")
        pic_up = st.file_uploader("Add a picture for the AI to look at (optional)", type=["png", "jpg", "jpeg"],
                                  key="lab_pic")
        if st.form_submit_button("Ask") and typed_q.strip():
            q = typed_q.strip()
            user_pic = pic_up
    user_part, user_bytes = None, None
    if user_pic is not None and image_part is not None:
        try:
            user_part, user_bytes = image_part(user_pic), user_pic.getvalue()
        except Exception:
            st.warning("That picture couldn't be read, so the question is sent without it.")

    if q:
        if not spots_now:
            ss.spots = scan_bright_spots(slice_pack, sensitivity)
            spots_now = ss.spots
        point = (il_used, xl_used, t_used)
        desc = (f"{vol}: map at {t_used:.0f} ms, inline {il_used}, crossline {xl_used}, "
                f"{ts[0]:.0f}-{ts[-1]:.0f} ms, magenta cross at inline {il_used}, crossline {xl_used}, {t_used:.0f} ms")
        first = not ss.lab_chat
        if first:
            kb = _knowledge(competition)
            sent = ((f"FIELD KNOWLEDGE (use it to interpret what you see):\n{kb}\n\n" if kb else "")
                    + build_prompt(q, il_used, xl_used, t_used, spots_now)
                    + "\nThe magenta cross on all three panels marks the selected point. "
                      "Cite the measured EVIDENCE lines like [V2] whenever you use them.")
        else:
            sent = q + CAND_RULE

        evidence = ""
        with st.spinner("The AI is looking at the map and both sections..."):
            if competition:
                try:
                    cp, cx = _evidence_engine()
                    evidence = cx.point_evidence(cp, int(il_used), int(xl_used), float(t_used))
                except Exception:
                    st.caption("Measured evidence not available right now: the AI reads the picture only.")
            try:
                png = _views_png(il_pack, xl_pack, slice_pack, point, spots_now)
                history = [{"role": m["role"], "content": m["sent"]} for m in ss.lab_chat]   # earlier turns only
            except Exception as e:
                st.error(f"Couldn't draw the picture for the AI: {e}")
                return
        try:
            if live is not None:
                with st.chat_message("user"):
                    st.markdown(q)
                    if user_bytes:
                        st.image(user_bytes, width=260)
                with st.chat_message("assistant"):
                    raw = live(lambda img: _add_picture(_section_messages(api, png, sent, desc, evidence, history,
                                                                          examples), user_part),
                               lambda: api.ask_section(png, sent, desc, evidence, history=history or None,
                                                       examples=examples),
                               max_tokens=1400, hide="CANDIDATES:")
            else:
                with st.spinner("The AI is looking at the map and both sections..."):
                    raw = api.ask_section(png, sent, desc, evidence, history=history or None, examples=examples)
        except Exception:
            st.error("The AI is busy right now. Please try again in a moment.")
            return

        answer, cands = split_candidates(raw, (il_pos[0], il_pos[-1]), (xl_pos[0], xl_pos[-1]), (ts[0], ts[-1]))
        start = len(ss.cands)
        for k, c in enumerate(cands, start=1):
            c["id"] = f"T{start + k}"
            c["line"] = il_used if c["panel"] == "inline" else xl_used
        ss.cands = ss.cands + cands
        ss.lab_chat.append({"role": "user", "content": q + (" (picture attached)" if user_bytes else ""),
                            "sent": sent, "evidence": evidence, "img": user_bytes})
        ss.lab_chat.append({"role": "assistant", "content": clean(answer) if answer else "(no answer)", "sent": raw,
                            "cands": cands})
        st.rerun()
