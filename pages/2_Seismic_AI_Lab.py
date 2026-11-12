import numpy as np
import streamlit as st
from plots import (seismic_inline_range, crossline_range, get_section, sample_times,
                   wells_on_seismic, nearest_lines, read_well_list, il_xl_to_xy)
from viewer import (section_figure, map_figure, birdseye_figure, aerial_figure,
                    add_candidate_boxes, add_findings_to_map)
from seismic_ai import (save_upload, list_all_seismic, scan_bright_spots, views_png,
                        build_prompt, split_candidates, SYSTEM, crop_slice, crop_section)
from llm import ask_with_image

st.set_page_config(page_title="Seismic AI Lab", page_icon="🤖", layout="wide")
st.title("🤖 Seismic AI Lab")
st.caption("Load any seismic volume, explore it, and ask the AI what traps it can see. "
           "AI findings are a first pass: a geoscientist must confirm them.")


# ---------- Cached readers (max_entries stops memory filling up) ----------
@st.cache_data(max_entries=4)
def ranges(path):
    return seismic_inline_range(path), crossline_range(path)

@st.cache_data(max_entries=40)
def section(path, which, number):
    return get_section(path, which, number)

@st.cache_data(max_entries=4)
def times(path):
    return sample_times(path)

@st.cache_data(max_entries=20)
def cropped_slice(path, time_ms, il_rng, xl_rng):
    return crop_slice(path, time_ms, il_rng, xl_rng)

@st.cache_data(max_entries=4)
def coef(path):
    return il_xl_to_xy(path)

@st.cache_data(max_entries=4)
def wells_table(path):
    df = wells_on_seismic(path)
    return df[df.dist_m < 50].reset_index(drop=True)      # only wells that sit inside this survey

@st.cache_data(max_entries=1)
def well_xy():
    return read_well_list()


def snap(v, lo, hi, step):
    v = lo + round((v - lo) / step) * step
    return int(min(max(v, lo), hi))


def snap_range(r, lo, hi, step):
    """Keep a (from, to) range on real lines, inside the survey, and at least 2 steps wide."""
    a, b = snap(min(r), lo, hi, step), snap(max(r), lo, hi, step)
    if b - a < 2 * step:
        a, b = max(lo, a - 2 * step), min(hi, b + 2 * step)
    return (a, b)


ss = st.session_state
ss.setdefault("extra_paths", [])
ss.setdefault("chat", [])          # questions and answers about the seismic
ss.setdefault("cands", [])         # features the AI found (T1, T2...)
ss.setdefault("spots", [])         # bright spots from the auto-scan (B1, B2...)

# ---------- Sidebar: choose or add a volume ----------
with st.sidebar:
    with st.expander("➕ Add your own seismic volume", expanded=False):
        up = st.file_uploader("Upload a SEG-Y file (.sgy / .segy)", type=["sgy", "segy"])
        if up is not None and ss.get("last_upload") != up.name:
            with st.spinner("Saving the file on the HPC..."):
                save_upload(up)
            ss.last_upload = up.name
            ss.pick_name = "📤 " + up.name
            st.success(f"Saved: {up.name}")
        st.caption("Big file already on the HPC? Paste its full path instead:")
        typed = st.text_input("File path on the HPC", placeholder="/data/epic20/APGCE/.../my_volume.sgy")
        if st.button("Add this file", use_container_width=True) and typed.strip():
            if typed.strip() not in ss.extra_paths:
                ss.extra_paths.append(typed.strip())
            ss.pick_name = "📁 " + typed.strip().split("/")[-1]

    vols = list_all_seismic(tuple(ss.extra_paths))
    if not vols:
        st.error("No seismic volumes found.")
        st.stop()
    names = list(vols.keys())
    if ss.get("pick_name") in names:
        ss.vol_pick = ss.pop("pick_name")
    if ss.get("vol_pick") not in names:
        saved_vol = ss.get("saved_vol")
        ss.vol_pick = saved_vol if saved_vol in names else next((n for n in names if "Full_Stack" in n), names[0])
    vol = st.selectbox("Seismic volume", names, key="vol_pick")
    ss.saved_vol = vol
    path = vols[vol]

try:
    (il_lo, il_hi, il_step), (xl_lo, xl_hi, xl_step) = ranges(path)
    t = times(path)
except Exception as e:
    st.error(f"Couldn't read this volume: {e}")
    st.stop()
t0, t1 = int(t[0]), int(t[-1])
dt = max(1, int(round(t[1] - t[0]))) if len(t) > 1 else 4

# New volume picked: clear old AI findings, saved position and crop
if ss.get("vol_path") != path:
    ss.vol_path = path
    ss.cands, ss.spots = [], []
    ss.saved_pos, ss.saved_crop = {}, {}
    for k in ("il", "xl", "map_t", "crop_il", "crop_xl", "crop_t"):
        ss.pop(k, None)

# Streamlit forgets widget values when you visit another page, so restore them
saved, saved_crop = ss.get("saved_pos", {}), ss.get("saved_crop", {})
if "crop_il" not in ss:
    ss.crop_il = saved_crop.get("il", (il_lo, il_hi))
if "crop_xl" not in ss:
    ss.crop_xl = saved_crop.get("xl", (xl_lo, xl_hi))
if "crop_t" not in ss:
    ss.crop_t = saved_crop.get("t", (t0, t1))
if "il" not in ss:
    ss.il = saved.get("il", il_lo + ((il_hi - il_lo) // 2 // il_step) * il_step)
if "xl" not in ss:
    ss.xl = saved.get("xl", xl_lo + ((xl_hi - xl_lo) // 2 // xl_step) * xl_step)
if "map_t" not in ss:
    ss.map_t = saved.get("map_t", int(t[len(t) // 2]))

# Jumps requested by clicks / "Go" buttons in the previous run
if "map_jump" in ss:
    ss.il, ss.xl = ss.pop("map_jump")
if "time_jump" in ss:
    ss.map_t = int(round(ss.pop("time_jump")))

# Keep everything inside the survey and inside the crop
ss.crop_il = snap_range(ss.crop_il, il_lo, il_hi, il_step)
ss.crop_xl = snap_range(ss.crop_xl, xl_lo, xl_hi, xl_step)
ss.crop_t = snap_range(ss.crop_t, t0, t1, dt)
(cil0, cil1), (cxl0, cxl1), (ct0, ct1) = ss.crop_il, ss.crop_xl, ss.crop_t
ss.il = snap(ss.il, cil0, cil1, il_step)
ss.xl = snap(ss.xl, cxl0, cxl1, xl_step)
ss.map_t = snap(ss.map_t, ct0, ct1, dt)
ss.il_box, ss.xl_box = ss.il, ss.xl

try:
    wells = wells_table(path)
except Exception:
    wells = None

with st.sidebar:
    def il_from_box():
        ss.il = snap(ss.il_box, cil0, cil1, il_step)

    def xl_from_box():
        ss.xl = snap(ss.xl_box, cxl0, cxl1, xl_step)

    def reset_crop():
        ss.crop_il, ss.crop_xl, ss.crop_t = (il_lo, il_hi), (xl_lo, xl_hi), (t0, t1)

    def crop_to_wells():
        if wells is not None and len(wells):
            m = ss.get("crop_margin", 40)
            ss.crop_il = (int(wells.inline.min()) - m * il_step, int(wells.inline.max()) + m * il_step)
            ss.crop_xl = (int(wells.crossline.min()) - m * xl_step, int(wells.crossline.max()) + m * xl_step)

    if wells is not None and len(wells):
        def go_to_well():
            w = ss.well_pick
            if w != "-":
                row = wells[wells.well == w].iloc[0]
                if not (cil0 <= row.inline <= cil1 and cxl0 <= row.crossline <= cxl1):
                    reset_crop()                       # well is outside the crop: show everything
                ss.il, ss.xl = int(row.inline), int(row.crossline)
        st.selectbox("📍 Jump to well", ["-"] + wells.well.tolist(), key="well_pick", on_change=go_to_well)

    # ----- ✂️ Crop: show only the area you need -----
    cropped = (ss.crop_il != (il_lo, il_hi) or ss.crop_xl != (xl_lo, xl_hi) or ss.crop_t != (t0, t1))
    with st.expander("✂️ Crop area" + ("  (on)" if cropped else ""), expanded=False):
        st.slider("Inlines", il_lo, il_hi, step=il_step, key="crop_il")
        st.slider("Crosslines", xl_lo, xl_hi, step=xl_step, key="crop_xl")
        st.slider("Time window (ms)", t0, t1, step=dt, key="crop_t")
        if wells is not None and len(wells):
            st.number_input("Margin around wells (lines)", 5, 400, 40, 5, key="crop_margin")
            st.button("🎯 Crop to the wells", on_click=crop_to_wells, use_container_width=True)
        st.button("↩️ Show the whole survey", on_click=reset_crop, use_container_width=True)

    st.divider()
    st.number_input(f"Inline ({cil0} to {cil1})", min_value=cil0, max_value=cil1,
                    step=il_step, key="il_box", on_change=il_from_box)
    st.slider("Inline slider", cil0, cil1, step=il_step, key="il", label_visibility="collapsed")
    st.number_input(f"Crossline ({cxl0} to {cxl1})", min_value=cxl0, max_value=cxl1,
                    step=xl_step, key="xl_box", on_change=xl_from_box)
    st.slider("Crossline slider", cxl0, cxl1, step=xl_step, key="xl", label_visibility="collapsed")

    st.divider()
    st.number_input(f"🗺️ Map time (ms, {ct0} to {ct1})", min_value=ct0, max_value=ct1, step=dt, key="map_t")
    dot = st.slider("Top view: dot size", 2, 12, 5)
    z_stretch = st.slider("Aerial view: wall height", 0.2, 1.5, 0.5, 0.1)
    colours = st.radio("Colours", ["Grey", "Red-Blue"], horizontal=True)
    st.divider()
    sensitivity = st.slider("Bright-spot sensitivity (lower = finds more)", 1.2, 5.0, 2.0, 0.1)

il, xl, map_t = ss.il, ss.xl, ss.map_t
ss.saved_pos = {"il": il, "xl": xl, "map_t": map_t}                      # remembered across pages
ss.saved_crop = {"il": ss.crop_il, "xl": ss.crop_xl, "t": ss.crop_t}
crop_il, crop_xl, crop_t = ss.crop_il, ss.crop_xl, ss.crop_t

# ---------- Read data (only the cropped area) ----------
with st.spinner("Reading seismic (the first time can take up to a minute)..."):
    il_pack = crop_section(section(path, "inline", il), crop_xl, crop_t)
    xl_pack = crop_section(section(path, "crossline", xl), crop_il, crop_t)
    slice_pack, pts = cropped_slice(path, map_t, crop_il, crop_xl)
il_data, il_pos, ts, il_used = il_pack
xl_data, xl_pos, _, xl_used = xl_pack
t_used = slice_pack[3]

try:
    cf = coef(path)                          # real X/Y available?
except Exception:
    cf = None

# Wells inside the crop only
wells_in = None
if wells is not None and len(wells):
    wells_in = wells[(wells.inline.between(*crop_il)) & (wells.crossline.between(*crop_xl))]
wxy = None
if cf is not None and wells_in is not None and len(wells_in):
    try:
        wxy = well_xy()
        wxy = wxy[wxy.well.astype(str).isin(wells_in.well)]
    except Exception:
        wxy = None

# Only show findings that belong to what is on screen now
spots_now = [s for s in ss.spots if s["time"] == t_used]
cands_il = [c for c in ss.cands if c["panel"] == "inline" and c["line"] == il_used]
cands_xl = [c for c in ss.cands if c["panel"] == "crossline" and c["line"] == xl_used]

if cropped:
    st.info(f"✂️ Cropped to inlines {crop_il[0]}-{crop_il[1]}, crosslines {crop_xl[0]}-{crop_xl[1]}, "
            f"{crop_t[0]}-{crop_t[1]} ms. Change it under ✂️ Crop area in the sidebar.")

# ---------- Map: Top view / Aerial view ----------
view = st.radio("View", ["🗺️ Top view", "🦅 Aerial view (3D)"], horizontal=True, label_visibility="collapsed")
top = view.startswith("🗺️")


def to_xy(IL, XL):
    (a0, a1, a2), (b0, b1, b2) = cf
    IL, XL = np.asarray(IL, float), np.asarray(XL, float)
    return a0 + a1 * IL + a2 * XL, b0 + b1 * IL + b2 * XL


if top and cf is not None and pts is not None:
    fig = birdseye_figure(pts, to_xy([il, il], crop_xl), to_xy(crop_il, [xl, xl]),
                          tuple(float(v) for v in to_xy(il, xl)), wxy, il, xl, colours, dot)
    fig = add_findings_to_map(fig, spots_now, ss.cands, cf)
    st.caption("🖱️ Click anywhere on the map to jump there. Pink circles = bright spots, red lines = AI findings.")
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
    st.info("This volume has no real-world X/Y in its headers, so the map is shown in inline/crossline numbers.")
    grid, uil, uxl, _ = slice_pack
    event = st.plotly_chart(map_figure(grid, uil, uxl, t_used, il, xl, colours, wells_in),
                            use_container_width=True, key="lab_grid", on_select="rerun", selection_mode="points")
    sel = event.selection.points if event and event.selection else []
    if sel:
        click = (round(sel[0]["y"]), round(sel[0]["x"]))
        if click != ss.get("last_click"):
            ss.last_click = click
            ss.map_jump = click
            st.rerun()
elif cf is not None:
    st.caption("🖱️ Drag to rotate, scroll to zoom. Move the slices with the sidebar.")
    st.plotly_chart(aerial_figure(slice_pack, il_pack, xl_pack, cf, wxy, crop_t, colours, z_stretch),
                    use_container_width=True, key="lab_3d")
else:
    st.info("The 3D view needs real-world X/Y in the trace headers, and this volume has none.")

# ---------- Section panels (with AI boxes) ----------
il_wells, xl_wells = [], []
if wells_in is not None:
    il_wells = [(r.well, r.crossline) for r in wells_in.itertuples() if abs(r.inline - il_used) <= 10]
    xl_wells = [(r.well, r.inline) for r in wells_in.itertuples() if abs(r.crossline - xl_used) <= 10]

left, right = st.columns(2)
with left:
    f1 = section_figure(il_data, il_pos, ts, f"Inline {il_used}", "Crossline",
                        colours, il_wells, xl_used, f"CL {xl_used}", t_used)
    st.plotly_chart(add_candidate_boxes(f1, cands_il, "inline"), use_container_width=True)
with right:
    f2 = section_figure(xl_data, xl_pos, ts, f"Crossline {xl_used}", "Inline",
                        colours, xl_wells, il_used, f"IL {il_used}", t_used)
    st.plotly_chart(add_candidate_boxes(f2, cands_xl, "crossline"), use_container_width=True)

# =====================================================================
#                      🤖 ASK THE AI ABOUT THIS SEISMIC
# =====================================================================
st.divider()
st.subheader("🤖 Ask the AI about this seismic")
if cropped:
    st.caption("The AI only looks at the cropped area, so it focuses on what matters and answers faster.")

# ----- Bright-spot auto-scan (inside the crop) -----
c1, c2 = st.columns([1, 3])
with c1:
    if st.button("🔎 Scan this time slice for bright spots", use_container_width=True):
        ss.spots = scan_bright_spots(slice_pack, sensitivity)
        spots_now = ss.spots
        if not spots_now:
            st.info("No bright spots found at this time. Try another map time or a lower sensitivity.")
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


def go_to_candidate(c):
    mid_x, mid_t = (c["from"] + c["to"]) / 2, (c["t_from"] + c["t_to"]) / 2
    ss.map_jump = (c["line"], mid_x) if c["panel"] == "inline" else (mid_x, c["line"])
    ss.time_jump = mid_t


# ----- Chat history -----
for i, msg in enumerate(ss.chat):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("cands"):
            st.markdown("**Features marked on the sections** (red dashed boxes, red lines on the map):")
            for c in msg["cands"]:
                a, b = st.columns([5, 1])
                where = (f"inline {c['line']}, crosslines {c['from']:.0f}-{c['to']:.0f}" if c["panel"] == "inline"
                         else f"crossline {c['line']}, inlines {c['from']:.0f}-{c['to']:.0f}")
                a.markdown(f"**{c['id']}: {c['type']}** ({c['confidence']} confidence): {where}, "
                           f"{c['t_from']:.0f}-{c['t_to']:.0f} ms. {c['reason']}")
                if b.button("Go", key=f"go_{i}_{c['id']}"):
                    go_to_candidate(c)
                    st.rerun()

if ss.chat and st.button("🗑️ Clear AI chat and markings"):
    ss.chat, ss.cands, ss.spots = [], [], []
    st.rerun()

# ----- Quick questions + free question -----
st.caption("Try:")
q = None
qc = st.columns(3)
for col, text in zip(qc, ["What traps can you see here?",
                          "Are there any other possible traps?",
                          "Explain the bright spots and the risks."]):
    if col.button(text, use_container_width=True):
        q = text
q = st.chat_input("Ask about this seismic, e.g. 'Is there a fault trap on this inline?'") or q

if q:
    if not spots_now:                                  # give the AI the bright-spot scan too
        ss.spots = scan_bright_spots(slice_pack, sensitivity)
        spots_now = ss.spots
    history = ""
    for msg in ss.chat[-4:]:
        history += f"\n{'User' if msg['role'] == 'user' else 'You (earlier)'}: {msg['content'][:600]}"
    if history:
        history = "\nEARLIER IN THIS CONVERSATION:" + history + "\n"

    ss.chat.append({"role": "user", "content": q})
    with st.spinner("The AI is looking at the map and both sections..."):
        try:
            png = views_png(il_pack, xl_pack, slice_pack, spots_now)
            raw = ask_with_image(SYSTEM, build_prompt(q, il_used, xl_used, t_used, spots_now, history), png)
        except Exception as e:
            ss.chat.append({"role": "assistant", "content": f"⚠️ Couldn't reach the AI model: {e}"})
            st.rerun()

    answer, cands = split_candidates(raw, (il_pos[0], il_pos[-1]), (xl_pos[0], xl_pos[-1]), (ts[0], ts[-1]))
    start = len(ss.cands)
    for k, c in enumerate(cands, start=1):
        c["id"] = f"T{start + k}"
        c["line"] = il_used if c["panel"] == "inline" else xl_used
    ss.cands = ss.cands + cands
    ss.chat.append({"role": "assistant", "content": answer or "(no answer)", "cands": cands})
    st.rerun()