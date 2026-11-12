import numpy as np
import plotly.graph_objects as go

COLOURS = {"Grey": "Greys", "Red-Blue": "RdBu"}


# ================= SECTION PANELS =================
def _thin(data, pos, t, max_cols=700, max_rows=900):
    """Skip some points if the section is huge, so the browser stays fast."""
    cs = max(1, data.shape[1] // max_cols)
    rs = max(1, data.shape[0] // max_rows)
    return data[::rs, ::cs], pos[::cs], t[::rs]


def section_figure(data, pos, t, title, xlabel, colours="Grey",
                   wells=(), cross_at=None, cross_label="", time_line=None):
    """Interactive seismic section with hover, wells, intersection line and time line."""
    clip = float(np.percentile(np.abs(data), 98)) or 1.0
    z, x, y = _thin(data, pos, t)

    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y,
        colorscale=COLOURS[colours], zmin=-clip, zmax=clip, showscale=False,
        hovertemplate=f"{xlabel}: %{{x}}<br>TWT: %{{y}} ms<br>Amplitude: %{{z:.3g}}<extra></extra>",
    ))

    for name, p in wells:
        fig.add_vline(x=p, line=dict(color="orange", width=1.5, dash="dot"),
                      annotation_text=name, annotation_position="top",
                      annotation_font_color="orange")
    if cross_at is not None:
        fig.add_vline(x=cross_at, line=dict(color="#4DA3FF", width=1.5),
                      annotation_text=cross_label, annotation_position="bottom")
    if time_line is not None:
        fig.add_hline(y=time_line, line=dict(color="yellow", width=1.5))

    fig.update_yaxes(autorange="reversed", title="TWT (ms)")
    fig.update_xaxes(title=xlabel)
    fig.update_layout(title=title, height=620, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ================= TOP VIEW (VIDEO-GAME MAP) =================
def birdseye_figure(m, il_xy, xl_xy, here, wells_xy, il, xl, colours="Grey", dot=5):
    """Game-style top-down map in real X/Y metres, north up."""
    clip = float(np.nanpercentile(np.abs(m["v"]), 98)) or 1.0

    fig = go.Figure(go.Scattergl(
        x=m["x"], y=m["y"], mode="markers",
        marker=dict(color=m["v"], colorscale=COLOURS[colours], cmin=-clip, cmax=clip,
                    size=dot, symbol="square"),
        customdata=np.stack([m["il"], m["xl"]], axis=1),
        hovertemplate=("Inline: %{customdata[0]}<br>Crossline: %{customdata[1]}<br>"
                       "X: %{x:,.0f} m<br>Y: %{y:,.0f} m<br><i>Click to view here</i><extra></extra>"),
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(x=il_xy[0], y=il_xy[1], mode="lines",
                             line=dict(color="#4DA3FF", width=3), name=f"Inline {il}", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xl_xy[0], y=xl_xy[1], mode="lines",
                             line=dict(color="#00C2A8", width=3), name=f"Crossline {xl}", hoverinfo="skip"))

    if wells_xy is not None and len(wells_xy):
        fig.add_trace(go.Scatter(
            x=wells_xy["x"], y=wells_xy["y"], text=wells_xy["well"],
            mode="markers+text", textposition="top center",
            marker=dict(size=10, color="orange", line=dict(color="black", width=1)),
            textfont=dict(color="orange", size=12), name="Wells",
            hovertemplate="Well %{text}<br>X: %{x:,.0f} m<br>Y: %{y:,.0f} m<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=[here[0]], y=[here[1]], mode="markers", name="You are here",
        marker=dict(size=20, color="red", symbol="star", line=dict(color="white", width=2)),
        hovertemplate=f"You are here<br>Inline {il}, Crossline {xl}<extra></extra>",
    ))

    fig.add_annotation(text="▲<br><b>N</b>", xref="paper", yref="paper", x=0.99, y=0.98,
                       showarrow=False, font=dict(size=18))

    fig.update_xaxes(title="X (m)")
    fig.update_yaxes(title="Y (m)", scaleanchor="x", scaleratio=1)
    fig.update_layout(title=f"🗺️ Top view: time slice at {m['t']:.0f} ms",
                      height=600, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.12))
    return fig


# ================= AERIAL VIEW (3D) =================
def _hover_text(IL, XL):
    a = np.char.add("Inline ", IL.astype(int).astype(str))
    b = np.char.add("<br>Crossline ", XL.astype(int).astype(str))
    return np.char.add(a, b).tolist()


def aerial_figure(slice_pack, il_pack, xl_pack, coef, wells_xy, t_range, colours="Grey", z_stretch=0.5):
    """3D 'chair display': time slice as the ground, inline + crossline as walls, wells as poles."""
    grid, uil, uxl, t_used = slice_pack
    il_data, il_pos, ts, il_used = il_pack
    xl_data, xl_pos, _, xl_used = xl_pack
    cx, cy = coef

    def to_xy(IL, XL):
        return cx[0] + cx[1] * IL + cx[2] * XL, cy[0] + cy[1] * IL + cy[2] * XL

    clip = float(np.nanpercentile(np.abs(il_data), 98)) or 1.0
    look = dict(colorscale=COLOURS[colours], cmin=-clip, cmax=clip, showscale=False)
    tip = "%{text}<br>Time: %{z:.0f} ms<extra></extra>"

    # 1. The ground: time slice
    r, c = max(1, grid.shape[0] // 150), max(1, grid.shape[1] // 150)
    g = grid[::r, ::c]
    IL, XL = np.meshgrid(uil[::r], uxl[::c], indexing="ij")
    X, Y = to_xy(IL, XL)
    Z = np.where(np.isnan(g), np.nan, t_used)
    fig = go.Figure(go.Surface(x=X, y=Y, z=Z, surfacecolor=np.nan_to_num(g),
                               text=_hover_text(IL, XL), hovertemplate=tip, **look))

    # 2. Wall 1: the inline
    r, c = max(1, il_data.shape[0] // 250), max(1, il_data.shape[1] // 250)
    T, XLs = ts[::r], il_pos[::c]
    XL2 = np.tile(XLs, (len(T), 1))
    IL2 = np.full_like(XL2, il_used)
    X, Y = to_xy(IL2, XL2)
    Z = np.tile(T[:, None], (1, len(XLs)))
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=il_data[::r, ::c],
                             text=_hover_text(IL2, XL2), hovertemplate=tip, **look))

    # 3. Wall 2: the crossline
    r, c = max(1, xl_data.shape[0] // 250), max(1, xl_data.shape[1] // 250)
    T, ILs = ts[::r], xl_pos[::c]
    IL3 = np.tile(ILs, (len(T), 1))
    XL3 = np.full_like(IL3, xl_used)
    X, Y = to_xy(IL3, XL3)
    Z = np.tile(T[:, None], (1, len(ILs)))
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=xl_data[::r, ::c],
                             text=_hover_text(IL3, XL3), hovertemplate=tip, **look))

    # 4. Wells as orange poles with names on top
    if wells_xy is not None and len(wells_xy):
        for _, w in wells_xy.iterrows():
            fig.add_trace(go.Scatter3d(
                x=[w["x"], w["x"]], y=[w["y"], w["y"]], z=[t_range[0], t_range[1]],
                mode="lines", line=dict(color="orange", width=6), showlegend=False,
                hovertemplate=f"Well {w['well']}<extra></extra>"))
            fig.add_trace(go.Scatter3d(
                x=[w["x"]], y=[w["y"]], z=[t_range[0]], mode="text", text=[str(w["well"])],
                textfont=dict(color="orange", size=13), showlegend=False, hoverinfo="skip"))

    # Keep the true shape: same scale for X and Y
    IL_c = np.array([uil[0], uil[0], uil[-1], uil[-1]], float)
    XL_c = np.array([uxl[0], uxl[-1], uxl[0], uxl[-1]], float)
    xs, ys = to_xy(IL_c, XL_c)
    dx, dy = np.ptp(xs) or 1.0, np.ptp(ys) or 1.0
    mx = max(dx, dy)

    fig.update_layout(
        title=f"🦅 Aerial view: inline {il_used}, crossline {xl_used}, time slice {t_used:.0f} ms",
        height=680, margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)",
            zaxis=dict(title="TWT (ms)", autorange="reversed"),
            aspectmode="manual", aspectratio=dict(x=dx / mx, y=dy / mx, z=z_stretch),
            camera=dict(eye=dict(x=1.4, y=-1.4, z=0.9)),
        ),
    )
    return fig

    # ================= GRID MAP (backup when a volume has no X/Y) =================
def map_figure(grid, uil, uxl, time_used, il, xl, colours="Grey", wells=None):
    """Map view in inline/crossline numbers. Click anywhere to jump there."""
    clip = float(np.nanpercentile(np.abs(grid), 98)) or 1.0
    rs = max(1, grid.shape[0] // 500)
    cs = max(1, grid.shape[1] // 500)
    z, y, x = grid[::rs, ::cs], uil[::rs], uxl[::cs]

    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=COLOURS[colours],
                               zmin=-clip, zmax=clip, showscale=False, hoverinfo="skip"))

    gx, gy = np.meshgrid(np.linspace(uxl[0], uxl[-1], 80), np.linspace(uil[0], uil[-1], 80))
    fig.add_trace(go.Scatter(
        x=gx.ravel(), y=gy.ravel(), mode="markers",
        marker=dict(size=10, opacity=0), showlegend=False,
        hovertemplate="Inline: %{y:.0f}<br>Crossline: %{x:.0f}<br><i>Click to view here</i><extra></extra>",
    ))

    if wells is not None and len(wells):
        fig.add_trace(go.Scatter(
            x=wells["crossline"], y=wells["inline"], text=wells["well"],
            mode="markers+text", textposition="top center",
            marker=dict(size=9, color="orange", line=dict(color="black", width=1)),
            textfont=dict(color="orange"), showlegend=False,
            hovertemplate="Well %{text}<br>Inline: %{y}<br>Crossline: %{x}<extra></extra>",
        ))

    fig.add_hline(y=il, line=dict(color="#4DA3FF", width=2),
                  annotation_text=f"IL {il}", annotation_position="right")
    fig.add_vline(x=xl, line=dict(color="#4DA3FF", width=2),
                  annotation_text=f"CL {xl}", annotation_position="top")

    fig.update_xaxes(title="Crossline")
    fig.update_yaxes(title="Inline")
    fig.update_layout(title=f"Map view (inline/crossline grid): time slice at {time_used:.0f} ms",
                      height=520, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ================= AI FINDINGS ON THE PICTURES =================
RED = "#FF3B3B"
PINK = "#FF66CC"


def add_candidate_boxes(fig, cands, panel):
    """Draw the AI's suggested features as dashed red boxes on a section panel."""
    for c in cands:
        if c["panel"] != panel:
            continue
        fig.add_shape(type="rect", x0=c["from"], x1=c["to"], y0=c["t_from"], y1=c["t_to"],
                      line=dict(color=RED, width=2, dash="dash"))
        fig.add_annotation(x=(c["from"] + c["to"]) / 2, y=min(c["t_from"], c["t_to"]),
                           text=f"<b>{c['id']}</b> {c['type']}", showarrow=False, yshift=12,
                           font=dict(color=RED, size=13), bgcolor="rgba(0,0,0,0.5)")
    return fig


def add_findings_to_map(fig, spots, cands, coef):
    """Top view: bright spots as pink circles, AI features as red lines along the section they're on."""
    cx, cy = coef

    def to_xy(IL, XL):
        IL, XL = np.asarray(IL, float), np.asarray(XL, float)
        return cx[0] + cx[1] * IL + cx[2] * XL, cy[0] + cy[1] * IL + cy[2] * XL

    if spots:
        x, y = to_xy([s["inline"] for s in spots], [s["crossline"] for s in spots])
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers+text", name="Bright spots (auto-scan)",
            text=[s["id"] for s in spots], textposition="middle right",
            marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color=PINK, width=3)),
            textfont=dict(color=PINK, size=13),
            customdata=[[s["inline"], s["crossline"], s["strength"]] for s in spots],
            hovertemplate=("Bright spot %{text}<br>Inline %{customdata[0]}, Crossline %{customdata[1]}"
                           "<br>%{customdata[2]}x stronger than typical reflections<extra></extra>"),
        ))
    for c in cands:
        if c["panel"] == "inline":                       # feature lies along an inline
            x, y = to_xy([c["line"], c["line"]], [c["from"], c["to"]])
        else:                                            # feature lies along a crossline
            x, y = to_xy([c["from"], c["to"]], [c["line"], c["line"]])
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+text", name=f"{c['id']} {c['type']}", text=[c["id"], ""],
            textposition="top center", line=dict(color=RED, width=6), textfont=dict(color=RED, size=13),
            hovertemplate=f"{c['id']}: {c['type']} ({c['confidence']} confidence)<extra></extra>",
        ))
    return fig