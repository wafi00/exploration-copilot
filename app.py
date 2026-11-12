"""Exploration Copilot - Streamlit demo app (tabs 1-5).

Reads only the master/ folder through CopilotAPI. No seismic is read here.
Run:  python -m streamlit run app.py --server.port 8601
"""
import sys, json, os
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.insert(0, "/data/epic20/APGCE/apgce_team_01")   # folder with copilot_api.py
from copilot_api import CopilotAPI

MASTER = "/data/epic20/APGCE/apgce_team_01/copilot/master"
IMAGES = os.path.join(MASTER, "images")

st.set_page_config(page_title="Exploration Copilot", layout="wide")


# ======================================================================
#                              HELPERS
# ======================================================================
@st.cache_resource
def get_api():
    api = CopilotAPI(MASTER)
    # Team B's knowledge ("training"): add any part that master/knowledge.json is missing (in memory only)
    try:
        import copilot_knowledge as ck
        sp = api.k.get("system_prompt", "")
        add = [block for marker, block in (("TRAP TYPES", ck.TRAP_CATALOGUE), ("EXTRA CAUTIONS", ck.TRAP_CAUTIONS))
               if marker not in sp]
        if add:
            api.k["system_prompt"] = sp + "\n\n" + "\n\n".join(add)
    except Exception:
        pass
    return api


@st.cache_data
def load_demo_answers():
    path = os.path.join(MASTER, "demo_answers.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def safe(fn, default=None):
    """Run fn(); return default instead of crashing."""
    try:
        return fn()
    except Exception:
        return default


def show_image(path, caption=None, width=None):
    if path and os.path.isfile(path):
        if width:
            st.image(path, caption=caption, width=width)
        else:
            st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"Picture not available: {os.path.basename(path) if path else '(none)'}")


def val(row, col, default="n/a"):
    """Value from a pandas row, or 'n/a' when the column is missing or empty."""
    if row is None or col not in row.index:
        return default
    v = row[col]
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return default
    return v


BADGE = {"HIGH": "#2e7d32", "MEDIUM": "#ef8f00", "LOW": "#757575",
         "debated": "#ef8f00", "agreed": "#2e7d32", "combined": "#1565c0",
         "automated": "#546e7a", "dry": "#757575"}


def badge(label, value):
    colour = BADGE.get(str(value), BADGE.get(str(value).upper(), "#455a64"))
    return (f"<span style='background:{colour};color:white;padding:3px 10px;border-radius:12px;"
            f"font-size:0.85rem;margin-right:6px'>{label}: <b>{value}</b></span>")


def colour_cell(v):
    s = str(v)
    if s in ("HIGH", "MEDIUM", "LOW", "debated"):
        return f"background-color:{BADGE[s]};color:white"
    return ""


def styled(df):
    cols = [c for c in df.columns if "trust" in c.lower() or "status" in c.lower()]
    if not cols:
        return df
    sty = df.style
    try:
        return sty.map(colour_cell, subset=cols)          # pandas >= 2.1
    except AttributeError:
        return sty.applymap(colour_cell, subset=cols)


def rejected_mask(df):
    m = pd.Series(False, index=df.index)
    if "trap_test" in df.columns:
        m |= df["trap_test"].astype(str).str.lower().str.contains("drop", na=False)
    if "qc_flag" in df.columns:
        m |= df["qc_flag"].astype(str).str.lower().str.contains("not ranked", na=False)
    return m


ss = st.session_state
ss.setdefault("answers", {})     # (tab, well, question) -> markdown answer
ss.setdefault("chats", {})       # conversation key -> list of {"role", "content"}


def run_ai(key, call, demo_key=None):
    """Button-triggered AI call. Uses the cached demo answer when Demo mode is on."""
    demo = load_demo_answers()
    if ss.demo_mode and demo_key and demo_key in demo:
        ss.answers[key] = demo[demo_key]
        return
    with st.spinner("The AI is thinking..."):
        try:
            ss.answers[key] = call()
        except Exception:
            if demo_key and demo_key in demo:
                st.error("The AI is busy, showing the saved answer")
                ss.answers[key] = demo[demo_key]
            else:
                st.error("The AI is busy right now. Please try again in a moment.")


def follow_up(chat_key, first_q, first_a, send):
    """Follow-up chat under an answer. send(question, history) -> answer."""
    msgs = ss.chats.get(chat_key)
    if not msgs or msgs[0]["content"] != first_q or msgs[1]["content"] != first_a:
        msgs = [{"role": "user", "content": first_q}, {"role": "assistant", "content": first_a}]
        ss.chats[chat_key] = msgs
    for m in msgs[2:]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    with st.form(f"form_{chat_key}", clear_on_submit=True):
        q = st.text_input("Follow-up question", placeholder="e.g. Why not structural?",
                          label_visibility="collapsed")
        c1, c2 = st.columns([1, 1])
        sent = c1.form_submit_button("Ask follow-up", use_container_width=True)
        reset = c2.form_submit_button("New conversation", use_container_width=True)
    if reset:
        ss.chats[chat_key] = msgs[:2]
        st.rerun()
    if sent and q.strip():
        with st.spinner("The AI is thinking..."):
            try:
                a = send(q.strip(), list(msgs))
            except Exception:
                a = "The AI is busy right now. Please try again in a moment."
        msgs += [{"role": "user", "content": q.strip()}, {"role": "assistant", "content": a}]
        st.rerun()


def report_button(name, parts, key):
    text = "\n\n".join(p for p in parts if p)
    st.download_button("Download report (.md)", text, file_name=f"{name}.md",
                       mime="text/markdown", key=key)


# ======================================================================
#                              LOAD
# ======================================================================
try:
    api = get_api()
except Exception as e:
    st.error(f"Could not load the copilot data from {MASTER}: {e}")
    st.stop()

wells_df = safe(lambda: api.wells, pd.DataFrame())
well_names = [str(w) for w in wells_df.index] if wells_df is not None else []
survey = safe(api.survey_leads, pd.DataFrame())
if survey is None:
    survey = pd.DataFrame()
leads = safe(lambda: api.leads, pd.DataFrame())
if leads is None:
    leads = pd.DataFrame()
if "risked_area_km2" in leads.columns:
    leads = leads.sort_values("risked_area_km2", ascending=False)
recommendation = safe(api.recommend_leads, "") or ""
knowledge = safe(lambda: api.k, {}) or {}
test_wells = [str(w) for w in (knowledge.get("test_wells") or [])]


def well_row(w):
    try:
        return wells_df.loc[w]
    except Exception:
        return None


def is_dry(w):
    return str(val(well_row(w), "final_verdict", "")).strip().lower() == "dry"


# ======================================================================
#                              SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown("### Exploration Copilot")
    st.toggle("Demo mode", value=True, key="demo_mode",
              help="Shows saved answers instantly for the demo questions. Turn off to ask Gemma live.")
    st.caption("How to use: start on Field overview, pick a well in Well copilot, "
               "then follow Where else? and Leads to mature.")
    if not load_demo_answers():
        st.caption("Saved demo answers not available: every question goes to the live AI.")


# ======================================================================
#                              TABS
# ======================================================================
st.title("Exploration Copilot")
st.markdown("An AI geoscientist that reasons from evidence, says how much to trust itself, "
            "and tells you what to check.")

t1, t2, t3, t4, t5, t6 = st.tabs(["Field overview", "Well copilot", "Where else?",
                                  "Leads to mature", "How we checked the AI", "Seismic viewer"])

# ---------------------------------------------------------------- 1. FIELD OVERVIEW
with t1:
    n_wells = len(wells_df) if wells_df is not None else 0
    n_disc = sum(1 for w in well_names if not is_dry(w)) if "final_verdict" in wells_df.columns else "n/a"
    n_look = int((~rejected_mask(survey)).sum()) if len(survey) else "n/a"
    if len(leads) and "lead" in leads.columns:
        top = leads.iloc[0]
        top_txt = str(top["lead"])
        top_sub = f"{top['risked_area_km2']:.2f} km² risked" if "risked_area_km2" in leads.columns else None
    else:
        top_txt, top_sub = "n/a", None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Wells analysed", n_wells)
    m2.metric("Discoveries", n_disc)
    m3.metric("Undrilled look-alikes found", n_look)
    m4.metric("Top lead", top_txt, top_sub, delta_color="off")

    show_image(os.path.join(IMAGES, "survey_bodies.png"),
               "Whole survey screened in under a minute. Green = drilled discoveries, "
               "red = undrilled look-alikes.")

    st.subheader("Verdict for every well")
    table = safe(api.table)
    if table is not None and len(table):
        st.dataframe(styled(table), use_container_width=True)
    else:
        st.info("Verdict table not available.")

# ---------------------------------------------------------------- 2. WELL COPILOT
with t2:
    if not well_names:
        st.info("Well list not available.")
    else:
        well = st.selectbox("Well", well_names,
                            index=well_names.index("no-1") if "no-1" in well_names else 0, key="well_pick")
        row = well_row(well)

        blind = False
        if well in test_wells:
            blind = st.toggle("Blind test (hide Team B's interpretation from the AI)", key=f"blind_{well}")

        # Verdict card
        with st.container(border=True):
            st.markdown(f"## {str(val(row, 'final_verdict')).capitalize()}")
            st.markdown(badge("Status", val(row, "status")) +
                        badge("Evidence trust", val(row, "evidence_quality_trust")) +
                        badge("Trap-type confidence", val(row, "trap_type_confidence")),
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.markdown(f"**Pay:** {val(row, 'pay_fluid')} · net gas {val(row, 'net_gas_m')} m · "
                        f"net oil {val(row, 'net_oil_m')} m · pay MD {val(row, 'pay_md')}")
            c2.markdown(f"**Location:** inline {val(row, 'il')} · crossline {val(row, 'xl')} · "
                        f"{val(row, 'pay_twt_ms')} ms")

        qc = str(val(row, "qc_checks", ""))
        for check in [c.strip() for c in qc.split(" | ") if c.strip()]:
            st.warning(check)

        p1, p2 = st.columns(2)
        with p1:
            show_image(safe(lambda: api.image_path(well)), f"{well}: trap picture")
        with p2:
            show_image(os.path.join(IMAGES, f"faults_{well}.png"), f"{well}: faults and channel edges")

        # Questions
        st.subheader(f"Ask the copilot about {well}")
        QUESTIONS = ["What is the trap here and why?", "What would change your mind?",
                     "What should I check first in DUG?"]
        qcols = st.columns(3)
        asked = None
        for col, text in zip(qcols, QUESTIONS):
            if col.button(text, use_container_width=True, key=f"q_{well}_{text}"):
                asked = text
        with st.form(f"free_{well}", clear_on_submit=True):
            free = st.text_input("Your own question", placeholder="Ask anything about this well")
            if st.form_submit_button("Ask") and free.strip():
                asked = free.strip()

        if asked:
            demo_key = f"{well} trap" if asked == QUESTIONS[0] and not blind else None
            run_ai(("well", well, "last"),
                   lambda: api.ask(well, asked, test_mode=blind), demo_key)
            ss.answers[("well", well, "last_q")] = asked

        ans = ss.answers.get(("well", well, "last"))
        last_q = ss.answers.get(("well", well, "last_q"))
        if ans:
            with st.chat_message("user"):
                st.markdown(last_q)
            with st.chat_message("assistant"):
                st.markdown(ans)

            def send_well(q, history):
                if hasattr(api, "chat"):
                    return api.chat(well, q, history, test_mode=blind)
                return api.ask(well, q, test_mode=blind)

            follow_up(f"well_{well}", last_q, ans, send_well)

        # Debate
        st.subheader("Debate between two AI geoscientists")
        if st.button("Start the debate", key=f"debate_{well}"):
            run_ai(("debate", well, "debate"), lambda: api.debate(well), f"{well} debate")
        debate = ss.answers.get(("debate", well, "debate"))
        if debate:
            with st.container(border=True):
                st.markdown(debate)
            ev = safe(lambda: api.evidence(well, test_mode=blind), "") or ""
            follow_up(f"debate_{well}", "Debate", debate,
                      lambda q, h: api.followup(debate + "\n\n" + ev, q, h))

        with st.expander("Evidence the AI used"):
            ev = safe(lambda: api.evidence(well, test_mode=blind))
            if ev:
                st.code(ev, language=None)
            else:
                st.info("Evidence not available.")

        with st.expander("Human vs code"):
            h, c = st.columns(2)
            h.markdown("**Team B (human interpretation)**")
            for col in ["teamB_trap", "teamB_confidence", "teamB_family", "teamB_use"]:
                h.markdown(f"- {col}: {val(row, col)}")
            c.markdown("**Code trap test**")
            for col in ["code_verdict", "closure_ms", "anomaly_edge_std_ms", "anomaly_inside_closure_pct",
                        "t2_nearest_fault_m", "t2_nearest_fault_dir", "t2_contact",
                        "t2_contact_conformance_iou", "t2_trap_elements"]:
                c.markdown(f"- {col}: {val(row, col)}")

        with st.expander("All columns for this well"):
            if row is not None:
                st.dataframe(row.to_frame("value").astype(str), use_container_width=True)

        report_button(f"copilot_{well}", [
            f"# Exploration Copilot report: {well}",
            f"**Verdict:** {val(row, 'final_verdict')} | status {val(row, 'status')} | "
            f"trust {val(row, 'evidence_quality_trust')} | confidence {val(row, 'trap_type_confidence')}",
            f"## Question\n{last_q}\n\n## Answer\n{ans}" if ans else "",
            f"## Debate\n{debate}" if debate else "",
            "## Evidence\n```\n" + (safe(lambda: api.evidence(well, test_mode=blind), "") or "") + "\n```",
        ], key=f"dl_{well}")

# ---------------------------------------------------------------- 3. WHERE ELSE?
with t3:
    st.markdown("Similar traps yet to be tested, found by screening the whole survey.")
    show_image(os.path.join(IMAGES, "survey_bodies.png"), width=650)

    if len(survey):
        show_rej = st.checkbox("Show rejected bodies", value=False)
        cols = [c for c in ["id", "analogue", "similarity_pct", "area_km2", "thick_ms", "trap_test",
                            "edge_tests_ms", "qc_flag", "dist_to_well_km"] if c in survey.columns]
        view = survey if show_rej else survey[~rejected_mask(survey)]
        st.dataframe(view[cols] if cols else view, use_container_width=True, hide_index=True)

        if "id" in survey.columns and len(view):
            body = st.selectbox("Show pictures for body", view["id"].astype(str).tolist())
            b1, b2 = st.columns(2)
            with b1:
                show_image(os.path.join(IMAGES, f"section_{body}.png"), f"{body}: section")
            with b2:
                show_image(os.path.join(IMAGES, f"map_{body}.png"), f"{body}: map")
    else:
        st.info("Survey look-alikes not available.")

    st.subheader("Find look-alikes of a discovery")
    disc = [w for w in well_names if not is_dry(w)] or well_names
    if disc:
        dw = st.selectbox("Discovery", disc, index=disc.index("s-1") if "s-1" in disc else 0, key="disc_pick")
        if st.button("Find look-alikes", key=f"sim_{dw}"):
            run_ai(("where", dw, "similar"), lambda: api.similar_to(dw), f"where else {dw}")
        sim = ss.answers.get(("where", dw, "similar"))
        if sim:
            with st.container(border=True):
                st.markdown(sim)
            follow_up(f"where_{dw}", "Find look-alikes", sim,
                      lambda q, h: api.followup(sim, q, h))
            report_button(f"lookalikes_{dw}", [f"# Look-alikes of {dw}", sim], key=f"dl_where_{dw}")

# ---------------------------------------------------------------- 4. LEADS TO MATURE
with t4:
    st.markdown("Risked area = chance of gas from the blind look-back × area. "
                "A weak signal cannot push a lead to the top.")
    if len(leads):
        top8 = leads.head(8).copy()
        cols = [c for c in ["lead", "analogue_well", "area_km2", "closure_ms", "trap", "p_hc_calibrated",
                            "risked_area_km2", "il", "xl", "twt_ms"] if c in top8.columns]
        cfg = {}
        if "p_hc_calibrated" in top8.columns:
            top8["p_hc_calibrated"] = pd.to_numeric(top8["p_hc_calibrated"], errors="coerce") * 100
            cfg["p_hc_calibrated"] = st.column_config.ProgressColumn(
                "Chance of gas/oil", min_value=0, max_value=100, format="%.0f%%")
        if "risked_area_km2" in top8.columns:
            cfg["risked_area_km2"] = st.column_config.NumberColumn("Risked area (km²)", format="%.2f")
        st.dataframe(top8[cols] if cols else top8, column_config=cfg,
                     use_container_width=True, hide_index=True)
    else:
        st.info("Leads table not available.")

    st.subheader("Recommendation")
    if st.button("Ask the AI: which leads should we mature first?", key="leads_ai"):
        run_ai(("leads", "-", "ai"),
               lambda: api.followup(recommendation, "Which leads should we mature first and why?", []),
               "leads")
    lead_ans = ss.answers.get(("leads", "-", "ai"))
    if lead_ans:
        with st.container(border=True):
            st.markdown(lead_ans)
        follow_up("leads", "Which leads should we mature first?", lead_ans,
                  lambda q, h: api.followup(lead_ans + "\n\n" + recommendation, q, h))
    if recommendation:
        st.markdown(recommendation)
    else:
        st.info("Recommendation not available.")

# ---------------------------------------------------------------- 5. HOW WE CHECKED THE AI
with t5:
    lookback = ""
    for para in recommendation.split("\n\n"):
        if "look-back" in para.lower() or "lookback" in para.lower() or "blind" in para.lower():
            lookback = para.strip()
            break
    show_image(os.path.join(IMAGES, "lookback.png"))
    if lookback:
        st.caption(lookback.replace("#", "").strip())

    st.subheader("QC principles")
    st.markdown(
        "- **Evidence only:** every answer cites numbered evidence like [E4]; no evidence, no claim.\n"
        "- **Two confidences:** how good the evidence is, and how sure the trap type is, shown separately.\n"
        "- **Two-point trap test:** the code tests every survey body from two points and keeps only calls that agree.\n"
        "- **Kill criteria are fixed rules, not AI:** the checklist that rejects a lead is geoscience rules in code.")

    st.warning("**Limitations:** thin pays under about 10 m are below seismic reliability. There are no sonic "
               "or density logs, so well ties use the sand pattern. Image-only answers stay at MEDIUM confidence at most.")

    st.subheader("Well data: the 13 analysed wells")
    show_image(os.path.join(IMAGES, "well_panel.png"), "Logs, sands, fluids and contacts for all wells")
    tie_cols = [c for c in ["tie_method", "tie_ms", "tie_corr", "checkshot", "moved_from_wellhead_m"]
                if c in wells_df.columns]
    if tie_cols:
        st.dataframe(wells_df[tie_cols], use_container_width=True)
    st.caption("New wells: add LAS, checkshot and deviation files to the project folder and re-run the "
               "pipeline; the app reads the result.")

    with st.expander("Glossary (for non-geoscientists)"):
        g = knowledge.get("glossary")
        if isinstance(g, dict):
            for term, meaning in g.items():
                st.markdown(f"- **{term}:** {meaning}")
        elif isinstance(g, list):
            for item in g:
                st.markdown(f"- {item}")
        elif g:
            st.markdown(str(g))
        else:
            st.info("Glossary not available.")

    with st.expander("Test questions used to check the AI"):
        tests = knowledge.get("tests")
        if isinstance(tests, list) and tests:
            for t in tests:
                st.markdown(f"- {t if not isinstance(t, dict) else json.dumps(t)}")
        else:
            st.info("Test questions not available.")

# ---------------------------------------------------------------- 6. SEISMIC VIEWER
with t6:
    try:
        import seismic_tab
    except Exception as e:
        st.error(f"The seismic viewer could not start: {e}")
    else:
        seismic_tab.render(api)
