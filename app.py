import streamlit as st
from llm import ask, ask_with_image, ask_stream, ask_with_image_stream
import re
from plots import list_wells, plot_well, list_seismic, seismic_inline_range, plot_seismic_inline
from docs import read_pdf, build_context, page_image, find_page_number

BASE_PROMPT = (
    "You are an exploration geoscientist assistant. "
    "You have no tools: never output JSON, function calls or tool calls. "
    "Always reply in plain text."
)

DRAW_WORDS = ["image", "diagram", "draw", "picture", "sketch", "illustrate", "visualise", "visualize"]

SVG_RULES = (
    "For this question, start your reply with ONE simple, clearly labelled diagram "
    "written as SVG code between <svg> and </svg>. Use width='700' height='400', "
    "flat colours, and readable text labels. After the SVG, explain the diagram in words."
)

def split_svg(text):
    """Separate the <svg>...</svg> drawing from the explanation text."""
    m = re.search(r"<svg.*?</svg>", text, re.S | re.I)
    if not m:
        return None, text
    rest = text[:m.start()] + text[m.end():]
    for fence in ["```svg", "```xml", "```html", "```"]:
        rest = rest.replace(fence, "")
    return m.group(0), rest.strip()

from llm import ask, ask_with_image
from plots import list_wells, plot_well

# ---------- 1. Page setup ----------
st.set_page_config(page_title="Exploration Copilot", page_icon="🪨", layout="wide")
st.title("🪨 Exploration Copilot")
st.caption("AI assistant that helps geoscientists interpret seismic and well data.")

# ---------- 2. Memory ----------
if "msgs" not in st.session_state:
    st.session_state.msgs = []
# ---------- Tone options ----------
TONES = {
    "🔬 Expert": "Answer like a senior geoscientist talking to another expert. Be precise and use proper technical terms.",
    "💼 Manager": "Answer for a business manager with no geoscience background. Start with a one-sentence bottom line, then explain the risk and what it means for the decision. Avoid jargon.",
    "🎓 Student": "Answer like a patient teacher. Explain technical terms in simple words and use an everyday comparison where it helps.",
    "⚡ Quick": "Answer in 3 bullet points or fewer. No introduction.",
}

@st.cache_data
def cached_range(path):
    return seismic_inline_range(path)
    
# ---------- 3. Sidebar ----------
tone = list(TONES.keys())[0]   # default style, in case the picker below isn't reached

with st.sidebar:

    if st.session_state.get("doc"):
        st.header("📄 Active document")
        st.caption(st.session_state.doc["name"])
        if st.button("Remove document", use_container_width=True):
            del st.session_state.doc
            st.rerun()
        st.divider()
    
    st.header("📊 Well log viewer")
    wells = list_wells()
    if wells:
        chosen_well = st.selectbox("Pick a well", list(wells.keys()))
        if st.button("Plot & explain", use_container_width=True):
            st.session_state.plot_request = chosen_well
    else:
        st.caption("No .las files found. Check DATA_DIR in plots.py.")
    st.divider()

    st.header("🌊 Seismic viewer")
    seis = list_seismic()
    if seis:
        chosen_seis = st.selectbox("Seismic volume", list(seis.keys()))
        try:
            lo, hi, step = cached_range(seis[chosen_seis])
            start = lo + ((hi - lo) // 2 // step) * step
            chosen_il = st.slider("Inline", lo, hi, start, step)
            if st.button("Show & explain section", use_container_width=True):
                st.session_state.seis_request = (chosen_seis, chosen_il)
        except Exception as e:
            st.caption(f"Couldn't read this volume's inlines: {e}")
    else:
        st.caption("No SEG-Y files found. Check DATA_DIR in plots.py.")
    st.divider()

    st.header("💡 Try asking")
    examples = [
        "Why do you think this is a stratigraphic trap instead of a structural trap?",
        "Which known fields resemble this prospect and why?",
        "What are the key geological risks that should be investigated next?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    st.divider()

    st.header("🗣️ Answer style")
    tone = st.radio("Who is reading?", list(TONES.keys()))
    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.msgs = []
        st.rerun()
        
        
    
# ---------- 4. Welcome message ----------
welcome = st.empty()
if not st.session_state.msgs:
    welcome.info("👋 Ask a question about a well or a lead, or pick an example from the sidebar.")

# ---------- 4b. If the last answer was stopped, keep what was written ----------
if st.session_state.get("generating"):
    partial = st.session_state.pop("partial", "")
    st.session_state.generating = False
    st.session_state.msgs.append(
        {"role": "assistant", "content": (partial or "...") + "\n\n*⏹ Stopped by user.*"}
    )
    
# ---------- 5. Chat history ----------
for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        if m.get("image"):
            st.image(m["image"], width=300)
        if m.get("svg"):
            st.markdown(m["svg"], unsafe_allow_html=True)
            st.caption("AI-generated sketch for illustration only. Not to scale, not from the dataset.")
        st.write(m["content"])
        
# ---------- 6. New question ----------
def remember(gen):
    """Pass the answer through, while saving it piece by piece in case Stop is clicked."""
    st.session_state.partial = ""
    for piece in gen:
        st.session_state.partial += piece
        yield piece


inp = st.chat_input(
    "Ask a question... (📎 attach an image or PDF)",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "pdf"],
)
pending = st.session_state.pop("pending", None)

# Work out the question text and the attachment (if any)
if inp:
    q = inp.text
    upload = inp.files[0] if inp.files else None
elif pending:
    q, upload = pending, None
else:
    q, upload = None, None

img, pdf_name = None, None
if upload is not None:
    if upload.name.lower().endswith(".pdf"):
        pdf_bytes = upload.getvalue()
        pages = read_pdf(pdf_bytes)
        st.session_state.doc = {"name": upload.name, "pages": pages, "bytes": pdf_bytes}
        pdf_name = upload.name
        if not pages:
            st.info("📷 No text found. This looks like a scanned PDF, so I'll read the pages as pictures. "
                    "Mention a page number in your question, e.g. 'page 3'.")
        if not q:
            q = "Give me a short summary of this document." if pages else "Describe page 1 of this document."
    else:
        img = upload
        if not q:
            q = "Describe this image."

if q:
    welcome.empty()
    img_bytes = img.getvalue() if img else None
    shown_q = q + (f"\n\n📄 *{pdf_name}*" if pdf_name else "")

    st.session_state.msgs.append({"role": "user", "content": shown_q, "image": img_bytes})
    with st.chat_message("user"):
        if img_bytes:
            st.image(img_bytes, width=300)
        st.write(shown_q)

    system = BASE_PROMPT + " " + TONES[tone]
    wants_drawing = any(w in q.lower() for w in DRAW_WORDS) and not img_bytes
    svg, source_note, page_png = None, "", None

    # ----- If a PDF is active, decide: read a page as a picture, or search the text -----
    question_for_ai = q
    doc = st.session_state.get("doc")
    if doc and not img_bytes and not wants_drawing:
        page_no = find_page_number(q)
        if page_no is None and not doc["pages"]:
            page_no = 1                                   # scanned PDF with no page mentioned
        if page_no and doc.get("bytes"):
            page_png = page_image(doc["bytes"], page_no)

        if page_png:                                      # picture mode: sees figures, maps, tables
            page_text = next((p["text"] for p in doc["pages"] if p["page"] == page_no), "")
            question_for_ai = (
                f"This image is page {page_no} of the document '{doc['name']}', "
                "including any figures, maps and tables. "
                + (f"The extracted text of this page is:\n{page_text[:4000]}\n\n" if page_text else "")
                + f"Question: {q}\n"
                "Describe figures carefully and only say what you can actually see."
            )
            source_note = f"📄 *Source: {doc['name']}, page {page_no} (read as image)*"
        elif doc["pages"]:                                # text mode: best matching pages
            ctx, used = build_context(doc["pages"], q)
            question_for_ai = (
                f"Answer using the document '{doc['name']}' below. "
                "Cite page numbers like (p. 3). If the answer is not in the document, say so clearly.\n\n"
                f"{ctx}\n\nQuestion: {q}"
            )
            source_note = f"📄 *Source: {doc['name']}, pages {', '.join(map(str, used))}*"

    try:
        if wants_drawing:
            with st.spinner("Gemma is drawing..."):
                svg, a = split_svg(ask(system + " " + SVG_RULES, q))
            with st.chat_message("assistant"):
                if svg:
                    st.markdown(svg, unsafe_allow_html=True)
                    st.caption("AI-generated sketch for illustration only. Not to scale, not from the dataset.")
                st.write(a)
        else:
            stop_box = st.empty()
            stop_box.button("⏹ Stop generating")
            st.session_state.generating = True
            with st.chat_message("assistant"):
                if page_png:
                    st.image(page_png, width=450, caption="Page the copilot looked at")
                if img_bytes:
                    gen = ask_with_image_stream(system, q, img_bytes, img.type)
                elif page_png:
                    gen = ask_with_image_stream(system, question_for_ai, page_png)
                else:
                    gen = ask_stream(system, question_for_ai)
                a = st.write_stream(remember(gen))
                if source_note:
                    st.caption(source_note)
            st.session_state.generating = False
            st.session_state.pop("partial", None)
            stop_box.empty()
    except Exception:
        st.session_state.generating = False
        st.error("⚠️ Couldn't reach the AI model. Please try again in a moment.")
        st.stop()

    # Safety net: hide fake tool calls
    if a.strip().startswith("{") and '"action"' in a:
        a = "Sorry, I couldn't do that. Try rephrasing, or use the Well log viewer in the sidebar."
        st.session_state.msgs.append({"role": "assistant", "content": a})
        st.rerun()

    if source_note:
        a = a + "\n\n" + source_note
    st.session_state.msgs.append({"role": "assistant", "content": a, "svg": svg, "image": page_png})
    
    # ---------- 7. Plot a well and explain it ----------
well_name = st.session_state.pop("plot_request", None)
if well_name:
    welcome.empty()
    q = f"Show and explain the logs for well {well_name}"
    st.session_state.msgs.append({"role": "user", "content": q})
    st.chat_message("user").write(q)

    with st.spinner("Drawing the plot and asking Gemma..."):
        try:
            png, curves = plot_well(wells[well_name])
            system = BASE_PROMPT + " " + TONES[tone]
            prompt = (
                f"This image shows well log curves ({', '.join(curves)}) for well {well_name}. "
                "Depth increases downward. Explain what the logs suggest, point out any "
                "interesting zones, and say what should be checked next. "
                "Only describe what you can actually see, and say clearly when you are unsure."
            )
            a = ask_with_image(system, prompt, png)
        except Exception as e:
            st.error(f"⚠️ Couldn't make the plot: {e}")
            st.stop()

    st.session_state.msgs.append({"role": "assistant", "content": a, "image": png})
    with st.chat_message("assistant"):
        st.image(png)
        st.write(a)

        # ---------- 8. Show a seismic section and explain it ----------
req = st.session_state.pop("seis_request", None)
if req:
    welcome.empty()
    seis_name, il_num = req
    q = f"Show and explain inline {il_num} of {seis_name}"
    st.session_state.msgs.append({"role": "user", "content": q})
    st.chat_message("user").write(q)

    with st.spinner("Reading the seismic and asking Gemma..."):
        try:
            png, il_used = plot_seismic_inline(seis[seis_name], il_num)
            system = BASE_PROMPT + " " + TONES[tone]
            prompt = (
                f"This image is a seismic inline section (inline {il_used}). "
                "The horizontal axis is crossline, the vertical axis is time or depth increasing downward, "
                "and the colours show reflection amplitude (red and blue are opposite polarities). "
                "Describe the visible features: continuous or broken reflectors, possible faults, "
                "folds or domes, and very bright or dim spots, and where in the image they are. "
                "Only describe what you can actually see, say clearly when you are unsure, "
                "and remind the user that a geoscientist should confirm the interpretation."
            )
            a = ask_with_image(system, prompt, png)
        except Exception as e:
            st.error(f"⚠️ Couldn't read the seismic: {e}")
            st.stop()

    st.session_state.msgs.append({"role": "assistant", "content": a, "image": png})
    with st.chat_message("assistant"):
        st.image(png)
        st.write(a)