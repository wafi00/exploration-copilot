import io, re
from pypdf import PdfReader


def read_pdf(file_bytes):
    """Extract text from each page. Returns [{'page': 1, 'text': '...'}, ...]."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, p in enumerate(reader.pages, start=1):
        text = (p.extract_text() or "").strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


def _words(s):
    return set(w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2)


def build_context(pages, question, k=5, max_chars=12000):
    """Pick the pages most related to the question. Returns (text for Gemma, page numbers used)."""
    if sum(len(p["text"]) for p in pages) <= max_chars:
        chosen = pages                                     # short PDF: send all of it
    else:
        qw = _words(question)
        ranked = sorted(pages, key=lambda p: len(qw & _words(p["text"])), reverse=True)
        chosen, total = [], 0
        for p in ranked[:k]:
            t = p["text"][:4000]
            if total + len(t) > max_chars:
                break
            chosen.append({"page": p["page"], "text": t})
            total += len(t)
        chosen.sort(key=lambda p: p["page"])
    ctx = "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in chosen)
    return ctx, [p["page"] for p in chosen]

    # ---------- Read a page as a picture (for figures, maps, tables, scanned PDFs) ----------
import fitz  # PyMuPDF


def page_image(file_bytes, page_no, zoom=2):
    """Turn one page into a PNG picture. Returns None if the page doesn't exist."""
    with fitz.open(stream=file_bytes, filetype="pdf") as d:
        if not 1 <= page_no <= len(d):
            return None
        pix = d[page_no - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")


def find_page_number(question):
    """Find 'page 5', 'p. 5' or 'pg 5' in the question. Returns 5, or None."""
    m = re.search(r"\b(?:page|pg\.?|p\.)\s*(\d+)", question, re.I)
    return int(m.group(1)) if m else None