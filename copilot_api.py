"""
Exploration Copilot: API for the app (IT teammate uses ONLY this file + the master folder)

No seismic reading here: everything comes from master/ (made by copilot_merge.export_master). Fast and simple.

    from copilot_api import CopilotAPI
    api = CopilotAPI("/data/epic20/APGCE/apgce_team_01/copilot/master")
    api.table()                                              # final verdict for every well
    print(api.ask("no-1", "What is the trap at no-1, and what should I check?"))
    print(api.debate("no-1"))
    print(api.recommend_leads())
    print(api.ask("p-1", "What is the trap here?", test_mode=True))   # scoring: hides Team B's answer for test wells
"""

import os, json, base64
import pandas as pd
import openai

GEMMA_URL = os.environ.get("GEMMA_URL", "http://<vllm-host>:8000/v1")
GEMMA_MODEL = "google/gemma-4-26B-A4B-it"


class CopilotAPI:
    def __init__(self, master_dir="/data/epic20/APGCE/apgce_team_01/copilot/master"):
        self.dir = master_dir
        self.k = json.load(open(f"{master_dir}/knowledge.json"))
        self.wells = pd.read_csv(f"{master_dir}/master_wells.csv").set_index("well")
        p = f"{master_dir}/leads.csv"
        self.leads = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
        self.client = openai.OpenAI(base_url=GEMMA_URL, api_key="token-not-needed")

    # ------------------------------------------------------------------ data for the screen
    def table(self):
        return self.wells[["final_verdict", "status", "evidence_quality_trust", "trap_type_confidence",
                           "pay_md", "pay_fluid", "net_gas_m", "net_oil_m", "il", "xl", "pay_twt_ms"]]

    def image_path(self, well):
        img = self.wells.loc[well, "image"]
        return f"{self.dir}/{img}" if isinstance(img, str) and img else None

    def evidence(self, well, test_mode=False, prefix="E"):
        """Numbered evidence: code measurements, then Team B's interpretation (hidden for test wells in test_mode)."""
        lines = open(f"{self.dir}/evidence/{well}.txt").read().splitlines()
        hide = test_mode and well in self.k.get("test_wells", [])
        tb = f"{self.dir}/evidence/{well}_teamB.txt"
        if not hide and os.path.exists(tb):
            lines += [l for l in open(tb).read().splitlines() if l.strip()]
        return "\n".join(f"{prefix}{i + 1} {l}" for i, l in enumerate(lines))

    # ------------------------------------------------------------------ prompt building
    def _img(self, path):
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()}}

    def messages(self, well, question, examples=True, image=True, test_mode=False):
        msgs = [{"role": "system", "content": self.k["system_prompt"]}]
        if examples:                                        # Team B's worked examples as earlier turns (never the asked well)
            for ex in self.k["examples"]:
                if ex["well"] == well:
                    continue
                msgs.append({"role": "user", "content": f"EVIDENCE:\n{self.evidence(ex['well'])}\n\nQUESTION: What is the "
                                                        f"trap at {ex['well']}?"})
                msgs.append({"role": "assistant", "content": ex["answer"]})
        content = [{"type": "text", "text": f"EVIDENCE:\n{self.evidence(well, test_mode)}\n\nQUESTION: {question}"}]
        p = self.image_path(well)
        if image and p and os.path.exists(p):
            content.append(self._img(p))
        msgs.append({"role": "user", "content": content})
        return msgs

    def _chat(self, msgs, temperature=0.2):
        return self.client.chat.completions.create(model=GEMMA_MODEL, temperature=temperature,
                                                   messages=msgs).choices[0].message.content

    # ------------------------------------------------------------------ the three things the app needs
    def ask(self, well, question, examples=True, image=True, test_mode=False, temperature=0.2):
        try:
            return self._chat(self.messages(well, question, examples, image, test_mode), temperature)
        except Exception:                                   # image input not available: answer from text
            return self._chat(self.messages(well, question, examples, False, test_mode), temperature)

    def debate(self, well):
        ev = self.evidence(well)
        adv = ("You are a geoscientist arguing that the trap is {side}. Make the strongest honest case using only the "
               "EVIDENCE, citing IDs in [brackets], and admit the weakest point of your own case. Max 150 words.")
        s = self._chat([{"role": "system", "content": adv.format(side="STRUCTURAL")}, {"role": "user", "content": ev}], 0.4)
        g = self._chat([{"role": "system", "content": adv.format(side="STRATIGRAPHIC")}, {"role": "user", "content": ev}], 0.4)
        j = self._chat([{"role": "system", "content": "You are a chief geoscientist. Weigh both arguments against the "
                         "EVIDENCE. Decide: structural, stratigraphic, combination or undetermined. Your confidence must "
                         "not exceed the TRUST level. Cite IDs. Max 150 words."},
                        {"role": "user", "content": f"EVIDENCE:\n{ev}\n\nSTRUCTURAL CASE:\n{s}\n\nSTRATIGRAPHIC CASE:\n{g}"}], 0.2)
        return f"## Structural advocate\n{s}\n\n## Stratigraphic advocate\n{g}\n\n## Chief geoscientist\n{j}"

    def survey_leads(self):
        p = f"{self.dir}/survey_leads.csv"
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    def similar_to(self, well, n=5, temperature=0.2):
        """'Where else is there a trap like the one at this well?' Undrilled survey bodies whose closest analogue is
        this well, explained by the AI."""
        S = self.survey_leads()
        if S.empty:
            return "Survey screening has not been run yet."
        # QC first: bodies the trap test dropped, or a geoscientist flagged out, never reach the AI
        tt = S["trap_test"].fillna("") if "trap_test" in S else pd.Series("", index=S.index)
        fl = S["qc_flag"].fillna("") if "qc_flag" in S else pd.Series("", index=S.index)
        dropped = S[tt.str.contains("drop") | fl.str.contains("not ranked")]
        S = S.drop(dropped.index)
        m = S[S.analogue.fillna("").str.split().apply(lambda ws: well in ws)].head(n)
        if m.empty:
            m = S.head(n)
            note = f"No undrilled body has {well} as its closest analogue; these are the top-ranked undrilled bodies instead."
        else:
            note = f"Undrilled bodies whose closest analogue is {well}."
        if len(dropped):
            note += (" Already removed by QC (do not rank or recommend): "
                     + ", ".join(f"{dropped.at[i, 'id']} ({fl[i] or tt[i]})" for i in dropped.index) + ".")

        def g(r, c, default="n/a"):
            v = getattr(r, c, default)
            return default if (v is None or (isinstance(v, float) and pd.isna(v))) else v
        rows = "\n".join(
            f"[{r.id}] crest inline {r.crest_il} / crossline {r.crest_xl}, top {r.top_ms} ms, body {r.area_km2} km2, "
            f"{r.thick_ms} ms thick, elongation {r.elongation}, sand {r.sand}, Vp/Vs {r.vpvs} (drop {r.vpvs_contrast}), "
            f"AVO relative to background {r.avo_relative}, closest analogue {r.analogue} (similarity {r.similarity_pct}%, "
            f"similar in {r.similar_in}), {r.dist_to_well_km} km from the nearest well. Trap test from two points: "
            f"{g(r, 'trap_test', 'not trap-tested yet (lower confidence)')}, closure {g(r, 'closure_ms')} ms, gas patch {g(r, 'anomaly_km2')} km2, edge variation "
            f"{g(r, 'edge_tests_ms')} ms (6 ms or less = flat). Geoscientist QC flag: {g(r, 'qc_flag', 'none')}."
            for r in m.itertuples())
        system = (
            "You are a senior exploration geoscientist in the Malay Basin. You screen UNTESTED sand bodies for traps like a "
            f"discovery. The discovery well is called {well}. Its evidence lines are numbered D1, D2... (these are evidence "
            "labels, not well names). Cite discovery evidence as [D4] and bodies as [S02]. Use only the numbers given.\n"
            "Rules: untested bodies are at most MEDIUM confidence. Never recommend drilling an untested body; recommend "
            "which one to MATURE first and the checks needed before it can become a drillable prospect. Never rank a body "
            "removed by QC. A flat edge seen in only one of two tests is 'possible contact, check by hand', not a contact. "
            "Seismic gas indicators are unreliable in sands thinner than about 10 m (field look-back), and AVO relative "
            "values above about 5 can be unstable where near-offset amplitude is weak, so treat them as supportive only. "
            "Trap logic: in a stratigraphic trap the gas edge follows the sand outline, so a NON-flat edge is normal and does not kill the lead; a flat edge in both tests would upgrade it to a combination trap (sand plus contact). What kills a sand-controlled lead: (a) no updip seal - the sand continues updip without a shale or pinch-out, so gas would leak; (b) the anomaly is a coal - bright on near AND far, low sand probability, Vp/Vs not low; (c) the gas indicators vanish when the horizon is re-picked by hand. "
            "Coals can mimic gas. Body thickness is in ms two-way time, not metres: at these depths 1 ms is about 1.3-1.5 m, so convert before comparing with the 10 m limit (e.g. 8 ms is about 11 m).\n"
            "Answer in this format, max 300 words, no other headings:\n"
            f"**The {well} trap** - two sentences: trap type and the 2-3 measurements that define it, cited.\n"
            "**Look-alikes** - one bullet per body, best first: why it resembles the discovery, how it differs, main risk.\n"
            "**Recommendation** - one or two sentences: the one body to mature first and why. Do NOT write a list of "
            "checks: the copilot adds a standard checklist after your answer.")
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": f"DISCOVERY {well}:\n{self.evidence(well, prefix='D')}\n\n{note}\n"
                                            f"UNDRILLED BODIES:\n{rows}\n\n"
                                            f"QUESTION: Where else in the survey is there a trap like the one at {well}?"}]
        answer = self._chat(msgs, temperature)
        # the checklist is written by code from fixed geoscience rules, not by the AI
        rec = answer.split("Recommendation")[-1]
        ids = [i for i in m["id"] if f"[{i}]" in rec or f" {i} " in rec or f" {i}." in rec or f" {i}," in rec]
        pick = m[m["id"] == ids[0]].iloc[0] if ids else m.iloc[0]
        return answer + "\n\n" + self.checklist(pick)

    @staticmethod
    def checklist(r):
        """Standard checks before an untested sand body can become a drillable prospect (rule-based, not AI)."""
        loc = f"inline {r['crest_il']}, crossline {r['crest_xl']}, about {r['top_ms']} ms"
        get = lambda c, d="": r[c] if c in r and isinstance(r[c], str) else d
        lines = [f"**Checks on {r['id']} before it can become a prospect** (standard checklist, written by the copilot's rules)",
                 f"1. **Updip seal** at {loc}: on the inline and crossline through the crest, follow the sand updip. "
                 "It must pinch out or be capped by shale. *Kills the lead:* the sand continues updip into another sand "
                 "(the gas would leak away).",
                 f"2. **Coal or gas?** at {loc}: compare near and far stacks with the sand-probability and Vp/Vs volumes. "
                 "Gas: low Vp/Vs, high sand probability, brighter on far than near. *Kills the lead:* bright on near AND "
                 "far with low sand probability (a coal).",
                 "3. **Re-pick by hand**: pick the top of the body in DUG on 3 inlines and 3 crosslines and repeat the "
                 "measurements. *Kills the lead:* the gas response disappears or the body breaks into small patches."]
        edge, tt = get("edge_tests_ms"), get("trap_test")
        if "possible contact" in tt:
            lines.append(f"4. **Contact or not?** The gas edge was flat in one of two tests ({edge} ms; 6 ms or less = "
                         "flat). If a flat event cuts across the layering at a constant time, it is a fluid contact and "
                         "the trap is a combination trap (sand plus contact). If not, it stays stratigraphic. Neither "
                         "result kills the lead.")
        try:
            thick_m = float(r["thick_ms"]) * 1.4
            if thick_m < 10:
                lines.append(f"{len(lines)}. **Thickness**: about {thick_m:.0f} m ({r['thick_ms']} ms), below the ~10 m "
                             "limit where seismic gas indicators work in this field (blind look-back). Treat the gas "
                             "evidence as weak.")
        except (TypeError, ValueError):
            pass
        return "\n".join(lines)

    # ------------------------------------------------------------------ follow-up questions (chat with memory)
    def chat(self, well, question, history=None, test_mode=False, image=True, temperature=0.2):
        """Ask about a well with follow-ups. history = earlier turns of THIS conversation, oldest first:
        [{"role": "user", "content": q1}, {"role": "assistant", "content": a1}, ...]. The evidence is sent once,
        with the first question; later questions are answered with the whole conversation in view."""
        history = [h for h in (history or []) if h.get("content")]
        first_q = history[0]["content"] if history else question

        def build(img):
            msgs = self.messages(well, first_q, image=img, test_mode=test_mode)
            if history:
                msgs += [{"role": h["role"], "content": h["content"]} for h in history[1:]]
                msgs.append({"role": "user", "content": question})
            return msgs
        try:
            return self._chat(build(image), temperature)
        except Exception:
            return self._chat(build(False), temperature)

    def followup(self, context, question, history=None, temperature=0.2):
        """Follow-up on any earlier answer (debate, 'where else', leads). context = the text the user is looking
        at (e.g. the earlier answer plus its evidence); history as in chat()."""
        msgs = [{"role": "system", "content": self.k["system_prompt"] + "\nAnswer follow-up questions about the "
                 "CONTEXT below. Use only its numbers and IDs, cite them in [brackets], and say plainly when the "
                 "context does not contain the answer."},
                {"role": "user", "content": f"CONTEXT:\n{context}"},
                {"role": "assistant", "content": "Understood. What would you like to know?"}]
        msgs += [{"role": h["role"], "content": h["content"]} for h in (history or []) if h.get("content")]
        msgs.append({"role": "user", "content": question})
        return self._chat(msgs, temperature)

    # ------------------------------------------------------------------ seismic view questions and trap images
    SYSTEM_SECTION = (
        "You are an experienced exploration geoscientist. The user shows a seismic display (image) and may mark a "
        "point. Rules: (1) Separate what you SEE in the image, written '(image)', from measured EVIDENCE, cited as "
        "[V2]. (2) Look for trap features: faults (reflectors offset along a line), anticlines or domes (reflectors "
        "arching up), channels and lenses, pinch-outs and onlap, unconformities, flat spots cutting across dipping "
        "reflectors, bright amplitudes. (3) Say whether the area looks structural, stratigraphic, combination or no "
        "trap, and argue the main alternative in one or two sentences. (4) Confidence: an image alone justifies at "
        "most MEDIUM; without well data you cannot confirm the fluid. (5) End with 'What to check', naming lines, "
        "times or volumes. (6) Never invent wells, numbers or field names. Max 250 words.")

    def _img_any(self, x):
        """PNG/JPG as bytes or a file path -> image message part."""
        if isinstance(x, str):
            x = open(x, "rb").read()
        kind = "jpeg" if x[:3] == b"\xff\xd8\xff" else "png"
        return {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64," + base64.b64encode(x).decode()}}

    def _example_parts(self, examples, max_examples=4):
        """Team C style worked examples shown to the AI before the question (in-context, no training)."""
        parts = []
        for i, ex in enumerate((examples or [])[:max_examples]):
            parts.append({"type": "text", "text": f"WORKED EXAMPLE {i + 1}: raw seismic"
                          + (" and the same section with the trap marked by a geoscientist" if ex.get("marked") else "")
                          + f". Trap type: {ex.get('label', 'unknown')}. Why: {ex.get('note', '')}"})
            parts.append(self._img_any(ex["image"]))
            if ex.get("marked"):
                parts.append(self._img_any(ex["marked"]))
        return parts

    def ask_section(self, image, question, desc="", evidence="", history=None, examples=None, temperature=0.2):
        """Interpret a seismic view (any volume, including one the user uploaded).
        image: PNG/JPG bytes or path of what is on screen (draw the selected point on it first).
        desc: e.g. 'Inline 1634, full stack, 1500-2200 ms, magenta cross at crossline 2327, 1680 ms'.
        evidence: optional measured lines (V1, V2...) for the point, e.g. copilot_extras.point_evidence(...) when
        the competition volumes are loaded. examples: optional Team C worked examples (see trap_examples())."""
        history = [h for h in (history or []) if h.get("content")]
        first_q = history[0]["content"] if history else question
        content = self._example_parts(examples)
        content.append({"type": "text", "text": f"DISPLAY: {desc or 'seismic section'}\n\nEVIDENCE:\n"
                                                f"{evidence or 'none (image only)'}\n\nQUESTION: {first_q}"})
        content.append(self._img_any(image))
        msgs = [{"role": "system", "content": self.SYSTEM_SECTION}, {"role": "user", "content": content}]
        if history:                                            # follow-up: earlier turns, then the new question
            msgs += [{"role": h["role"], "content": h["content"]} for h in history[1:]]
            msgs.append({"role": "user", "content": question})
        return self._chat(msgs, temperature)

    @staticmethod
    def trap_examples(folder):
        """Load Team C's labelled images from a folder of pairs and a labels file:
        folder/labels.csv with columns  name,label,note   (label = structural / stratigraphic / combination / none)
        folder/<name>_raw.png  and optionally  folder/<name>_marked.png"""
        L = pd.read_csv(f"{folder}/labels.csv")
        out = []
        for r in L.itertuples():
            raw, marked = f"{folder}/{r.name}_raw.png", f"{folder}/{r.name}_marked.png"
            if os.path.exists(raw):
                out.append({"name": r.name, "image": raw, "marked": marked if os.path.exists(marked) else None,
                            "label": str(r.label).strip().lower(), "note": "" if pd.isna(r.note) else str(r.note)})
        return out

    def image_test(self, examples, n_shots=3):
        """Leave-one-out test on Team C's images: for each image, show the AI n_shots OTHER labelled examples and
        ask it to classify the raw image. Returns a table and the accuracy with and without examples."""
        rows = []
        q = ("Classify the trap in this seismic image. Start your answer with exactly one word: STRUCTURAL, "
             "STRATIGRAPHIC, COMBINATION or NONE, then one sentence of reasoning.")
        for i, ex in enumerate(examples):
            others = [e for j, e in enumerate(examples) if j != i][:n_shots]
            res = {}
            for mode, shots in (("no_examples", None), ("with_examples", others)):
                try:
                    a = self.ask_section(ex["image"], q, examples=shots, temperature=0)
                    word = a.strip().split()[0].strip("*.:,").lower() if a.strip() else ""
                except Exception as e:
                    a, word = f"failed: {e}", ""
                res[mode] = word
                res[mode + "_answer"] = a
            rows.append({"name": ex["name"], "true_label": ex["label"], **res})
        T = pd.DataFrame(rows)
        acc = {m: round(float((T[m] == T.true_label).mean()), 2) for m in ("no_examples", "with_examples")} if len(T) else {}
        return T, acc

    def recommend_leads(self, n=3):
        p = f"{self.dir}/recommendation.md"
        if os.path.exists(p):
            return open(p).read()
        if self.leads.empty:
            return "No leads available."
        return self.leads.head(n).to_string(index=False)