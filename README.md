# Exploration Copilot

APGCE 2026 GeoHackathon, Challenge 1.

An AI geoscientist that classifies hydrocarbon traps as **structural, stratigraphic or combination** from
measured evidence, says **how much to trust itself**, **debates itself** (structural vs stratigraphic advocate,
judged by a chief geoscientist), **finds untested look-alike traps** across the whole survey, and **ranks
leads by risked area** (chance of gas from a blind look-back x area). Every answer cites numbered evidence
([E4], [V2]) and ends with what a geoscientist should check.

## Architecture

1. **Notebook pipeline** (`copilot_pipeline.py`, `copilot_extras.py`, `copilot_traps2.py`, `copilot_survey.py`):
   well deviation, well-seismic tie, sands and contacts from the logs, auto-picked horizons, two-point trap
   test, whole-survey look-alike screen, blind look-back.
2. **`master/` folder**: one verdict per well, numbered evidence files, leads, survey look-alikes, pictures,
   cached demo answers (made on the HPC; competition data, not in this repo).
3. **`copilot_api.py`**: the only interface between the app and the data / AI (ask, chat, debate,
   similar_to, followup, ask_section, trap_examples, image_test).
4. **`app.py`** (Streamlit) + **Gemma 4** served with vLLM on the HPC. Five sections:
   - **Seismic viewer** (`seismic_tab.py`, using `plots.py`, `viewer.py`, `seismic_ai.py`): map view, 3D view,
     inline and crossline, crop, bright-amplitude and sand-probability highlights, ask the AI about any point;
     well-log (LAS) upload with curve plots and questions.
   - **Field overview**: every well's trap verdict and trust level, the whole-survey map.
   - **Well copilot**: verdict card, one-click questions, debate, the evidence the AI used, well data.
   - **Similar traps**: undrilled look-alikes of a discovery with a rule-based checklist.
   - **Leads to mature**: leads ranked by risked area and the written recommendation.

   Every chat box accepts a picture for Gemma to look at, and the sidebar sets the AI's tone (expert
   geoscientist, plain English, executive summary, teacher, quick answer).

Gemma's weights are not changed: the copilot learns **in context** (the geoscientists' reasoning steps, field
calibration, trap catalogue and worked examples on every request; labelled trap images from a published paper
as examples).

## How to run (on the HPC)

```bash
pip install -r requirements.txt
export GEMMA_URL="http://<vllm-host>:8000/v1"      # the competition's Gemma endpoint
# 1. run the notebook pipeline cells once (they write the master/ folder)
# 2. start the app
python -m streamlit run app.py --server.port 8603
```
Keep **Demo mode** on for a live demo: cached answers appear instantly.

## QC and validation

- **Well ties** with checkshots and the sand pattern; wells placed at reservoir depth (deviation).
- **Two-point trap test**: every survey body is tested from two points; only calls that agree are kept.
- **Blind look-back** calibrates the chance of gas used to rank leads.
- **Rule-based kill checklist**: fixed geoscience rules in code, not AI, decide what rejects a lead.
- **Two confidences**: evidence quality vs trap type; the AI may not exceed the trust level.
- **Blind test mode** hides the human interpretation for test wells.
- **Human in the loop**: Agree / Disagree feedback on every AI answer.

## Limits

- Thin pays under about 10 m are below seismic reliability.
- No sonic or density logs, so ties use the sand pattern.
- Image-only answers (uploaded seismic) stay at MEDIUM confidence at most.
- `AI_Chat.py` and `pages/` are the earlier prototype; `app.py` is the submitted app.

## Team

<!-- Fill in: name - role -->
- Code and app: ...
- Seismic and well interpretation: ...
- Trap image library: ...
