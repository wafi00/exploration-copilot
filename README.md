# \# Exploration Copilot

# 

# AI geoscience copilot for APGCE 2026 GeoHackathon (Challenge 1).

# Chat screen (Streamlit) + Gemma 4 LLM hosted on the DUG HPC.

# 

# \## Files

# \- `app.py` – chat screen

# \- `llm.py` – sends questions to Gemma (only works inside the HPC)

# 

# \## Run on a laptop (chat screen only, placeholder answers)

# 1\. Install Python (tick "Add Python to PATH").

# 2\. Open Command Prompt and run:

# &#x20;  python -m pip install streamlit openai

# 3\. Go to the project folder:

# &#x20;  cd exploration-copilot

# 4\. Start the app:

# &#x20;  python -m streamlit run app.py

# 5\. The browser opens at http://localhost:8501. Press Ctrl+C in Command Prompt to stop.

# 

# \## Run on the HPC (real Gemma answers)

# 1\. Copy app.py and llm.py into the team folder on the HPC.

# 2\. pip install --user streamlit openai

# 3\. streamlit run app.py --server.port 8601

# 4\. Open http://localhost:8601 in Firefox inside RustDesk.

# 

# \## Rules

# \- No competition data in this repo (code only). Data stays on the HPC.

