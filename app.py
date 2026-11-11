import streamlit as st
from llm import ask

st.set_page_config(page_title="Exploration Copilot", layout="wide")
st.title("Exploration Copilot")
if "msgs" not in st.session_state:
    st.session_state.msgs = []
for m in st.session_state.msgs:
    st.chat_message(m["role"]).write(m["content"])
q = st.chat_input("Ask about a well or a lead...")
if q:
    st.session_state.msgs.append({"role": "user", "content": q})
    st.chat_message("user").write(q)
    with st.spinner("Gemma is thinking..."):
        a = ask("You are an exploration geoscientist.", q)
    st.session_state.msgs.append({"role": "assistant", "content": a})
    st.chat_message("assistant").write(a)