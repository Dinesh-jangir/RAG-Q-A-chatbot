import streamlit as st
from rag import rag_answer

st.set_page_config(page_title="Loan Q&A Chatbot")
st.title("Loan Approval Q&A (Gemini)")

if 'history' not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask me anything about loan approvals:")
if query:
    answer = rag_answer(query)
    st.session_state.history.append((query, answer))

for q,a in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")