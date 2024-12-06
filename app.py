import streamlit as st
from utils import extract_text
from summarizer import summarize_document

st.title("RAG-Based Document Summarizer")
st.write(
    "Let's start speeding up our lives"
)

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])
query = st.text_input("Enter your query or topic for summarization")
summary_style = st.radio("Choose summary style:", ["short", "detailed", "bullet-point"])

if uploaded_file and query:
    raw_text = extract_text(uploaded_file)  # This function extracts text from uploaded file
    summary = summarize_document(raw_text, query, style=summary_style)
    st.write("### Summary:")
    st.text(summary)
