import streamlit as st

st.set_page_config(page_title="PDF AI assistant",page_icon="⚡🤖⚡︎ ",layout="wide")

st.title("⚡🤖⚡ PDF AI assistant ⚡🤖⚡")
st.text("Upload your PDF Document here to get answer related to your document\nThis bot will help you summarize your document")


st.sidebar.subheader("Upload your PDF file and ask he questions")

file=st.sidebar.file_uploader("Upload your PDF file",type=["pdf"])

text_input=st.text_input("Ask Questions Related to your PDF document")

if st.button("Ask AI"):
    if file is None:
        st.error("Please upload your PDF Document First")
        
