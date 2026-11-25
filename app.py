# Libraries
import hashlib
import io
import streamlit as st
import torch
import faiss
import numpy as np
from pypdf import PDFReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer



#-------------- Model loaded and installation-----------------

# Settings

EMDED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL="google/flan-t5-base"   # google/flan-t5-large for better response
CHUNK_SIZE=600 # charachter per chunk
CHUNK_OVERLAP=100
TOP_K=3
MAX_INPUT_TOKENS=512

# ------------------- Models loaded once then cached----------------
@st.cache_resource(show_spinner="Loading Embedding Model...")
def load_embedder():
    return SentenceTransformer(EMDED_MODEL)

@st.cache_resource(show_spinnder="Loading LLM Model...")

def load_LLM():
    device="cuda" if torch.cuda.is_available() else "cpu"
    tokenizer=AutoTokenizer.from_pretrained(LLM_Model)
    model=AutoModelForSeq2SeqLM.from_pretrained(LLM_Model).to(device)
    model.eval()
    return tokenizer,model,device



# ---------------------PDF Processing---------------------------

def extract_pages(file_bytes):
    reader=PDFReader(io.BytesIO(file_bytes))

    pages=[]
    for i,page in enumerate(reader.pages,start=1):
        text=(page.extract_text() or "").strip()
        if text:
            pages.append((i,text))

        return pages

def build_index(pages):
    # Split pages into chunks,embed them ,and store them into a FAISS index
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP

    )

    chunks=[]
    for page_num,text in pages:
        for peice in splitter.split_text(text):
            chunks.append({"page":page_num,"text":peice})

    embedder =load_embedder()
    embeddings=embedder.encode(
        [c['text'] for c in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,   # False
    ).astype("float32")
    index=faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index,chunks

def retreive(question,index,chunks,k=TOP_K):
    # Find most similar question related to chunks
    embedder=load_embedder()
    q_emb=embedder.encode(
        [question],normalize_embeddings=True,convert_to_numpy=True

    ).astype("float32")
    scores,ids=index.search(q_emb,min(k,len(chunks)))
    return [{**chunks[i],"score":float(s)} for s,i in zip(scores[0],ids[0] if i!=-1)]

            























# ------------------------UI----------------------
st.set_page_config(page_title="PDF AI assistant",page_icon="⚡🤖⚡︎ ",layout="wide")

st.title("⚡🤖⚡ PDF AI assistant ⚡🤖⚡")
st.text("Upload your PDF Document here to get answer related to your document\nThis bot will help you summarize your document")


st.sidebar.subheader("Upload your PDF file and ask he questions")

file=st.sidebar.file_uploader("Upload your PDF file",type=["pdf"])

text_input=st.text_input("Ask Questions Related to your PDF document")

if st.button("Ask AI"):
    if file is None:
        st.error("Please upload your PDF Document First")

    elif not  text_input.strip :
        st.warning("Plz ask a Question First")
    else:
        st.success("Text")
