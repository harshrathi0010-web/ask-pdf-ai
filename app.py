# Libraries
import hashlib
import io
import streamlit as st
import torch
import faiss
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer



#-------------- Model loaded and installation-----------------

# Settings

EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL="google/flan-t5-base"
CHUNK_SIZE=600
CHUNK_OVERLAP=100
TOP_K=3
MAX_INPUT_TOKENS=512

# ------------------- Models loaded once then cached----------------
@st.cache_resource(show_spinner="Loading Embedding Model...")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner="Loading Answer Model...")
def load_llm():
    device="cuda" if torch.cuda.is_available() else "cpu"
    tokenizer=AutoTokenizer.from_pretrained(LLM_MODEL)
    model=AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL).to(device)
    model.eval()
    return tokenizer, model, device



# ---------------------PDF Processing---------------------------

def extract_pages(file_bytes):
    # return a list of (page_number,text) for pages that contain text.....
    reader=PdfReader(io.BytesIO(file_bytes))

    pages=[]
    for i,page in enumerate(reader.pages,start=1):
        text=(page.extract_text() or "").strip()
        if text:
            pages.append((i,text))

    return pages

def build_index(pages):
    # Split pages into chunks, embed them, and store them in a FAISS index
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP
    )

    chunks=[]
    for page_num,text in pages:
        for piece in splitter.split_text(text):
            chunks.append({"page":page_num,"text":piece})

    if not chunks:
        return None, []

    embedder =load_embedder()
    embeddings=embedder.encode(
        [c['text'] for c in chunks],  # normalize vector-->inner product =cosine similarity
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    index=faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index,chunks

def retrieve(question,index,chunks,k=TOP_K):
    # Find chunks most similar to the question
    embedder=load_embedder()
    q_emb=embedder.encode(
        [question],normalize_embeddings=True,convert_to_numpy=True
    ).astype("float32")
    scores,ids=index.search(q_emb,min(k,len(chunks)))
    return [
        {**chunks[i],"score":float(s)}
        for s,i in zip(scores[0],ids[0])
        if i != -1
    ]

# Text generation function

def generate(prompt,max_new_tokens=200):
    tokenizer,model,device=load_llm()
    inputs=tokenizer(
        prompt,return_tensors="pt",truncation=True,max_length=MAX_INPUT_TOKENS).to(device)
    with torch.no_grad():
        output=model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,



        )
    return tokenizer.decode(output[0],skip_special_tokens=True).strip()


def answer_question(question,index,chunks):
    hits=retrieve(question,index,chunks)
    context="\n\n".join(h["text"] for h in hits)
    prompt=(
        "Answer the question using only the context below. Give a complete answer "
        "in one or more sentences. If the answer is not in the context, reply "
        "\"I could not find this in the document.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return generate(prompt),hits

def summarize_document(chunks, max_parts=8):
    """Summarize evenly spaced chunks, then combine the partial summaries."""
    if len(chunks) <= max_parts:
        selected = chunks
    else:
        step = len(chunks) / max_parts
        selected = [chunks[int(i * step)] for i in range(max_parts)]

    partials = [
        generate(f"Summarize the following text:\n\n{c['text']}", max_new_tokens=80)
        for c in selected
    ]
    combined = " ".join(partials)
    return generate(
        f"Write a clear summary of this document in a few sentences:\n\n{combined}",
        max_new_tokens=250,
    )




# ------------------------UI----------------------
st.set_page_config(page_title="PDF AI assistant",page_icon="⚡🤖⚡︎ ",layout="wide")

st.title("⚡🤖⚡ PDF AI assistant ⚡🤖⚡")
st.text("Upload your PDF Document here to get answer related to your document\nThis bot will help you summarize your document")


st.sidebar.subheader("Upload your PDF file and ask he questions")

file=st.sidebar.file_uploader("Upload your PDF file",type=["pdf"])

question=st.text_input("Ask questions related to your PDF document")

if st.button("Ask AI"):
    if file is None:
        st.error("Please upload your PDF document first.")
    elif not question.strip():
        st.warning("Please ask a question first.")
    else:
        with st.spinner("Reading PDF and finding relevant sections..."):
            try:
                pages=extract_pages(file.getvalue())
                if not pages:
                    st.error("This PDF does not contain readable text.")
                    st.stop()

                index,chunks=build_index(pages)
                if index is None:
                    st.error("No text chunks could be created from this PDF.")
                    st.stop()

                matches=retrieve(question,index,chunks)
                context="\n\n".join(
                    f"[Page {match['page']}] {match['text']}"
                    for match in matches
                )
                prompt=f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

                tokenizer,llm,device=load_llm()
                inputs=tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_INPUT_TOKENS,
                ).to(device)
                outputs=llm.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
                answer=tokenizer.decode(outputs[0],skip_special_tokens=True).strip()

                st.subheader("Answer")
                st.write(answer or "The model did not produce an answer.")

                st.subheader("Sources")
                for match in matches:
                    st.caption(
                        f"Page {match['page']} - similarity {match['score']:.3f}"
                    )
                    st.write(match["text"])
            except Exception as exc:
                st.error(f"Could not answer the question: {exc}")
