# Libraries
import hashlib
import io
import streamlit as st
import torch
import faiss
import numpy as np
from pypdf import PdfReader
from docx import Document
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
    # Return page-number/text pairs for PDF pages that contain text.
    reader=PdfReader(io.BytesIO(file_bytes))

    pages=[]
    for i,page in enumerate(reader.pages,start=1):
        text=(page.extract_text() or "").strip()
        if text:
            pages.append((i,text))

    return pages

def extract_document_sources(file_bytes, file_name):
    extension = file_name.lower().rsplit(".", 1)[-1]
    if extension == "pdf":
        return [
            (f"Page {page_number}", text)
            for page_number, text in extract_pages(file_bytes)
        ]
    if extension == "docx":
        document = Document(io.BytesIO(file_bytes))
        content = [paragraph.text.strip() for paragraph in document.paragraphs]
        content.extend(
            " | ".join(cell.text.strip() for cell in row.cells)
            for table in document.tables
            for row in table.rows
        )
        text = "\n".join(part for part in content if part)
        return [("Word document", text)] if text else []
    raise ValueError("Unsupported document type. Upload a PDF or DOCX file.")

def build_index(sources):
    # Split document sources into chunks, embed them, and store them in FAISS.
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP
    )

    chunks=[]
    for source,text in sources:
        for piece in splitter.split_text(text):
            chunks.append({"source":source,"text":piece})

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
    tokenizer,_,_=load_llm()
    instruction=(
        "Answer the question using only the document excerpts. Give a direct, "
        "clear answer and do not repeat the excerpts. If the answer is not "
        "present, say \"I could not find this in the document.\"\n\n"
        "Document excerpts:\n"
    )
    question_prompt=f"\n\nQuestion: {question}\nAnswer:"
    fixed_tokens=len(tokenizer.encode(
        instruction+question_prompt,
        add_special_tokens=True,
    ))
    remaining=max(0,MAX_INPUT_TOKENS-fixed_tokens-32)
    excerpts=[]
    for hit in hits:
        source_label=f"[{hit['source']}] "
        label_tokens=tokenizer.encode(source_label,add_special_tokens=False)
        text_tokens=tokenizer.encode(hit["text"],add_special_tokens=False)
        text_budget=remaining-len(label_tokens)-2
        if text_budget <= 0:
            break
        excerpt=tokenizer.decode(
            label_tokens+text_tokens[:text_budget],
            skip_special_tokens=True,
        )
        excerpts.append(excerpt)
        remaining-=len(label_tokens)+min(len(text_tokens),text_budget)+2
    context="\n\n".join(excerpts)
    prompt=instruction+context+question_prompt
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
st.text("Upload a PDF or Word document to ask questions or request a summary.")


st.sidebar.subheader("Upload a PDF or Word document")

file=st.sidebar.file_uploader("Upload a PDF or Word document",type=["pdf","docx"])
if file is not None:
    st.sidebar.success(f"Uploaded {file.name} successfully.")

question=st.text_input("Ask a question about your document")

with st.container(horizontal=True):
    ask_clicked = st.button("Ask AI")
    summarize_clicked = st.button("Summarize PDF")

if ask_clicked or summarize_clicked:
    if file is None:
        st.error("Please upload a document first.")
    elif ask_clicked and not question.strip():
        st.warning("Please ask a question first.")
    else:
        action = "summarize" if summarize_clicked else "answer"
        spinner_text = (
            "Reading document and generating summary..."
            if summarize_clicked
            else "Reading document and finding relevant sections..."
        )
        with st.spinner(spinner_text):
            try:
                sources=extract_document_sources(file.getvalue(), file.name)
                if not sources:
                    st.error("This document does not contain readable text.")
                    st.stop()

                index,chunks=build_index(sources)
                if index is None:
                    st.error("No text chunks could be created from this document.")
                    st.stop()

                if summarize_clicked:
                    summary = summarize_document(chunks)
                    st.subheader("Document summary")
                    st.write(summary or "The model did not produce a summary.")
                else:
                    answer,matches=answer_question(question,index,chunks)

                    st.subheader("Answer")
                    st.write(answer or "The model did not produce an answer.")

                    st.subheader("Sources")
                    for match in matches:
                        st.caption(
                            f"{match['source']} - similarity {match['score']:.3f}"
                        )
                        st.write(match["text"])
            except Exception as exc:
                st.error(f"Could not {action} the document: {exc}")
