# Ask PDF AI

Ask PDF AI is a Streamlit application for asking questions about a text-based PDF or Word `.docx` document. Instead of asking a language model to answer from general knowledge alone, the app retrieves relevant passages from the uploaded document, uses those passages as context, and shows their sources alongside its answer.

## What problem does it solve?

Long documents can be difficult to search manually. This project provides a simple way to find and understand information in a document by asking questions in natural language. It is intended for personal exploration of reports, study material, manuals, and other documents with extractable text.

The app is a helpful reading aid, not a guarantee of factual or complete answers. Always check important answers against the cited passages in the original document.

## How it works

The app uses retrieval-augmented generation (RAG): search the document for relevant text first, then give that text to a language model to draft an answer.

1. **Upload:** The user selects one PDF or DOCX file in the Streamlit sidebar. The app confirms the selected file there.
2. **Extract text:** `pypdf` reads PDF text page by page. `python-docx` reads DOCX paragraphs and table rows. Empty content is skipped; PDF passages keep page-number sources, while DOCX passages are identified as coming from the Word document.
3. **Split into chunks:** `RecursiveCharacterTextSplitter` divides page text into chunks of 600 characters with 100 characters of overlap. The overlap helps preserve context across neighboring chunks.
4. **Create embeddings:** `sentence-transformers/all-MiniLM-L6-v2` converts each chunk into a normalized vector.
5. **Build a search index:** FAISS `IndexFlatIP` indexes those vectors. Because they are normalized, inner product ranks chunks by cosine similarity.
6. **Retrieve context:** The question is embedded with the same model, and the three most similar chunks are selected.
7. **Generate an answer:** The app sends the question and retrieved excerpts to `google/flan-t5-base`. The prompt instructs the model to answer from the excerpts and to say when the document does not contain the answer. Input is limited to 512 tokens.
8. **Show evidence:** The app displays the generated answer, the matching passages, their page numbers, and similarity scores.

The embedding and answer models are cached by Streamlit after they are loaded. The document's text, chunks, and FAISS index are currently rebuilt when the user asks a question or requests a summary.

## Current features and limitations

### Included

- Upload one PDF or DOCX file at a time, with an upload confirmation in the sidebar.
- Ask a question about its extracted text.
- Generate a summary from selected document chunks.
- Retrieve relevant passages and show their source pages.
- Run inference on CUDA when available, or fall back to CPU.
- Cache the loaded models within the Streamlit process.

### Important limitations

- **Scanned PDFs:** The app does not perform OCR. Pages that contain images instead of an embedded text layer may appear empty.
- **Legacy Word files:** Only `.docx` is supported; older binary `.doc` files must be converted to `.docx` first.
- **Word pagination:** DOCX files do not have stable page numbers in this app, so citations identify the Word document rather than a page.
- **Tables and layout:** PDF text extraction may lose table structure, reading order, headers, or other visual context. DOCX table text is extracted, but table layout and paragraph/table order may not be preserved.
- **Answer quality:** Retrieval can miss relevant passages, and a language model can still produce an incomplete or unsupported answer. Similarity scores are ranking signals, not calibrated confidence values.
- **Document size:** The full document is reprocessed for each question. Large PDFs can take time and use substantial memory, especially on CPU.
- **Context length:** Only a limited amount of retrieved text fits into the answer model's input.
- **Summary coverage:** Summaries are built from at most eight evenly spaced chunks. For long documents, this can omit details and should not be treated as exhaustive.
- **Single document and session:** There is no multi-file library, conversation history, saved index, or persistent document storage.
- **Model downloads:** Hugging Face models are downloaded on first use and require network access unless they are already cached. Inference is performed by the app process; consider where the app is hosted and how uploads are handled before using sensitive documents.

## Requirements

- Python with versions supported by the installed PyTorch, FAISS, and Transformers packages.
- Internet access on first run to download the models, unless they are already present in the Hugging Face cache.
- CPU is supported; a compatible CUDA setup can accelerate inference.

## Setup and run

From the project directory, create and activate a virtual environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project dependencies and start the app:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints a local URL in the terminal. Open it in a browser and upload a PDF or `.docx` file using the sidebar. A success message appears after selection. Then either enter a question and select **Ask AI**, or select **Summarize PDF**. The first request may take longer while the models are downloaded and loaded.

Run Streamlit as a Python module (`python -m streamlit`) rather than using the `streamlit` launcher if the project or virtual environment has been moved. Virtual environments are not portable; recreate the environment at its new location if its Python executable is also missing.

If dependency installation fails, check that your Python version and platform are supported by the selected PyTorch and FAISS wheels. For CUDA, install the PyTorch build that matches your CUDA environment; otherwise, the app uses CPU.

## Project structure

```text
.
├── app.py            # Streamlit UI, PDF processing, retrieval, and generation
├── requirements.txt  # Python dependencies
└── README.md         # Project overview and setup
```

## Improvement roadmap

Suggested order, starting with correctness and the most visible usability gaps:

1. **Add focused tests and a small evaluation set.** Test PDF/DOCX extraction, chunk/source association, retrieval, and prompt construction. Evaluate representative questions against known answers and supporting sources before changing model or chunk settings.
2. **Improve summary coverage.** The app summarizes up to eight evenly spaced chunks; evaluate coverage on long documents and consider a hierarchical summarization strategy before presenting the result as exhaustive.
3. **Avoid rebuilding the index for every question.** Store extracted pages, chunks, and the index in per-session state, and invalidate them when the uploaded file changes. Use a content hash if you later add a safe, bounded cache across sessions.
4. **Improve evidence quality.** Make chunk size, overlap, and retrieval count configurable only after measuring them. Consider a relevance threshold, reranking, and displaying the exact source excerpts. Do not treat raw similarity as answer confidence.
5. **Handle more document types.** Add OCR for scanned PDFs and improve layout-aware extraction if those document types matter. Show clear messages for encrypted, malformed, empty, or unsupported files.
6. **Strengthen the interface.** Support follow-up questions, multiple documents, visible processing status, and a reset/remove-document action. Keep document state scoped to the user's session.
7. **Plan for deployment.** Apply upload size and processing limits, avoid logging document contents, define retention and deletion behavior, and use appropriate isolation and access controls for sensitive files.
8. **Measure performance and quality.** Record latency and resource use, test CPU and GPU paths, and compare model and retrieval changes against the same evaluation set.

## Recommended approach

For this project's goal, a small RAG pipeline is a better starting point than fine-tuning a language model: document-specific information can change, and retrieval makes it possible to show where answers came from. Keep the pipeline simple and improve it in this order:

1. Establish a test set of PDFs, questions, expected answers, and supporting pages.
2. Make ingestion reliable and reuse the per-session index.
3. Improve retrieval and citation behavior based on measured failures.
4. Add OCR or layout-aware parsing only for document types that need it.
5. Optimize model size, hosting, and caching against actual quality, latency, cost, and privacy requirements.

This keeps answers grounded in document evidence while avoiding the cost and complexity of training a custom model prematurely.

## Models and libraries

- [Streamlit](https://streamlit.io/) for the web interface
- [pypdf](https://pypi.org/project/pypdf/) for PDF text extraction
- [python-docx](https://python-docx.readthedocs.io/) for DOCX text extraction
- [LangChain text splitters](https://pypi.org/project/langchain-text-splitters/) for chunking
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Transformers](https://huggingface.co/docs/transformers/) and [PyTorch](https://pytorch.org/) for answer generation

