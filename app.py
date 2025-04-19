import streamlit as st
import pdfplumber
import pytesseract
import fitz  # PyMuPDF
import requests
import tempfile
from PIL import Image
from io import BytesIO
import openai
import os
import faiss
import numpy as np
import tiktoken

# --- CONFIG ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
EMBED_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50
TOP_K = 5

# --- HELPERS ---
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text):
    return len(tokenizer.encode(text))

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk))
    return chunks

def get_embedding(text):
    res = openai.Embedding.create(
        input=[text],
        model=EMBED_MODEL
    )
    return np.array(res['data'][0]['embedding'])

def download_pdf_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return BytesIO(response.content)

def extract_text_from_pdf(file_like):
    try:
        with pdfplumber.open(file_like) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
    except Exception:
        return ""

def extract_text_from_scanned_pdf(file_like):
    text = ""
    doc = fitz.open(stream=file_like.read(), filetype="pdf")
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(BytesIO(pix.tobytes()))
        text += pytesseract.image_to_string(img)
    return text.strip()

# --- STATE ---
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'index' not in st.session_state:
    st.session_state.index = faiss.IndexFlatL2(1536)
    st.session_state.embedded_chunks = []
    st.session_state.chunk_sources = []

# --- UI ---
st.title("📚 Multi-PDF Analyzer + AI Q&A")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
url_input = st.text_area("Or paste FDA PDF URLs (one per line):")

if uploaded_files or url_input:
    with st.spinner("Processing PDFs..."):
        all_text = ""

        # Process uploaded files
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            if not text or len(text) < 100:
                file.seek(0)
                text = extract_text_from_scanned_pdf(file)
            if text:
                chunks = chunk_text(text)
                embeddings = [get_embedding(chunk) for chunk in chunks]
                st.session_state.index.add(np.array(embeddings).astype("float32"))
                st.session_state.embedded_chunks.extend(chunks)
                st.session_state.chunk_sources.extend([file.name] * len(chunks))
                st.success(f"Processed {file.name} with {len(chunks)} chunks.")

        # Process URLs
        for url in url_input.splitlines():
            if not url.strip():
                continue
            try:
                pdf_bytes = download_pdf_from_url(url.strip())
                text = extract_text_from_pdf(pdf_bytes)
                if not text or len(text) < 100:
                    pdf_bytes.seek(0)
                    text = extract_text_from_scanned_pdf(pdf_bytes)
                if text:
                    chunks = chunk_text(text)
                    embeddings = [get_embedding(chunk) for chunk in chunks]
                    st.session_state.index.add(np.array(embeddings).astype("float32"))
                    st.session_state.embedded_chunks.extend(chunks)
                    st.session_state.chunk_sources.extend([url] * len(chunks))
                    st.success(f"Processed {url.strip()} with {len(chunks)} chunks.")
            except Exception as e:
                st.error(f"Error fetching {url.strip()}: {e}")

# --- Q&A ---
if st.session_state.embedded_chunks:
    question = st.text_input("Ask a question across all documents:")
    if question:
        with st.spinner("Finding the answer..."):
            q_embed = get_embedding(question).astype("float32")
            D, I = st.session_state.index.search(np.array([q_embed]), k=TOP_K)
            retrieved_chunks = [st.session_state.embedded_chunks[i] for i in I[0]]
            retrieved_sources = [st.session_state.chunk_sources[i] for i in I[0]]

            context = "\n\n---\n\n".join(retrieved_chunks)
            prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}
"""

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You're a helpful assistant that answers questions about documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            answer = response.choices[0].message.content.strip()
            st.markdown(f"**Answer:** {answer}")

            with st.expander("🔍 Sources and Context Chunks"):
                for src, chunk in zip(retrieved_sources, retrieved_chunks):
                    st.markdown(f"**Source:** {src}\n\n```{chunk[:1000]}```")

