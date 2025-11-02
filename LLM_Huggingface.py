# Hugging Face RAG PDF Summarizer
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
from huggingface_hub import InferenceClient
import nltk
import tempfile
import os

# -------------------------------------------------------------------
# NLTK setup (tokenizers and POS tagger)
# -------------------------------------------------------------------
nltk_resources = {
    "punkt": "tokenizers",
    "punkt_tab": "tokenizers",
    "averaged_perceptron_tagger_eng": "taggers"
}
for resource, folder in nltk_resources.items():
    try:
        nltk.data.find(f"{folder}/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# -------------------------------------------------------------------
# Hugging Face setup
# -------------------------------------------------------------------
HF_TOKEN = "XXXXXXXX"   # ← Replace with your Hugging Face token

# Chat model (for answering questions / summarizing)
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Embedding model for vector database
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize Hugging Face LLM client
llm_client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)

# Initialize embedding model (using LangChain’s wrapper)
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# -------------------------------------------------------------------
# Streamlit UI setup
# -------------------------------------------------------------------
st.set_page_config(page_title="RAG PDF Summarizer")
st.title("🤖 Hugging Face RAG-powered PDF Summarizer")

upload_file = st.file_uploader("Upload a PDF", type="pdf")

if upload_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(upload_file.read())
        temp_file_path = temp_file.name

    # Step 1: Extract text
    st.info("📄 Extracting text from PDF...")
    raw_text = extract_text(temp_file_path)

    # Step 2: Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)

    # Step 3: Create embeddings and store in Chroma
    with st.spinner("🔍 Indexing document..."):
        vectordb = Chroma.from_texts(chunks, embedding_model, persist_directory="/chroma_index")
        vectordb.persist()

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Step 4: Define helper for LLM response using Hugging Face chat_completion
    def hf_chat(prompt):
        response = llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        return response.choices[0].message["content"]

    # Step 5: Generate RAG summary
    with st.spinner("🧠 Running summarization using RAG..."):
        docs = retriever.invoke("Summarize this document.")
        context = "\n\n".join([d.page_content for d in docs])
        summary_prompt = f"Summarize the following document sections clearly and concisely:\n\n{context}"
        summary = hf_chat(summary_prompt)

    st.subheader("📘 Summary")
    st.write(summary)
