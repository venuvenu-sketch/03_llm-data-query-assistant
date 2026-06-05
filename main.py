"""
LLM-Powered Data Query Assistant
=================================
Query any CSV dataset using plain English. Combines a FAISS vector store
(built from chunked dataset rows) with Google Gemini to deliver context-aware,
grounded answers — no SQL, no manual filtering required.

Pipeline
--------
CSV upload  →  Data cleaning  →  Text chunking  →  HuggingFace embeddings
→  FAISS vector store  →  Semantic retrieval  →  Gemini LLM  →  Answer

Author : V.J.M. Venu Gopal
Tools  : Python, LangChain, FAISS, HuggingFace Transformers, Google Gemini, Pandas
"""

import os
import sys
import pandas as pd
import numpy as np
import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


# ── Configuration ─────────────────────────────────────────────────────────────

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"   # Fast, lightweight, accurate
GEMINI_MODEL     = "gemini-2.5-flash-preview-05-20"
CHUNK_SIZE_ROWS  = 100          # Rows per CSV chunk (controls memory usage)
CHUNK_SIZE_CHARS = 2000         # Characters per LangChain text chunk
CHUNK_OVERLAP    = 200          # Overlap between consecutive text chunks
TOP_K_RESULTS    = 2            # Number of retrieved context chunks sent to Gemini


# ── 1. API Key Setup ──────────────────────────────────────────────────────────

def configure_api() -> str:
    """
    Loads the Google API key from environment variable GOOGLE_API_KEY.
    Raises a clear error if the key is missing.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set.\n"
            "Run: export GOOGLE_API_KEY='your_key_here'  (Linux/Mac)\n"
            "  or: set GOOGLE_API_KEY=your_key_here       (Windows)"
        )
    genai.configure(api_key=api_key)
    return api_key


# ── 2. Data Cleaning ──────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises column names, handles missing values, and drops
    columns with more than 50% nulls.

    Steps:
      - Strip and lowercase column names; replace spaces with underscores
      - Replace common null-like strings with NaN
      - Drop columns where >50% values are missing
      - Fill numeric nulls with column mean
      - Fill categorical nulls with column mode
    """
    # Normalise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Replace common null-like strings
    df.replace(["", " ", "na", "n/a", "nan", None], np.nan, inplace=True)

    # Drop columns with >50% missing values
    threshold = len(df) * 0.5
    df.dropna(thresh=threshold, axis=1, inplace=True)

    # Impute numeric columns with mean
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].mean())

    # Impute categorical columns with mode
    for col in df.select_dtypes(exclude=np.number).columns:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ── 3. Vector Store Builder ───────────────────────────────────────────────────

def build_vector_store(file_path: str) -> FAISS:
    """
    Reads the CSV in chunks, cleans each chunk, converts rows to text,
    splits into overlapping text chunks, embeds with HuggingFace, and
    stores in a FAISS vector index.

    Chunked reading keeps memory usage low even for large datasets.

    Returns a FAISS vectorstore ready for retrieval.
    """
    print(f"Loading dataset: {file_path}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP
    )

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    all_docs = []
    total_rows = 0

    # Read and process CSV in chunks
    for i, chunk_df in enumerate(pd.read_csv(file_path, chunksize=CHUNK_SIZE_ROWS)):
        cleaned_df  = clean_dataframe(chunk_df)
        chunk_text  = cleaned_df.to_string(index=False)
        text_chunks = text_splitter.split_text(chunk_text)
        chunk_docs  = [Document(page_content=tc) for tc in text_chunks]
        all_docs.extend(chunk_docs)
        total_rows += len(cleaned_df)

        if (i + 1) % 10 == 0:
            print(f"  Processed {total_rows:,} rows, {len(all_docs)} document chunks so far...")

    print(f"\nDataset loaded: {total_rows:,} rows → {len(all_docs)} document chunks")
    print("Building FAISS vector index...")

    vectorstore = FAISS.from_documents(all_docs, embeddings)
    print("Vector index ready.\n")
    return vectorstore


# ── 4. Query Engine ───────────────────────────────────────────────────────────

def ask_question(query: str, retriever) -> str:
    """
    Retrieves the most relevant dataset context for the query using
    semantic search, then sends the context + question to Google Gemini
    for a grounded, accurate answer.

    Parameters
    ----------
    query     : Natural language question about the dataset
    retriever : FAISS retriever object

    Returns
    -------
    Gemini's response as a plain string.
    """
    # Step 1: Semantic retrieval from vector store
    results = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in results[:TOP_K_RESULTS]])

    # Step 2: Build grounded prompt
    prompt = f"""
You are a skilled data analyst assistant.
Use only the dataset context provided below to answer the question.
If the answer cannot be determined from the context, say so clearly.

Dataset Context:
{context}

Question:
{query}

Provide a clear, concise answer with reasoning. Include calculations if needed.
"""

    # Step 3: Call Gemini
    model    = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()


# ── 5. Main Pipeline ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("     LLM-POWERED DATA QUERY ASSISTANT")
    print("=" * 60)

    # ── API key ──────────────────────────────────────────────────────────────
    try:
        configure_api()
        print("API key configured.\n")
    except EnvironmentError as e:
        print(f"[Error] {e}")
        sys.exit(1)

    # ── Dataset path ─────────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter path to your CSV dataset: ").strip()

    if not os.path.isfile(file_path):
        print(f"[Error] File not found: {file_path}")
        sys.exit(1)

    # ── Build vector store ───────────────────────────────────────────────────
    try:
        vectorstore = build_vector_store(file_path)
        retriever   = vectorstore.as_retriever()
    except Exception as e:
        print(f"[Error] Failed to build vector store: {e}")
        sys.exit(1)

    # ── Interactive Q&A loop ─────────────────────────────────────────────────
    print("Dataset indexed and ready.")
    print("Ask any question about your dataset. Type 'exit' to quit.\n")
    print("-" * 60)

    while True:
        try:
            user_query = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break

        if not user_query:
            continue

        if user_query.lower() in {"exit", "quit", "q"}:
            print("Exiting Q&A. Goodbye!")
            break

        try:
            answer = ask_question(user_query, retriever)
            print(f"\nAnswer:\n{answer}")
            print("-" * 60)
        except Exception as e:
            print(f"[Error] {e}")


if __name__ == "__main__":
    main()
