# app/rag_retriever.py
import os
import pickle
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

PDF_FOLDER = os.path.join(os.path.dirname(__file__), "../pdfs")
INDEX_FILE = os.path.join(os.path.dirname(__file__), "faiss_index.pkl")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load or create FAISS index
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, 'rb') as f:
        index, docs = pickle.load(f)
else:
    docs = []
    embeddings = []

    # Read PDFs
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    docs.append(text)
                    embeddings.append(model.encode(text))

    # Build FAISS index
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    # Save index
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump((index, docs), f)

# Function to retrieve top-k relevant docs
def retrieve(query, top_k=3):
    q_emb = model.encode(query)
    D, I = index.search(np.array([q_emb]).astype("float32"), top_k)
    results = [docs[i] for i in I[0]]
    return results

