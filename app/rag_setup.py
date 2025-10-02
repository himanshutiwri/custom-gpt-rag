import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import pickle  # to save  FAISS index and chunks

# 1. Load PDF files
pdf_folder = os.path.join(os.path.dirname(__file__), '../pdfs')
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

chunks = []
for file in pdf_files:
    reader = PdfReader(os.path.join(pdf_folder, file))
    for page in reader.pages:
        text = page.extract_text()
        if text:
            # Split into smaller paragraphs
            for paragraph in text.split('\n\n'):
                paragraph = paragraph.strip()
                if paragraph:
                    chunks.append(paragraph)

print(f"Total chunks: {len(chunks)}")

# 2. Create embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(chunks)

# 3. Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 4. Save index and chunks for later use
with open(os.path.join(os.path.dirname(__file__), 'faiss_index.pkl'), 'wb') as f:
    pickle.dump((index, chunks), f)

print("FAISS index saved!")
