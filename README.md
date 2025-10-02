
---

# **CustomGPT Chat Application with RAG**

CustomGPT is an AI-powered chat application that allows users to interact with a conversational AI model. This project also integrates **RAG (Retrieval-Augmented Generation)** to provide answers based on specific PDF documents, making responses more accurate and context-aware.

### **Features:**

* User-friendly chat interface
* Concise responses for general queries
* Detailed explanations for programming or technical questions
* **RAG-enabled answers using PDFs**
* Code block support with syntax highlighting
* Contextual responses based on uploaded documents

### **RAG Implementation:**

This project uses **RAG (Retrieval-Augmented Generation)** to enhance AI responses.

* The chatbot can search through PDFs and retrieve relevant information to answer user queries.
* **PDFs used in this project:**

  * Climate Change
  * Machine Learning

#### **RAG Workflow:**

```
User Query
    │
    ▼
ChatManager receives message
    │
    ▼
RAG Retriever searches PDF embeddings (FAISS index)
    │
    ▼
Relevant content fetched from PDFs
    │
    ▼
Groq API (or AI Model) generates answer using retrieved content
    │
    ▼
Answer displayed to user
```

* This ensures that answers are **contextual and sourced from the PDFs** you added.

### **Prerequisites:**

* Anaconda or Miniconda
* Python 3.10

### **Setup Instructions:**

1. **Create a Conda Environment:**

   ```
   conda create -n gptenv python==3.10 -y
   ```

2. **Activate the Environment:**

   ```
   conda activate gptenv
   ```

3. **Install Requirements:**

   ```
   pip install -r requirements.txt
   ```

4. **Run the Application:**

   ```
   python app.py
   ```

### **Usage:**

* Open your browser and navigate to `http://127.0.0.1:5000/`.
* Ask general or technical questions.
* Queries related to **Climate Change** or **Machine Learning** will utilize **RAG** to provide answers directly from the PDFs.

---



