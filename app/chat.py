from app.api_client import GroqClient
from app.rag_retriever import retrieve  # Import your RAG function

class ChatManager:
    def __init__(self):
        self.client = GroqClient()
        self.conversation_history = []

    def add_message(self, role, content):
        self.conversation_history.append({"role": role, "content": content})

    def get_response(self, user_message):
        self.add_message("user", user_message)

        # Step 1: Retrieve relevant documents from PDFs
        relevant_docs = retrieve(user_message, top_k=3)
        context = "\n".join(relevant_docs)

        # Step 2: Prepare the prompt for GPT
        prompt = f"Answer the question based on the following documents:\n{context}\n\nQuestion: {user_message}"

        messages = [{"role": "user", "content": prompt}]

        # Step 3: Get response from Groq API
        ai_response = self.client.get_response(messages)
        self.add_message("assistant", ai_response)

        return ai_response
