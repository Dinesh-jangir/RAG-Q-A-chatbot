import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai  # updated import for Gemini API client

# Load environment variables
load_dotenv()
# Initialize Gemini client (picks API key from GEMINI_API_KEY env)
client = genai.Client()

# Load FAISS index and documents
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
index = faiss.read_index(os.path.join(base_dir, 'data', 'faiss.index'))
docs = np.load(os.path.join(base_dir, 'data', 'docs.npy'), allow_pickle=True)
# SentenceTransformer for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')


def retrieve(question, top_k=5):
    """
    Retrieve top_k relevant docs for the question.
    """
    q_emb = model.encode([question])
    D, I = index.search(np.array(q_emb), top_k)
    return [docs[i] for i in I[0]]


def ask_gemini(question, context):
    """
    Send a prompt (context + question) to Gemini via generate_content.
    """
    prompt = f"Context: {context}
Question: {question}
Answer:"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


def rag_answer(question):
    """
    Complete RAG pipeline: retrieve, then generate answer.
    """
    snippets = retrieve(question)
    ctx = "
".join(snippets)
    return ask_gemini(question, ctx)


if __name__ == '__main__':
    while True:
        q = input("You: ")
        if q.lower() in ('exit', 'quit'):
            break
        print("Bot:", rag_answer(q))