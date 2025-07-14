import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google.generativeai import client as gemini
from dotenv import load_dotenv

# Load env
load_dotenv()
gemini.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Load index & docs
index = faiss.read_index('data/faiss.index')
docs = np.load('data/docs.npy', allow_pickle=True)
model = SentenceTransformer('all-MiniLM-L6-v2')


def retrieve(question, top_k=5):
    q_emb = model.encode([question])
    D, I = index.search(np.array(q_emb), top_k)
    return [docs[i] for i in I[0]]


def ask_gemini(question, context):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"  
    response = gemini.chat(messages=[{'role':'user','content':prompt}])
    return response.choices[0].message.content


def rag_answer(question):
    snippets = retrieve(question)
    ctx = "\n".join(snippets)
    return ask_gemini(question, ctx)


if __name__ == '__main__':
    while True:
        q = input("You: ")
        if q.lower() in ('exit','quit'):
            break
        print("Bot:", rag_answer(q))