import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_index(docs_file):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with open(docs_file) as f:
        docs = [line.strip() for line in f]
    embeds = model.encode(docs)
    dim = embeds.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeds))
    faiss.write_index(index, 'data/faiss.index')
    # save docs too
    np.save('data/docs.npy', np.array(docs))

if __name__ == '__main__':
    build_index('../data/docs.txt')