from langchain_core.embeddings import Embeddings
import random

class SimpleEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[random.random() for _ in range(384)] for _ in texts]

    def embed_query(self, text):
        return [random.random() for _ in range(384)]

def get_embeddings():
    return SimpleEmbeddings()