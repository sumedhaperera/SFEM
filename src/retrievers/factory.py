# retrievers/factory.py
from src.retrievers.base import Retriever
from src.retrievers.qdrant_retriever import QdrantRetriever

def get_retriever() -> Retriever:
    # In the future, switch on env or config to return different backends
    return QdrantRetriever()

