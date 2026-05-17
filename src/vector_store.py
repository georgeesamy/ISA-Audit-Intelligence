import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_DB_PATH, COLLECTION_NAME, SIMILARITY_METRIC


class ISAVectorIndex:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": SIMILARITY_METRIC},
        )

    def similarity_search(self, query: str) -> str:
        query_vector = self._encoder.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=1,
            include=["documents", "metadatas"],
        )
        if results["documents"] and results["documents"][0]:
            text = results["documents"][0][0]
            meta = results["metadatas"][0][0] if results["metadatas"] else {}
            source = meta.get("source", "")
            para_id = meta.get("para_id", "")
            m = re.match(r'ISA\s*(\d+)', os.path.basename(source), re.IGNORECASE)
            isa_name = f"ISA {m.group(1)}" if m else os.path.splitext(os.path.basename(source))[0]
            citation = f"[{isa_name}, Para {para_id}]" if isa_name and para_id else ""
            return f"{citation}\n{text}" if citation else text
        return ""
