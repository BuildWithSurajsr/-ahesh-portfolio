from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from app.src.config import Config


class EmbeddingManager:
    def __init__(self):
        self.client = None
        self.collection = None
        self._initialize()

    def _initialize(self):
        embedding_fn = GoogleGenerativeAiEmbeddingFunction(
            api_key=Config.GOOGLE_API_KEY,
            model_name="models/gemini-embedding-001"
        )

        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.collection = self.client.get_collection(
            name=Config.COLLECTION_NAME,
            embedding_function=embedding_fn
        )

    def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        if not k:
            k = Config.TOP_K
        if not self.collection:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )

            documents = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    documents.append({
                        "content": doc,
                        "metadata": metadata
                    })
            return documents
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []
