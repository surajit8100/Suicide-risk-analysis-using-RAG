# Retrieval pipeline--src/retriever.py
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ===========================================================================
# 1. Configuration Data Class
# ===========================================================================
@dataclass
class RetrieverConfig:
    """Configuration settings for connecting to and querying Chroma DB."""
    persist_directory: str = "vectorstore/chroma_db"
    collection_name: str = "train_collection"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    default_top_k: int = 3
    deduplicate_by_parent: bool = True


# ===========================================================================
# 2. Main Systematic Retriever Class
# ===========================================================================
class SimilarityRetriever:
    """
    Modular retriever to connect to an existing persisted Chroma vector store,
    perform similarity search, MMR diversity search, metadata filtering,
    and parent-document deduplication.
    """

    def __init__(self, config: Optional[RetrieverConfig] = None):
        self.config = config or RetrieverConfig()
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vector_store: Optional[Chroma] = None
        
        # Connect to Chroma on initialization
        self._connect_vector_store()

    # -----------------------------------------------------------------------
    # Step A: Load Embeddings and Connect to Vector Store
    # -----------------------------------------------------------------------
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Initializes and returns the cached Hugging Face embedding function."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config.model_name,
                model_kwargs={"device": self.config.device},
                encode_kwargs={"normalize_embeddings": True}
            )
        return self._embeddings

    def _connect_vector_store(self) -> Chroma:
        """Connects to the persisted Chroma database index on disk."""
        db_path = Path(self.config.persist_directory)
        if not db_path.exists():
            raise FileNotFoundError(
                f"[Error] Chroma database directory not found at: '{db_path.resolve()}'. "
                f"Please run ingestion first."
            )

        print(f"[*] Connecting to persisted Chroma DB at: '{db_path.resolve()}'")
        embeddings = self.get_embeddings()

        self._vector_store = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=embeddings,
            persist_directory=str(db_path)
        )
        print(f"[✓] Connected successfully to collection: '{self.config.collection_name}'\n")
        return self._vector_store

    # -----------------------------------------------------------------------
    # Step B: Standard Similarity Search (with deduplication)
    # -----------------------------------------------------------------------
    def similarity_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        label_filter: Optional[str] = None,
        deduplicate: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes similarity search with distance score calculation.
        """
        k = top_k or self.config.default_top_k
        should_dedup = deduplicate if deduplicate is not None else self.config.deduplicate_by_parent
        
        fetch_k = k * 3 if should_dedup else k
        where_filter = {"label": label_filter} if label_filter else None

        raw_results = self._vector_store.similarity_search_with_score(
            query=query,
            k=fetch_k,
            filter=where_filter
        )

        results: List[Dict[str, Any]] = []
        seen_parents = set()

        for doc, score in raw_results:
            parent_id = doc.metadata.get("original_id", "N/A")

            if should_dedup and parent_id in seen_parents:
                continue

            seen_parents.add(parent_id)

            results.append({
                "chunk_id": doc.metadata.get("id", "N/A"),
                #"original_id": parent_id,
                "label": doc.metadata.get("label", "N/A"),
                #"chunk_index": doc.metadata.get("chunk_index", 0) + 1,
                #"total_chunks": doc.metadata.get("total_chunks", 1),
                "distance_score": round(float(score), 4),
                "content": doc.page_content.strip()
            })

            if len(results) >= k:
                break

        return results

    def format_context_for_prompt(self, retrieved_posts: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved posts into a structured, labeled text block
        ready for insertion into the LLM prompt's {context} variable.
        """
        if not retrieved_posts:
            return "No reference historical posts available."

        context_blocks = []
        for rank, post in enumerate(retrieved_posts, start=1):
            label = str(post.get("label", "unknown")).upper()
            content = post.get("content", post.get("text", "")).strip()
            score = post.get("distance_score", post.get("score", None))

            score_str = f" | Distance Score: {score:.4f}" if score is not None else ""

            block = (
                f"--- Reference Post #{rank} [True Severity: {label}{score_str}] ---\n"
                f"\"{content}\""
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)
