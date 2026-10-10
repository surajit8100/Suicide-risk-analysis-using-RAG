import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from src.ingestion import load_existing_vector_store, CHROMA_PERSIST_DIR, COLLECTION_NAME

class SeverityRetriever:
    """Retriever wrapper for fetching relevant labeled posts from Chroma DB."""
    
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = COLLECTION_NAME):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.vectorstore = None
        
    def _ensure_loaded(self):
        if self.vectorstore is None:
            if not os.path.exists(self.persist_dir):
                raise FileNotFoundError(f"Chroma DB vector store directory not found at {self.persist_dir}. Run ingestion.py first!")
            self.vectorstore = load_existing_vector_store(self.persist_dir, self.collection_name)

    def retrieve_similar_posts(
        self, 
        query: str, 
        top_k: int = 5, 
        use_mmr: bool = False,
        label_filter: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k similar posts for a given query text.
        
        Returns:
            List of dicts: [{'id': str, 'text': str, 'label': str, 'score': float}]
        """
        self._ensure_loaded()
        
        search_kwargs = {}
        if label_filter:
            search_kwargs["filter"] = {"label": label_filter}
            
        results = []
        if use_mmr:
            docs = self.vectorstore.max_marginal_relevance_search(
                query=query, 
                k=top_k, 
                fetch_k=top_k * 3,
                **search_kwargs
            )
            for doc in docs:
                results.append({
                    "id": doc.metadata.get("id", "unknown"),
                    "text": doc.page_content,
                    "label": doc.metadata.get("label", "unknown"),
                    "score": None
                })
        else:
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query=query, 
                k=top_k,
                **search_kwargs
            )
            for doc, score in docs_and_scores:
                results.append({
                    "id": doc.metadata.get("id", "unknown"),
                    "text": doc.page_content,
                    "label": doc.metadata.get("label", "unknown"),
                    "score": float(score)
                })
                
        return results

    def format_context_for_prompt(self, retrieved_posts: List[Dict[str, Any]]) -> str:
        """Formats retrieved labeled posts into a clean context string for LLM prompting."""
        if not retrieved_posts:
            return "No historical reference posts found."
            
        formatted_blocks = []
        for i, post in enumerate(retrieved_posts, start=1):
            block = (
                f"Reference Example #{i}:\n"
                f"Historical Severity Label: {post['label'].upper()}\n"
                f"Post Text: \"{post['text']}\"\n"
            )
            formatted_blocks.append(block)
            
        return "\n-------------------\n".join(formatted_blocks)

if __name__ == "__main__":
    retriever = SeverityRetriever()
    test_query = "I feel so lonely and tired of failing every single day."
    print(f"[*] Testing retriever with query: '{test_query}'")
    try:
        posts = retriever.retrieve_similar_posts(test_query, top_k=3)
        for p in posts:
            print(f"[{p['label'].upper()}] Score: {p['score']} | {p['text'][:80]}...")
    except Exception as e:
        print("[!] Retrieval test error:", e)
