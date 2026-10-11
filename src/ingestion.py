#Final
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import torch

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Maximize multi-core CPU performance
torch.set_num_threads(os.cpu_count() or 4)

@dataclass
class IngestionConfig:
    csv_path: str = "data/processed/train.csv"
    text_column: str = "text"
    id_column: str = "id"
    label_column: str = "label"

    # SPEED TWEAK 1: 700 chars reduces chunk count by ~40% while staying well under 256 tokens
    chunk_size: int = 700
    chunk_overlap: int = 70
    separators: List[str] = field(default_factory=lambda: [
        "\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""
    ])

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # SPEED TWEAK 2: 256 internal batch size for PyTorch
    embed_batch_size: int = 256
    
    # SPEED TWEAK 3: 2,000 batch size for Chroma to minimize disk writes
    persist_directory: str = "vectorstore/chroma_db"
    collection_name: str = "train_collection"
    batch_size: int = 2000


class IngestionPipeline:
    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        self._splitter: Optional[RecursiveCharacterTextSplitter] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vector_store: Optional[Chroma] = None

    def load_csv(self) -> pd.DataFrame:
        csv_file = Path(self.config.csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"[Error] CSV not found at: {csv_file.resolve()}")

        print(f"[1/4] Loading CSV from: {csv_file}")
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=[self.config.text_column])
        print(f"      Loaded {len(df)} records.")
        return df

    def convert_to_documents(self, df: pd.DataFrame) -> List[Document]:
        print(f"[2/4] Chunking text (chunk_size={self.config.chunk_size})...")
        if self._splitter is None:
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=self.config.separators
            )

        documents: List[Document] = []
        for _, row in df.iterrows():
            content = str(row[self.config.text_column]).strip()
            base_id = str(row[self.config.id_column])
            label = str(row[self.config.label_column]) if pd.notnull(row[self.config.label_column]) else ""

            chunks = self._splitter.split_text(content)
            for idx, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "id": f"{base_id}_{idx}",
                        "original_id": base_id,
                        "label": label,
                        "chunk_index": idx,
                        "total_chunks": len(chunks)
                    }
                ))

        print(f"      Total chunks generated: {len(documents)}")
        return documents

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            print(f"[3/4] Initializing embeddings on device: {self.config.device}...")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config.model_name,
                model_kwargs={"device": self.config.device},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": self.config.embed_batch_size  # Fast batching
                }
            )
        return self._embeddings

    def build_vector_store(self, documents: List[Document]) -> Chroma:
        persist_path = Path(self.config.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        embeddings = self.get_embeddings()
        print(f"[4/4] Ingesting {len(documents)} chunks into Chroma (batch size = {self.config.batch_size})...")

        vector_store = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_path)
        )

        total_docs = len(documents)
        batch_size = self.config.batch_size

        for i in range(0, total_docs, batch_size):
            batch = documents[i : i + batch_size]
            vector_store.add_documents(documents=batch)
            print(f"      -> Ingested {min(i + batch_size, total_docs)} / {total_docs} chunks...")

        print(f"      Ingestion complete! Persisted to '{persist_path}'.")
        self._vector_store = vector_store
        return vector_store

    def run(self) -> Chroma:
        df = self.load_csv()
        documents = self.convert_to_documents(df)
        return self.build_vector_store(documents)

