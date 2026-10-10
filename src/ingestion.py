import os
import json
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROCESSED_DATA_DIR = r"c:\depression-suicide\data\processed"
CHROMA_PERSIST_DIR = r"c:\depression-suicide\vectorstore\chroma_db"
COLLECTION_NAME = "depression_severity_posts"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_train_documents(train_csv_path: str = None, max_docs: int = 3000) -> List[Document]:
    if train_csv_path is None:
        train_csv_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
        
    print(f"[*] Loading training posts from: {train_csv_path}", flush=True)
    df = pd.read_csv(train_csv_path)
    
    if len(df) > max_docs:
        print(f"[*] Sampling {max_docs} balanced posts for fast vector database indexing...", flush=True)
        sampled_dfs = []
        n_per_class = max_docs // len(df['label'].unique())
        for label, group in df.groupby('label'):
            sampled_dfs.append(group.sample(n=min(len(group), n_per_class), random_state=42))
        df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    
    for idx, row in df.iterrows():
        doc_id = str(row['id']) if 'id' in row and pd.notnull(row['id']) else f"post_{idx}"
        label = str(row['label']) if 'label' in row else "unknown"
        text = str(row['text_clean']) if pd.notnull(row['text_clean']) else str(row['text'])
        
        if len(text.strip()) == 0:
            continue
            
        metadata = {
            "id": doc_id,
            "label": label,
            "char_len": len(text)
        }
        
        if len(text) <= 800:
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        else:
            chunks = text_splitter.split_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_meta = metadata.copy()
                chunk_meta["chunk_id"] = f"{doc_id}_c{chunk_idx}"
                documents.append(Document(page_content=chunk, metadata=chunk_meta))
                
    print(f"[+] Loaded and chunked {len(documents)} document objects from training set.", flush=True)
    return documents

def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    print(f"[*] Initializing HuggingFace Embeddings model: {model_name}...", flush=True)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

def build_vector_store(
    documents: List[Document] = None,
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 500
) -> Chroma:
    if documents is None:
        documents = load_train_documents()
        
    embeddings = get_embedding_model()
    
    print(f"[*] Building Chroma DB vector collection at: {persist_dir}", flush=True)
    os.makedirs(persist_dir, exist_ok=True)
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"[*] Adding {len(documents)} documents to Chroma in batches of {batch_size}...", flush=True)
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        ids = [f"{doc.metadata.get('id', 'doc')}_{i+j}" for j, doc in enumerate(batch)]
        
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"    -> Indexed {min(i + batch_size, len(documents))}/{len(documents)} documents...", flush=True)
        
    print(f"[+] Successfully built and persisted Chroma DB vector store at: {persist_dir}", flush=True)
    return vectorstore

def load_existing_vector_store(persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = COLLECTION_NAME) -> Chroma:
    embeddings = get_embedding_model()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore

if __name__ == "__main__":
    build_vector_store()
