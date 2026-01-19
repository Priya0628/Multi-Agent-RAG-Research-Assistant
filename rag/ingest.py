"""
Document Ingestion Pipeline for RAG System
===========================================
Loads text documents, chunks them into semantic units, generates embeddings,
and stores in ChromaDB vector database for semantic search.

Architecture:
    Documents → Chunking → Embeddings → Vector DB Storage
    
Cost: 100% FREE (uses local SentenceTransformers model)
"""

from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from configs.settings import settings


def load_documents(data_dir: str = "data") -> List[Dict[str, Any]]:
    """
    Load all .txt files from data directory.
    
    Args:
        data_dir: Path to directory containing text files
        
    Returns:
        List of dicts with 'text' and 'source' keys
    """
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    txt_files = list(data_path.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {data_dir}")
    
    for file_path in txt_files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        documents.append({
            "text": text,
            "source": file_path.name
        })
        print(f"  ✓ Loaded: {file_path.name} ({len(text):,} chars)")
    
    return documents


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start += (chunk_size - overlap)
    
    return chunks


def ingest_documents(data_dir: str = "data", collection_name: str = "knowledge_base"):
    """
    Complete ingestion pipeline: Load → Chunk → Embed → Store
    
    Args:
        data_dir: Path to documents directory
        collection_name: ChromaDB collection name
    """
    print("\n" + "=" * 60)
    print("📚 Document Ingestion Pipeline")
    print("=" * 60)
    
    # Step 1: Load documents
    print(f"\n📂 Loading documents from: {data_dir}")
    documents = load_documents(data_dir)
    print(f"✅ Loaded {len(documents)} documents")
    
    # Step 2: Chunk documents
    print(f"\n✂️  Chunking documents...")
    all_chunks = []
    metadatas = []
    ids = []
    
    chunk_id = 0
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            metadatas.append({"source": doc["source"]})
            ids.append(f"chunk_{chunk_id}")
            chunk_id += 1
    
    print(f"✅ Created {len(all_chunks)} chunks")
    
    # Step 3: Initialize ChromaDB
    print(f"\n💾 Storing in vector database: {settings.chroma_dir}")
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
        print(f"🗑️  Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create new collection with embedding function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"description": "RAG knowledge base for multi-agent research system"}
    )
    
    # Step 4: Add documents to collection
    collection.add(
        documents=all_chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Stored {len(all_chunks)} chunks in collection '{collection_name}'")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Run ingestion when executed directly
    ingest_documents()
