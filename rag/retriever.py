"""
Semantic Search Retriever for RAG System
=========================================
Performs similarity search on vector database to find relevant document chunks
for a given query. Uses cosine similarity on embedded vectors.

Architecture:
    Query → Embedding → Vector Search → Top-K Results
    
Cost: 100% FREE (local ChromaDB + local embeddings)
"""

from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from configs.settings import settings


def retrieve(query: str, k: int = 5, collection_name: str = "knowledge_base") -> List[Dict[str, Any]]:
    """
    Retrieve top-K most relevant document chunks for a query.
    
    Args:
        query: Natural language question or search query
        k: Number of top results to return
        collection_name: ChromaDB collection to search
        
    Returns:
        List of dicts with 'text', 'source', and 'distance' keys
        
    Example:
        results = retrieve("What is model collapse?", k=3)
        for result in results:
            print(f"Source: {result['source']}")
            print(f"Text: {result['text'][:200]}...")
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    
    # Load embedding function (same as ingestion)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    
    # Get collection
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )
    except Exception as e:
        raise ValueError(f"Collection '{collection_name}' not found. Run ingestion first.") from e
    
    # Perform similarity search
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    
    # Format results
    formatted_results = []
    if results["documents"] and len(results["documents"]) > 0:
        for i, doc in enumerate(results["documents"][0]):
            formatted_results.append({
                "text": doc,
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i] if results["distances"] else None
            })
    
    print(f"🔍 Found {len(formatted_results)} relevant chunks")
    return formatted_results


if __name__ == "__main__":
    # Test retrieval
    test_query = "What is model collapse in AI?"
    print(f"\n📝 Test query: {test_query}\n")
    results = retrieve(test_query, k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"📄 Source: {result['source']}")
        print(f"� Distance: {result['distance']:.4f}")
        print(f"📝 Text: {result['text'][:200]}...")
