"""
Multi-Agent RAG Research Assistant
Main entry point
"""
import sys
from pathlib import Path
from rag.ingest import ingest_documents
from rag.retriever import retrieve
from crew.orchestrator import run_research_crew, save_outputs

def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print("🤖  Multi-Agent RAG Research Assistant")
    print("=" * 70)
    print("Powered by: CrewAI + OpenAI + ChromaDB")
    print("=" * 70)
    print()

def main():
    """Main application logic."""
    print_banner()
    
    # Check for --ingest flag
    if "--ingest" in sys.argv:
        print("📚 Starting document ingestion...\n")
        ingest_documents()
        return
    
    # Get query from command line or prompt
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your research question: ").strip()
    
    if not query:
        print("❌ Error: No query provided")
        print("\nUsage:")
        print("  python3 main.py 'Your research question here'")
        print("  python3 main.py --ingest  # To ingest documents")
        return
    
    print(f"\n🚀 Starting research on: {query}\n")
    
    # Step 1: RAG Retrieval
    print("📚 Retrieving context...")
    results = retrieve(query, k=5)
    
    if not results:
        print("❌ Error: No relevant context found. Run with --ingest first.")
        return
    
    # Format results into context string
    context = "\n\n".join([f"Source: {r['source']}\n{r['text']}" for r in results])
    
    # Step 2: Multi-Agent Processing
    print("\n🤖 Running agents...")
    print("-" * 70)
    
    result = run_research_crew(query, context)
    
    # Step 3: Save outputs
    save_outputs(result["markdown"], result["linkedin_post"])
    
    print("\n" + "-" * 70)
    print("\n⚠️  Using fallback format")
    print("\n" + "=" * 70)
    print("✅ Complete!")
    print("=" * 70)
    print()
    print("📄 Brief: artifacts/brief.md")
    print("📱 Post:  artifacts/linkedin_post.md")
    print()
    
    print("\n✅ Complete! Check artifacts/ folder.")

if __name__ == "__main__":
    main()