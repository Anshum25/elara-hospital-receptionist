import sys
from pathlib import Path
import logging

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Ensure the hospital-video-agent directory is in the Python path
sys.path.append(str(Path(__file__).parent))

from backend.rag_pipeline import HospitalRAGPipeline

def test():
    print("\n" + "="*50)
    print("🧪 Testing RAG Pipeline")
    print("="*50)
    
    # 1. Initialize (will load JSON and build ChromaDB on first run)
    print("\n[1] Initializing Pipeline (This may take a minute if it's the first time)...")
    pipeline = HospitalRAGPipeline()
    
    # 2. Check stats
    print("\n[2] Knowledge Base Stats:")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"  - {key}: {value}")
        
    # 3. Test a query
    print("\n[3] Testing Query...")
    query = "What should i do if i have a fever?"
    print(f"\n🗣️ Question: {query}")
    
    result = pipeline.query(query)
    
    print(f"\n🤖 Answer: {result['answer']}")
    
    print("\n📚 Sources used:")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n--- Source {i} ---")
        print(source[:200] + "..." if len(source) > 200 else source)
        
    print("\n" + "="*50)
    print("✅ Test Complete!")

if __name__ == '__main__':
    test()
