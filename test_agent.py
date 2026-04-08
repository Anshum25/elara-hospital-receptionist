import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

sys.path.append(str(Path(__file__).parent))

from backend.rag_pipeline import HospitalRAGPipeline
from backend.agent import HospitalAgent

def run_chat_loop():
    print("\n" + "="*50)
    print("🤖 Starting Hospital Agent Chat Test")
    print("="*50)
    
    # 1. Initialize core systems
    print("\nInitializing RAG Pipeline... (Loading DB)")
    rag = HospitalRAGPipeline()
    
    print("Initializing Agent... (Equipping Tools)")
    agent = HospitalAgent(rag)
    
    print("\n" + "="*50)
    print("✅ System Ready! Type 'quit' to exit.")
    print("Try asking it to book an appointment for Cardiology tomorrow.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("\n👨‍🦰 PATIENT: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            print("\n🤖 AGENT THINKING...")
            # This triggers the agent execution
            result = agent.process_message(user_input)
            
            print(f"\n👩‍⚕️ MEDBOT: {result['answer']}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    run_chat_loop()
