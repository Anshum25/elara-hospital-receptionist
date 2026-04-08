
import logging
import sys
from backend.rag_pipeline import HospitalRAGPipeline
from backend.agent import HospitalAgent
from backend.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_agent_greeting():
    print("\n--- Testing Agent Greeting ---")
    rag = HospitalRAGPipeline()
    agent = HospitalAgent(rag)
    
    test_input = "Hi."
    print(f"User: {test_input}")
    
    result = agent.process_message(test_input)
    print(f"Agent: {result['answer']}")

if __name__ == "__main__":
    test_agent_greeting()
