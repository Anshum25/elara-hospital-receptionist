"""
Configuration Settings for Hospital Video Agent

WHY this file?
--------------
Instead of hardcoding values like "llama3.2:3b" or "../data/chroma_db" in every file,
we put them here. If you want to switch to a different model or change the database 
location, you change ONE file — not 10.

HOW it works:
-------------
Other files import from here:
    from backend.config import Settings
    settings = Settings()
    print(settings.LLM_MODEL)  # "llama3.2:3b"

WHAT each setting means:
------------------------
Explained with comments below.
"""

import os
from pathlib import Path


# Get the absolute path of this project's root directory
# Path(__file__) = this file's path (backend/config.py)
# .parent = backend/
# .parent.parent = hospital-video-agent/ (project root)
PROJECT_ROOT = Path(__file__).parent.parent


class Settings:
    """Central configuration for the entire application."""
    
    # ==================== LLM Settings ====================
    # 
    # LLM_MODEL: Which AI model Ollama should run
    #   - "llama3.2:3b" → 3 billion params, ~2GB RAM, fast (OUR CHOICE for 8GB Mac)
    #   - "llama3.1:8b" → 8 billion params, ~5GB RAM, smarter but needs 16GB
    #   - "mistral:7b"  → 7 billion params, ~4.5GB RAM, alternative option
    #   - "phi3:mini"   → 3.8 billion params, ~2.5GB RAM, Microsoft's model
    #
    LLM_MODEL: str = "llama3.2:3b"
    
    # TEMPERATURE: Controls randomness of responses (0.0 to 1.0)
    #   - 0.0 = always same answer (deterministic) — good for factual Q&A
    #   - 0.5 = balanced (OUR CHOICE)
    #   - 1.0 = very creative/random — bad for medical info
    LLM_TEMPERATURE: float = 0.5
    
    # BASE_URL: Where Ollama is running. Default is localhost (your Mac).
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # ==================== Embedding Settings ====================
    #
    # EMBEDDING_MODEL: Converts text to number vectors for similarity search
    #   - "all-MiniLM-L6-v2" → Fast, 384-dim vectors, good quality (OUR CHOICE)
    #   - "all-mpnet-base-v2" → Better quality, 768-dim, but slower
    #   - "paraphrase-multilingual-MiniLM-L12-v2" → If you need multilingual support
    #
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # ==================== Vector Database Settings ====================
    #
    # CHROMA_DB_DIR: Where ChromaDB stores its data on disk
    #   - This is a folder that ChromaDB creates and manages
    #   - Delete this folder to "reset" the knowledge base
    #
    CHROMA_DB_DIR: str = str(PROJECT_ROOT / "data" / "chroma_db")
    
    # COLLECTION_NAME: Name of the "table" inside ChromaDB
    #   - Like a table name in a regular database
    #
    CHROMA_COLLECTION_NAME: str = "hospital_knowledge"
    
    # ==================== RAG Settings ====================
    #
    # CHUNK_SIZE: How many characters per "chunk" when splitting documents
    #   WHY chunk? LLMs have a limited "context window" (how much text they can read).
    #   We split long documents into smaller pieces so we can feed ONLY the relevant
    #   pieces to the LLM, not the entire document.
    #   - 500 = good for Q&A (small, focused chunks)
    #   - 1000 = good for detailed answers (more context per chunk)
    #
    CHUNK_SIZE: int = 500
    
    # CHUNK_OVERLAP: How many characters overlap between consecutive chunks
    #   WHY overlap? If a sentence gets split across two chunks, the overlap ensures
    #   both chunks have the complete sentence. Without overlap, you might lose context.
    #   Rule of thumb: 10-20% of chunk_size
    #
    CHUNK_OVERLAP: int = 50
    
    # RETRIEVAL_K: How many relevant chunks to retrieve for each question
    #   - 3 = standard (OUR CHOICE) — gives enough context without overwhelming
    #   - 5 = more context but slower and might dilute relevance
    #   - 1 = minimal, might miss important info
    #
    RETRIEVAL_K: int = 3
    
    # ==================== Data Paths ====================
    HOSPITAL_DATA_PATH: str = str(PROJECT_ROOT / "data" / "hospital_knowledge.json")
    
    # ==================== Server Settings ====================
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # ==================== System Prompt ====================
    # This tells the LLM WHO it is and HOW to behave.
    # Think of it as the "job description" for the AI.
    SYSTEM_PROMPT: str = """You are a friendly and professional AI receptionist for MedCare General Hospital. 
Your name is MedBot.

Your responsibilities:
1. Answer questions about hospital services, departments, doctors, and facilities
2. Help patients book appointments
3. Provide information about health packages and checkups
4. Guide patients to the right department based on their symptoms
5. Handle general inquiries about visiting hours, parking, insurance, etc.

Important rules:
- Always be polite, empathetic, and professional
- If you're unsure about something, say so and suggest contacting the hospital directly
- NEVER provide medical diagnosis or treatment advice — always recommend consulting a doctor
- Use the information provided in the context to give accurate answers
- If the patient seems to have an emergency, immediately direct them to call +91 22-4567-8911
- Keep responses concise but helpful (2-3 sentences when possible)
- If the context doesn't contain the answer, be honest about it

Remember: You represent MedCare General Hospital. Be warm, helpful, and accurate."""


# Create a singleton instance for easy importing
settings = Settings()
