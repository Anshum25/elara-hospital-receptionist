"""
RAG Pipeline for Hospital Video Agent

=== WHAT IS THIS FILE? ===
This is the "brain" of our hospital agent. It implements RAG (Retrieval-Augmented Generation):
1. Loads hospital data (services, doctors, FAQs, etc.)
2. Stores it in a vector database (ChromaDB) for fast semantic search
3. When a patient asks a question, finds relevant info and generates an answer

=== HOW DOES RAG WORK? (Step by Step) ===

Imagine you're a new receptionist at a hospital. A patient walks in and asks:
"Do you have a cardiologist available on Monday?"

Without RAG (pure LLM): The AI would GUESS — "Most hospitals have cardiologists..."
With RAG:
  Step 1: Search your hospital's info folder for "cardiologist" + "Monday"
  Step 2: Find: "Dr. Rajesh Sharma, Cardiology, available Mon/Wed/Fri"
  Step 3: Answer: "Yes! Dr. Rajesh Sharma is our cardiologist, available on Mondays."

That's EXACTLY what this code does, but with vectors and math instead of a physical folder.

=== KEY CONCEPTS ===

1. EMBEDDING: Turning text → numbers so computers can compare meaning
   "heart doctor" → [0.8, 0.2, 0.9, ...]  (384 numbers)
   "cardiologist" → [0.79, 0.21, 0.88, ...] (similar numbers = similar meaning!)

2. VECTOR STORE: A database that stores these number-arrays and can quickly
   find "which stored text is most similar to this new text?"

3. TEXT SPLITTING: LLMs can only read ~4000 tokens at a time. If our hospital
   data is 10,000 words, we split it into ~20 chunks of 500 chars each.
   Then we only send the 3 MOST RELEVANT chunks to the LLM, not all 20.

4. RETRIEVAL CHAIN: LangChain's way of connecting the steps:
   Question → Search Vector DB → Get top 3 chunks → Feed to LLM → Answer
"""

import json
import logging
import os
from typing import List, Dict, Optional, Tuple

# Prevent ChromaDB from sending anonymous telemetry (fixes posthog errors in the console)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# LangChain imports — each one explained:
#
# HuggingFaceEmbeddings: Converts text to number vectors using open-source models
#   - Downloads and runs the model LOCALLY on your Mac
#   - No API key needed, no internet needed after first download
#
from langchain_community.embeddings import HuggingFaceEmbeddings

# Chroma: The vector database — stores embeddings and searches them
#   - Saves to disk so you don't re-process data every restart
#   - Uses "cosine similarity" to find the closest matches
#
from langchain_community.vectorstores import Chroma

# OllamaLLM: Connects to the Ollama app running on your Mac to use the LLM
#   - Ollama manages the model download, loading, and inference
#   - LangChain just sends prompts and gets responses
#
from langchain_ollama import OllamaLLM as Ollama

# RecursiveCharacterTextSplitter: Splits long text into smaller chunks
#   - "Recursive" means it tries to split on paragraphs first, then sentences,
#     then words, then characters — keeping meaning intact as much as possible
#
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document: LangChain's standard format for a piece of text + metadata
#   - page_content = the actual text
#   - metadata = extra info like {"source": "faq", "category": "insurance"}
#
from langchain_core.documents import Document

# PromptTemplate: A template for the prompt we send to the LLM
#   - Includes placeholders like {context} and {question}
#   - Ensures consistent, well-structured prompts every time
#
from langchain_core.prompts import PromptTemplate

# RetrievalQA: The chain that ties everything together
#   - Takes a question → retrieves relevant docs → generates answer
#
from langchain.chains import RetrievalQA

from backend.config import settings

logger = logging.getLogger(__name__)


class HospitalRAGPipeline:
    """
    The main RAG (Retrieval-Augmented Generation) pipeline.
    
    Think of this class as a smart librarian that:
    1. Organizes all hospital information into a searchable catalog (vector store)
    2. When asked a question, finds the most relevant pages
    3. Hands those pages to an AI (LLM) to write a helpful answer
    """
    
    def __init__(self):
        """
        Initialize the RAG pipeline.
        
        This runs when you create the object: pipeline = HospitalRAGPipeline()
        It sets up three things:
        1. The embedding model (text → numbers converter)
        2. The LLM (the AI that generates answers)
        3. The vector store (loaded from disk or created fresh)
        """
        logger.info("🚀 Initializing Hospital RAG Pipeline...")
        
        # ---- Step 1: Set up the Embedding Model ----
        # 
        # WHAT: Downloads "all-MiniLM-L6-v2" (a small 80MB model) from HuggingFace
        # WHY: We need to convert text to numbers for similarity search
        # HOW: Uses the sentence-transformers library under the hood
        #
        # First run: Downloads the model (~80MB) — takes ~30 seconds
        # After that: Loads from cache — instant
        #
        logger.info("📦 Loading embedding model (first time takes ~30 seconds)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            # model_kwargs: Options passed to the underlying PyTorch model
            model_kwargs={"device": "cpu"},  # Use CPU (safe for all Macs)
            # encode_kwargs: Options for the encoding process
            encode_kwargs={"normalize_embeddings": True}  # Normalize for better cosine similarity
        )
        logger.info("✅ Embedding model loaded!")
        
        # ---- Step 2: Set up the LLM ----
        #
        # WHAT: Connects to Ollama (which is running Llama 3.2 on your Mac)
        # WHY: This is the "brain" that reads context + question and writes answers
        # HOW: Sends HTTP requests to Ollama's local API (localhost:11434)
        #
        logger.info("🤖 Connecting to Ollama LLM...")
        self.llm = Ollama(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            # num_ctx: Context window size (how many tokens the LLM can "see" at once)
            # More = LLM sees more of our retrieved docs, but uses more RAM
            num_ctx=2048,
        )
        logger.info(f"✅ Connected to LLM: {settings.LLM_MODEL}")
        
        # ---- Step 3: Set up the Vector Store ----
        self.vectorstore = None
        self.qa_chain = None
        self._load_or_create_vectorstore()
    
    def _load_or_create_vectorstore(self):
        """
        Try to load an existing vector store from disk.
        If it doesn't exist, create a new one from the hospital data.
        
        WHY do we save to disk?
        Without persistence, we'd have to re-process all hospital data 
        every time we restart the server (embed all text again = ~30 seconds).
        With persistence, restart is instant.
        """
        try:
            # Try loading from disk
            self.vectorstore = Chroma(
                collection_name=settings.CHROMA_COLLECTION_NAME,
                persist_directory=settings.CHROMA_DB_DIR,
                embedding_function=self.embeddings,
            )
            
            # Check if the collection actually has data
            count = self.vectorstore._collection.count()
            
            if count > 0:
                logger.info(f"✅ Loaded existing vector store with {count} chunks")
                self._build_qa_chain()
            else:
                logger.info("📂 Vector store exists but is empty. Loading hospital data...")
                self._initialize_from_hospital_data()
                
        except Exception as e:
            logger.info(f"📂 No existing vector store found. Creating new one... ({e})")
            self._initialize_from_hospital_data()
    
    def _initialize_from_hospital_data(self):
        """
        Load hospital data from JSON and build the vector store.
        
        This is the "indexing" phase — it happens once:
        1. Read the JSON file
        2. Convert each piece into a text Document
        3. Split into chunks
        4. Embed each chunk (text → numbers)
        5. Store in ChromaDB
        """
        try:
            # Load the JSON file we created
            with open(settings.HOSPITAL_DATA_PATH, "r") as f:
                hospital_data = json.load(f)
            
            logger.info("📄 Loaded hospital data from JSON")
            
            # Convert JSON into text documents that the RAG system can use
            documents = self._prepare_documents(hospital_data)
            logger.info(f"📝 Prepared {len(documents)} documents")
            
            # Split documents into smaller chunks
            chunks = self._split_documents(documents)
            logger.info(f"✂️ Split into {len(chunks)} chunks")
            
            # Create vector store from chunks
            # This is where the magic happens:
            # - Each chunk gets converted to 384 numbers (embedding)
            # - ChromaDB stores the text + numbers + metadata
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=settings.CHROMA_COLLECTION_NAME,
                persist_directory=settings.CHROMA_DB_DIR,
            )
            
            # Save to disk so we don't have to do this again
            logger.info(f"💾 Vector store created and saved with {len(chunks)} chunks")
            
            # Build the QA chain
            self._build_qa_chain()
            
        except FileNotFoundError:
            logger.error(f"❌ Hospital data file not found: {settings.HOSPITAL_DATA_PATH}")
            logger.error("Please create the hospital data file first!")
        except Exception as e:
            logger.error(f"❌ Error initializing vector store: {e}")
            raise
    
    def _prepare_documents(self, data: dict) -> List[Document]:
        """
        Convert hospital JSON data into LangChain Document objects.
        
        WHY do we need this conversion?
        The JSON has structured data (nested dictionaries), but the RAG system 
        needs plain text to embed and search. We convert each piece of data 
        into a natural-language text document.
        
        For example, this JSON:
            {"name": "Dr. Rajesh Sharma", "department": "Cardiology", ...}
        Becomes this text:
            "Dr. Rajesh Sharma is a doctor in the Cardiology department..."
        
        This way, when someone asks "Who is the heart doctor?", the embedding 
        of the question will be similar to the embedding of this document.
        """
        documents = []
        
        # ---- Hospital General Info ----
        info = data.get("hospital_info", {})
        if info:
            doc_text = f"""MedCare General Hospital Information:
Hospital Name: {info.get('name', '')}
Tagline: {info.get('tagline', '')}
Address: {info.get('address', '')}
Phone: {info.get('phone', '')}
Emergency Number: {info.get('emergency', '')}
Email: {info.get('email', '')}
Website: {info.get('website', '')}
Established: {info.get('established', '')}
Total Beds: {info.get('beds', '')}
Accreditation: {info.get('accreditation', '')}"""
            
            hours = info.get("operating_hours", {})
            if hours:
                doc_text += f"""
Operating Hours:
- OPD (Outpatient): {hours.get('opd', '')}
- Emergency: {hours.get('emergency', '')}
- Pharmacy: {hours.get('pharmacy', '')}
- Lab: {hours.get('lab', '')}"""
            
            documents.append(Document(
                page_content=doc_text,
                metadata={"source": "hospital_info", "category": "general"}
            ))
        
        # ---- Departments ----
        # Each department becomes its own document (so search can find the specific one)
        for dept in data.get("departments", []):
            doc_text = f"""Department: {dept['name']}
Description: {dept['description']}
Department Head: {dept.get('head', 'Not specified')}
Services Offered: {', '.join(dept.get('services', []))}"""
            
            documents.append(Document(
                page_content=doc_text,
                metadata={"source": "departments", "category": dept["name"].lower()}
            ))
        
        # ---- Doctors ----
        # Each doctor is a separate document
        for doctor in data.get("doctors", []):
            doc_text = f"""Doctor Information:
Name: {doctor['name']}
Department: {doctor['department']}
Qualification: {doctor['qualification']}
Experience: {doctor['experience']}
Specialization: {doctor['specialization']}
Available Days: {doctor['available_days']}
Consultation Fee: Rs. {doctor['consultation_fee']}
Languages Spoken: {', '.join(doctor.get('languages', []))}"""
            
            documents.append(Document(
                page_content=doc_text,
                metadata={"source": "doctors", "category": doctor["department"].lower()}
            ))
        
        # ---- FAQs ----
        for faq in data.get("faqs", []):
            doc_text = f"""Frequently Asked Question:
Q: {faq['question']}
A: {faq['answer']}"""
            
            documents.append(Document(
                page_content=doc_text,
                metadata={"source": "faqs", "category": "general"}
            ))
        
        # ---- Health Packages ----
        for package in data.get("health_packages", []):
            doc_text = f"""Health Package: {package['name']}
Price: Rs. {package['price']}
Duration: {package['duration']}
Includes: {', '.join(package.get('includes', []))}"""
            
            documents.append(Document(
                page_content=doc_text,
                metadata={"source": "health_packages", "category": "packages"}
            ))
        
        # ---- Appointment Rules ----
        rules = data.get("appointment_rules", {})
        if rules:
            doc_text = f"""Appointment Rules and Policies:
Advance Booking: {rules.get('booking_advance', '')}
Cancellation Policy: {rules.get('cancellation_policy', '')}
Rescheduling Policy: {rules.get('rescheduling', '')}
Walk-in Policy: {rules.get('walk_in', '')}
Follow-up Policy: {rules.get('follow_up', '')}
Documents Required: {rules.get('documents_required', '')}"""
            
            documents.append(Document(
                page_content=doc_text,
                metadata={"source": "appointment_rules", "category": "appointments"}
            ))
        
        return documents
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        WHY split?
        -----------
        Imagine searching a 1000-page book for "cardiologist availability".
        It's easier to find the answer in a specific paragraph than 
        searching all 1000 pages, right?
        
        Same idea: smaller chunks = more precise search results.
        
        HOW RecursiveCharacterTextSplitter works:
        ------------------------------------------
        It tries to split text at natural boundaries, in this order:
        1. First, try splitting at paragraph breaks ("\\n\\n")
        2. If chunks are still too big, split at line breaks ("\\n")
        3. Then at sentences (".")
        4. Last resort: split at word boundaries (" ")
        
        This preserves meaning better than blindly cutting every 500 characters.
        
        WHAT is chunk_overlap?
        ----------------------
        If a sentence gets split across two chunks, the overlap ensures
        both chunks have the complete sentence:
        
        Without overlap:
          Chunk 1: "Dr. Sharma is available on Monday, Wednesday,"
          Chunk 2: "and Friday. His fee is Rs. 1200."
        
        With overlap (50 chars):
          Chunk 1: "Dr. Sharma is available on Monday, Wednesday, and Friday."
          Chunk 2: "Wednesday, and Friday. His fee is Rs. 1200."
        
        Now both chunks have the complete availability info!
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            # These are the separators, tried in order:
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,  # Count characters (not tokens)
        )
        
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def _build_qa_chain(self):
        """
        Build the Question-Answering chain.
        
        This creates the pipeline:
        Question → Search Vector DB → Get Context → Feed to LLM → Answer
        
        The PROMPT TEMPLATE is crucial — it tells the LLM:
        - WHO it is (hospital receptionist)
        - WHAT context it has (retrieved hospital data)
        - HOW to answer (be helpful, don't make stuff up)
        """
        
        # This is the prompt that gets sent to the LLM for EVERY question.
        # {context} gets replaced with the retrieved hospital data chunks
        # {question} gets replaced with the patient's question
        prompt_template = PromptTemplate(
            template="""You are MedBot, a friendly AI receptionist for MedCare General Hospital.

Use the following pieces of hospital information to answer the patient's question.
If you don't know the answer based on the provided information, say "I don't have that specific information, but you can contact us at +91 22-4567-8900 for more details."

IMPORTANT: Only answer based on the provided context. Do NOT make up information.

Hospital Information:
{context}

Patient's Question: {question}

Helpful Answer:""",
            input_variables=["context", "question"]
        )
        
        # RetrievalQA chain — connects everything:
        # 1. Takes the question
        # 2. Uses the retriever to search ChromaDB for top-K similar chunks
        # 3. Puts those chunks into {context} in the prompt
        # 4. Sends the complete prompt to the LLM
        # 5. Returns the LLM's response
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" = stuff all retrieved docs into one prompt
            # Other chain_types:
            #   "map_reduce" = summarize each doc separately, then combine
            #   "refine" = iteratively refine the answer with each doc
            #   "stuff" is simplest and works great for small context (OUR CHOICE)
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": settings.RETRIEVAL_K}  # Return top 3 matches
            ),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True,  # Also return which docs were used
        )
        
        logger.info("✅ QA Chain built and ready!")
    
    def query(self, question: str) -> Dict:
        """
        Ask a question and get an answer based on hospital data.
        
        This is the main method that the API endpoints will call.
        
        Args:
            question: The patient's question (e.g., "What are your visiting hours?")
        
        Returns:
            dict with:
                - "answer": The generated answer text
                - "sources": List of source chunks that were used
                - "source_metadata": Metadata about where the info came from
        """
        if self.qa_chain is None:
            return {
                "answer": "I'm sorry, my knowledge base is not initialized yet. "
                          "Please try again in a moment.",
                "sources": [],
                "source_metadata": []
            }
        
        try:
            logger.info(f"🔍 Processing question: {question}")
            
            # Run the chain!
            # Internally this does:
            # 1. Embed the question → [0.12, 0.89, 0.34, ...]
            # 2. Search ChromaDB for 3 most similar chunks
            # 3. Build prompt: system + context + question
            # 4. Send to Ollama → Llama generates answer
            # 5. Return answer + source documents
            result = self.qa_chain.invoke({"query": question})
            
            # Extract the answer
            answer = result.get("result", "I'm sorry, I couldn't generate a response.")
            
            # Extract source documents (what info was used to answer)
            source_docs = result.get("source_documents", [])
            sources = [doc.page_content for doc in source_docs]
            source_metadata = [doc.metadata for doc in source_docs]
            
            logger.info(f"✅ Answer generated using {len(sources)} source chunks")
            
            return {
                "answer": answer,
                "sources": sources,
                "source_metadata": source_metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing question: {e}")
            return {
                "answer": f"I'm sorry, I encountered an error: {str(e)}. "
                          "Please try again or contact us at +91 22-4567-8900.",
                "sources": [],
                "source_metadata": []
            }
    
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base."""
        if self.vectorstore is None:
            return {"status": "not_initialized", "chunk_count": 0}
        
        count = self.vectorstore._collection.count()
        return {
            "status": "ready",
            "chunk_count": count,
            "model": settings.LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
        }
    
    def reset_knowledge_base(self):
        """
        Delete and recreate the vector store.
        Useful if you update the hospital data and want to re-index.
        """
        import shutil
        
        logger.info("🗑️ Resetting knowledge base...")
        
        # Delete the ChromaDB directory
        try:
            shutil.rmtree(settings.CHROMA_DB_DIR)
            logger.info("Deleted old vector store")
        except FileNotFoundError:
            pass
        
        # Reinitialize
        self.vectorstore = None
        self.qa_chain = None
        self._initialize_from_hospital_data()
        
        logger.info("✅ Knowledge base reset complete!")
