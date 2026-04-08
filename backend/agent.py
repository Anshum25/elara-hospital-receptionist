"""
LangChain ReAct Agent Setup

=== WHAT IS AN AGENT? ===
An Agent is an LLM combined with a loop of reasoning:
1. Observe user input
2. THOUGHT: "What should I do?"
3. ACTION: Call a tool (e.g. check_doctor_availability)
4. OBSERVATION: Get result from the tool
5. THOUGHT: "Now I have the answer, I will formulate a response."
6. RESPONSE: Talk to user

This pattern is called ReAct (Reasoning and Acting).

=== WHY DO WE NEED IT? ===
Because real receptionists do more than answer questions — they perform actions.
This file connects our LLM to the Tools we built in `agent_tools.py`.
"""

import logging
from typing import List, Dict

# Core LangChain Agent tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Model and Tools
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory

from backend.config import settings
from backend.agent_tools import get_all_tools, set_rag_pipeline
from backend.rag_pipeline import HospitalRAGPipeline

logger = logging.getLogger(__name__)

class HospitalAgent:
    """
    The main Agent class.
    This manages the LLM, the Tools, and the memory of the conversation.
    """
    
    def __init__(self, rag_pipeline: HospitalRAGPipeline):
        """Build the agent."""
        logger.info("🧠 Powering up the Hospital LangChain Agent...")
        
        # 1. Provide the RAG engine to the tools so they can use it
        set_rag_pipeline(rag_pipeline)
        self.tools = get_all_tools()
        
        # 2. Setup the LLM. 
        # Note we use ChatOllama (not OllamaLLM) because Agents require "Chat Models"
        # that specifically support tool-calling functionality.
        self.llm = ChatOllama(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            num_ctx=4096, # Give it plenty of context window for tool history
        )
        
        # Enable tool calling natively on the model
        # Llama 3.1 and 3.2 both have native tool-calling capabilities!
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # 3. Create the System Prompt
        # This gives the LLM its "personality" and rules for using tools.
        system_prompt = """You are Elara, a professional and extremely helpful AI video receptionist for MedCare General Hospital.

CORE RULES:
1. Be concise! Your responses are being converted to speech for a video call. Keep answers short and conversational (1-2 sentences max).
2. If a patient asks a general medical or hospital question, ALWAYS use the `search_hospital_information` tool to get accurate data. Do not guess.
3. If a patient wants to book an appointment:
   - Ask for their name, preferred department, and preferred date if they haven't provided them.
   - Use `check_doctor_availability` to find open slots.
   - Present times to the patient.
   - Once they pick a time, use `book_appointment` to finalize it.
4. If it is an emergency, tell them to call +91 22-4567-8911 immediately and use the `transfer_to_human` tool.
5. Focus ONLY on answering the patient's newest user message. Do not repeat previous confirmations.
"""

        # 4. Create the Prompt Template
        # This represents the structure of the entire conversation.
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"), # Injects memory
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"), # Where the agent "thinks"
        ])
        
        # 5. Create the Agent
        # create_tool_calling_agent connects the prompt, llm, and tools together
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        
        # 6. Create the Agent Executor
        # The Executor runs the React loop (Think -> Act -> Observe -> Think -> Respond)
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True,  # Set to true so we can watch the agent "think" in the terminal!
            handle_parsing_errors=True # If the LLM makes a mistake, try again
        )
        
        # 7. Setup Memory
        # The agent needs to remember previous messages in the current conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        logger.info("✅ Agent is fully equipped and ready!")

    def process_message(self, user_input: str) -> Dict[str, str]:
        """
        Takes the user message, runs the agent logic, and returns the response.
        """
        try:
            logger.info(f"🗣️ User says: {user_input}")
            
            # Load previous conversation history
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            
            # Execute the agent logic
            response = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            agent_answer = response["output"]
            
            # Hotfix: Sometimes Llama 3B will hallucinate and output the raw JSON tool schema
            # into the chat instead of invoking the tool. If we detect this, gracefully recover!
            if '{"name":' in agent_answer or '{"type":' in agent_answer:
                logger.warning(f"⚠️ Caught LLM JSON spillage: {agent_answer}")
                agent_answer = "I'm checking the systems for you, could you please clarify your question?"
            
            # Save the new message pair to memory
            self.memory.save_context(
                {"input": user_input},
                {"output": agent_answer}
            )
            
            return {
                "answer": agent_answer,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ Agent execution error: {e}")
            return {
                "answer": "I'm having a little trouble connecting to my systems right now. Let me transfer you to a human.",
                "status": "error"
            }
    
    def reset_memory(self):
        """Clears the conversation history (e.g. for a new patient call)"""
        self.memory.clear()
        logger.info("🧹 Conversation memory cleared.")
