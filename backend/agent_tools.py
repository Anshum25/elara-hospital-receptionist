"""
Agent Tools Definition

=== WHAT IS THIS FILE? ===
This file defines the "Toolbox" for our AI Agent.

Unlike a pure RAG pipeline (which can only READ information and answer questions),
an Agent can TAKE ACTIONS. The LLM acts as the "brain," and these tools act
as the "hands."

=== HOW DO TOOLS WORK IN LANGCHAIN? ===
We use the @tool decorator. This does something very clever:
It takes the Python function name, the docstring, and the type hints, and sends them
to the LLM. 

When the LLM reads a user's prompt (e.g. "Book an appointment for me"), the LLM
looks at its toolbox and thinks: "Ah, I have a tool called 'book_appointment'. The
docstring says 'Books an appointment for a patient'. I should output a command to use
this tool, and ask the user for the 'patient_name' and 'date' since they are required."

This is how AI logic is connected to real-world code!
"""

import logging
import json
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# We import the RAG pipeline so one of our tools can be "Searching the Knowledge Base"
from backend.rag_pipeline import HospitalRAGPipeline

logger = logging.getLogger(__name__)

# We need a shared instance of the RAG pipeline for the tools to use
# We'll set this up from the main agent file.
rag_pipeline = None

def set_rag_pipeline(pipeline_instance):
    """Inject the RAG pipeline instance into the tools module."""
    global rag_pipeline
    rag_pipeline = pipeline_instance


# ==================== TOOL 1: Knowledge Base Search ====================

# We establish strict schemas for tool inputs so the LLM knows EXACTLY what to provide.
class SearchKnowledgeBaseInput(BaseModel):
    query: str = Field(description="The question to ask about hospital policies, doctors, or services. Must be a detailed question.")

@tool("search_hospital_information", args_schema=SearchKnowledgeBaseInput)
def search_hospital_information(query: str) -> str:
    """
    Search the hospital knowledge base. USE THIS TOOL whenever the patient asks general questions 
    about services, doctors, visiting hours, prices, insurance, or hospital policies.
    """
    logger.info(f"🛠️ TOOL CALLED: search_hospital_information(query='{query}')")
    
    if not rag_pipeline:
        return "System error: Knowledge base is not connected. Please ask the patient to try again later."
    
    result = rag_pipeline.query(query)
    return f"Information found: {result['answer']}"


# ==================== TOOL 2: Check Availability ====================

class CheckAvailabilityInput(BaseModel):
    department: Optional[str] = Field(default="", description="The medical department (e.g., Cardiology, General Medicine)")
    date: Optional[str] = Field(default="", description="The date to check in YYYY-MM-DD format, or 'today', 'tomorrow'")

@tool("check_doctor_availability", args_schema=CheckAvailabilityInput)
def check_doctor_availability(department: str, date: str) -> str:
    """
    Check the live availability of doctors for a specific department and date.
    USE THIS before booking an appointment to verify there are open slots.
    """
    if not date or date == "None":
        return "ERROR: Cannot check availability. You MUST explicitly ask the patient what date they prefer."
    if not department or department == "None":
        return "ERROR: Cannot check availability. You MUST explicitly ask the patient which department or doctor they need."
        
    logger.info(f"🛠️ TOOL CALLED: check_doctor_availability(department='{department}', date='{date}')")
    
    # In a real app, you would make an HTTP request to your client's SQL database here!
    # For this demo, we simulate a database check:
    department = department.lower()
    
    # Simulate some logic
    if "cardiology" in department:
        return f"Dr. Rajesh Sharma (Cardiology) has slots available at 10:00 AM, 2:00 PM, and 4:30 PM on {date}."
    elif "ortho" in department:
        return f"Dr. Priya Malhotra (Orthopedics) has slots available at 9:00 AM and 11:30 AM on {date}."
    elif "medicine" in department or "general" in department:
        return f"Dr. Anand Krishnan (General Medicine) has open slots every hour from '9:00 AM' to '4:00 PM' on {date}. He is widely available."
    else:
        # Default response
        return f"There are general slots available in {department} throughout the day on {date}. What time does the patient prefer?"


# ==================== TOOL 3: Book Appointment ====================

class BookAppointmentInput(BaseModel):
    patient_name: Optional[str] = Field(default="", description="The full name of the patient. Essential.")
    department: Optional[str] = Field(default="", description="The medical department")
    time_slot: Optional[str] = Field(default="", description="The preferred time slot (e.g. 10:00 AM). Essential.")
    date: str = Field(description="The date of the appointment")

@tool("book_appointment", args_schema=BookAppointmentInput)
def book_appointment(patient_name: str, department: str, time_slot: str, date: str) -> str:
    """
    Actually books the appointment in the hospital system. 
    USE THIS ONLY when the patient has confirmed their name, department, date, and specific time.
    """
    # Defensive programming: If the LLM tries to call this without the required info,
    # we return a string telling the LLM to ask the user. (This stops Pydantic from crashing!)
    if not patient_name or not time_slot or patient_name == "None" or time_slot == "None":
        return "ERROR: Cannot book yet. You MUST explicitly ask the patient for their missing Name or preferred Time Slot before calling this tool."
    if not department or department == "None":
        return "ERROR: Cannot book yet. You MUST explicitly ask the patient which department they want to book."
        
    logger.info(f"🛠️ TOOL CALLED: book_appointment(patient_name='{patient_name}', dept='{department}', time='{time_slot}', date='{date}')")
    
    # In a real app, this writes to the database or calls a booking API!
    
    # Save the appointment locally for demo purposes
    booking_details = {
        "id": f"APT-{int(datetime.now().timestamp())}",
        "patient_name": patient_name,
        "department": department,
        "date": date,
        "time_slot": time_slot,
        "status": "confirmed"
    }
    
    # Log it for our backend visibility
    logger.info(f"✅ APPOINTMENT BOOKED SECURELY: {json.dumps(booking_details)}")
    
    # Make sure to return a very clear message to the LLM so it relays the success
    return f"SUCCESS. Appointment is booked. The Booking ID is {booking_details['id']}. Tell this to the patient."


# ==================== TOOL 4: Human Handoff ====================

@tool("transfer_to_human")
def transfer_to_human() -> str:
    """
    Transfers the chat to a live human representative.
    USE THIS if the patient asks to speak to a real person, if there's a severe emergency, or if you cannot help them.
    """
    logger.info("🛠️ TOOL CALLED: transfer_to_human")
    
    # In a real application, this would trigger an event on your WebSockets or Twilio
    # that alerts a human agent dashboard.
    
    return "ACTION COMPLETE. Tell the patient that a real human staff member has been notified and will join the video call in a few moments."


# Export the list of tools for the agent to consume
def get_all_tools():
    """Return the toolbox that the LLM is allowed to use."""
    return [
        search_hospital_information,
        check_doctor_availability,
        book_appointment,
        transfer_to_human
    ]
