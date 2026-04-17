import json
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from config import Config

# Initialize configuration
config = Config()

REPORT_GENERATION_PROMPT = """You are an expert medical scribe and clinical assistant. Your task is to generate a comprehensive, structured consultation report based on the provided conversation history between a patient and a medical AI assistant.

The report must be in strict JSON format with the following fields:
- "chief_complaint": A concise statement of the patient's primary reason for the consultation.
- "summary": A detailed summary of the consultation, including the patient's history, the AI's analysis (including any image analysis results), and the discussion.
- "symptoms": A list of symptoms reported by the patient or identified during the consultation.
- "medications": A list of recommended medications or treatments mentioned.
- "advice": A list of care instructions, lifestyle recommendations, or follow-up steps provided.

**Input Conversation:**
{conversation_text}

**Instructions:**
1. Analyze the entire conversation history.
2. Extract relevant medical information.
3. Synthesize a professional clinical summary.
4. Identify all symptoms, medications, and advice given.
5. Ensure the output is valid JSON. Do not include markdown formatting (like ```json) in the response, just the raw JSON string.
"""

def generate_consultation_report(conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates a structured consultation report from conversation history.
    
    Args:
        conversation_history: List of message dictionaries containing 'role' and 'content'.
        
    Returns:
        Dict containing the structured report.
    """
    try:
        # Format conversation for the LLM
        conversation_text = ""
        for msg in conversation_history:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            # Skip system messages or empty content
            if not content or role == "SYSTEM":
                continue
            conversation_text += f"{role}: {content}\n\n"
            
        if not conversation_text.strip():
            return {
                "chief_complaint": "No consultation data",
                "summary": "The consultation history was empty.",
                "symptoms": [],
                "medications": [],
                "advice": []
            }

        # prompt = REPORT_GENERATION_PROMPT.format(conversation_text=conversation_text)
        
        # Use the conversation LLM (or a specialized one if available)
        llm = config.conversation.llm
        
        messages = [
            SystemMessage(content=REPORT_GENERATION_PROMPT.replace("{conversation_text}", conversation_text)),
            HumanMessage(content="Generate the consultation report now.")
        ]
        
        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up response if it contains markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].strip()
            
        # Parse JSON
        try:
            report_data = json.loads(response_text)
        except json.JSONDecodeError:
            print(f"Error parsing JSON report: {response_text}")
            # Fallback structure
            report_data = {
                "chief_complaint": "Error generating report",
                "summary": response_text,
                "symptoms": [],
                "medications": [],
                "advice": []
            }
            
        return report_data
        
    except Exception as e:
        print(f"Error generating consultation report: {e}")
        return {
            "chief_complaint": "Error",
            "summary": f"An error occurred while generating the report: {str(e)}",
            "symptoms": [],
            "medications": [],
            "advice": []
        }
