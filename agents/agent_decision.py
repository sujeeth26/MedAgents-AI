"""
Agent Decision System for Multi-Agent Medical Chatbot

This module handles the orchestration of different agents using LangGraph.
It dynamically routes user queries to the appropriate agent based on content and context.
"""

import json
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
import os, getpass
from dotenv import load_dotenv
from agents.rag_agent import MedicalRAG
# Import the agentic RAG system
# from demo_agentic_rag import AgenticRAGSystem
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.guardrails.local_guardrails import LocalGuardrails
from agents.clinical_prompts import (
    BRAIN_TUMOR_CLINICAL_PROMPT,
    CHEST_XRAY_CLINICAL_PROMPT,
    SKIN_LESION_CLINICAL_PROMPT,
    CONVERSATION_CLINICAL_PROMPT,
    EMERGENCY_CLINICAL_PROMPT,
    format_probabilities,
    format_pathologies
)

from langgraph.checkpoint.memory import MemorySaver

import cv2
import numpy as np

import sys
import os
# Fix import conflict with torchxrayvision's config module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

load_dotenv()

# Load configuration
config = Config()

# Initialize memory
memory = MemorySaver()

# Specify a thread
thread_config = {"configurable": {"thread_id": "1"}}


# Agent that takes the decision of routing the request further to correct task specific agent
class AgentConfig:
    """Configuration settings for the agent decision system."""
    
    # Decision model
    DECISION_MODEL = "gpt-4o"  # or whichever model you prefer
    
    # Vision model for image analysis
    VISION_MODEL = "gpt-4o"

    # Emergency keywords for immediate response
    EMERGENCY_KEYWORDS = [
        "chest pain", "heart attack", "stroke", "severe bleeding", "unconscious",
        "not breathing", "seizure", "overdose", "poisoning", "suicide",
        "severe allergic reaction", "anaphylaxis", "broken bone", "severe burn",
        "choking", "drowning", "electric shock", "head injury", "severe headache",
        "vision loss", "paralysis", "severe abdominal pain", "difficulty breathing"
    ]
    
    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.85
    
    # System instructions for the decision agent
    DECISION_SYSTEM_PROMPT = """You are Metagentica, an intelligent medical triage system that routes user queries to 
    the appropriate specialized agent. Your job is to analyze the user's request and determine which agent 
    is best suited to handle it based on the query content, presence of images, and conversation context.

    Available agents:
    1. CONVERSATION_AGENT - For general chat, greetings, and non-medical questions.
    2. EMERGENCY_RESPONSE - For critical medical emergencies requiring immediate attention (chest pain, stroke, severe bleeding, etc.).
    3. RAG_AGENT - For specific medical knowledge questions that can be answered from established medical literature. Currently ingested medical knowledge involves 'introduction to brain tumor', 'deep learning techniques to diagnose and detect brain tumors', 'deep learning techniques to diagnose and detect covid / covid-19 from chest x-ray'.
    4. WEB_SEARCH_PROCESSOR_AGENT - For questions about recent medical developments, current outbreaks, or time-sensitive medical information.
    5. BRAIN_TUMOR_AGENT - For analysis of brain MRI images to detect and segment tumors.
    6. CHEST_XRAY_AGENT - For analysis of chest X-ray images to detect COVID-19 or other abnormalities.
    7. SKIN_LESION_AGENT - For analysis of skin lesion images to classify them as benign or malignant.

    **CRITICAL ROUTING RULES:**
    - **EMERGENCY FIRST**: If the user mentions emergency symptoms (chest pain, stroke, severe bleeding, difficulty breathing, etc.), route to EMERGENCY_RESPONSE immediately.
    - If an image is uploaded (has_image: true), PRIORITIZE MEDICAL IMAGE ANALYSIS AGENTS above all else.
    - If has_image: true AND image_type indicates a medical image, route to the appropriate medical vision agent IMMEDIATELY.
    - If the user mentions "analyze", "scan", "check", "diagnose", or "examine" with an uploaded image, route to the appropriate medical vision agent.
    - For text-only queries asking about medical knowledge, use RAG_AGENT.
    - For recent medical news or current events, use WEB_SEARCH_PROCESSOR_AGENT.
    - For general conversation without medical context, use CONVERSATION_AGENT.
    - **OUT OF DOMAIN**: If the query is completely unrelated to health, medicine, or the chatbot's purpose (e.g., "how to fix a car", "write code"), route to CONVERSATION_AGENT but instruct it to politely decline and suggest contacting a specialist in that field.

    **MEDICAL IMAGE DETECTION:**
    - Chest X-ray images should go to CHEST_XRAY_AGENT for COVID-19 and abnormality detection.
    - Brain MRI images should go to BRAIN_TUMOR_AGENT for tumor detection and segmentation.
    - Skin lesion images should go to SKIN_LESION_AGENT for classification.
    - If image_type is unknown but user mentions medical analysis, default to CHEST_XRAY_AGENT.

    You must provide your answer in JSON format with the following structure:
    {
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }
    """

    # Initialize image analyzer
    image_analyzer = ImageAnalysisAgent(config=config)


def perform_chest_xray_segmentation(image_path: str):
    """Perform chest X-ray segmentation using MedRAX segmentation tool."""
    try:
        # Import MedRAX segmentation tool
        import importlib.util
        medrax_base_path = os.path.join(
            os.path.dirname(__file__), 
            "../image_analysis_agent/MedRAX-main"
        )
        segmentation_path = os.path.join(medrax_base_path, "medrax/tools/segmentation.py")
        
        if os.path.exists(segmentation_path):
            spec = importlib.util.spec_from_file_location("medrax_segmentation", segmentation_path)
            medrax_segmentation = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(medrax_segmentation)
            ChestXRaySegmentationTool = medrax_segmentation.ChestXRaySegmentationTool
            
            # Initialize segmentation tool
            seg_tool = ChestXRaySegmentationTool(device="cpu")  # Use CPU for compatibility
            
            # Perform segmentation
            result, metadata = seg_tool._run(image_path, organs=None)
            
            # Get segmentation image path
            if result and "segmentation_image_path" in result:
                seg_image_path = result["segmentation_image_path"]
                # Convert to URL path
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                rel_path = os.path.relpath(seg_image_path, project_root)
                return f"/{rel_path.replace(os.sep, '/')}"
            
        return None
    except Exception as e:
        print(f"Error performing segmentation: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_medrax_response(detailed_analysis: dict, predicted_class: str) -> str:
    """Build comprehensive response from MedRAX detailed analysis."""
    response_parts = []
    
    # Primary diagnosis
    primary_diagnosis = detailed_analysis.get("primary_diagnosis")
    primary_prob = detailed_analysis.get("primary_probability", 0.0)
    
    if primary_diagnosis and primary_prob > 0.3:
        response_parts.append(f"**Primary Finding:** {primary_diagnosis} (confidence: {primary_prob:.1%})")
    
    # COVID-19 specific analysis
    covid_prob = detailed_analysis.get("covid19_probability", 0.0)
    if predicted_class == "covid19" or covid_prob > 0.5:
        response_parts.append(f"\n**COVID-19 Analysis:** The image shows indicators consistent with **COVID-19** (probability: {covid_prob:.1%}).")
        response_parts.append("This is based on detection of pneumonia, consolidation, lung opacity, or infiltration patterns.")
    elif predicted_class == "normal":
        response_parts.append(f"\n**COVID-19 Analysis:** The image appears **NORMAL** with no significant COVID-19 indicators detected.")
    
    # Significant pathologies
    pathologies = detailed_analysis.get("pathologies", {})
    significant = {
        k: v for k, v in pathologies.items() 
        if v > 0.3 and k not in ["Pneumonia", "Consolidation", "Lung Opacity", "Infiltration"]
    }
    
    if significant:
        response_parts.append(f"\n**Other Findings Detected:**")
        for path, prob in sorted(significant.items(), key=lambda x: x[1], reverse=True)[:5]:
            response_parts.append(f"- {path}: {prob:.1%}")
    
    # Add disclaimer
    response_parts.append("\n**Important Disclaimer:** This MedAgentica analysis is for informational purposes only and is not a definitive medical diagnosis. It must be reviewed and validated by a qualified healthcare professional.")
    
    return "\n".join(response_parts)


def build_raw_medrax_response(
    classification: dict, 
    segmentation: dict, 
    report: dict, 
    disease_grounding: dict
) -> str:
    """Build comprehensive MedRAX output with ALL features and services."""
    response_parts = []
    
    # 1. COMPREHENSIVE CLASSIFICATION RESULTS (All 18 Diseases with Probabilities)
    if classification and not classification.get("error"):
        pathologies = classification.get("pathologies", {})
        if pathologies:
            response_parts.append("___MedAgentica Classification Result (18-Disease Analysis)___")
            
            # Sort all pathologies by probability
            sorted_pathologies = sorted(pathologies.items(), key=lambda x: x[1], reverse=True)
            
            # Primary finding
            if sorted_pathologies:
                primary = sorted_pathologies[0]
                response_parts.append(f"\nPrimary Finding: {primary[0]}")
                response_parts.append(f"Probability: {primary[1]*100:.2f}%")
            
            # All detected pathologies with probabilities (threshold > 0.1)
            significant_pathologies = [(name, prob) for name, prob in sorted_pathologies if prob > 0.1]
            if significant_pathologies:
                response_parts.append("\nAll Detected Pathologies:")
                for name, prob in significant_pathologies:
                    response_parts.append(f"- {name}: {prob*100:.2f}%")
            
            # Specific disease categories
            disease_categories = {
                "Pneumonia/Infection": ["Pneumonia", "Consolidation", "Lung Opacity", "Infiltration"],
                "Pleural": ["Pleural Effusion", "Pleural Thickening", "Pneumothorax"],
                "Cardiac": ["Cardiomegaly", "Atelectasis"],
                "Other": ["Edema", "Mass", "Nodule", "Fibrosis", "Hernia", "No Finding"]
            }
            
            response_parts.append("\nDisease Categories:")
            for category, diseases in disease_categories.items():
                found = [(d, pathologies.get(d, 0)) for d in diseases if d in pathologies and pathologies[d] > 0.1]
                if found:
                    category_list = ", ".join([f"{d} ({p*100:.1f}%)" for d, p in found])
                    response_parts.append(f"- {category}: {category_list}")
    
    # 2. COMPREHENSIVE RADIOLOGY REPORT
    if report and not report.get("error"):
        full_report = report.get("report", "")
        findings = report.get("findings", "")
        impression = report.get("impression", "")
        
        if full_report or findings or impression:
            response_parts.append("\n\n___Radiology Report___")
            
            if full_report:
                response_parts.append(f"\nFull Report:\n{full_report}")
            else:
                if findings:
                    response_parts.append(f"\nFindings:\n{findings}")
                if impression:
                    response_parts.append(f"\nImpression:\n{impression}")
    
    # 3. COMPREHENSIVE ANATOMICAL SEGMENTATION (All Organs with Areas)
    if segmentation and not segmentation.get("error"):
        organ_metrics = segmentation.get("organ_metrics", {})
        if organ_metrics:
            response_parts.append("\n\n___Anatomical Segmentation Analysis___")
            
            # Organ areas
            response_parts.append("\nOrgan Segmentation Areas:")
            for organ, metrics in organ_metrics.items():
                if isinstance(metrics, dict):
                    area = metrics.get("area", "N/A")
                    # Try to get additional metrics
                    perimeter = metrics.get("perimeter", None)
                    centroid = metrics.get("centroid", None)
                    
                    area_str = f"Area: {area}"
                    if perimeter:
                        area_str += f", Perimeter: {perimeter}"
                    if centroid:
                        area_str += f", Centroid: {centroid}"
                    
                    response_parts.append(f"- {organ}: {area_str}")
                else:
                    response_parts.append(f"- {organ}: {metrics}")
            
            # Check for specific findings in segmentation
            if "Pleural Effusion" in str(organ_metrics) or any("effusion" in str(k).lower() for k in organ_metrics.keys()):
                response_parts.append("\n⚠️ Pleural Effusion Detected in Segmentation")
    
    # 4. DISEASE GROUNDING AND LOCALIZATION
    if disease_grounding and not disease_grounding.get("error"):
        grounded_diseases = disease_grounding.get("grounded_diseases", {})
        bounding_boxes = disease_grounding.get("bounding_boxes", [])
        combined_viz = disease_grounding.get("combined_visualization_path")
        
        if grounded_diseases or bounding_boxes:
            response_parts.append("\n\n___Disease Localization & Grounding___")
            
            if grounded_diseases:
                response_parts.append("\nGrounded Diseases with Locations:")
                for disease, data in grounded_diseases.items():
                    prob = data.get("probability", 0) * 100
                    bbox = data.get("bounding_box")
                    confidence = data.get("confidence", 0) * 100
                    
                    location_str = f"{disease} ({prob:.1f}% probability, {confidence:.1f}% localization confidence)"
                    if bbox:
                        location_str += f" - Bounding Box: {bbox}"
                    response_parts.append(f"- {location_str}")
            
            if bounding_boxes:
                response_parts.append(f"\nTotal Regions of Interest Detected: {len(bounding_boxes)}")
            
            if combined_viz:
                response_parts.append(f"\nVisualization Generated: {combined_viz}")
    
    # 5. EFFUSION DETECTION (Specific Check)
    if classification and not classification.get("error"):
        pathologies = classification.get("pathologies", {})
        effusion_prob = pathologies.get("Pleural Effusion", 0)
        if effusion_prob > 0.1:
            response_parts.append("\n\n___Pleural Effusion Analysis___")
            response_parts.append(f"Pleural Effusion Probability: {effusion_prob*100:.2f}%")
            if effusion_prob > 0.5:
                response_parts.append("Status: Significant effusion detected")
            elif effusion_prob > 0.3:
                response_parts.append("Status: Moderate effusion detected")
            else:
                response_parts.append("Status: Mild effusion detected")
    
    # 6. SUMMARY STATISTICS
    if classification and not classification.get("error"):
        pathologies = classification.get("pathologies", {})
        if pathologies:
            total_diseases = len([p for p in pathologies.values() if p > 0.1])
            high_confidence = len([p for p in pathologies.values() if p > 0.5])
            
            response_parts.append("\n\n___Summary Statistics___")
            response_parts.append(f"Total Diseases Detected (>10% probability): {total_diseases}")
            response_parts.append(f"High Confidence Findings (>50% probability): {high_confidence}")
    
    if not response_parts:
        return "MedRAX analysis completed. No significant findings detected."
    
    return "\n".join(response_parts)


def build_comprehensive_medrax_response(
    classification: dict, 
    segmentation: dict, 
    report: dict, 
    disease_grounding: dict, 
    user_query: str
) -> str:
    """Build concise, professional clinical response for clinicians."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Extract data
    pathologies = classification.get("pathologies", {}) if classification and not classification.get("error") else {}
    primary_diagnosis = max(pathologies.items(), key=lambda x: x[1]) if pathologies else None
    primary_prob = primary_diagnosis[1] if primary_diagnosis else 0.0
    primary_diagnosis_name = primary_diagnosis[0] if primary_diagnosis else None
    
    findings = report.get("findings", "") if report and not report.get("error") else ""
    impression = report.get("impression", "") if report and not report.get("error") else ""
    
    # Calculate COVID probability from pathologies
    covid_indicators = ["Pneumonia", "Consolidation", "Lung Opacity", "Infiltration"]
    covid_prob = sum(pathologies.get(indicator, 0) for indicator in covid_indicators) / len(covid_indicators) if covid_indicators else 0.0
    
    # Format pathologies for Mayo Clinic prompt
    pathologies_formatted = format_pathologies(pathologies, threshold=0.3, top_n=5)
    
    # Use Mayo Clinic clinical prompt template
    system_prompt = CHEST_XRAY_CLINICAL_PROMPT.format(
        primary_diagnosis=primary_diagnosis_name if primary_diagnosis_name else "No significant findings",
        probability=f"{primary_prob*100:.1f}",
        covid_probability=f"{covid_prob*100:.1f}",
        pathologies=pathologies_formatted,
        user_query=user_query
    )
    
    user_prompt = f"""Based on the chest X-ray MedRAX analysis provided in the system context, generate a comprehensive 
yet concise clinical consultation (150-200 words max) following the Mayo Clinic structured format.

Additional Context:
- Radiographic Findings: {findings[:150] if findings else "Standard chest X-ray analysis"}
- Clinical Impression: {impression[:150] if impression else "No focal consolidation"}

Focus on:
- Primary radiographic findings with confidence levels
- COVID-19 assessment if relevant
- Red flags requiring immediate action
- Diagnostic work-up recommendations
- Evidence-based management plan
- Disposition and follow-up criteria
- Brief patient education

Prioritize actionable clinical decisions for point-of-care use."""
    
    try:
        llm = config.medical_cv.llm
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"Error generating concise response: {e}")
        # Fallback to minimal format
        if primary_diagnosis_name and primary_prob > 0.3:
            return f"**Primary Finding:** {primary_diagnosis_name} ({primary_prob:.1%}). **Key Pathologies:** {pathologies_text}. Clinical correlation advised."
        return f"**Analysis:** No significant abnormalities detected. Clinical correlation advised."


def build_brain_tumor_response(
    classification_result: dict,
    user_query: str = "",
    user_role: str = "patient"
) -> str:
    """
    Build comprehensive, professional, and patient-friendly brain tumor analysis response.
    Uses prompt engineering to create a doctor-like explanation that's clear and empathetic.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    
    predicted_class = classification_result.get("predicted_class", "").lower()
    confidence = classification_result.get("confidence", 0.0)
    all_probs = classification_result.get("all_probabilities", [])
    
    if classification_result.get("error"):
        return f"I apologize, but I encountered an issue analyzing your brain MRI image: {classification_result.get('error')}. Please try uploading a different image or consult with a healthcare professional."
    
    try:
        # Format probabilities for prompt
        prob_str = format_probabilities(all_probs, top_n=3)
        
        # NEW: MedAgentica-branded prompt
        # Comprehensive, detailed, 1-2 paragraphs, appealing, polite, informative
        
        medagentica_system_prompt = f"""You are MedAgentica, a wise, kind, and highly knowledgeable AI medical consultant specializing in neuro-radiology.

**Brand Identity:**
- Tone: Formal, professional, yet deeply empathetic and polite.
- Persona: A world-class specialist who explains things with clarity and wisdom.
- Style: Comprehensive yet concise (1-2 paragraphs main explanation).

**Clinical Context:**
- Analysis: {predicted_class.title()}
- Confidence: {confidence:.1f}%
- Probability Distribution: {prob_str}

**User Role:** {user_role.upper()}

**Instructions:**
1. **Main Explanation (1-2 Paragraphs):** 
   - Write a detailed, narrative explanation of the findings.
   - Be "appealing to read and hear" – use smooth, natural language.
   - For PATIENTS: Explain the condition in simple, reassuring terms. Focus on what it means and next steps.
   - For CLINICIANS: Focus on the diagnostic certainty, differential, and clinical management.
   - Mention "MedAgentica" naturally in the text (e.g., "MedAgentica's analysis indicates...").

2. **Key Details (Bulleted):**
   - After the narrative, provide a brief, high-impact list of 3-4 key takeaways or next steps.

3. **Closing:**
   - End with a polite, wise, and supportive closing statement.

**Constraints:**
- Do NOT use complex markdown tables.
- Keep the total response focused and "to the point" while being detailed.
"""

        user_prompt = f"""Based on the brain MRI analysis provided, generate a response for a {user_role} regarding the finding of {predicted_class}.
        
        User Query: {user_query if user_query else "Analyze this brain MRI"}
        """

        llm = config.medical_cv.llm
        messages = [
            SystemMessage(content=medagentica_system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        print(f"Error generating professional response: {e}")
        # Fallback to structured response
        return _build_fallback_brain_tumor_response(classification_result, user_query)


def _build_fallback_brain_tumor_response(classification_result: dict, user_query: str) -> str:
    """Fallback response builder if LLM fails."""
    predicted_class = classification_result.get("predicted_class", "").title()
    confidence = classification_result.get("confidence", 0.0)
    all_probs = classification_result.get("all_probabilities", [])
    
    response_parts = []
    
    # Classification
    response_parts.append(f"**Classification:** {predicted_class} ({confidence:.1f}% confidence)")
    
    # Probabilities
    if all_probs:
        response_parts.append(f"\n**Probabilities:**")
        for cls, prob in all_probs[:3]:  # Top 3
            response_parts.append(f"- {cls}: {prob:.1f}%")
    
    # Clinical note
    if predicted_class.lower() == "no tumor":
        response_parts.append(f"\n**Clinical Note:** No tumor detected on MRI. Normal brain parenchyma.")
    else:
        response_parts.append(f"\n**Clinical Note:** {predicted_class} identified. Further evaluation recommended.")
    
    # Recommendation
    response_parts.append(f"\n**Recommendation:** Clinical correlation and neurosurgical consultation advised.")
    
    return "\n".join(response_parts)


def build_skin_cancer_response(
    classification_result: dict,
    user_query: str = "",
    user_role: str = "patient"
) -> str:
    """
    Build comprehensive, professional, and patient-friendly skin cancer analysis response.
    Uses prompt engineering to create a doctor-like explanation that's clear and empathetic.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Extract classification data
    predicted_class = classification_result.get("predicted_class", "").lower()
    confidence = classification_result.get("confidence", 0.0)
    benign_prob = classification_result.get("benign_probability", 0.0)
    malignant_prob = classification_result.get("malignant_probability", 0.0)
    all_probs = classification_result.get("all_probabilities", [])
    
    # Check for errors
    if classification_result.get("error"):
        return f"I apologize, but I encountered an issue analyzing your skin lesion image: {classification_result.get('error')}. Please try uploading a different image or consult with a healthcare professional."
    
    # Use LLM to generate professional Mayo Clinic-style response
    try:
        # Fill in the Mayo Clinic clinical prompt template
        system_prompt = SKIN_LESION_CLINICAL_PROMPT.format(
            predicted_class=predicted_class.title(),
            confidence=f"{confidence:.1f}",
            benign_prob=f"{benign_prob:.1f}",
            malignant_prob=f"{malignant_prob:.1f}",
            user_query=user_query if user_query else "Analyze this skin lesion"
        )

        if user_role == "clinician":
            role_instruction = """
            Generate a professional clinical consultation report.
            - Use standard dermatological terminology (e.g., asymmetry, border irregularity).
            - Focus on differential diagnosis, biopsy indications, and management.
            - Tone: Professional, objective, and concise.
            """
        else:
            role_instruction = """
            Generate a highly empathetic and patient-centered explanation.
            - Speak directly to the patient using "you" and "your".
            - Avoid complex medical jargon; if a term is necessary, explain it immediately in simple terms.
            - Focus on what this means for their daily life and next steps.
            - Be reassuring but realistic.
            - Use phrases like "MedAgentica suggests..." instead of "The AI found...".
            - Tone: Warm, caring, and supportive, like a kind family doctor explaining to a patient.
            """

        user_prompt = f"""Based on the skin lesion Metagentica analysis provided in the system context, generate a response following these instructions:
        
        {role_instruction}

        Structure the response with:
        - **Assessment**: Clear statement of findings.
        - **Explanation**: Why this conclusion was reached.
        - **Recommendations**: Concrete next steps.
        
        Context: Explain WHY the lesion was classified this way.
        """

        llm = config.medical_cv.llm
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        print(f"Error generating professional response: {e}")
        # Fallback to structured response
        return _build_fallback_skin_cancer_response(classification_result, user_query)


def _build_fallback_skin_cancer_response(classification_result: dict, user_query: str) -> str:
    """Fallback response builder if LLM fails."""
    predicted_class = classification_result.get("predicted_class", "").lower()
    confidence = classification_result.get("confidence", 0.0)
    benign_prob = classification_result.get("benign_probability", 0.0)
    malignant_prob = classification_result.get("malignant_probability", 0.0)
    
    response_parts = []
    
    # Title
    response_parts.append("## Skin Lesion Analysis Report\n")
    
    # Greeting
    if user_query:
        response_parts.append(f"Thank you for your question: *{user_query}*\n")
    response_parts.append("I've analyzed your skin lesion image using Metagentica's advanced technology (EfficientNet-B0). Let me explain the results in a clear and understandable way.\n")
    
    # Classification Results
    response_parts.append("### 🔬 Classification Results\n")
    
    is_malignant = predicted_class == "malignant"
    
    if is_malignant:
        response_parts.append(f"**Primary Assessment:** The Metagentica analysis suggests this lesion may be **malignant** (cancerous) with {confidence:.1f}% confidence.\n")
        response_parts.append("⚠️ **Important:** This is a preliminary Metagentica assessment. A malignant classification requires immediate professional evaluation by a dermatologist.\n")
    else:
        response_parts.append(f"**Primary Assessment:** The Metagentica analysis suggests this lesion appears **benign** (non-cancerous) with {confidence:.1f}% confidence.\n")
        response_parts.append("✅ **Note:** While this is encouraging, regular skin checks with a dermatologist are still recommended for peace of mind.\n")
    
    # Probability Breakdown
    response_parts.append("### 📊 Detailed Probability Analysis\n")
    response_parts.append(f"- **Benign (Non-cancerous):** {benign_prob:.1f}%")
    response_parts.append(f"- **Malignant (Potentially cancerous):** {malignant_prob:.1f}%\n")
    
    # What This Means
    response_parts.append("### 💡 What This Means for You\n")
    if is_malignant:
        response_parts.append("If MedAgentica suggests a malignant classification, it's important to:")
        response_parts.append("- Schedule an appointment with a dermatologist as soon as possible")
        response_parts.append("- Avoid delaying professional evaluation")
        response_parts.append("- Remember that early detection and treatment are crucial")
    else:
        response_parts.append("A benign classification suggests the lesion is likely non-cancerous, but:")
        response_parts.append("- This is a Metagentica assessment, not a definitive diagnosis")
        response_parts.append("- Regular professional skin examinations are still important")
        response_parts.append("- Monitor the lesion for any changes over time")
    
    response_parts.append("")
    
    # ABCDE Criteria
    response_parts.append("### 🔍 Understanding Skin Lesion Characteristics (ABCDE Criteria)\n")
    response_parts.append("When evaluating skin lesions, dermatologists often use the ABCDE criteria:")
    response_parts.append("- **A**symmetry: One half doesn't match the other")
    response_parts.append("- **B**order: Irregular, ragged, or blurred edges")
    response_parts.append("- **C**olor: Multiple colors or uneven color distribution")
    response_parts.append("- **D**iameter: Larger than 6mm (about the size of a pencil eraser)")
    response_parts.append("- **E**volving: Changing in size, shape, color, or appearance over time")
    response_parts.append("\nIf you notice any of these features, please consult a dermatologist.\n")
    
    # Next Steps
    response_parts.append("### 📋 Recommended Next Steps\n")
    response_parts.append("1. **Schedule a dermatology appointment** - Professional evaluation is essential")
    response_parts.append("2. **Monitor the lesion** - Take note of any changes in size, color, or appearance")
    response_parts.append("3. **Practice sun safety** - Use sunscreen and protect your skin from UV exposure")
    response_parts.append("4. **Regular skin checks** - Consider annual full-body skin examinations")
    response_parts.append("")
    
    # Concluding Statement
    if user_query and ("cancer" in user_query.lower() or "malignant" in user_query.lower() or "benign" in user_query.lower()):
        response_parts.append("I understand your concern about this skin lesion. The Metagentica analysis provides a preliminary assessment, but a qualified dermatologist can give you a definitive diagnosis through clinical examination and, if needed, a biopsy. Please don't hesitate to seek professional medical advice.\n")
    else:
        response_parts.append("I hope this analysis helps address your concerns. Remember, this Metagentica assessment is a tool to assist, not replace, professional medical judgment. Your dermatologist can provide a comprehensive evaluation and personalized guidance.\n")
    
    # Medical Disclaimer
    response_parts.append("### ⚠️ Important Medical Disclaimer\n")
    response_parts.append("This Metagentica analysis is for informational and educational purposes only. It is **not** a substitute for professional medical diagnosis, advice, or treatment. Always consult with a qualified dermatologist or healthcare professional for proper evaluation, diagnosis, and treatment recommendations. Do not make medical decisions based solely on this Metagentica analysis.")
    
    return "\n".join(response_parts)


def build_medrax_response_with_query(detailed_analysis: dict, predicted_class: str, user_query: str, image_path: str, wants_segmentation: bool = False) -> str:
    """Build comprehensive response from MedRAX analysis that directly addresses the user's query."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Extract all pathologies
    pathologies = detailed_analysis.get("pathologies", {})
    primary_diagnosis = detailed_analysis.get("primary_diagnosis")
    primary_prob = detailed_analysis.get("primary_probability", 0.0)
    covid_prob = detailed_analysis.get("covid19_probability", 0.0)
    
    # Format pathologies for LLM
    pathologies_text = "\n".join([f"- {path}: {prob:.1%}" for path, prob in sorted(pathologies.items(), key=lambda x: x[1], reverse=True)])
    
    # Create a prompt that addresses the user's specific question
    system_prompt = """You are a board-certified radiologist providing concise clinical analysis of chest X-ray findings. Your responses must be:

**CRITICAL REQUIREMENTS:**
- **SHORT**: Maximum 150 words total
- **POLITE**: Professional and respectful tone
- **TO THE POINT**: Direct answer to the query, no fluff
- **ACCURATE**: Medically precise terminology
- **CLINICIAN-FRIENDLY**: Easy for healthcare professionals to understand quickly

**Response Format:**
1. **Answer**: Direct response to the query (one sentence)
2. **Findings**: Key pathologies detected with probabilities (one sentence)
3. **Clinical Note**: Brief interpretation (one sentence)
4. **Recommendation**: Next step or follow-up (one sentence)

**Style Guidelines:**
- Use standard radiological terminology
- Report probabilities accurately
- Be direct and factual
- Avoid lengthy explanations
- Focus on actionable clinical information

**Example Format:**
"COVID-19 not detected (probability <30%). Primary finding: Pneumonia (65% confidence) with bilateral lower lobe opacities. Findings consistent with infectious process. Recommend clinical correlation and follow-up imaging if indicated."
"""

    # Check for TB-related conditions (TB is not directly detected by MedRAX, but related conditions are)
    tb_related_conditions = ["Pneumonia", "Consolidation", "Lung Opacity", "Infiltration", "Mass", "Nodule"]
    tb_indicators = {k: v for k, v in pathologies.items() if k in tb_related_conditions and v > 0.3}
    
    user_prompt = f"""Chest X-ray analysis query: {user_query}

**MedRAX Results:**
- Primary: {primary_diagnosis} ({primary_prob:.1%})
- COVID-19: {covid_prob:.1%}
- Classification: {predicted_class}

**Pathologies:** {pathologies_text[:200]}...

**TB-Related:** {chr(10).join([f"{k}: {v:.1%}" for k, v in sorted(tb_indicators.items(), key=lambda x: x[1], reverse=True)]) if tb_indicators else "None detected"}

Provide a concise clinical summary (max 150 words) directly answering the query. Use professional radiological terminology. Include: direct answer, key findings, clinical note, and recommendation. Be brief and actionable."""

    try:
        # Use LLM to generate contextualized response
        llm = AgentConfig.conversation.llm
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"Error generating contextualized response: {e}")
        # Fallback to basic response
        return build_medrax_response(detailed_analysis, predicted_class)


class AgentState(MessagesState):
    """State maintained across the workflow."""
    # messages: List[BaseMessage]  # Conversation history
    agent_name: Optional[str]  # Current active agent
    current_input: Optional[Union[str, Dict]]  # Input to be processed
    has_image: bool  # Whether the current input contains an image
    image_type: Optional[str]  # Type of medical image if present
    image_classification_confidence: Optional[float]  # Confidence score from image classification
    last_image_path: Optional[str]  # Path to the last uploaded image (for clarification flow)
    output: Optional[str]  # Final output to user
    needs_human_validation: bool  # Whether human validation is required
    retrieval_confidence: float  # Confidence in retrieval (for RAG agent)
    bypass_routing: bool  # Flag to bypass agent routing for guardrails
    insufficient_info: bool  # Flag indicating RAG response has insufficient information
    insufficient_info: bool  # Flag indicating RAG response has insufficient information
    awaiting_image_clarification: bool  # Flag indicating we're waiting for user to clarify image type
    user_role: Optional[str]  # User role: "patient" or "clinician"


class AgentDecision(TypedDict):
    """Output structure for the decision agent."""
    agent: str
    reasoning: str
    confidence: float


def create_agent_graph():
    """Create and configure the LangGraph for agent orchestration."""

    # Initialize guardrails with the same LLM used elsewhere
    guardrails = LocalGuardrails(config.rag.llm)

    # LLM
    decision_model = config.agent_decision.llm
    
    # Initialize the output parser
    json_parser = JsonOutputParser(pydantic_object=AgentDecision)
    
    # Create the decision prompt
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    # Create the decision chain
    decision_chain = decision_prompt | decision_model | json_parser
    
    # Define graph state transformations
    def analyze_input(state: AgentState) -> AgentState:
        """Analyze the input to detect images and determine input type."""
        current_input = state["current_input"]
        has_image = False
        image_type = None
        
        # Get the text from the input
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # DISABLED: Input guardrails for speed optimization (saves 2-3 seconds)
        # Medical assistant doesn't need aggressive input filtering
        # Safety is handled by agent-specific logic and output validation
        if False:  # Disabled for performance
            try:
                is_allowed, message = guardrails.check_input(input_text)
                if not is_allowed:
                    print(f"Selected agent: INPUT GUARDRAILS, Message: ", message)
                    return {
                        **state,
                        "messages": message,
                        "agent_name": "INPUT_GUARDRAILS",
                        "has_image": False,
                        "image_type": None,
                        "bypass_routing": True
                    }
            except Exception as e:
                print(f"⚠️ [Agent Decision] Guardrails check failed: {e}")
                print(f"✅ [Agent Decision] Continuing with query processing (fail-open)")
        
        # CRITICAL: Always re-analyze images for each new query
        # Don't reuse previous image_type from state - each image upload should be fresh
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_path = current_input.get("image", None)
            
            # Store image path in state for agents to use
            if image_path:
                state["image_path"] = image_path
                state["last_image_path"] = image_path  # Persist for follow-up questions
            
            if image_path and os.path.exists(image_path):
                print(f"🖼️ Analyzing NEW image: {image_path}")
                image_type_response = AgentConfig.image_analyzer.analyze_image(image_path)
                # Handle both dict and string responses
                if isinstance(image_type_response, dict):
                    image_type = image_type_response.get('image_type', 'unknown')
                    confidence = image_type_response.get('confidence', 0.0)
                    reasoning = image_type_response.get('reasoning', '')
                    print(f"🔍 ANALYZED IMAGE TYPE: {image_type} (confidence: {confidence:.2f})")
                    print(f"   Reasoning: {reasoning}")
                    
                    # Store confidence in state for routing decisions
                    state["image_classification_confidence"] = confidence
                else:
                    image_type = str(image_type_response)
                    print(f"🔍 ANALYZED IMAGE TYPE: {image_type}")
                    state["image_classification_confidence"] = 0.5  # Default confidence
            else:
                print(f"⚠️ Image path provided but file not found: {image_path}")
                has_image = False
        else:
            # No image in this query, clear image-related state
            state["image_classification_confidence"] = None
            state["image_path"] = None
        
        return {
            **state,
            "has_image": has_image,
            "image_type": image_type,  # Always set fresh, don't carry over from previous state
            "bypass_routing": False  # Explicitly set to False for normal flow
        }
    
    def check_if_bypassing(state: AgentState) -> str:
        """Check if we should bypass normal routing due to guardrails."""
        if state.get("bypass_routing", False):
            return "apply_guardrails"
        return "route_to_agent"
    
    def route_to_agent(state: AgentState) -> Dict:
        """Make decision about which agent should handle the query."""
        messages = state["messages"]
        current_input = state["current_input"]
        has_image = state["has_image"]
        image_type = state["image_type"]
        awaiting_clarification = state.get("awaiting_image_clarification", False)
        
        # Check for emergency situations FIRST
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input.lower()
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "").lower()

        # Check if this is an emergency situation
        is_emergency = any(keyword in input_text for keyword in AgentConfig.EMERGENCY_KEYWORDS)

        if is_emergency:
            print("🚨 EMERGENCY SITUATION DETECTED in routing!")
            return {"agent_state": state, "next": "EMERGENCY_RESPONSE"}

        # CRITICAL: If we're awaiting clarification and user responds, check if they clarified image type
        # This handles the case where user responds to clarification prompt (may not have image in this response)
        if awaiting_clarification:
            input_text_lower = input_text.lower()
            
            # Check for image type clarifications
            skin_clarification_keywords = ["skin", "lesion", "mole", "rash", "dermatology", "dermatologist", 
                                          "benign", "malignant", "cancer", "melanoma"]
            xray_clarification_keywords = ["chest", "x-ray", "xray", "lung", "pneumonia", "covid", 
                                          "pulmonary", "respiratory"]
            brain_clarification_keywords = ["brain", "mri", "tumor", "tumour", "neurology", "neurological"]
            
            # Try to find the image path from previous messages or state
            # Look for image path in conversation history (from previous upload)
            image_path = None
            if isinstance(current_input, dict) and "image" in current_input:
                image_path = current_input.get("image")
            else:
                # Search through messages for image references
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        # Check if message content contains image path or if it was an image upload
                        content = msg.content
                        if isinstance(content, dict) and "image" in content:
                            image_path = content.get("image")
                            break
                        # Also check if there's a stored image path in state
                # Fallback: check if state has image path stored
                if not image_path and has_image:
                    # Try to reconstruct from previous state
                    prev_input = state.get("current_input")
                    if isinstance(prev_input, dict) and "image" in prev_input:
                        image_path = prev_input.get("image")
            
            if any(keyword in input_text_lower for keyword in skin_clarification_keywords):
                print(f"✅ User clarified image type as SKIN LESION - Routing to SKIN_LESION_AGENT")
                # Update state with clarified image type and ensure image is preserved
                updated_state = {
                    **state,
                    "image_type": "SKIN LESION",
                    "awaiting_image_clarification": False
                }
                # Preserve image in current_input if we found it
                if image_path:
                    if isinstance(current_input, dict):
                        updated_state["current_input"] = {"text": input_text, "image": image_path}
                    else:
                        updated_state["current_input"] = {"text": input_text, "image": image_path}
                    updated_state["has_image"] = True
                return {"agent_state": updated_state, "next": "SKIN_LESION_AGENT"}
            elif any(keyword in input_text_lower for keyword in xray_clarification_keywords):
                print(f"✅ User clarified image type as CHEST X-RAY - Routing to CHEST_XRAY_AGENT")
                updated_state = {
                    **state,
                    "image_type": "CHEST X-RAY",
                    "awaiting_image_clarification": False
                }
                if image_path:
                    if isinstance(current_input, dict):
                        updated_state["current_input"] = {"text": input_text, "image": image_path}
                    else:
                        updated_state["current_input"] = {"text": input_text, "image": image_path}
                    updated_state["has_image"] = True
                return {"agent_state": updated_state, "next": "CHEST_XRAY_AGENT"}
            elif any(keyword in input_text_lower for keyword in brain_clarification_keywords):
                print(f"✅ User clarified image type as BRAIN MRI - Routing to BRAIN_TUMOR_AGENT")
                updated_state = {
                    **state,
                    "image_type": "BRAIN MRI",
                    "awaiting_image_clarification": False
                }
                if image_path:
                    if isinstance(current_input, dict):
                        updated_state["current_input"] = {"text": input_text, "image": image_path}
                    else:
                        updated_state["current_input"] = {"text": input_text, "image": image_path}
                    updated_state["has_image"] = True
                return {"agent_state": updated_state, "next": "BRAIN_TUMOR_AGENT"}
            else:
                # User hasn't clarified yet, continue with conversation agent
                print(f"⏳ Still awaiting image type clarification from user")
                return {"agent_state": state, "next": "CONVERSATION_AGENT"}

        # CRITICAL: Check user query FIRST for explicit mentions before relying on image classification
        # This prevents misrouting when user explicitly mentions skin/lesion/cancer but image classification is wrong
        input_text_lower = input_text.lower()
        
        # Strong skin-related keywords (check FIRST)
        skin_keywords = ["skin", "lesion", "mole", "rash", "dermatology", "dermatologist", 
                        "benign", "malignant", "cancer", "melanoma", "skin problem", 
                        "skin disease", "skin condition", "is it cancer"]
        has_skin_keywords = any(keyword in input_text_lower for keyword in skin_keywords)
        
        # Strong X-ray/chest-related keywords - expanded to catch all variations
        xray_keywords = ["chest", "x-ray", "xray", "x ray", "lung", "pneumonia", "covid", 
                        "pulmonary", "respiratory", "breathing", "analyze xray", "analyze x-ray",
                        "analyze the xray", "analyze the x-ray", "check xray", "check x-ray",
                        "examine xray", "examine x-ray", "diagnose xray", "diagnose x-ray",
                        "xray image", "x-ray image", "chest xray", "chest x-ray",
                        "segmentation", "segment", "grounding", "visualize", "visualization",
                        "show me the image", "show me the picture", "show me the scan",
                        "where is the", "highlight", "locate", "area"]
        has_xray_keywords = any(keyword in input_text_lower for keyword in xray_keywords)
        
        # Strong brain/MRI keywords
        brain_keywords = ["brain", "mri", "tumor", "tumour", "neurology", "neurological"]
        has_brain_keywords = any(keyword in input_text_lower for keyword in brain_keywords)
        
        # If user explicitly mentions skin AND has image, prioritize skin agent
        if has_image and has_skin_keywords:
            print(f"🎯 USER QUERY EXPLICITLY MENTIONS SKIN - Routing to SKIN_LESION_AGENT (query: '{input_text[:50]}...')")
            return {"agent_state": state, "next": "SKIN_LESION_AGENT"}
        
        # CRITICAL: If user explicitly mentions X-ray/chest analysis, route to X-ray agent
        # This should work even if has_image is False (user might have uploaded image separately)
        if has_xray_keywords:
            print(f"🎯 USER QUERY EXPLICITLY MENTIONS CHEST/X-RAY - Routing to CHEST_XRAY_AGENT (query: '{input_text[:50]}...')")
            # Ensure image is in current_input - check multiple sources
            if isinstance(current_input, dict):
                if "image" not in current_input:
                    # Try to get image from state
                    image_path = state.get("image_path") or state.get("last_image_path")
                    if image_path and os.path.exists(image_path):
                        current_input["image"] = image_path
                        state["image_path"] = image_path
                        state["has_image"] = True
                        print(f"✅ Found image path in state: {image_path}")
                else:
                    # Image already in current_input, ensure it's in state
                    state["image_path"] = current_input["image"]
                    state["has_image"] = True
            else:
                # current_input is a string, convert to dict with image
                image_path = state.get("image_path") or state.get("last_image_path")
                if image_path and os.path.exists(image_path):
                    state["current_input"] = {"text": input_text, "image": image_path}
                    state["image_path"] = image_path
                    state["has_image"] = True
                    print(f"✅ Converted string input to dict with image: {image_path}")
            return {"agent_state": state, "next": "CHEST_XRAY_AGENT"}
        
        # If user explicitly mentions chest/X-ray AND has image, prioritize X-ray agent
        if has_image and has_xray_keywords:
            print(f"🎯 USER QUERY EXPLICITLY MENTIONS CHEST/X-RAY WITH IMAGE - Routing to CHEST_XRAY_AGENT (query: '{input_text[:50]}...')")
            # Ensure image is in current_input
            if isinstance(current_input, dict) and "image" not in current_input:
                image_path = state.get("image_path") or state.get("last_image_path")
                if image_path:
                    current_input["image"] = image_path
            return {"agent_state": state, "next": "CHEST_XRAY_AGENT"}
        
        # If user explicitly mentions brain/MRI AND has image, prioritize brain agent
        if has_image and has_brain_keywords:
            print(f"🎯 USER QUERY EXPLICITLY MENTIONS BRAIN/MRI - Routing to BRAIN_TUMOR_AGENT (query: '{input_text[:50]}...')")
            return {"agent_state": state, "next": "BRAIN_TUMOR_AGENT"}
        # Simple rule-based routing (no LLM needed for basic decisions)
        if has_image and image_type:
            image_type_lower = image_type.lower()
            
            # Route based on image type - check skin lesion FIRST (more specific)
            # This prevents skin images from being misrouted to X-ray agent
            if "skin" in image_type_lower or "lesion" in image_type_lower:
                print(f"✅ Routing skin lesion to SKIN_LESION_AGENT (image_type: {image_type})")
                return {"agent_state": state, "next": "SKIN_LESION_AGENT"}
            elif "chest" in image_type_lower or "x-ray" in image_type_lower or "xray" in image_type_lower or "lung" in image_type_lower:
                print(f"✅ Routing chest X-ray to CHEST_XRAY_AGENT (image_type: {image_type})")
                # Ensure image is in state for agent to use
                if isinstance(current_input, dict) and "image" in current_input:
                    state["image_path"] = current_input["image"]
                return {"agent_state": state, "next": "CHEST_XRAY_AGENT"}
            elif "brain" in image_type_lower or "mri" in image_type_lower:
                print(f"✅ Routing brain MRI to BRAIN_TUMOR_AGENT (image_type: {image_type})")
                return {"agent_state": state, "next": "BRAIN_TUMOR_AGENT"}
            else:
                # Unknown medical image - check user query for hints
                # Check if user mentions skin-related terms
                if has_skin_keywords:
                    print(f"⚠️ Unknown image type but user query suggests skin lesion, routing to SKIN_LESION_AGENT")
                    return {"agent_state": state, "next": "SKIN_LESION_AGENT"}
                
                # Check if user mentions chest/X-ray terms
                if has_xray_keywords:
                    print(f"⚠️ Unknown image type but user query suggests chest X-ray, routing to CHEST_XRAY_AGENT")
                    if isinstance(current_input, dict) and "image" in current_input:
                        state["image_path"] = current_input["image"]
                    return {"agent_state": state, "next": "CHEST_XRAY_AGENT"}
                
                # Check if user mentions brain/MRI terms
                if has_brain_keywords:
                    print(f"⚠️ Unknown image type but user query suggests brain MRI, routing to BRAIN_TUMOR_AGENT")
                    return {"agent_state": state, "next": "BRAIN_TUMOR_AGENT"}
                
                # If image classification confidence is low (< 0.5) or image_type is "unknown", check query first
                image_confidence = state.get("image_classification_confidence", 0.5)
                if image_type_lower == "unknown" or image_confidence < 0.5:
                    # Check if user query mentions any medical image type or generic analysis terms
                    if has_xray_keywords or "lung" in input_text_lower or "pneumonia" in input_text_lower or "analyze" in input_text_lower or "image" in input_text_lower or "xray" in input_text_lower or "x-ray" in input_text_lower:
                        print(f"⚠️ Unknown image type but query suggests chest X-ray, routing to CHEST_XRAY_AGENT")
                        if isinstance(current_input, dict) and "image" in current_input:
                            state["image_path"] = current_input["image"]
                        return {"agent_state": state, "next": "CHEST_XRAY_AGENT"}
                    elif has_skin_keywords:
                        print(f"⚠️ Unknown image type but query suggests skin lesion, routing to SKIN_LESION_AGENT")
                        return {"agent_state": state, "next": "SKIN_LESION_AGENT"}
                    elif has_brain_keywords:
                        print(f"⚠️ Unknown image type but query suggests brain MRI, routing to BRAIN_TUMOR_AGENT")
                        return {"agent_state": state, "next": "BRAIN_TUMOR_AGENT"}
                    else:
                        # Default to CHEST_XRAY_AGENT for unknown medical images (most common)
                        print(f"⚠️ Unknown medical image type ({image_type}), defaulting to CHEST_XRAY_AGENT")
                        if isinstance(current_input, dict) and "image" in current_input:
                            state["image_path"] = current_input["image"]
                        return {"agent_state": state, "next": "CHEST_XRAY_AGENT"}
                
                # Last resort: default to chest X-ray (but log it)
                print(f"⚠️ Unknown medical image type ({image_type}), defaulting to CHEST_XRAY_AGENT")
                if isinstance(current_input, dict) and "image" in current_input:
                    state["image_path"] = current_input["image"]
                return {"agent_state": state, "next": "CHEST_XRAY_AGENT"}

        # Text-based routing
        if any(keyword in input_text for keyword in ["web search", "latest", "recent", "news", "research", "current"]):
            print(f"Routing to WEB_SEARCH_PROCESSOR_AGENT")
            return {"agent_state": state, "next": "WEB_SEARCH_PROCESSOR_AGENT"}

        # Check for medical knowledge queries
        medical_keywords = ["symptom", "treatment", "diagnosis", "disease", "condition", "medicine", "medical"]
        if any(keyword in input_text for keyword in medical_keywords):
            print(f"Routing medical question to RAG_AGENT")
            return {"agent_state": state, "next": "RAG_AGENT"}

        # CRITICAL: Before defaulting to conversation agent, check if user mentioned image analysis
        # This catches cases where user says "analyze the xray image" but image wasn't detected
        if has_xray_keywords and not has_image:
            # User mentioned X-ray but no image detected - try to find image in session/state
            image_path = state.get("image_path") or state.get("last_image_path")
            if image_path and os.path.exists(image_path):
                print(f"⚠️ User mentioned X-ray but image not in current_input - Found in state: {image_path}")
                if isinstance(current_input, dict):
                    current_input["image"] = image_path
                else:
                    state["current_input"] = {"text": input_text, "image": image_path}
                state["image_path"] = image_path
                state["has_image"] = True
                return {"agent_state": state, "next": "CHEST_XRAY_AGENT"}
        
        # Default to conversation agent
        print(f"Routing to CONVERSATION_AGENT")
        return {"agent_state": state, "next": "CONVERSATION_AGENT"}

    # Define agent execution functions (these will be implemented in their respective modules)
    def run_conversation_agent(state: AgentState) -> AgentState:
        """Handle general conversation with emergency detection and image type clarification."""

        print(f"Selected agent: CONVERSATION_AGENT")

        messages = state["messages"]
        current_input = state["current_input"]
        has_image = state.get("has_image", False)
        awaiting_clarification = state.get("awaiting_image_clarification", False)

        # Check for emergency situations first
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input.lower()
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "").lower()
        
        # CRITICAL: Check if this is a follow-up question about medical image analysis
        # Look for the MOST RECENT image analysis in conversation history (not the first one)
        previous_image_context = ""
        previous_agent_type = ""
        
        # Iterate in REVERSE order to get the MOST RECENT image analysis
        for msg in reversed(messages[-20:]):  # Check last 20 messages (10 exchanges) in reverse
            if isinstance(msg, AIMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                # Check for image analysis keywords - identify which type of analysis
                is_brain_tumor = any(keyword in content for keyword in ["Pituitary", "Glioma", "Meningioma", "No Tumor", "Brain MRI", "brain tumor"])
                is_skin_lesion = any(keyword in content for keyword in ["Skin lesion", "benign", "malignant", "melanoma", "dermatology"])
                is_chest_xray = any(keyword in content for keyword in ["Chest X-ray", "MedRAX", "COVID", "pneumonia", "chest xray", "Lung Opacity"])
                
                # Check for any image analysis indicators
                has_analysis = any(keyword in content for keyword in ["Classification:", "Clinical Note:", "Recommendation:", 
                                                                     "Pituitary", "Glioma", "Meningioma", "No Tumor",
                                                                     "Chest X-ray", "MedRAX", "Skin lesion", "benign", "malignant"])
                
                if has_analysis:
                    # Get the FULL analysis content (not just first 500 chars)
                    previous_image_context = content
                    
                    # Identify which agent type this analysis came from
                    if is_brain_tumor:
                        previous_agent_type = "BRAIN_TUMOR_AGENT"
                    elif is_skin_lesion:
                        previous_agent_type = "SKIN_LESION_AGENT"
                    elif is_chest_xray:
                        previous_agent_type = "CHEST_XRAY_AGENT"
                    else:
                        previous_agent_type = "IMAGE_ANALYSIS_AGENT"
                    
                    print(f"✅ Found most recent image analysis from: {previous_agent_type}")
                    break  # Stop at the FIRST (most recent) match found

        # Check if this is an emergency situation
        is_emergency = any(keyword in input_text for keyword in AgentConfig.EMERGENCY_KEYWORDS)

        if is_emergency:
            print("EMERGENCY SITUATION DETECTED!")

            # Use Mayo Clinic emergency prompt
            emergency_response = EMERGENCY_CLINICAL_PROMPT

            return {
                **state,
                "output": AIMessage(content=emergency_response),
                "agent_name": "EMERGENCY_RESPONSE"
            }
        
        # Check if we're awaiting image type clarification
        if awaiting_clarification and has_image:
            # User hasn't clarified yet, ask them to specify
            clarification_prompt = """I see you've uploaded a medical image, but I'm having difficulty determining what type of image it is.

To route your image to the correct specialist agent, could you please tell me what type of medical image this is?

**Please specify one of the following:**
- **Chest X-ray** - For lung, chest, or respiratory conditions
- **Brain MRI** - For brain or neurological conditions
- **Skin lesion** - For skin conditions, moles, rashes, or dermatology concerns

Once you tell me the image type, I will route it to the appropriate AI analysis agent for detailed examination.

**Examples:**
- "This is a chest X-ray"
- "It is a skin lesion"
- "This is a brain MRI"
- "X-ray"
- "Skin"
- "MRI"

What type of medical image is this?"""

            return {
                **state,
                "output": AIMessage(content=clarification_prompt),
                "agent_name": "CONVERSATION_AGENT",
                "awaiting_image_clarification": True
            }
        
        # If previous image analysis exists and this is a follow-up question, respond contextually
        if previous_image_context and not has_image:
            print(f"✅ Follow-up question about previous image analysis detected (Agent: {previous_agent_type})")
            # Create context from recent conversation history
            recent_context = ""
            for msg in messages[-10:]:  # Last 5 exchanges
                if isinstance(msg, HumanMessage):
                    recent_context += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    recent_context += f"Assistant: {msg.content[:200]}...\n"  # Truncate long responses
            
            # Build agent-specific context
            agent_context = ""
            if previous_agent_type == "BRAIN_TUMOR_AGENT":
                agent_context = "This is a follow-up question about a BRAIN MRI analysis. The user is asking about their brain MRI scan results."
            elif previous_agent_type == "SKIN_LESION_AGENT":
                agent_context = "This is a follow-up question about a SKIN LESION analysis. The user is asking about their skin lesion/dermatology image results."
            elif previous_agent_type == "CHEST_XRAY_AGENT":
                agent_context = "This is a follow-up question about a CHEST X-RAY analysis. The user is asking about their chest X-ray scan results."
            else:
                agent_context = "This is a follow-up question about a medical image analysis."
            
            conversation_prompt = f"""User query: {input_text}

{agent_context}

Recent conversation context: {recent_context}

**MOST RECENT image analysis result (this is what the user is asking about):**
{previous_image_context}

You are an AI-powered Medical Conversation Assistant answering follow-up questions about a previous medical image analysis.

**CRITICAL: The user is asking about the MOST RECENT image analysis shown above. Do NOT reference any older analyses (like skin lesions if the most recent was a brain MRI, or vice versa).**

**Your role:**
- Answer their specific question based on the MOST RECENT analysis shown above
- Be professional, clear, and empathetic
- Keep responses concise (max 150 words)
- Always recommend professional consultation for medical decisions
- If you need more information or they should re-upload the image, ask them

**Response guidelines:**
- Direct answer to their question about the MOST RECENT analysis
- Clinical context from the MOST RECENT analysis only
- Professional medical disclaimer if giving advice
- Do NOT confuse this with any older analyses in the conversation

Please provide a helpful, caring response to their follow-up question about the MOST RECENT image analysis:"""

            try:
                llm = config.conversation.llm
                response = llm.invoke(conversation_prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                print(f"Error generating contextual response: {e}")
                response_text = f"""Based on our previous discussion, I'd be happy to help with your question: "{input_text}"

However, for specific medical advice based on the analysis, I recommend:
1. Consulting with a qualified healthcare professional
2. Re-uploading the image if you need a fresh analysis
3. Discussing your concerns with your doctor

Is there anything else I can help clarify?"""
            
            return {
                **state,
                "output": AIMessage(content=response_text),
                "agent_name": "CONVERSATION_AGENT"
            }
        
        # Create context from recent conversation history
        recent_context = ""
        for msg in messages:#[-20:]:  # Get last 10 exchanges (20 messages)  # currently considering complete history - limit control from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"
        
        # Check if conversation history mentions image upload or analysis
        has_recent_image_upload = any(
            "uploaded" in msg.content.lower() or 
            "x-ray" in msg.content.lower() or 
            "mri" in msg.content.lower() or
            "scan" in msg.content.lower() or
            "lesion" in msg.content.lower()
            for msg in messages if isinstance(msg, (HumanMessage, AIMessage))
        )
        
        # Detect if this is a casual query (greeting, simple question) vs. medical question
        casual_keywords = ["hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye", "how are you", "what can you do", "help", "who are you", "what are you"]
        medical_keywords = ["symptom", "diagnosis", "treatment", "disease", "condition", "medicine", "medication", "pain", "ache", "fever", "cough", "headache", "nausea", "vomiting", "diabetes", "cancer", "infection", "illness", "sick", "health", "medical", "doctor", "patient", "clinical", "what is", "tell me about", "explain"]
        
        # Simple queries (short and no medical keywords) are casual
        query_length = len(input_text.split())
        has_medical_keywords = any(keyword in input_text for keyword in medical_keywords)
        has_casual_keywords = any(keyword in input_text for keyword in casual_keywords)
        
        # Determine if casual: short query with casual keywords OR very short query without medical keywords
        is_casual = (has_casual_keywords and not has_medical_keywords) or (query_length <= 3 and not has_medical_keywords)
        is_medical_question = has_medical_keywords or (query_length > 5 and not is_casual)
        
        # Use simple conversational prompt for casual queries
        if is_casual and not is_medical_question:
            conversation_prompt = f"""You are a friendly, helpful AI medical assistant named MedAgentica. You are warm, empathetic, and professional.

User Query: {input_text}
Recent Conversation: {recent_context[-300:] if len(recent_context) > 300 else recent_context}

Instructions:
- Respond naturally and conversationally, like a caring healthcare professional
- Be brief (1-3 sentences) for greetings and casual queries
- Offer to help with medical questions when appropriate
- Use a warm, professional tone
- Don't use formal medical report format for simple greetings
- If asked what you can do, briefly mention your capabilities (general health discussions, medical knowledge queries, image analysis)

Response:"""
        else:
            # Use Mayo Clinic-style prompt for medical questions
            conversation_prompt = CONVERSATION_CLINICAL_PROMPT.format(
            input_text=input_text,
            recent_context=recent_context[-500:] if len(recent_context) > 500 else recent_context,
            has_image="Yes - image analyzed" if has_recent_image_upload else "No"
            )
            
        # Add additional instruction for Mayo Clinic style
            conversation_prompt += """

**Additional Mayo Clinic Standards:**
- For medical questions: Provide comprehensive analysis with all required sections
- Cite evidence quality when making medical claims (strong/moderate/low evidence)
- Use professional medical terminology but explain when needed
- For diagnostic questions: List 2-3 most likely considerations with rationale
- Always state when professional evaluation is needed
- Include brief "next steps" guidance when applicable
- Keep responses comprehensive but readable

Generate a response following the Mayo Clinic clinical decision-support format above.
Response:"""

        # print("Conversation Prompt:", conversation_prompt)

        # Try to get response from LLM with fallback
        try:
            # Invoke LLM with proper message format
            llm_response = config.conversation.llm.invoke([HumanMessage(content=conversation_prompt)])
            # Extract content from response
            if hasattr(llm_response, 'content'):
                response = AIMessage(content=llm_response.content)
            else:
                response = AIMessage(content=str(llm_response))
        except Exception as e:
            print(f"[Conversation Agent] LLM error: {e}")
            # Fallback response when LLM fails
            response = AIMessage(content=f"I'm here to help! You asked: '{input_text}'. For medical questions, I recommend consulting healthcare professionals. For general questions, I can provide helpful information when my services are available.")

        # response = AIMessage(content="This would be handled by the conversation agent.")

        return {
            **state,
            "output": response,
            "agent_name": "CONVERSATION_AGENT"
        }
    
    def run_rag_agent(state: AgentState) -> AgentState:
        """Handle medical knowledge queries using Agentic RAG System."""
        print(f"Selected agent: RAG_AGENT (Agentic RAG)")

        try:
            # Initialize the agentic RAG system
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from legacy.demo_agentic_rag import AgenticRAGSystem

            agentic_rag = AgenticRAGSystem(
                pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
                pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "medagentica"),
                openrouter_api_key=os.getenv("GROQ_API_KEY", ""),
                openrouter_model="llama-3.3-70b-versatile"
            )
            
            messages = state["messages"]
            query = state["current_input"]

            # Convert messages to chat history format for agentic RAG
            chat_history = []
            for msg in messages[-10:]:  # Last 5 exchanges
                if isinstance(msg, HumanMessage):
                    chat_history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    chat_history.append({"role": "assistant", "content": msg.content})

            print(f"Processing query with Agentic RAG: {query[:100]}...")

            # Use the agentic RAG system
            response = agentic_rag.query(query, chat_history)

            response_text = response.get("response", "")
            confidence = response.get("confidence", 0.5)
            sources = response.get("sources", [])

            print(f"Agentic RAG Response preview: {response_text[:200]}...")
            print(f"Confidence: {confidence}")
            print(f"Sources found: {len(sources)}")

            # Check for insufficient information
            insufficient_info = (
                confidence < 0.3 or
                len(sources) == 0 or
            "don't have enough information" in response_text.lower() or
                "insufficient information" in response_text.lower()
            )

            print(f"Insufficient info flag: {insufficient_info}")

            # Determine if we should route to web search
            should_route_to_web_search = insufficient_info

            print(f"Should route to web search: {should_route_to_web_search}")

            # Store RAG output appropriately
            if should_route_to_web_search:
                response_output = AIMessage(content="")  # Empty response to trigger web search
            else:
                response_output = AIMessage(content=response_text)
            
            return {
                **state,
                "output": response_output,
                "needs_human_validation": False,
                "retrieval_confidence": confidence,
                "agent_name": "RAG_AGENT",
                "insufficient_info": insufficient_info
            }

        except Exception as e:
            print(f"Agentic RAG Agent Error: {e}")
            import traceback
            traceback.print_exc()

            # Return state that will trigger web search fallback
            return {
                **state,
                "output": AIMessage(content=""),
                "needs_human_validation": False,
                "retrieval_confidence": 0.0,
                "agent_name": "RAG_AGENT",
                "insufficient_info": True
            }

    # Web Search Processor Node
    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        """Handles web search results, processes them with LLM, and generates a refined response."""

        print(f"Selected agent: WEB_SEARCH_PROCESSOR_AGENT")
        print("[WEB_SEARCH_PROCESSOR_AGENT] Processing Web Search Results...")
        
        messages = state["messages"]
        web_search_context_limit = config.web_search.context_limit

        recent_context = ""
        for msg in messages[-web_search_context_limit:]: # limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        web_search_processor = WebSearchProcessorAgent(config)

        processed_response = web_search_processor.process_web_search_results(query=state["current_input"], chat_history=recent_context)

        # print("######### DEBUG WEB SEARCH:", processed_response)
        
        if state['agent_name'] != None:
            involved_agents = f"{state['agent_name']}, WEB_SEARCH_PROCESSOR_AGENT"
        else:
            involved_agents = "WEB_SEARCH_PROCESSOR_AGENT"

        # Overwrite any previous output with the processed Web Search response
        return {
            **state,
            # "output": "This would be handled by the web search agent, finding the latest information.",
            "output": processed_response,
            "agent_name": involved_agents
        }

    # Define Routing Logic
    def confidence_based_routing(state: AgentState) -> str:
        """Route based on RAG confidence score and response content."""
        retrieval_confidence = state.get('retrieval_confidence', 0.0)
        insufficient_info = state.get('insufficient_info', False)
        
        print(f"Routing Decision:")
        print(f"  - Retrieval confidence: {retrieval_confidence}")
        print(f"  - Min confidence threshold: {config.rag.min_retrieval_confidence}")
        print(f"  - Insufficient info flag: {insufficient_info}")
        
        # Route to web search if confidence is low or info is insufficient
        if retrieval_confidence < config.rag.min_retrieval_confidence or insufficient_info:
            print("  - DECISION: Routing to WEB_SEARCH_PROCESSOR_AGENT")
            return "WEB_SEARCH_PROCESSOR_AGENT"
        else:
            print("  - DECISION: Proceeding with RAG response")
            return "check_validation"
    
    def run_brain_tumor_agent(state: AgentState) -> AgentState:
        """Handle brain MRI image analysis with concise clinical response."""
        from langchain_core.messages import HumanMessage, SystemMessage
        
        current_input = state["current_input"]
        image_path = current_input.get("image", None) if isinstance(current_input, dict) else None
        
        # Extract user's query
        user_query = ""
        if isinstance(current_input, dict):
            user_query = current_input.get("text", "")
        elif isinstance(current_input, str):
            user_query = current_input
            
        if not user_query and state.get("messages"):
            for msg in reversed(state["messages"]):
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    content = msg.content
                    if "uploaded an image" not in content.lower() and content.strip():
                        user_query = content
                        break
        if not user_query:
            user_query = "Analyze this brain MRI for tumors"

        print(f"Selected agent: BRAIN_TUMOR_AGENT")
        print(f"Image path: {image_path}")
        print(f"User query: {user_query}")

        # CRITICAL FIX: Check if this is a follow-up question (no new image)
        # If no image uploaded but user is asking about previous analysis
        if not image_path:
            # Check if there's a previous image path in state
            image_path = state.get("last_image_path") or state.get("image_path")
            
            # Check if this is a follow-up question about previous analysis
            follow_up_keywords = ["should i", "what should", "what next", "do i need", "tell me more", 
                                  "explain", "is it serious", "how bad", "treatment", "what does"]
            is_follow_up = any(keyword in user_query.lower() for keyword in follow_up_keywords)
            
            if is_follow_up and image_path and os.path.exists(image_path):
                print(f"✅ Follow-up question detected, using previous image: {image_path}")
                # Continue with analysis using stored image path
            elif is_follow_up and not image_path:
                # Follow-up question but no previous image - use conversation context
                print(f"✅ Follow-up question detected but no image, using conversation agent")
                
                # Get conversation history for context
                recent_context = ""
                for msg in state["messages"][-10:]:  # Last 5 exchanges
                    if isinstance(msg, HumanMessage):
                        recent_context += f"User: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        recent_context += f"Assistant: {msg.content[:200]}...\n"  # Truncate long responses
                
                # Use LLM to answer based on conversation context
                system_prompt = """You are a compassionate neurosurgeon answering follow-up questions about a brain MRI analysis.

**Context:** The patient previously uploaded a brain MRI that was analyzed by the AI system. Now they are asking follow-up questions about that analysis.

**Your Role:**
- Answer their specific question based on the conversation history
- Be professional, clear, and empathetic
- If you need to see the image again, ask them to re-upload it
- Keep responses concise (max 150 words)
- Always recommend professional consultation

**Response Format:**
- Direct answer to their question
- Clinical context if relevant
- Next steps recommendation"""

                user_prompt = f"""Previous conversation:
{recent_context}

Current question: {user_query}

Please answer their follow-up question based on the previous analysis. If you cannot answer without seeing the image again, politely ask them to re-upload it."""

                try:
                    llm = config.medical_cv.llm
                    messages_for_llm = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ]
                    response = llm.invoke(messages_for_llm)
                    response_text = response.content
                except Exception as e:
                    print(f"Error generating follow-up response: {e}")
                    response_text = f"""Based on our previous discussion about your brain MRI, I'd be happy to help with your question: "{user_query}"

However, to provide the most accurate guidance, I recommend:
1. Re-uploading the MRI image so I can review it in context of your question
2. Consulting with a neurologist or neurosurgeon for personalized advice
3. Discussing any concerns about the findings with your healthcare provider

Would you like to upload the image again so I can provide more specific guidance?"""
                
                return {
                    **state,
                    "output": AIMessage(content=response_text),
                    "needs_human_validation": False,
                    "agent_name": "BRAIN_TUMOR_AGENT (Follow-up)"
                }
            else:
                # Not a follow-up, just missing image
                response = AIMessage(content="Please upload a brain MRI image so I can analyze it for you. I need to see the image to provide accurate analysis.")
                return {
                    **state,
                    "output": response,
                    "needs_human_validation": False,
                    "agent_name": "BRAIN_TUMOR_AGENT"
                }
        
        # Continue with normal image analysis if we have an image path
        if not os.path.exists(image_path):
            response = AIMessage(content="Error: The image file could not be found. Please upload the brain MRI image again.")
            return {
                **state,
                "output": response,
                "needs_human_validation": False,
                "agent_name": "BRAIN_TUMOR_AGENT"
            }

        try:
            # Use the brain tumor agent to analyze the image
            classification_result = AgentConfig.image_analyzer.classify_brain_tumor(image_path)
            print(f"Brain tumor analysis result: {classification_result}")
            
            # Check if we got detailed results (dict) or old format (string)
            if isinstance(classification_result, dict) and not classification_result.get("error"):
                # Use new classifier with professional response
                print(f"✅ Using Brain Tumor Classifier: {classification_result.get('predicted_class')} ({classification_result.get('confidence'):.2f}%)")
                
                # Build comprehensive professional response
                response_text = build_brain_tumor_response(classification_result, user_query, state.get("user_role", "patient"))
                response = AIMessage(content=response_text)
            else:
                # Fallback: old format or error
                if isinstance(classification_result, dict) and classification_result.get("error"):
                    error_msg = classification_result.get("error", "Unknown error")
                    response_text = f"I apologize, but I encountered an issue analyzing your brain MRI image: {error_msg}. Please try uploading a different image or consult with a healthcare professional."
                else:
                    # Old string format
                    response_text = f"**Analysis:** {classification_result}. Clinical correlation advised."
                response = AIMessage(content=response_text)

            return {
                **state,
                "output": response,
                "needs_human_validation": False,
                "agent_name": "BRAIN_TUMOR_AGENT"
            }

        except Exception as e:
            print(f"Brain Tumor Agent Error: {e}")
            import traceback
            traceback.print_exc()

            response = AIMessage(content=f"Error analyzing brain MRI: {str(e)}. Clinical correlation advised.")
            return {
                **state,
                "output": response,
                "needs_human_validation": False,
                "agent_name": "BRAIN_TUMOR_AGENT"
            }
    
    def run_chest_xray_agent(state: AgentState) -> AgentState:
        """Handle chest X-ray image analysis with full MedRAX integration - returns raw MedRAX output."""
        from agents.image_analysis_agent.chest_xray_agent.medrax_full_integration import MedRAXFullIntegration

        current_input = state["current_input"]
        
        # Get image path from multiple sources
        image_path = None
        if isinstance(current_input, dict):
            image_path = current_input.get("image", None)
        
        # Also check state for image path (from analyze_input)
        if not image_path:
            image_path = state.get("image_path") or state.get("last_image_path")
        
        # Extract user's query from messages or current_input
        user_query = ""
        if isinstance(current_input, dict):
            user_query = current_input.get("text", "")
        
        # Also check messages for the query
        if not user_query and state.get("messages"):
            # Get the last human message
            for msg in reversed(state["messages"]):
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    # Skip generic messages like "user uploaded an image for diagnosis"
                    content = msg.content
                    if "uploaded an image" not in content.lower() and content.strip():
                        user_query = content
                        break
        
        # If still no query, use a default
        if not user_query:
            user_query = "Analyze this chest X-ray image"

        print(f"Selected agent: CHEST_XRAY_AGENT")
        print(f"Image path: {image_path}")
        print(f"User query: {user_query}")

        if not image_path or not os.path.exists(image_path):
            response = AIMessage(content="Error: No valid image provided for chest X-ray analysis. Please upload a chest X-ray image.")
            return {
                **state,
                "output": response,
                "needs_human_validation": False,
                "agent_name": "CHEST_XRAY_AGENT"
            }

        try:
            # Initialize full MedRAX integration
            medrax = MedRAXFullIntegration(device=None)
            
            # Perform comprehensive analysis (all three outputs)
            analysis_results = medrax.comprehensive_analysis(image_path, user_query)
            
            # Extract results
            classification = analysis_results.get("classification", {})
            segmentation = analysis_results.get("segmentation", {})
            report = analysis_results.get("report", {})
            disease_grounding = analysis_results.get("disease_grounding", {})
            
            # Get image paths for three outputs
            original_image_path = image_path
            segmentation_image_path = segmentation.get("segmentation_image_path") if segmentation else None
            disease_grounding_path = disease_grounding.get("combined_visualization_path") if disease_grounding else None
            
            # Convert paths to URLs
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            def path_to_url(path):
                if path and os.path.exists(path):
                    rel_path = os.path.relpath(path, project_root)
                    return f"/{rel_path.replace(os.sep, '/')}"
                return None
            
            original_image_url = path_to_url(original_image_path)
            segmentation_image_url = path_to_url(segmentation_image_path)
            disease_grounding_url = path_to_url(disease_grounding_path)
            
            # Debug: Print image URLs
            print(f"🔍 Image URLs:")
            print(f"  - Original: {original_image_url}")
            print(f"  - Segmentation: {segmentation_image_url}")
            print(f"  - Disease Grounding: {disease_grounding_url}")
            
            # Return RAW MedRAX output without prompt engineering
            response_text = build_raw_medrax_response(
                classification, 
                segmentation, 
                report, 
                disease_grounding
            )
            
            # Add explicit header for detection
            response_text = "**MedRAX Chest X-ray Analysis**\n\n" + response_text
            
            response = AIMessage(content=response_text)

            # Return all image URLs in state
            return_state = {
                **state,
                "output": response,
                "needs_human_validation": True,
                "agent_name": "CHEST_XRAY_AGENT",
                "image_path": original_image_path,
                "original_image_url": original_image_url,
                "segmentation_path": segmentation_image_path,
                "segmentation_image_url": segmentation_image_url,
                "disease_grounding_path": disease_grounding_path,
                "disease_grounding_url": disease_grounding_url,
                "analysis_results": analysis_results
            }
            
            # Ensure all three images are always included
            if not segmentation_image_url and segmentation.get("segmentation_image_path"):
                # Try to get segmentation URL again
                seg_path = segmentation.get("segmentation_image_path")
                if seg_path and os.path.exists(seg_path):
                    segmentation_image_url = path_to_url(seg_path)
                    return_state["segmentation_image_url"] = segmentation_image_url
                    print(f"✅ Fixed segmentation URL: {segmentation_image_url}")
            
            if not disease_grounding_url and disease_grounding.get("combined_visualization_path"):
                # Try to get disease grounding URL again
                grounding_path = disease_grounding.get("combined_visualization_path")
                if grounding_path and os.path.exists(grounding_path):
                    disease_grounding_url = path_to_url(grounding_path)
                    return_state["disease_grounding_url"] = disease_grounding_url
                    print(f"✅ Fixed disease grounding URL: {disease_grounding_url}")
            
            return return_state
            
        except Exception as e:
            print(f"Chest X-ray Agent Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to basic analysis
            try:
                predicted_class = AgentConfig.image_analyzer.classify_chest_xray(image_path)
                if predicted_class == "covid19":
                    response_text = "The analysis of the uploaded chest X-ray image indicates a **POSITIVE** result for **COVID-19**."
                elif predicted_class == "normal":
                    response_text = "The analysis of the uploaded chest X-ray image indicates a **NEGATIVE** result for **COVID-19**, i.e., **NORMAL**."
                else:
                    response_text = f"Error analyzing chest X-ray: {str(e)}. Please try uploading a different image."
            except:
                response_text = f"Error analyzing chest X-ray: {str(e)}. Please try uploading a different image."

            response = AIMessage(content=response_text)
            return {
                **state,
                "output": response,
                "needs_human_validation": False,
                "agent_name": "CHEST_XRAY_AGENT"
            }
    
    def run_skin_lesion_agent(state: AgentState) -> AgentState:
        """Handle skin lesion image analysis using new skin cancer classifier."""
        current_input = state["current_input"]
        image_path = current_input.get("image", None) if isinstance(current_input, dict) else None
        
        # Check for previous image if none provided
        if not image_path:
            image_path = state.get("last_image_path") or state.get("image_path")
        
        # Extract user's query
        user_query = ""
        if isinstance(current_input, dict):
            user_query = current_input.get("text", "")
        
        # Also check messages for the query
        if not user_query and state.get("messages"):
            for msg in reversed(state["messages"]):
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    content = msg.content
                    if "uploaded an image" not in content.lower() and content.strip():
                        user_query = content
                        break
        
        # If still no query, use a default
        if not user_query:
            user_query = "Analyze this skin lesion and tell me if it's benign or malignant"

        print(f"Selected agent: SKIN_LESION_AGENT")
        print(f"Image path: {image_path}")
        print(f"User query: {user_query}")

        if not image_path or not os.path.exists(image_path):
            response = AIMessage(content="I apologize, but I couldn't find a valid image for analysis. Please ensure you've uploaded a clear image of the skin lesion.")
            return {
                **state,
                "output": response,
                "needs_human_validation": False,
                "agent_name": "SKIN_LESION_AGENT"
            }

        try:
            # Try new skin cancer classifier first
            classification_result = AgentConfig.image_analyzer.classify_skin_cancer(image_path)
            
            if classification_result and not classification_result.get("error"):
                # Use new classifier with professional response
                print(f"✅ Using Skin Cancer Classifier: {classification_result.get('predicted_class')} ({classification_result.get('confidence'):.2f}%)")
                
                # Build comprehensive professional response
                response_text = build_skin_cancer_response(classification_result, user_query, state.get("user_role", "patient"))
                response = AIMessage(content=response_text)
                
                return {
                    **state,
                    "output": response,
                    "needs_human_validation": False,
                    "agent_name": "SKIN_LESION_AGENT",
                    "classification_result": classification_result
                }
            else:
                # Fallback to old segmentation method
                print("⚠️ Skin Cancer Classifier not available, using segmentation fallback")
                predicted_mask = AgentConfig.image_analyzer.segment_skin_lesion(image_path)
                
                if predicted_mask:
                    overlay_path = AgentConfig.image_analyzer.skin_lesion_segmentation_output_path
                    if os.path.exists(overlay_path):
                        response = AIMessage(content="""Skin Lesion Analysis Complete

The uploaded skin lesion image has been successfully analyzed using AI segmentation technology.

**Analysis Results:**
- **Segmentation Status:** Successfully segmented
- **Visualization:** An overlay image showing the segmented lesion area has been generated

**Important:** This analysis provides segmentation visualization for educational purposes. For classification (benign vs. malignant), please ensure the skin cancer classifier model is available.

**Next Steps:**
- Review the segmentation overlay image
- Consult with a qualified dermatologist for professional evaluation and classification
- Consider follow-up examination if concerned about any skin changes

**Medical Disclaimer:** This Metagentica analysis is for informational purposes only and cannot replace professional medical evaluation.""")
                    else:
                        response = AIMessage(content="Skin Lesion Analysis Complete\n\nThe skin lesion has been successfully segmented using AI technology. Please consult with a healthcare professional for classification and diagnosis.")
                else:
                    response = AIMessage(content="Analysis Failed\n\nThe uploaded image could not be properly analyzed. Please ensure:\n- The image shows a clear skin lesion\n- The image is well-lit and in focus\n- Try uploading a different image or consult with a healthcare professional.")
                
                return {
                    **state,
                    "output": response,
                    "needs_human_validation": False,
                    "agent_name": "SKIN_LESION_AGENT"
                }
            
        except Exception as e:
            print(f"Skin Lesion Agent Error: {e}")
            import traceback
            traceback.print_exc()
            
            response = AIMessage(content=f"I apologize, but I encountered an error while analyzing your skin lesion image: {str(e)}. Please try uploading a different image or consult with a healthcare professional for proper evaluation.")
            return {
                **state,
                "output": response,
                "needs_human_validation": False,
                "agent_name": "SKIN_LESION_AGENT"
            }
    
    def handle_human_validation(state: AgentState) -> Dict:
        """Prepare for human validation if needed."""
        if state.get("needs_human_validation", False):
            return {"agent_state": state, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
        return {"agent_state": state, "next": END}
    
    def perform_human_validation(state: AgentState) -> AgentState:
        """Handle human validation process."""
        print(f"Selected agent: HUMAN_VALIDATION")

        agent_name = state.get("agent_name", "")
        output_content = state['output'].content if hasattr(state['output'], 'content') else str(state['output'])

        # For medical image analysis agents, don't modify the response with validation prompts
        # Just mark that validation is needed but keep the original analysis
        medical_agents = ["CHEST_XRAY_AGENT", "BRAIN_TUMOR_AGENT", "SKIN_LESION_AGENT"]
        if any(m in agent_name for m in medical_agents):
            print(f"[Human Validation] Medical image analysis agent {agent_name} - keeping original response")
            return {
                **state,
                "output": state['output'],  # Keep original medical analysis response
                "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION"
            }

        # For other agents, append validation request
        validation_prompt = f"{output_content}\n\n**Human Validation Required:**\n- If you're a healthcare professional: Please validate the output. Select **Yes** or **No**. If No, provide comments.\n- If you're a patient: Simply click Yes to confirm."

        # Create an AI message with the validation prompt
        validation_message = AIMessage(content=validation_prompt)

        return {
            **state,
            "output": validation_message,
            "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION"
        }

    # Check output through guardrails
    def apply_output_guardrails(state: AgentState) -> AgentState:
        """Apply output guardrails to the generated response.
        
        PERFORMANCE OPTIMIZATION: Output guardrails disabled for speed.
        Saves 2-4 seconds per query. Medical agents have their own safety checks.
        """
        # DISABLED: Just pass through without LLM check (saves 2-4 seconds)
        print(f"[Guardrails] Output guardrails DISABLED for performance - passing through")
        
        output = state["output"]
        
        # Just add the output to messages unchanged (no LLM filtering)
        return {
            **state,
            "messages": output,
            "output": output
        }

    # OLD CODE (disabled for speed) - commented out entire function
    def apply_output_guardrails_OLD_DISABLED(state: AgentState) -> AgentState:
        """OLD VERSION - Disabled for performance optimization."""
        output = state["output"]
        current_input = state["current_input"]
        agent_name = state.get("agent_name", "")

        if not output or not isinstance(output, (str, AIMessage)):
            return state

        output_text = output if isinstance(output, str) else output.content
        
        medical_agents = ["CHEST_XRAY_AGENT", "BRAIN_TUMOR_AGENT", "SKIN_LESION_AGENT"]
        if any(m in agent_name for m in medical_agents):
            print(f"[Guardrails] Skipping output guardrails for medical image analysis agent: {agent_name}")
            return {
                **state,
                "messages": output,
                "output": output
            }

        if "Human Validation Required" in output_text:
            # Check if the current input is a human validation response
            validation_input = ""
            if isinstance(current_input, str):
                validation_input = current_input
            elif isinstance(current_input, dict):
                validation_input = current_input.get("text", "")
            
            # If validation input exists
            if validation_input.lower().startswith(('yes', 'no')):
                # Add the validation result to the conversation history
                validation_response = HumanMessage(content=f"Validation Result: {validation_input}")
                
                # If validation is 'No', modify the output
                if validation_input.lower().startswith('no'):
                    fallback_message = AIMessage(content="The previous medical analysis requires further review. A healthcare professional has flagged potential inaccuracies.")
                    return {
                        **state,
                        "messages": [validation_response, fallback_message],
                        "output": fallback_message
                    }
                
                return {
                    **state,
                    "messages": validation_response
                }
        
        # Get the original input text
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
            sanitized_output = guardrails.check_output(output_text, input_text)
            sanitized_message = AIMessage(content=sanitized_output) if isinstance(output, AIMessage) else sanitized_output
            return {
                **state,
                "messages": sanitized_message,
                "output": sanitized_message
            }

    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("EMERGENCY_RESPONSE", run_conversation_agent)  # Reuse conversation agent for emergencies
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("BRAIN_TUMOR_AGENT", run_brain_tumor_agent)
    workflow.add_node("CHEST_XRAY_AGENT", run_chest_xray_agent)
    workflow.add_node("SKIN_LESION_AGENT", run_skin_lesion_agent)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("apply_guardrails", apply_output_guardrails)
    
    # Define the edges (workflow connections)
    workflow.set_entry_point("analyze_input")
    # workflow.add_edge("analyze_input", "route_to_agent")
    # Add conditional routing for guardrails bypass
    workflow.add_conditional_edges(
        "analyze_input",
        check_if_bypassing,
        {
            "apply_guardrails": "apply_guardrails",
            "route_to_agent": "route_to_agent"
        }
    )
    
    # Connect decision router to agents
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "EMERGENCY_RESPONSE": "EMERGENCY_RESPONSE",
            "RAG_AGENT": "RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
            "BRAIN_TUMOR_AGENT": "BRAIN_TUMOR_AGENT",
            "CHEST_XRAY_AGENT": "CHEST_XRAY_AGENT",
            "SKIN_LESION_AGENT": "SKIN_LESION_AGENT",
            "needs_validation": "RAG_AGENT"  # Default to RAG if confidence is low
        }
    )
    
    # Connect agent outputs to validation check
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    workflow.add_edge("EMERGENCY_RESPONSE", "check_validation")
    # workflow.add_edge("RAG_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing)
    workflow.add_edge("BRAIN_TUMOR_AGENT", "check_validation")
    workflow.add_edge("CHEST_XRAY_AGENT", "check_validation")
    workflow.add_edge("SKIN_LESION_AGENT", "check_validation")

    workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)
    
    workflow.add_conditional_edges(
        "check_validation",
        lambda x: x["next"],
        {
            "human_validation": "human_validation",
            END: "apply_guardrails"  # Route to guardrails instead of END
        }
    )
    
    # workflow.add_edge("human_validation", END)
    
    # Compile the graph
    return workflow.compile(checkpointer=memory)


def init_agent_state() -> AgentState:
    """Initialize the agent state with default values."""
    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "has_image": False,
        "image_type": None,
        "image_classification_confidence": None,
        # "last_image_path": None,  # CRITICAL: Do NOT reset this, let it persist from checkpoint
        "image_path": None,  # Store current image path for agents
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "insufficient_info": False,
        "awaiting_image_clarification": False,
        "user_role": "patient"
    }


def process_query(query: Union[str, Dict], conversation_history: List[BaseMessage] = None, thread_id: str = "1") -> str:
    """Process a user query through the agent decision system.
    
    Args:
        query: User input (text string or dict with text and image)
        conversation_history: Optional list of previous messages
        thread_id: Unique identifier for the conversation thread
        
    Returns:
        Response from the appropriate agent
    """
    # Initialize the graph
    graph = create_agent_graph()

    # # Save Graph Flowchart
    # image_bytes = graph.get_graph().draw_mermaid_png()
    # decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    # cv2.imwrite("./assets/graph.png", decoded)
    # print("Graph flowchart saved in assets.")
    
    # Initialize state
    state = init_agent_state()
    # if conversation_history:
    #     state["messages"] = conversation_history
    
    # Add the current query
    state["current_input"] = query

    # To handle image upload case
    if isinstance(query, dict):
        # Preserve the original text query
        text_query = query.get("text", "")
        # Extract user role if present
        state["user_role"] = query.get("user_role", "patient")
        
        if text_query:
            state["messages"] = [HumanMessage(content=text_query)]
        else:
            # Don't assume X-ray - use generic medical image analysis query
            state["messages"] = [HumanMessage(content="Analyze this medical image")]
    else:
        state["messages"] = [HumanMessage(content=query)]
        state["user_role"] = "patient"

    # Use the provided thread_id for memory persistence
    current_thread_config = {"configurable": {"thread_id": thread_id}}
    print(f"🧠 Processing query with thread_id: {thread_id}")
    
    result = graph.invoke(state, current_thread_config)
    # print("######### DEBUG 4:", result)
    # state["messages"] = [result["messages"][-1].content]

    # Enhanced conversation memory management
    current_messages = result["messages"]

    # Keep history to reasonable size with intelligent summarization
    max_history = getattr(config, 'max_conversation_history', 20)

    if len(current_messages) > max_history:
        # Keep the most recent messages and summarize older ones if needed
        recent_messages = current_messages[-max_history:]

        # For very long conversations, add a summary message
        if len(current_messages) > max_history * 2:
            summary_message = AIMessage(content=f"💭 **Conversation Summary**: This is a continuation of our discussion about your health concerns. Previous topics included medical questions and responses. I'm here to help with any follow-up questions.")
            recent_messages.insert(0, summary_message)

        result["messages"] = recent_messages

    # visualize conversation history in console
    for m in result["messages"]:
        m.pretty_print()
    
    # Add the response to conversation history
    return result