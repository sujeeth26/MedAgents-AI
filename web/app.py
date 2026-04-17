
import os
import sys
import uuid
import tempfile
from typing import Dict, Union, Optional, List
import glob
import threading
import time
from io import BytesIO

# Add parent directory to path for imports (must be before other imports)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response, Cookie
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import uvicorn
import requests
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs

from config import Config
from agents.agent_decision import process_query

# Load configuration
config = Config()

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Medical Chatbot", version="2.0")

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the project root directory
import pathlib
project_root = pathlib.Path(__file__).parent.parent.absolute()

# Set up directories with absolute paths
UPLOAD_FOLDER = project_root / "uploads/backend"
FRONTEND_UPLOAD_FOLDER = project_root / "uploads/frontend"
SKIN_LESION_OUTPUT = project_root / "uploads/skin_lesion_output"
CHEST_XRAY_OUTPUT = project_root / "uploads/chest_xray_output"
SPEECH_DIR = project_root / "uploads/speech"

# Create directories if they don't exist
for directory in [UPLOAD_FOLDER, FRONTEND_UPLOAD_FOLDER, SKIN_LESION_OUTPUT, CHEST_XRAY_OUTPUT, SPEECH_DIR]:
    os.makedirs(directory, exist_ok=True)

# Convert back to strings for compatibility
UPLOAD_FOLDER = str(UPLOAD_FOLDER)
FRONTEND_UPLOAD_FOLDER = str(FRONTEND_UPLOAD_FOLDER)
SKIN_LESION_OUTPUT = str(SKIN_LESION_OUTPUT)
CHEST_XRAY_OUTPUT = str(CHEST_XRAY_OUTPUT)
SPEECH_DIR = str(SPEECH_DIR)

# Mount static files directory
data_dir = project_root / "data"
uploads_dir = project_root / "uploads"

# Only mount if directories exist
if data_dir.exists():
    app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")
if uploads_dir.exists():
    app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

# Set up templates - use absolute path relative to web directory
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Initialize ElevenLabs client
client = ElevenLabs(
    api_key=config.speech.eleven_labs_api_key,
)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_audio():
    """Deletes all .mp3 files in the uploads/speech folder every 5 minutes."""
    while True:
        try:
            files = glob.glob(f"{SPEECH_DIR}/*.mp3")
            for file in files:
                os.remove(file)
            print("Cleaned up old speech files.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        time.sleep(300)  # Runs every 5 minutes

# Start background cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_audio, daemon=True)
cleanup_thread.start()

# Session-based image persistence with expiry
# Store recently uploaded images per session (session_id -> {image_path, timestamp})
from datetime import datetime, timedelta
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

session_images: Dict[str, Dict] = {}
session_images_lock = threading.Lock()

# Security configuration
SESSION_EXPIRY_HOURS = 2  # Sessions expire after 2 hours
COOKIE_MAX_AGE = 7200  # 2 hours in seconds

def audit_log(action: str, session_id: str, details: str = ""):
    """Audit logging for security and compliance"""
    logger.info(f"AUDIT: action={action} session={session_id[:8]}... details={details}")

def store_session_image(session_id: str, image_path: str):
    """Store uploaded image path for a session with timestamp"""
    with session_images_lock:
        session_images[session_id] = {
            'image_path': image_path,
            'timestamp': datetime.now(),
            'access_count': 0
        }
        audit_log("IMAGE_STORED", session_id, f"path={image_path}")
        print(f"📸 Stored image for session {session_id}: {image_path}")

def get_session_image(session_id: str) -> Optional[str]:
    """Retrieve uploaded image path for a session if not expired"""
    with session_images_lock:
        session_data = session_images.get(session_id)
        
        if not session_data:
            return None
        
        # Check if session is expired
        age = datetime.now() - session_data['timestamp']
        if age > timedelta(hours=SESSION_EXPIRY_HOURS):
            logger.warning(f"Session {session_id[:8]}... expired (age: {age})")
            del session_images[session_id]
            return None
        
        image_path = session_data['image_path']
        
        if image_path and os.path.exists(image_path):
            session_data['access_count'] += 1
            audit_log("IMAGE_ACCESSED", session_id, f"path={image_path} count={session_data['access_count']}")
            print(f"✅ Retrieved image for session {session_id}: {image_path}")
            return image_path
        elif image_path:
            logger.error(f"Image path exists in session but file not found: {image_path}")
            return None
        return None

def clear_session_image(session_id: str):
    """Clear uploaded image path for a session"""
    with session_images_lock:
        if session_id in session_images:
            audit_log("IMAGE_CLEARED", session_id)
            print(f"🗑️ Cleared image for session {session_id}")
            del session_images[session_id]

def cleanup_expired_sessions():
    """Remove expired sessions (run periodically)"""
    with session_images_lock:
        expired = []
        current_time = datetime.now()
        
        for session_id, data in session_images.items():
            age = current_time - data['timestamp']
            if age > timedelta(hours=SESSION_EXPIRY_HOURS):
                expired.append(session_id)
        
        for session_id in expired:
            logger.info(f"Cleaning up expired session: {session_id[:8]}...")
            del session_images[session_id]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

# Start background cleanup thread
def session_cleanup_worker():
    """Background worker to clean up expired sessions every 30 minutes"""
    while True:
        try:
            time.sleep(1800)  # 30 minutes
            cleanup_expired_sessions()
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")

cleanup_thread_sessions = threading.Thread(target=session_cleanup_worker, daemon=True)
cleanup_thread_sessions.start()

class QueryRequest(BaseModel):
    query: str
    conversation_history: List = []
    user_role: str = "patient"
    conversation_id: str = "1"  # Default to "1" for backward compatibility

class SpeechRequest(BaseModel):
    text: str
    voice_id: str = "EXAMPLE_VOICE_ID"  # Default voice ID

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    """Basic health check endpoint for Docker health checks"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/health/ready")
async def readiness_check():
    """
    Comprehensive readiness check - verifies all dependencies are operational.
    Returns 200 if ready, 503 if not ready.
    """
    checks = {
        "server": "ok",
        "timestamp": datetime.now().isoformat()
    }
    
    all_healthy = True
    
    # Check if config is loaded
    try:
        from config import Config
        test_config = Config()
        checks["config"] = "ok"
    except Exception as e:
        checks["config"] = f"error: {str(e)}"
        all_healthy = False
    
    # Check if agent system is available
    try:
        from agents.agent_decision import init_agent_state
        test_state = init_agent_state()
        checks["agent_system"] = "ok"
    except Exception as e:
        checks["agent_system"] = f"error: {str(e)}"
        all_healthy = False
    
    # Check upload directories
    try:
        for directory in [UPLOAD_FOLDER, SKIN_LESION_OUTPUT, CHEST_XRAY_OUTPUT]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        checks["upload_directories"] = "ok"
    except Exception as e:
        checks["upload_directories"] = f"error: {str(e)}"
        all_healthy = False
    
    # Check session storage
    try:
        session_count = len(session_images)
        checks["session_storage"] = f"ok ({session_count} active sessions)"
    except Exception as e:
        checks["session_storage"] = f"error: {str(e)}"
        all_healthy = False
    
    status_code = 200 if all_healthy else 503
    checks["overall_status"] = "ready" if all_healthy else "not_ready"
    
    return JSONResponse(status_code=status_code, content=checks)

@app.get("/health/live")
def liveness_check():
    """Liveness check - server is running"""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}

@app.get("/metrics")
def metrics():
    """Basic metrics endpoint"""
    with session_images_lock:
        active_sessions = len(session_images)
        total_accesses = sum(data['access_count'] for data in session_images.values())
    
    return {
        "active_sessions": active_sessions,
        "total_image_accesses": total_accesses,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat")
def chat(
    request: QueryRequest, 
    response: Response, 
    session_id: Optional[str] = Cookie(None)
):
    """Process user text query through the multi-agent system."""
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        # Check if user is referring to a previously uploaded image
        query_lower = request.query.lower()
        # Refined keywords - only force image context if user EXPLICITLY refers to the image
        # Generic words like "check", "examine", "diagnose" are removed to prevent false positives
        image_analysis_keywords = [
            "the image", "this image", "uploaded image", "the picture", "this picture",
            "the scan", "this scan", "the x-ray", "this x-ray", "the xray", "this xray",
            "the mri", "this mri", "analyze image", "analyze picture", "analyze scan"
        ]
        
        # Check if query mentions image analysis
        mentions_image = any(keyword in query_lower for keyword in image_analysis_keywords)
        
        # If user mentions image analysis and we have a recent image in session, use it
        query_input = request.query
        if mentions_image and session_id:
            stored_image_path = get_session_image(session_id)
            if stored_image_path:
                print(f"🔍 User query explicitly mentions image - Using stored image: {stored_image_path}")
                # Create dict input with both text and image
                query_input = {"text": request.query, "image": stored_image_path, "user_role": request.user_role}
            else:
                print(f"⚠️ User mentions image but no stored image found for session {session_id}")
        # Also pass the image path if it exists, but let the router decide (optional enhancement)
        # For now, we stick to the explicit check to avoid confusing the router

        if isinstance(query_input, str):
             query_input = {"text": request.query, "user_role": request.user_role}
        
        # Use conversation_id from request or session_id as fallback
        thread_id = request.conversation_id if hasattr(request, "conversation_id") and request.conversation_id else session_id
        
        response_data = process_query(query_input, thread_id=thread_id)
        response_text = response_data['messages'][-1].content
        
        # Set secure session cookie
        response.set_cookie(
            key="session_id",
            value=session_id,
            max_age=COOKIE_MAX_AGE,
            httponly=True,  # Prevent XSS attacks
            samesite='strict'  # CSRF protection
            # Note: 'secure=True' should be enabled in production with HTTPS
        )
        
        audit_log("CHAT_REQUEST", session_id, f"query_length={len(request.query)}")

        # Check if the agent is skin lesion segmentation and find the image path
        result = {
            "status": "success",
            "response": response_text, 
            "agent": response_data["agent_name"],
            "thinking": f"🤔 Analyzing your request...\n📋 Selected: {response_data['agent_name'].replace('_', ' ').title()}",
            "confidence": response_data.get('retrieval_confidence', 0.95) if 'retrieval_confidence' in response_data else 0.95
        }
        
        # Add suggested follow-ups based on agent type
        agent_name = response_data["agent_name"].lower()
        if 'rag' in agent_name:
            result["suggestions"] = ["Can you explain more?", "What are the treatment options?", "Are there any side effects?"]
        elif 'web' in agent_name:
            result["suggestions"] = ["Show me latest research", "What are current guidelines?", "Any recent updates?"]
        elif 'medical' in agent_name or 'xray' in agent_name or 'lesion' in agent_name:
            result["suggestions"] = ["Is this serious?", "What should I do next?", "Should I see a doctor?"]
        else:
            result["suggestions"] = ["Tell me more", "What are my options?", "How can I prevent this?"]
        
        # If it's the skin lesion segmentation agent, check for output image
        if "SKIN_LESION" in response_data["agent_name"]:
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "overlayed_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/overlayed_plot.png"
            else:
                print("Skin Lesion Output path does not exist.")
        
        # For chest X-ray agent, include all three images
        if "CHEST_XRAY" in response_data["agent_name"]:
            print(f"🔍 Processing CHEST_XRAY response data keys: {list(response_data.keys())}")
            
            # All three images for display - ALWAYS include all available
            images = []
            
            # Original image
            if "original_image_url" in response_data and response_data["original_image_url"]:
                images.append({
                    "type": "original",
                    "url": response_data["original_image_url"],
                    "label": "Original X-ray"
                })
            elif "image_path" in response_data and response_data["image_path"]:
                 # Fallback if URL not in state but path is
                 img_path = response_data["image_path"]
                 if os.path.exists(img_path):
                     rel_path = os.path.relpath(img_path, project_root)
                     img_url = f"/{rel_path.replace(os.sep, '/')}"
                     images.append({
                        "type": "original",
                        "url": img_url,
                        "label": "Original X-ray"
                    })

            # Segmentation image
            if "segmentation_image_url" in response_data and response_data["segmentation_image_url"]:
                images.append({
                    "type": "segmentation",
                    "url": response_data["segmentation_image_url"],
                    "label": "Segmentation Overlay"
                })
            
            # Disease grounding image
            if "disease_grounding_url" in response_data and response_data["disease_grounding_url"]:
                images.append({
                    "type": "grounding",
                    "url": response_data["disease_grounding_url"],
                    "label": "Disease Grounding"
                })
            
            if images:
                result["all_images"] = images
                # For backward compatibility, set result_image to first available
                result["result_image"] = images[0]["url"]
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_image(
    response: Response,
    image: UploadFile = File(...), 
    text: str = Form(""),
    user_role: str = Form("patient"),
    conversation_id: str = Form("1"),
    session_id: Optional[str] = Cookie(None)
):
    """Process medical image uploads with optional text input."""
    # Validate file type
    if not allowed_file(image.filename):
        return JSONResponse(
            status_code=400, 
            content={
                "status": "error",
                "agent": "System",
                "response": "Unsupported file type. Allowed formats: PNG, JPG, JPEG"
            }
        )
    
    # Check file size before saving
    file_content = await image.read()
    if len(file_content) > config.api.max_image_upload_size * 1024 * 1024:  # Convert MB to bytes
        return JSONResponse(
            status_code=413, 
            content={
                "status": "error",
                "agent": "System",
                "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
            }
        )
    
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Save file securely
    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # Store image path in session for later reference
    store_session_image(session_id, file_path)
    
    try:
        query = {"text": text, "image": file_path, "user_role": user_role}
        
        # Use conversation_id from form or session_id as fallback
        thread_id = conversation_id if conversation_id and conversation_id != "1" else session_id
        
        response_data = process_query(query, thread_id=thread_id)
        response_text = response_data['messages'][-1].content
        
        # Clean response text - only remove data URLs and encoded strings, keep file paths for display
        if response_text:
            import re
            # Remove data URLs (base64 encoded images)
            response_text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{50,}', '', response_text)
            # Remove very long encoded strings (likely corrupted data)
            response_text = re.sub(r'[A-Za-z0-9+/=]{100,}', '', response_text)
            # Don't remove file paths - they might be needed for image display

        # Set secure session cookie
        response.set_cookie(
            key="session_id",
            value=session_id,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite='strict'
        )
        
        audit_log("IMAGE_UPLOAD", session_id, f"filename={image.filename}")

        # Get image URL for frontend (before deletion)
        # Convert file path to URL path
        image_url = None
        if file_path:
            # Get relative path from project root
            rel_path = os.path.relpath(file_path, project_root)
            image_url = f"/{rel_path.replace(os.sep, '/')}"

        # Check if the agent is skin lesion segmentation and find the image path
        result = {
            "status": "success",
            "response": response_text, 
            "agent": response_data["agent_name"],
            "thinking": f"🖼️ Analyzing image...\n📋 Selected: {response_data['agent_name'].replace('_', ' ').title()}",
            "confidence": 0.95
        }
        
        # Add the uploaded image URL to the response
        if image_url:
            result["uploaded_image"] = image_url
        
        # Add suggested follow-ups for image analysis
        result["suggestions"] = ["Is this serious?", "What should I do next?", "Should I see a doctor?"]
        
        # If it's the skin lesion segmentation agent, check for output image
        if "SKIN_LESION" in response_data["agent_name"]:
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "overlayed_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/overlayed_plot.png"
            else:
                print("Skin Lesion Output path does not exist.")
        
        # For chest X-ray agent, include all three images
        if "CHEST_XRAY" in response_data["agent_name"]:
            print(f"🔍 Processing CHEST_XRAY response data keys: {list(response_data.keys())}")
            
            # Original image - always use uploaded image URL
            if image_url:
                result["uploaded_image"] = image_url
                result["original_image"] = image_url
            # Also check original_image_url from state
            if "original_image_url" in response_data and response_data["original_image_url"]:
                result["original_image"] = response_data["original_image_url"]
            
            # Segmentation image - check multiple sources
            segmentation_image_url = None
            if "segmentation_image_url" in response_data and response_data["segmentation_image_url"]:
                segmentation_image_url = response_data["segmentation_image_url"]
            elif "segmentation_path" in response_data and response_data["segmentation_path"]:
                seg_path = response_data["segmentation_path"]
                if seg_path and os.path.exists(seg_path):
                    rel_path = os.path.relpath(seg_path, project_root)
                    segmentation_image_url = f"/{rel_path.replace(os.sep, '/')}"
            
            # Also check analysis_results for segmentation
            if not segmentation_image_url and "analysis_results" in response_data:
                analysis_results = response_data.get("analysis_results", {})
                segmentation = analysis_results.get("segmentation", {})
                if segmentation and not segmentation.get("error"):
                    seg_path = segmentation.get("segmentation_image_path")
                    if seg_path and os.path.exists(seg_path):
                        rel_path = os.path.relpath(seg_path, project_root)
                        segmentation_image_url = f"/{rel_path.replace(os.sep, '/')}"
            
            if segmentation_image_url:
                result["segmentation_image"] = segmentation_image_url
                print(f"✅ Segmentation image URL: {segmentation_image_url}")
            
            # Disease grounding image - check multiple sources
            disease_grounding_url = None
            if "disease_grounding_url" in response_data and response_data["disease_grounding_url"]:
                disease_grounding_url = response_data["disease_grounding_url"]
            elif "disease_grounding_path" in response_data and response_data["disease_grounding_path"]:
                grounding_path = response_data["disease_grounding_path"]
                if grounding_path and os.path.exists(grounding_path):
                    rel_path = os.path.relpath(grounding_path, project_root)
                    disease_grounding_url = f"/{rel_path.replace(os.sep, '/')}"
            
            # Also check analysis_results for disease grounding
            if not disease_grounding_url and "analysis_results" in response_data:
                analysis_results = response_data.get("analysis_results", {})
                disease_grounding = analysis_results.get("disease_grounding", {})
                if disease_grounding and not disease_grounding.get("error"):
                    grounding_path = disease_grounding.get("combined_visualization_path")
                    if grounding_path and os.path.exists(grounding_path):
                        rel_path = os.path.relpath(grounding_path, project_root)
                        disease_grounding_url = f"/{rel_path.replace(os.sep, '/')}"
            
            if disease_grounding_url:
                result["disease_grounding_image"] = disease_grounding_url
                print(f"✅ Disease grounding image URL: {disease_grounding_url}")
            
            # All three images for display - ALWAYS include all available
            images = []
            
            # Original image (always available)
            original_url = result.get("original_image") or result.get("uploaded_image")
            if original_url:
                images.append({
                    "type": "original",
                    "url": original_url,
                    "label": "Original X-ray"
                })
            
            # Segmentation image
            if result.get("segmentation_image"):
                images.append({
                    "type": "segmentation",
                    "url": result["segmentation_image"],
                    "label": "Segmentation Overlay"
                })
            
            # Disease grounding image
            if result.get("disease_grounding_image"):
                images.append({
                    "type": "grounding",
                    "url": result["disease_grounding_image"],
                    "label": "Disease Grounding"
                })
            
            print(f"📸 Total images to display: {len(images)}")
            for img in images:
                print(f"  - {img['label']}: {img['url']}")
            
            if images:
                result["all_images"] = images
                # For backward compatibility, set result_image to first available
                result["result_image"] = images[0]["url"]
            else:
                print("⚠️ No images found to display!")
        
        # Don't remove the file immediately - let it be served by static files
        # The cleanup can happen later or via a separate cleanup job
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate")
def validate_medical_output(
    response: Response,
    validation_result: str = Form(...), 
    comments: Optional[str] = Form(None),
    conversation_id: str = Form("1"),
    session_id: Optional[str] = Cookie(None)
):
    """Handle human validation for medical AI outputs."""
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        # Set secure session cookie
        response.set_cookie(
            key="session_id",
            value=session_id,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite='strict'
        )
        
        audit_log("VALIDATION_SUBMIT", session_id, f"result={validation_result}")
        
        # Re-run the agent decision system with the validation input
        validation_query = f"Validation result: {validation_result}"
        if comments:
            validation_query += f" Comments: {comments}"
        
        # Use conversation_id from form or session_id as fallback
        thread_id = conversation_id if conversation_id and conversation_id != "1" else session_id
        
        response_data = process_query(validation_query, thread_id=thread_id)

        if validation_result.lower() == 'yes':
            return {
                "status": "validated",
                "message": "**Output confirmed by human validator:**",
                "response": response_data['messages'][-1].content
            }
        else:
            return {
                "status": "rejected",
                "comments": comments,
                "message": "**Output requires further review:**",
                "response": response_data['messages'][-1].content
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Endpoint to transcribe speech using ElevenLabs API"""
    if not audio.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "No audio file selected"}
        )
    
    try:
        # Save the audio file temporarily
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.webm"
        
        # Read and save the file
        audio_content = await audio.read()
        with open(temp_audio, "wb") as f:
            f.write(audio_content)
        
        # Debug: Print file size to check if it's empty
        file_size = os.path.getsize(temp_audio)
        print(f"Received audio file size: {file_size} bytes")
        
        if file_size == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Received empty audio file"}
            )
        
        # Convert to MP3
        mp3_path = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.mp3"
        
        try:
            # Use pydub with format detection
            audio = AudioSegment.from_file(temp_audio)
            audio.export(mp3_path, format="mp3")
            
            # Debug: Print MP3 file size
            mp3_size = os.path.getsize(mp3_path)
            print(f"Converted MP3 file size: {mp3_size} bytes")

            with open(mp3_path, "rb") as mp3_file:
                audio_data = mp3_file.read()
            print(f"Converted audio file into byte array successfully!")

            transcription = client.speech_to_text.convert(
                file=audio_data,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="eng",
                diarize=True,
            )
            
            # Clean up temp files
            try:
                os.remove(temp_audio)
                os.remove(mp3_path)
                print(f"Deleted temp files: {temp_audio}, {mp3_path}")
            except Exception as e:
                print(f"Could not delete file: {e}")
            
            if transcription.text:
                return {"transcript": transcription.text}
            else:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"API error: {transcription}", "details": transcription.text}
                )

        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error processing audio: {str(e)}"}
            )
                
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
    """Endpoint to generate speech using ElevenLabs API"""
    try:
        text = request.text
        selected_voice_id = request.voice_id
        
        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Text is required"}
            )
        
        # Define API request to ElevenLabs
        elevenlabs_url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice_id}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": config.speech.eleven_labs_api_key
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        # Send request to ElevenLabs API
        response = requests.post(elevenlabs_url, headers=headers, json=payload)

        if response.status_code != 200:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to generate speech, status: {response.status_code}", "details": response.text}
            )
        
        # Save the audio file temporarily
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio_path = f"./{SPEECH_DIR}/{uuid.uuid4()}.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(response.content)

        # Return the generated audio file
        return FileResponse(
            path=temp_audio_path,
            media_type="audio/mpeg",
            filename="generated_speech.mp3"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Add exception handler for request entity too large
@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    return JSONResponse(
        status_code=413,
        content={
            "status": "error",
            "agent": "System",
            "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host=config.api.host, port=config.api.port)