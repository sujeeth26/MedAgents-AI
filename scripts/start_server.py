#!/usr/bin/env python3
"""
Startup script for the Multi-Agent Medical Assistant
Runs the FastAPI application from the root directory
"""

import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and run the app
from web.app import app
import uvicorn
from config import Config

if __name__ == "__main__":
    config = Config()
    print(f"🚀 Starting Multi-Agent Medical Assistant...")
    print(f"📍 Server: http://{config.api.host}:{config.api.port}")
    print(f"✅ MedRAX Integration: Ready")
    print(f"📊 Health Check: http://{config.api.host}:{config.api.port}/health")
    print("\n" + "="*60)
    uvicorn.run(app, host=config.api.host, port=config.api.port)














