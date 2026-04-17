#!/usr/bin/env python3
"""
Simple Python launcher for Neo-Aurora MedAgentica
This ensures the correct Python path is set before starting the server.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = str(project_root) + os.pathsep + os.environ.get('PYTHONPATH', '')

print("🌌 " + "="*58)
print("   Neo-Aurora MedAgentica")
print("   AI Medical Assistant")
print("="*58 + " 🌌\n")

# Check if .env exists
env_file = project_root / '.env'
if not env_file.exists():
    print("⚠️  Warning: .env file not found!")
    print("📝 Please create a .env file with your API keys")
    print("   See demo_env_template.txt for reference\n")

print("🚀 Starting Neo-Aurora Medical Assistant...")
print("🌐 Opening at: http://localhost:8000\n")
print("✨ Features:")
print("   • Multi-Agent System")
print("   • Real-time Image Analysis")
print("   • RAG-Powered Knowledge")
print("   • Beautiful Aurora Theme\n")
print("Press Ctrl+C to stop the server\n")

# Change to web directory
web_dir = project_root / 'web'
os.chdir(web_dir)

# Start the application
try:
    subprocess.run([sys.executable, 'app.py'])
except KeyboardInterrupt:
    print("\n\n✨ Neo-Aurora Medical Assistant stopped. Thank you! 🌌")
    sys.exit(0)


