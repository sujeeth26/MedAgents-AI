#!/bin/bash
# Startup script for the Multi-Agent Medical Assistant
# This script ensures the server starts properly on localhost

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Virtual environment not found. Please create it first:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check if port 8001 is already in use
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 8001 is already in use. Killing existing process..."
    pkill -f "python.*app.py"
    pkill -f "uvicorn.*app"
    sleep 2
fi

# Start the server
echo ""
echo "🚀 Starting Multi-Agent Medical Assistant..."
echo "📍 Server will be available at: http://localhost:8001"
echo "📊 Health Check: http://localhost:8001/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "="*60
echo ""

python scripts/start_server.py













