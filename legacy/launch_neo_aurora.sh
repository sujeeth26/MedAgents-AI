#!/bin/bash

# 🌌 Neo-Aurora MedAgentica Launcher
# Quick start script for the beautiful medical assistant interface

echo "🌌 =================================="
echo "   Neo-Aurora MedAgentica"
echo "   AI Medical Assistant"
echo "================================== 🌌"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found!"
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "venv/.requirements_installed" ]; then
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
    touch venv/.requirements_installed
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "📝 Please create a .env file with your API keys"
    echo "   See demo_env_template.txt for reference"
    echo ""
fi

# Set Python path to include parent directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Navigate to web directory
cd web

echo ""
echo "🚀 Starting Neo-Aurora Medical Assistant..."
echo "🌐 Opening at: http://localhost:8000"
echo ""
echo "✨ Features:"
echo "   • Multi-Agent System"
echo "   • Real-time Image Analysis"
echo "   • RAG-Powered Knowledge"
echo "   • Beautiful Aurora Theme"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the application
python app.py

