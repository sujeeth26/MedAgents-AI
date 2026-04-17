#!/bin/bash
# Start both servers (Modern Version)

# Function to kill processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    ./scripts/kill_servers.sh
    exit
}

# Trap Ctrl+C (SIGINT) to run cleanup
trap cleanup SIGINT

# Ensure we are in the project root
cd "$(dirname "$0")"

# Kill anything running first
./scripts/kill_servers.sh

echo "---------------------------------------------------"
echo "🚀 Starting Multi-Agent Medical Assistant"
echo "---------------------------------------------------"

# Start Backend
echo "🔌 Starting Backend Server (Port 8001)..."
./run_server.sh &
BACKEND_PID=$!

# Wait a moment for backend to initialize
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Start Frontend
echo "💻 Starting Frontend Server (Port 8000)..."
cd aurora-ai-main
# Ensure dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi
# Run Vite dev server on port 8000
PORT=8000 npm run dev &
FRONTEND_PID=$!

echo "---------------------------------------------------"
echo "✅ System is Online!"
echo "   - Frontend: http://localhost:8000"
echo "   - Backend:  http://localhost:8001"
echo "---------------------------------------------------"
echo "📝 Logs from both servers will appear below."
echo "❌ Press Ctrl+C to stop all servers."
echo "---------------------------------------------------"

# Wait for both processes to keep the script running
wait $BACKEND_PID $FRONTEND_PID
