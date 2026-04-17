#!/bin/bash
# Kill servers running on ports 8000 and 8001

echo "🔪 Killing servers..."

# Kill process on port 8000 (Frontend/API)
PID_8000=$(lsof -t -i:8000)
if [ -n "$PID_8000" ]; then
    echo "   - Found process on port 8000 (PID: $PID_8000). Killing..."
    kill -9 $PID_8000 2>/dev/null
else
    echo "   - No process found on port 8000."
fi

# Kill process on port 8001 (Backend)
PID_8001=$(lsof -t -i:8001)
if [ -n "$PID_8001" ]; then
    echo "   - Found process on port 8001 (PID: $PID_8001). Killing..."
    kill -9 $PID_8001 2>/dev/null
else
    echo "   - No process found on port 8001."
fi

# Kill any lingering python processes related to the app
echo "   - Cleaning up lingering Python processes..."
pkill -f "python.*start_server.py" 2>/dev/null
pkill -f "python.*app.py" 2>/dev/null

# Kill any lingering node processes related to the frontend
echo "   - Cleaning up lingering Node processes..."
pkill -f "vite" 2>/dev/null

echo "✅ All servers killed."
