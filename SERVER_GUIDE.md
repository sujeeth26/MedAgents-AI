# 🚀 Server Management Guide - Modern Version

Complete guide for starting and stopping the backend and frontend servers for MedAgentica.

---

## 📋 **Quick Reference**

| Server | Port | Start Command | Kill Command |
|--------|------|---------------|--------------|
| **Backend** | 8001 | `bash run_server.sh` | `lsof -ti:8001 \| xargs kill -9` |
| **Frontend** | 8000 | `cd aurora-ai-main && npm run dev` | `lsof -ti:8000 \| xargs kill -9` |

---

## 🖥️ **Backend Server (Port 8001)**

### **Starting the Backend**

#### **Option 1: Using the Shell Script (Recommended)**
```bash
# From project root directory
bash run_server.sh
```

**What it does:**
- ✅ Activates virtual environment automatically
- ✅ Checks if port 8001 is in use and kills existing processes
- ✅ Starts the FastAPI server
- ✅ Shows server URL and health check endpoint

#### **Option 2: Using Python Directly**
```bash
# From project root directory
# Make sure virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the server
python start_server.py
```

**Server will be available at:**
- 🌐 Main Server: http://localhost:8001
- 📊 Health Check: http://localhost:8001/health
- 📚 API Docs: http://localhost:8001/docs

### **Stopping the Backend**

#### **Method 1: Using Ctrl+C (If running in terminal)**
```bash
# Simply press Ctrl+C in the terminal where the server is running
```

#### **Method 2: Kill by Port (If server is running in background)**
```bash
# Kill process on port 8001
lsof -ti:8001 | xargs kill -9

# Or on Windows (PowerShell):
# netstat -ano | findstr :8001
# taskkill /PID <PID> /F
```

#### **Method 3: Kill by Process Name**
```bash
# Kill all Python processes running app.py or uvicorn
pkill -f "python.*app.py"
pkill -f "uvicorn.*app"

# Or more specifically:
pkill -f "start_server.py"
```

#### **Method 4: Find and Kill Manually**
```bash
# Find the process
lsof -i :8001

# Kill using the PID shown
kill -9 <PID>
```

---

## 🎨 **Frontend Server (Port 8000)**

### **Starting the Frontend**

#### **Step 1: Navigate to Frontend Directory**
```bash
cd aurora-ai-main
```

#### **Step 2: Install Dependencies (First Time Only)**
```bash
npm install
```

#### **Step 3: Start Development Server**
```bash
npm run dev
```

**Server will be available at:**
- 🌐 Frontend: http://localhost:8000
- 🔄 Hot Reload: Enabled (auto-refreshes on code changes)

### **Stopping the Frontend**

#### **Method 1: Using Ctrl+C (If running in terminal)**
```bash
# Simply press Ctrl+C in the terminal where the server is running
```

#### **Method 2: Kill by Port**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or on Windows (PowerShell):
# netstat -ano | findstr :8000
# taskkill /PID <PID> /F
```

#### **Method 3: Kill by Process Name**
```bash
# Kill all Node/Vite processes
pkill -f "vite"
pkill -f "node.*vite"

# Or more specifically:
pkill -f "npm run dev"
```

#### **Method 4: Find and Kill Manually**
```bash
# Find the process
lsof -i :8000

# Kill using the PID shown
kill -9 <PID>
```

---

## 🔄 **Starting Both Servers**

### **Using Two Terminal Windows**

#### **Terminal 1: Backend**
```bash
cd /path/to/Multi-Agent-Medical-Assistant
bash run_server.sh
```

#### **Terminal 2: Frontend**
```bash
cd /path/to/Multi-Agent-Medical-Assistant/aurora-ai-main
npm run dev
```

### **Using Background Processes**

#### **Start Backend in Background**
```bash
# Start backend in background
bash run_server.sh > backend.log 2>&1 &

# Or
nohup python start_server.py > backend.log 2>&1 &
```

#### **Start Frontend in Background**
```bash
# Start frontend in background
cd aurora-ai-main
npm run dev > ../frontend.log 2>&1 &
```

#### **Check Background Processes**
```bash
# Check if servers are running
lsof -i :8001  # Backend
lsof -i :8000  # Frontend

# Or check all processes
ps aux | grep -E "(uvicorn|vite|node)"
```

---

## 🛑 **Killing Both Servers**

### **Quick Kill Script**
```bash
# Kill both servers at once
lsof -ti:8001 | xargs kill -9 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null
echo "✅ Both servers stopped"
```

### **Create a Kill Script**

Create a file `kill_servers.sh`:
```bash
#!/bin/bash
echo "🛑 Stopping MedAgentica servers..."

# Kill backend (port 8001)
if lsof -ti:8001 >/dev/null 2>&1; then
    lsof -ti:8001 | xargs kill -9
    echo "✅ Backend stopped (port 8001)"
else
    echo "ℹ️  Backend not running"
fi

# Kill frontend (port 8000)
if lsof -ti:8000 >/dev/null 2>&1; then
    lsof -ti:8000 | xargs kill -9
    echo "✅ Frontend stopped (port 8000)"
else
    echo "ℹ️  Frontend not running"
fi

echo "🎉 Done!"
```

Make it executable:
```bash
chmod +x kill_servers.sh
```

Run it:
```bash
./kill_servers.sh
```

---

## 🔍 **Checking Server Status**

### **Check if Servers are Running**
```bash
# Check backend (port 8001)
lsof -i :8001

# Check frontend (port 8000)
lsof -i :8000

# Check both
lsof -i :8001 -i :8000
```

### **Test Server Health**
```bash
# Test backend health
curl http://localhost:8001/health

# Test frontend (should return HTML)
curl http://localhost:8000
```

### **Check Server Logs**
```bash
# Backend logs (if using background process)
tail -f backend.log

# Frontend logs (if using background process)
tail -f frontend.log
```

---

## 🐛 **Troubleshooting**

### **Port Already in Use**

#### **Backend (Port 8001)**
```bash
# Kill existing process
lsof -ti:8001 | xargs kill -9

# Or use the run_server.sh script (it handles this automatically)
bash run_server.sh
```

#### **Frontend (Port 8000)**
```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9

# Then start again
cd aurora-ai-main && npm run dev
```

### **Permission Denied**

```bash
# Make scripts executable
chmod +x run_server.sh
chmod +x kill_servers.sh
```

### **Virtual Environment Not Activated**

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Then start backend
python start_server.py
```

### **Node Modules Not Installed**

```bash
# Install frontend dependencies
cd aurora-ai-main
npm install
```

### **Server Won't Start**

```bash
# Check if ports are available
lsof -i :8001
lsof -i :8000

# Kill all related processes
pkill -f "uvicorn"
pkill -f "vite"
pkill -f "node.*dev"

# Try starting again
```

---

## 📝 **Quick Commands Cheat Sheet**

```bash
# ============================================
# START SERVERS
# ============================================

# Backend
bash run_server.sh

# Frontend
cd aurora-ai-main && npm run dev

# ============================================
# STOP SERVERS
# ============================================

# Kill backend
lsof -ti:8001 | xargs kill -9

# Kill frontend
lsof -ti:8000 | xargs kill -9

# Kill both
lsof -ti:8001 | xargs kill -9 && lsof -ti:8000 | xargs kill -9

# ============================================
# CHECK STATUS
# ============================================

# Check if running
lsof -i :8001  # Backend
lsof -i :8000  # Frontend

# Test health
curl http://localhost:8001/health
curl http://localhost:8000
```

---

## 🎯 **Recommended Workflow**

1. **Start Backend First:**
   ```bash
   bash run_server.sh
   ```

2. **In a New Terminal, Start Frontend:**
   ```bash
   cd aurora-ai-main
   npm run dev
   ```

3. **Access Application:**
   - Frontend: http://localhost:8000
   - Backend API: http://localhost:8001
   - API Docs: http://localhost:8001/docs

4. **When Done, Stop Both:**
   ```bash
   # Press Ctrl+C in each terminal, OR
   lsof -ti:8001 | xargs kill -9 && lsof -ti:8000 | xargs kill -9
   ```

---

## 💡 **Pro Tips**

1. **Use Terminal Tabs**: Open two tabs in your terminal - one for backend, one for frontend
2. **Use Background Processes**: Start servers in background if you need the terminal for other tasks
3. **Create Aliases**: Add these to your `~/.bashrc` or `~/.zshrc`:
   ```bash
   alias medagentica-backend="cd /path/to/Multi-Agent-Medical-Assistant && bash run_server.sh"
   alias medagentica-frontend="cd /path/to/Multi-Agent-Medical-Assistant/aurora-ai-main && npm run dev"
   alias medagentica-kill="lsof -ti:8001 | xargs kill -9 && lsof -ti:8000 | xargs kill -9"
   ```
4. **Use Process Managers**: Consider using `pm2` or `supervisor` for production-like management

---

**Need Help?** Check the main [README.md](README.md) or open an issue on GitHub.


