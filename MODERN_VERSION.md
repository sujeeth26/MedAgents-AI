# MedAgentica - Modern Version Quick Start

This is the **modern, actively maintained** version of MedAgentica with a premium UI and multi-agent system.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Virtual environment activated

### 1. Start Backend (Port 8001)
```bash
# From project root
bash run_server.sh
```

### 2. Start Frontend (Port 8000)
```bash
cd aurora-ai-main
npm install  # First time only
npm run dev
```

### 3. Access Application
Open your browser to: **http://localhost:8000**

## 📁 Modern Architecture

```
Multi-Agent-Medical-Assistant/
├── aurora-ai-main/          # Modern React Frontend (Premium UI)
├── web/                     # FastAPI Backend
│   └── app.py              # Main server
├── agents/                  # Multi-Agent System
│   ├── agent_decision.py   # Agent router
│   ├── image_analysis_agent/
│   ├── rag_agent/
│   └── web_search_processor_agent/
├── start_server.py         # Backend entry point
├── run_server.sh           # Backend launcher
├── config.py               # Configuration
└── legacy/                 # ⚠️ Archived old code (DO NOT USE)
```

## 🎨 Features

### Premium UI
- **Living Background**: Interactive particle system
- **Medical Glassmorphism**: Deep blur effects and mesh gradients
- **Agent Theming**: Distinct visual identity for each AI agent
- **Magnetic Interactions**: Fluid animations and hover effects

### Multi-Agent System
- **Conversation Agent**: General health discussions
- **RAG Agent**: Medical knowledge queries
- **Web Search Agent**: Latest medical research
- **Brain Tumor Agent**: MRI analysis
- **Chest X-ray Agent**: COVID-19 detection with MedRAX
- **Skin Lesion Agent**: Skin condition analysis

### Chat Features
- **Persistent History**: localStorage-based chat management
- **Rename/Delete**: Full chat management
- **Stop Generation**: Cancel ongoing AI responses
- **Image Upload**: Medical image analysis

## 🔧 Configuration

Edit `.env` file for API keys:
```bash
# Required
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key

# Optional
OPENROUTER_API_KEY=your_openrouter_key
ELEVEN_LABS_API_KEY=your_elevenlabs_key
```

## 📚 Documentation

- **Full README**: See main `README.md`
- **Legacy Code**: See `legacy/README_LEGACY.md`
- **Changelog**: See `CHANGELOG.md`

## ⚠️ Important Notes

1. **Port 8001**: Backend (FastAPI)
2. **Port 8000**: Frontend (React/Next.js)
3. **Legacy Code**: The `legacy/` directory contains old, deprecated code. Do not use it.

## 🐛 Troubleshooting

### Backend won't start
```bash
# Kill processes on port 8001
lsof -ti:8001 | xargs kill -9
bash run_server.sh
```

### Frontend won't start
```bash
# Kill processes on port 8000
lsof -ti:8000 | xargs kill -9
cd aurora-ai-main && npm run dev
```

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt
cd aurora-ai-main && npm install
```

## 📞 Support

For issues or questions, refer to the main README.md or check the documentation in `docs/`.
