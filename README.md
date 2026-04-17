<div align="center">

# 🏥 MedAgentica

### **🤖 Next-Generation Multi-Agent Medical Assistant**

**Revolutionary AI-Powered Healthcare System with Agentic RAG, Multi-Modal Image Analysis & Real-Time Research**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/Vasu2604/MedAgentica?style=for-the-badge&logo=github&color=yellow)](https://github.com/Vasu2604/MedAgentica)
[![GitHub Forks](https://img.shields.io/github/forks/Vasu2604/MedAgentica?style=for-the-badge&logo=github&color=blue)](https://github.com/Vasu2604/MedAgentica)

![MedAgentica Logo](assets/logo_rounded.png)

**[📖 Documentation](#-documentation) • [🚀 Quick Start](#-quick-start) • [🎯 Features](#-key-features) • [🏗️ Architecture](#️-system-architecture) • [🤝 Contributing](#-contributing)**


## 🎥 **Demonstration**



https://github.com/user-attachments/assets/88e3868c-9234-4cda-9954-116194d63c86



---

</div>

## 🌟 **What is MedAgentica?**

**MedAgentica** is a cutting-edge, production-ready **Multi-Agent Medical Assistant** that revolutionizes healthcare AI by combining:

- 🧠 **Agentic RAG (Retrieval-Augmented Generation)** - Self-correcting, intelligent document retrieval
- 🖼️ **Multi-Modal Image Analysis** - Brain MRI, Chest X-ray, and Skin Lesion detection
- 🌐 **Real-Time Web Search** - Latest medical research from PubMed and trusted sources
- 🎨 **Premium React UI** - Beautiful Neo-Aurora themed interface with glassmorphism design
- ⚡ **Production-Grade** - Comprehensive evaluation, monitoring, and error handling

### 🎯 **Why MedAgentica Stands Out**

| Feature | Traditional Systems | **MedAgentica** |
|---------|-------------------|-----------------|
| **RAG System** | Single-pass retrieval | ✅ **4-Agent Self-Correcting Workflow** |
| **Image Analysis** | Single model | ✅ **3 Specialized Medical Vision Agents** |
| **Response Quality** | Fixed retrieval | ✅ **Reflection & Re-retrieval** |
| **UI/UX** | Basic interfaces | ✅ **Premium Aurora Theme with Animations** |
| **Privacy** | Cloud-only | ✅ **Local Image Processing** |
| **Evaluation** | Manual testing | ✅ **15+ Automated Metrics** |

---

## ✨ **Key Features**

### 🤖 **Intelligent Agentic RAG System**

Our revolutionary **4-Agent Workflow** ensures accurate, grounded medical responses:

```
┌─────────────────┐
│ 1. Query        │ → Analyzes intent, extracts medical terms
│    Analysis     │
└────────┬────────┘
         │
┌────────▼────────┐
│ 2. Retrieval    │ → Searches vector database (Pinecone/ChromaDB)
└────────┬────────┘
         │
┌────────▼────────┐
│ 3. Reflection   │ → Evaluates quality, decides re-retrieval
└────────┬────────┘
         │
    ┌────▼────┐
    │ Good?  │ → YES → Continue | NO → Re-retrieve
    └────┬────┘
         │
┌────────▼────────┐
│ 4. Response     │ → Generates answer with Chain-of-Thought
│    Synthesis    │
└─────────────────┘
```

**Key Advantages:**
- ✅ **Self-Correcting**: Automatically improves retrieval if quality is low
- ✅ **Context-Aware**: Understands medical terminology and query intent
- ✅ **Hallucination Prevention**: Grounded responses with source citations
- ✅ **Adaptive**: Dynamically adjusts document count (3-10) based on complexity

### 🩺 **Advanced Medical Image Analysis**

#### 🧠 **Brain Tumor Detection Agent**
- **5-Class Classification**: No Tumor, Pituitary, Glioma, Meningioma, Other
- **MRI Segmentation**: Precise tumor localization and visualization
- **Follow-up Support**: Contextual answers to questions about analysis
- **Model**: Local MedGemma (privacy-preserving)

#### 🫁 **Chest X-Ray Analysis Agent (MedRAX)**
- **18+ Disease Detection**: COVID-19, Pneumonia, Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumothorax, and more
- **Anatomical Segmentation**: 15+ structures (lungs, heart, spine, ribs, etc.)
- **Disease Grounding**: Visual localization with bounding boxes
- **Professional Reports**: Radiology-style Findings and Impression sections
- **Triple Output**: Original X-ray, Segmentation Overlay, Disease Grounding

#### 🩺 **Skin Lesion Classification Agent**
- **Binary Classification**: Benign vs Malignant
- **EfficientNet-B0 Model**: State-of-the-art deep learning architecture
- **ABCDE Criteria**: Educational explanations about skin lesion characteristics
- **Professional Responses**: Doctor-style empathetic explanations

### 🌐 **Neo-Aurora Premium UI**

**Modern React Frontend** with stunning visual design:

| Feature | Description |
|---------|-------------|
| 🌌 **Animated Background** | 3 floating orbs with dynamic grid system |
| 🎨 **Glassmorphism** | Frosted glass panels with layered depth effects |
| 📊 **Live Dashboard** | Real-time KPIs (Active Agents, Queries, Response Time, Success Rate) |
| 🏷️ **Agent Badges** | Color-coded status indicators for each agent |
| 📝 **Markdown Support** | Full markdown rendering with syntax highlighting |
| 🖼️ **Image Handling** | Drag & drop upload, preview, and full-screen modal |
| 📱 **Responsive** | Mobile-first design with breakpoints |
| ⌨️ **Keyboard Shortcuts** | Enhanced accessibility features |

### 🔍 **7 Specialized AI Agents**

| Agent | Purpose | When to Use | Technology |
|-------|---------|-------------|------------|
| 💬 **Conversation** | General health discussions | Greetings, casual chat | Groq API (450 tok/s) |
| 📚 **RAG** | Medical knowledge queries | Specific medical questions | Agentic RAG + Pinecone |
| 🌐 **Web Search** | Latest medical research | Recent developments, news | OpenRouter + PubMed |
| 🧠 **Brain Tumor** | MRI analysis | Upload brain MRI images | Local MedGemma |
| 🫁 **Chest X-Ray** | X-ray analysis | Upload chest X-ray images | MedRAX + MedGemma |
| 🩺 **Skin Lesion** | Skin condition analysis | Upload skin lesion images | EfficientNet-B0 |
| 🚨 **Emergency** | Critical emergencies | Chest pain, stroke, severe bleeding | Groq API |

---

## 🚀 **Quick Start**

### **Prerequisites**

```bash
# Required
- Python 3.9+
- Node.js 18+
- npm or yarn
```

### **Installation (3 Steps)**

#### **Step 1: Clone & Setup**

```bash
# Clone the repository
git clone https://github.com/Vasu2604/MedAgentica.git
cd MedAgentica

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

#### **Step 2: Configure API Keys**

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your favorite editor
```

**Required API Keys:**
```env
# Vector Database (Required)
PINECONE_API_KEY=pcsk_your_key_here
PINECONE_INDEX_NAME=medagentica-demo-384

# LLM Provider (Choose one)
GROQ_API_KEY=gsk_your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# OR
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=deepseek/deepseek-chat-v3.1:free

# Optional Services
TAVILY_API_KEY=your_tavily_key      # Web search
ELEVEN_LABS_API_KEY=your_key         # Text-to-speech
```

**🔑 Get API Keys:**
- **Pinecone**: [Sign up here](https://app.pinecone.io/) (Free tier: 100k vectors)
- **Groq**: [Sign up here](https://console.groq.com/) (Free tier: 100k tokens/day)
- **OpenRouter**: [Sign up here](https://openrouter.ai/) (Pay-per-use)

#### **Step 3: Ingest Medical Documents (First Time Only)**

```bash
# Ingest documents into vector database
python legacy/demo_ingest_pinecone.py
```

### **Running the Application**

#### **Option 1: Modern Full-Stack Application (Recommended)**

```bash
# Terminal 1: Start Backend Server
bash run_server.sh
# Server runs on http://localhost:8001

# Terminal 2: Start React Frontend
cd aurora-ai-main
npm install  # First time only
npm run dev
# Frontend runs on http://localhost:8000
```

**🌐 Access**: Open [http://localhost:8000](http://localhost:8000) in your browser

#### **Option 2: Legacy Standalone RAG System**

```bash
# Run the standalone Agentic RAG system (CLI-based, no React frontend)
python legacy/demo_agentic_rag.py
```

> **Note**: The legacy version is located in the `legacy/` directory. See `legacy/README_LEGACY.md` for details.

---

## 📦 **Project Versions**

This project contains **two versions**:

### **1. 🆕 Modern Version** (Recommended)
- **Location**: Root directory (`aurora-ai-main/`, `web/app.py`)
- **Type**: Full-stack web application
- **Frontend**: React + TypeScript + Vite
- **Backend**: FastAPI (Python)
- **Features**: 
  - ✅ Premium Neo-Aurora UI
  - ✅ All 7 specialized agents
  - ✅ Real-time image analysis
  - ✅ Live KPI dashboard
  - ✅ Markdown rendering
  - ✅ Responsive design

**Quick Start:**
```bash
bash run_server.sh          # Backend
cd aurora-ai-main && npm run dev  # Frontend
```

### **2. 📜 Legacy Version** (Archived)
- **Location**: `legacy/` directory
- **Type**: Standalone Python scripts
- **Frontend**: None (CLI-based)
- **Use Case**: 
  - Testing RAG functionality
  - Understanding core architecture
  - Integration into other applications
  - Educational purposes

**Files:**
- `legacy/demo_agentic_rag.py` - Standalone Agentic RAG system
- `legacy/demo_ingest_pinecone.py` - Data ingestion script
- `legacy/interactive_demo.py` - Interactive CLI demo

> **⚠️ Important**: The legacy version is archived. For new development, use the modern version.

---

## 🏗️ **System Architecture**

### **High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                            │
│         (React Frontend / CLI / API)                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              AGENT DECISION ROUTER                          │
│         (Intelligent Query Routing)                        │
└───┬──────┬──────┬──────┬──────┬──────┬──────┬──────────────┘
    │      │      │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼      ▼      ▼
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│Conv │ │ RAG │ │ Web │ │Brain│ │Chest│ │Skin │ │Emerg│
│Agent│ │Agent│ │Srch │ │Tumor│ │Xray │ │Lesn │ │Resp │
└─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘
```

### **Agentic RAG Workflow**

```
User Query
    ↓
┌─────────────────────┐
│ 1. Query Analysis   │ ← Analyzes intent, extracts terms
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 2. Retrieval        │ ← Searches Pinecone vector DB
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 3. Reflection       │ ← Evaluates quality
└──────────┬──────────┘
           ↓
    ┌──────────────┐
    │ Sufficient?  │
    └──┬───────┬───┘
   YES │       │ NO
       ↓       ↓
    ┌────┐  ┌──────────────┐
    │    │  │ Re-retrieve  │
    │    │  │ (refined)    │
    └─┬──┘  └──────┬───────┘
      │            │
      ↓            ↓
┌─────────────────────┐
│ 4. Response         │ ← Generates answer with CoT
│    Synthesis        │
└──────────┬──────────┘
           ↓
    Final Response
```

**Visual Flowchart**: See [`assets/final-medical-assistant-flowchart-code.mermaid`](assets/final-medical-assistant-flowchart-code.mermaid)

---

## 📊 **Evaluation Framework**

### **Quick Evaluation (3 minutes)**

```bash
python evaluation/quick_evaluate.py
```

**Metrics:**
- ✅ Accuracy (BLEU, ROUGE, Semantic Similarity)
- ✅ RAG Quality (Faithfulness, Hallucination)
- ✅ Performance (Latency, Throughput)

### **Comprehensive Evaluation (10 minutes)**

```bash
python evaluation/evaluate_rag_llm.py
```

**Outputs:**
- 📄 `evaluation_results/evaluation_report_TIMESTAMP.html` - Interactive report
- 📊 `evaluation_results/evaluation_visualizations_TIMESTAMP.png` - Charts
- 📋 `evaluation_results/evaluation_results_TIMESTAMP.json` - Raw data

### **Success Criteria**

| Metric | Target | Good | Acceptable |
|--------|--------|------|------------|
| **BLEU Score** | > 0.7 | 0.5-0.7 | 0.3-0.5 |
| **Hallucination** | < 0.2 | 0.2-0.3 | 0.3-0.5 |
| **Faithfulness** | > 0.8 | 0.6-0.8 | 0.4-0.6 |
| **Latency** | < 2000ms | < 1500ms | 1500-3000ms |

---

## 📁 **Project Structure**

```
MedAgentica/
│
├── 🚀 MAIN ENTRY POINTS
│   ├── run_server.sh                # 🚀 Backend launcher script
│   ├── restart_servers.sh           # 🔄 Restart both servers
│   └── web/app.py                   # 🌐 FastAPI backend
│
├── 🎨 FRONTEND (Modern Version)
│   └── aurora-ai-main/              # React + TypeScript frontend
│       ├── src/
│       │   ├── pages/Chat.tsx       # Main chat interface
│       │   └── components/          # UI components
│       └── package.json
│
├── 🧠 AGENTS (Core System)
│   ├── agent_decision.py            # Main routing agent
│   ├── rag_agent/                   # RAG System
│   ├── image_analysis_agent/        # Medical Image Analysis
│   │   ├── brain_tumor_agent/
│   │   ├── chest_xray_agent/
│   │   └── skin_lesion_agent/
│   ├── web_search_processor_agent/  # Web Search
│   └── guardrails/                   # Safety & Validation
│
├── 📂 SCRIPTS & UTILS
│   ├── scripts/                     # Helper scripts
│   │   ├── start_server.py          # Backend starter
│   │   ├── kill_servers.sh          # Cleanup script
│   │   └── ingest_rag_data.py       # Data ingestion
│   └── utils/                       # Utility functions
│
├── 📊 EVALUATION
│   ├── evaluation/
│   │   ├── quick_evaluate.py
│   │   └── evaluate_rag_llm.py
│   └── evaluation_results/          # Generated reports (gitignored)
│
├── 🔧 CONFIGURATION
│   ├── config.py                     # System configuration
│   ├── .env.example                  # Template (copy to .env)
│   └── requirements.txt              # Python dependencies
│
├── 📚 DOCUMENTATION
│   ├── README.md                     # This file
│   ├── CHANGELOG.md                  # Version history
│   ├── MODERN_VERSION.md             # Modern version guide
│   └── docs/                         # Additional docs
│
├── 📜 LEGACY (Archived)
│   └── legacy/                       # Old standalone scripts & templates
│       ├── demo_agentic_rag.py
│       ├── web_templates/            # Old HTML templates
│       └── README_LEGACY.md
│
├── 🗄️ DATA & LOGS
│   ├── data/                         # Vector DB & raw data
│   ├── logs/                         # Server logs
│   └── uploads/                      # User uploads
│
└── 🎨 ASSETS
    ├── Demonstration/                # Demo videos
    ├── logo_rounded.png
    └── *.mermaid                     # Flowchart source
```

---

## ⚙️ **Configuration**

### **Environment Variables**

Create a `.env` file in the root directory:

```env
# Vector Database (Required)
PINECONE_API_KEY=pcsk_your_pinecone_key_here
PINECONE_INDEX_NAME=medagentica-demo-384

# LLM Provider (Choose one or more)
GROQ_API_KEY=gsk_your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# OR
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=deepseek/deepseek-chat-v3.1:free

# Optional Services
TAVILY_API_KEY=your_tavily_key      # Web search
ELEVEN_LABS_API_KEY=your_key         # Text-to-speech

# Local LLM (Optional - for image analysis)
USE_OLLAMA=true
OLLAMA_MODEL=alibayram/medgemma:4b
```

### **Configuration File**

Main configuration: `config.py`

**Key Settings:**
- LLM provider selection (Groq/OpenRouter/OpenAI/Ollama)
- Vector database selection (Pinecone/ChromaDB/Qdrant)
- Agent temperature settings
- Retrieval parameters
- Guardrails configuration

---

## 🔒 **Security & Best Practices**

### ⚠️ **Never Commit:**

- `.env` - Your API keys!
- `data/` - Large datasets
- `uploads/` - User uploads
- `evaluation_results/` - Generated outputs
- `venv/` - Python environment
- `*.pth`, `*.h5` - Model weights

### ✅ **Safe to Commit:**

- `.env.example` - Template (no secrets)
- All `.py` and `.md` files
- `requirements.txt`
- `.gitignore`

### **Security Features**

- ✅ **Environment Variables**: All secrets in `.env` (gitignored)
- ✅ **Input Validation**: Guardrails for user inputs
- ✅ **Output Filtering**: Content moderation
- ✅ **Session Management**: Secure session handling
- ✅ **CORS Configuration**: Proper cross-origin settings
- ✅ **Local Image Processing**: Medical images processed locally

---

## 🧪 **Testing & Validation**

### **Quick Test**

```bash
# Test backend health
curl http://localhost:8001/health

# Test chat endpoint
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello"}'
```

### **Full Evaluation**

```bash
# Comprehensive evaluation
python evaluation/evaluate_rag_llm.py

# View results
open evaluation_results/evaluation_report_*.html
```

### **Unit Tests**

```bash
# Run unit tests
python -m pytest tests/unit/

# Run integration tests
python -m pytest tests/integration/
```

---

## 📚 **Documentation**

### **Main Documentation**

- **README.md** (this file) - Complete project overview
- **CHANGELOG.md** - Version history and updates
- **MODERN_VERSION.md** - Modern version quick start guide
- **legacy/README_LEGACY.md** - Legacy version documentation

### **Additional Resources**

- **Architecture**: See `docs/ARCHITECTURE.txt` for detailed system design
- **Flowcharts**: Visual diagrams in `assets/` directory
- **Code Comments**: Comprehensive inline documentation

### **Getting Help**

1. Check this README first
2. Review `CHANGELOG.md` for recent changes
3. Check server logs: `tail -f server.log`
4. Open a GitHub issue for bugs or questions

---

## 🤝 **Contributing**

We welcome contributions! Here's how:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Test thoroughly**: Run `python evaluation/quick_evaluate.py`
5. **Commit**: `git commit -m 'Add amazing feature'`
6. **Push**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### **Contribution Guidelines**

- ✅ Follow existing code style
- ✅ Add tests for new features
- ✅ Update documentation
- ✅ Never commit `.env` or secrets
- ✅ Run evaluation before submitting

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎉 **Credits**

Built with:

- 🦙 **LangChain** - RAG framework and agent orchestration
- 📌 **Pinecone** - Vector database for semantic search
- ⚡ **Groq** - Fast LLM inference
- 🤗 **HuggingFace** - Embeddings and models
- 🎨 **FastAPI** - Modern web framework
- ⚛️ **React** - Frontend framework
- 🎭 **MedRAX** - Chest X-ray analysis
- 🧠 **MedGemma** - Medical LLM for image analysis

---

## 🚀 **Quick Commands Reference**

```bash
# Modern Full-Stack Application
bash run_server.sh                    # Backend
cd aurora-ai-main && npm run dev     # Frontend

# Legacy Standalone RAG System
python legacy/demo_agentic_rag.py

# Data Ingestion
python legacy/demo_ingest_pinecone.py

# Evaluation
python evaluation/quick_evaluate.py
python evaluation/evaluate_rag_llm.py

# Health Check
curl http://localhost:8001/health
```

---

<div align="center">

## ⭐ **Star This Repo**

If you find this project useful, please consider giving it a star! ⭐

**🌟 Made with ❤️ for the Medical AI Community 🌟**

[Documentation](#-documentation) • [Quick Start](#-quick-start) • [Features](#-key-features) • [Contributing](#-contributing)

**🔗 [GitHub Repository](https://github.com/Vasu2604/MedAgentica) • [Issues](https://github.com/Vasu2604/MedAgentica/issues) • [Discussions](https://github.com/Vasu2604/MedAgentica/discussions)**

---

**Built with cutting-edge AI technology to revolutionize healthcare** 🏥🤖

</div>
