# Legacy Code Archive

This directory contains **deprecated/legacy** code from earlier versions of MedAgentica. These files are kept for reference but are **not actively maintained**.

## ⚠️ Important Notice

**DO NOT USE THIS CODE FOR PRODUCTION OR DEVELOPMENT**

The modern, actively maintained version is in the parent directory:
- **Modern Frontend**: `../aurora-ai-main/`
- **Modern Backend**: `../web/app.py`
- **Modern Entry Point**: `../start_server.py` and `../run_server.sh`

## Contents

### Old Frontend
- **Nextjs React/** - Original Next.js implementation (replaced by aurora-ai-main)

### Demo Scripts (Standalone)
- **demo_agentic_rag.py** - Standalone RAG system demo (now integrated into agents/)
- **interactive_demo.py** - CLI-based interactive demo
- **demo_ingest_pinecone.py** - Data ingestion script

### Old Launchers
- **launch.py** - Old Python launcher (replaced by start_server.py)
- **launch_neo_aurora.sh** - Old shell launcher (replaced by run_server.sh)

## Why These Are Legacy

1. **Nextjs React/** - Replaced by the premium `aurora-ai-main/` UI with glassmorphism and modern design
2. **Demo scripts** - Standalone demos that are now integrated into the main agent system
3. **Old launchers** - Replaced by simplified `start_server.py` and `run_server.sh`

## Migration Guide

If you need functionality from legacy code:

| Legacy File | Modern Equivalent |
|------------|-------------------|
| `Nextjs React/` | `aurora-ai-main/` |
| `demo_agentic_rag.py` | `agents/` (RAG agent) |
| `launch.py` | `start_server.py` |
| `launch_neo_aurora.sh` | `run_server.sh` |

## Questions?

See the main `README.md` in the parent directory for current documentation.
