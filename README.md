# MedAI — Medical Report AI Agent

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/)

MedAI is an end-to-end AI agent that reads medical lab reports and explains them in plain English. It combines a custom BiomedBERT architecture with FAISS retrieval and a Gemini-powered ReAct agent.

## Architecture
Report (PDF/Text) → BiomedBERT Encoder → [NER Head + Risk Head] → FAISS RAG → Gemini Agent → Patient-Friendly Response

## Tech Stack

| Component | Technology |
|-----------|------------|
| Encoder | BiomedBERT (PubMedBERT) |
| NER Head | Custom token classifier (768→256→11) |
| Risk Head | Custom sequence classifier (768→128→3) |
| RAG | FAISS + sentence-transformers |
| Agent | LangChain ReAct + Gemini 2.5 Flash Lite |
| Backend | FastAPI |
| Frontend | HTML/CSS/JS |
| Deploy | Docker + Google Cloud Run |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /analyze | Analyze text report |
| POST | /analyze-pdf | Analyze uploaded PDF |
| POST | /chat | Chat with medical agent |
| POST | /build-index | Rebuild knowledge base index |
| GET | /health | Health check |

## Disclaimer

This is an educational project. It is **not** a substitute for professional medical advice, diagnosis, or treatment.

## Setup

```bash
pip install -r requirements.txt


Run Locally
uvicorn src.api:app --reload
License

MIT
