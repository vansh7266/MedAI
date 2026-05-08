"""FastAPI backend for MedAI report analysis and chat."""

import os
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

from src.agent import get_agent, run_agent_query
from src.model import MedicalReportModel, get_model_and_tokenizer
from src.rag_pipeline import MedicalRAG, build_and_save_kb


MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = timedelta(minutes=1)
RATE_LIMIT_STORE: Dict[str, List[datetime]] = {}


class AnalyzeRequest(BaseModel):
    """Request body for analyzing pasted report text."""

    text: str = Field(..., min_length=1, max_length=5000)


class AnalyzeResponse(BaseModel):
    """Structured response returned by report analysis endpoints."""

    entities: List[Dict]
    risk_level: str
    risk_probs: Dict
    explanation: str


class ChatRequest(BaseModel):
    """Request body for the MedAI chat endpoint."""

    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response body for agent chat interactions."""

    response: str
    tools_used: List[str]
    session_id: str


class HealthResponse(BaseModel):
    """Health status for deployment and frontend connectivity checks."""

    status: str
    model_loaded: bool
    index_loaded: bool
    version: str = "1.0.0"


class ModelManager:
    """Singleton-style cache for the model, tokenizer, RAG index, and agent."""

    def __init__(self) -> None:
        """Initialize unloaded components and select the best available device."""
        self.model: Optional[MedicalReportModel] = None
        self.tokenizer = None
        self.rag: Optional[MedicalRAG] = None
        self.agent = None
        self.rag_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> None:
        """Load the trained model checkpoint or a fresh model if no checkpoint exists."""
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            checkpoint_path = "models/best_model.pt"
            model_path = checkpoint_path if os.path.exists(checkpoint_path) else None
            self.model, self.tokenizer = get_model_and_tokenizer(
                model_path=model_path,
                device=self.device,
            )
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")
        except Exception as exc:
            print(f"Model loading failed: {exc}")
            self.model = None
            self.tokenizer = None

    def load_rag(self) -> None:
        """Load the saved medical knowledge index if it is available."""
        if self.rag is not None:
            return
        index_path = "data/medical_kb/medical.index"
        chunks_path = "data/medical_kb/chunks.pkl"
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            print("Warning: Medical knowledge index not found. Run /build-index to create it.")
            self.rag_loaded = False
            return
        try:
            self.rag = MedicalRAG(index_path=index_path, chunks_path=chunks_path)
            self.rag_loaded = True
            print("Medical knowledge index loaded successfully")
        except Exception as exc:
            print(f"RAG loading failed: {exc}")
            self.rag = None
            self.rag_loaded = False

    def load_agent(self) -> None:
        """Load the LangChain agent when Vertex AI configuration is available."""
        if self.agent is not None:
            return
        try:
            self.agent = get_agent()
            print("Agent loaded successfully")
        except Exception as exc:
            print(f"Agent not available: {exc}")
            self.agent = None

    def is_ready(self) -> bool:
        """Return whether the prediction model is loaded and ready."""
        return self.model is not None and self.tokenizer is not None


model_manager = ModelManager()
app = FastAPI(title="MedAI - Medical Report AI Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply a simple in-memory per-IP rate limit."""
    client_ip = request.client.host if request.client else "unknown"
    now = datetime.utcnow()
    recent_requests = [
        timestamp
        for timestamp in RATE_LIMIT_STORE.get(client_ip, [])
        if now - timestamp < RATE_LIMIT_WINDOW
    ]
    if len(recent_requests) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again soon."})
    recent_requests.append(now)
    RATE_LIMIT_STORE[client_ip] = recent_requests
    return await call_next(request)


@app.on_event("startup")
async def startup_event() -> None:
    """Load model, RAG, and agent components during application startup."""
    model_manager.load_model()
    model_manager.load_rag()
    model_manager.load_agent()


def build_explanation(prediction: Dict) -> str:
    """Create a concise patient-friendly explanation from prediction output."""
    entities = prediction.get("entities", [])
    risk_level = prediction.get("risk_level", "UNKNOWN")
    if entities:
        entity_list = ", ".join(
            f"{entity.get('text', '')} ({entity.get('type', '')})"
            for entity in entities
            if entity.get("text")
        )
    else:
        entity_list = "no key medical entities were detected"
    return f"Your report shows {risk_level} risk. Key findings: {entity_list}."


def analyze_report_text(text: str) -> AnalyzeResponse:
    """Run model inference for report text and return an API response model."""
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=422, detail="Report text must not be empty")

    id2ner = {index: label for index, label in enumerate(MedicalReportModel.NER_LABELS)}
    prediction = model_manager.model.predict(
        clean_text,
        model_manager.tokenizer,
        model_manager.device,
        id2ner=id2ner,
    )
    return AnalyzeResponse(
        entities=prediction["entities"],
        risk_level=str(prediction["risk_level"]),
        risk_probs=prediction["risk_probs"],
        explanation=build_explanation(prediction),
    )


def extract_pdf_text(file_path: str) -> str:
    """Extract and normalize text from a saved PDF file."""
    reader = PdfReader(file_path)
    page_texts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        stripped = text.strip()
        if stripped:
            page_texts.append(stripped)
    return "\n\n".join(page_texts).strip()


def rebuild_index_task() -> None:
    """Rebuild the medical knowledge base index and refresh the cached RAG object."""
    build_and_save_kb("data/medical_kb/")
    model_manager.rag = MedicalRAG(
        index_path="data/medical_kb/medical.index",
        chunks_path="data/medical_kb/chunks.pkl",
    )
    model_manager.rag_loaded = True


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze pasted medical report text."""
    return analyze_report_text(request.text)


@app.post("/analyze-pdf", response_model=AnalyzeResponse)
async def analyze_pdf(file: UploadFile = File(...)) -> AnalyzeResponse:
    """Analyze an uploaded PDF medical report."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    if len(contents) >= MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name
        extracted_text = extract_pdf_text(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()

    if not extracted_text:
        raise HTTPException(status_code=400, detail="No readable text found in PDF")
    return analyze_report_text(extracted_text)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat with the MedAI agent about a medical report or result."""
    if model_manager.agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")
    result = run_agent_query(model_manager.agent, request.message, request.session_id)
    return ChatResponse(
        response=str(result.get("response", "")),
        tools_used=list(result.get("tools_used", [])),
        session_id=str(result.get("session_id", request.session_id or uuid.uuid4())),
    )


@app.post("/build-index")
async def build_index(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Start rebuilding the built-in medical knowledge index in the background."""
    background_tasks.add_task(rebuild_index_task)
    return {
        "status": "Index rebuild started",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return current service health and component readiness."""
    ready = model_manager.is_ready()
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_loaded=ready,
        index_loaded=model_manager.rag is not None,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a concise 422 validation error payload."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid input", "errors": exc.errors()},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return consistent JSON for handled HTTP errors."""
    if exc.status_code == 413:
        return JSONResponse(status_code=413, content={"detail": "File too large. Max 10MB."})
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions and avoid leaking internal details."""
    print(f"Unhandled API error: {type(exc).__name__}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    uvicorn.run("src.api:app", host=host, port=port, reload=debug)
