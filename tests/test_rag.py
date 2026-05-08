"""Unit tests for the FAISS-based MedAI RAG pipeline."""

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")

from src import rag_pipeline
from src.rag_pipeline import MedicalRAG


class FakeSentenceTransformer:
    """Small deterministic embedding model used instead of downloading MiniLM."""

    def __init__(self, model_name: str) -> None:
        """Store the model name for interface compatibility."""
        self.model_name = model_name

    def encode(
        self,
        texts: list,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Return deterministic 384-dimensional vectors for any input text."""
        vectors = []
        for text in texts:
            vector = np.ones(384, dtype="float32")
            lowered = str(text).lower()
            if "diabetes" in lowered or "hba1c" in lowered:
                vector[0] = 4.0
            if "wbc" in lowered or "infection" in lowered:
                vector[1] = 4.0
            if "kidney" in lowered or "creatinine" in lowered:
                vector[2] = 4.0
            if normalize_embeddings:
                vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
        return np.asarray(vectors, dtype="float32")


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch SentenceTransformer with a deterministic local fake."""
    monkeypatch.setattr(rag_pipeline, "SentenceTransformer", FakeSentenceTransformer)


def build_small_rag(fake_embeddings: None) -> MedicalRAG:
    """Build a compact RAG index for tests."""
    documents = [
        "HbA1c is used to monitor diabetes and long term blood sugar control.",
        "WBC can rise when infection or inflammation is present.",
        "Creatinine and eGFR are used to evaluate kidney filtration.",
    ]
    rag = MedicalRAG(top_k=2)
    rag.build_index(documents, chunk_size=20, chunk_overlap=0)
    return rag


def test_build_index(fake_embeddings: None) -> None:
    """build_index should add embedded chunks to the FAISS index."""
    rag = build_small_rag(fake_embeddings)
    assert rag.index is not None
    assert rag.index.ntotal > 0
    assert len(rag.chunks) == rag.index.ntotal


def test_search(fake_embeddings: None) -> None:
    """search should return scored chunk dictionaries for relevant queries."""
    rag = build_small_rag(fake_embeddings)
    results = rag.search("HbA1c diabetes")
    assert isinstance(results, list)
    assert results
    assert results[0]["score"] > 0


def test_get_context(fake_embeddings: None) -> None:
    """get_context should include source markers and retrieved text."""
    rag = build_small_rag(fake_embeddings)
    context = rag.get_context("kidney creatinine")
    assert len(context) > 0
    assert "[Source 1]" in context


def test_save_load_index(fake_embeddings: None, tmp_path) -> None:
    """A saved RAG index should load with the same number of vectors."""
    rag = build_small_rag(fake_embeddings)
    index_path = tmp_path / "medical.index"
    chunks_path = tmp_path / "chunks.pkl"
    rag.save_index(str(index_path), str(chunks_path))

    loaded = MedicalRAG(index_path=str(index_path), chunks_path=str(chunks_path))
    assert loaded.index.ntotal == rag.index.ntotal
    assert loaded.chunks == rag.chunks
