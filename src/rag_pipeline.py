"""FAISS-backed semantic retrieval for MedAI medical knowledge."""

import os
import pickle
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class MedicalRAG:
    """Semantic search system for retrieving medical knowledge chunks with FAISS."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: Optional[str] = None,
        chunks_path: Optional[str] = None,
        top_k: int = 5,
    ) -> None:
        """Initialize the embedding model and optionally load a saved FAISS index."""
        self.embedding_model_name = embedding_model
        self.embedding_dim = 384
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks: List[str] = []
        self.top_k = top_k

        if index_path is not None and chunks_path is not None:
            if os.path.exists(index_path) and os.path.exists(chunks_path):
                self.load_index(index_path=index_path, chunks_path=chunks_path)

    def _chunk_document(
        self,
        document: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[str]:
        """Split one document into overlapping word windows."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        words = document.split()
        if not words:
            return []

        chunks: List[str] = []
        step = chunk_size - chunk_overlap
        for start in range(0, len(words), step):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end == len(words):
                break
        return chunks

    def build_index(
        self,
        documents: List[str],
        chunk_size: int = 400,
        chunk_overlap: int = 80,
    ) -> None:
        """Build a FAISS inner-product index from medical documents."""
        chunks: List[str] = []
        for document in documents:
            chunks.extend(self._chunk_document(document, chunk_size, chunk_overlap))

        if not chunks:
            raise ValueError("No text chunks were generated from the provided documents")

        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embeddings with dimension {self.embedding_dim}, got {embeddings.shape[1]}"
            )

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        self.chunks = chunks

    def search(self, query: str) -> List[Dict]:
        """Search the FAISS index and return relevant chunks above the score threshold."""
        if self.index is None or not self.chunks:
            return []
        if not query.strip():
            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_embedding = np.asarray(query_embedding, dtype="float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k = min(self.top_k, len(self.chunks))
        scores, indices = self.index.search(query_embedding, k=k)
        results: List[Dict] = []
        for score, chunk_index in zip(scores[0], indices[0]):
            if chunk_index < 0 or chunk_index >= len(self.chunks):
                continue
            if float(score) < 0.3:
                continue
            results.append(
                {
                    "text": self.chunks[int(chunk_index)],
                    "score": float(score),
                    "index": int(chunk_index),
                }
            )
        return results

    def get_context(self, query: str, max_tokens: int = 1500) -> str:
        """Return retrieved chunks formatted as source-marked context within a token budget."""
        results = self.search(query)
        max_words = max(1, int(max_tokens * 0.75))
        used_words = 0
        context_parts: List[str] = []

        for result in results:
            text = str(result["text"])
            source_number = len(context_parts) + 1
            source_text = f"[Source {source_number}] {text}"
            source_words = len(source_text.split())
            if used_words + source_words > max_words:
                break
            context_parts.append(source_text)
            used_words += source_words

        return "\n\n".join(context_parts)

    def save_index(self, index_path: str, chunks_path: str) -> None:
        """Persist the FAISS index and chunk list to disk."""
        if self.index is None:
            raise ValueError("Cannot save before building or loading an index")

        index_dir = os.path.dirname(index_path)
        chunks_dir = os.path.dirname(chunks_path)
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)
        if chunks_dir:
            os.makedirs(chunks_dir, exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(chunks_path, "wb") as chunks_file:
            pickle.dump(self.chunks, chunks_file)

    def load_index(self, index_path: str, chunks_path: str) -> None:
        """Load a FAISS index and its matching chunk list from disk."""
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as chunks_file:
            self.chunks = pickle.load(chunks_file)


def create_medical_kb() -> List[str]:
    """Create the built-in medical knowledge base used for retrieval."""
    return [
        "HbA1c is a blood test that estimates average blood sugar over the past two to three months and is commonly used to screen for or monitor diabetes. A normal HbA1c is below 5.7%, prediabetes is 5.7% to 6.4%, and diabetes is 6.5% or higher. Higher values suggest that blood glucose has been elevated for a sustained period and should be reviewed with a clinician.",
        "White blood cell count, or WBC, measures immune cells that help fight infection. A typical adult range is about 4,000 to 11,000 cells per microliter. Elevated WBC can occur with infection, inflammation, physical stress, medications, or some blood disorders, while low WBC can increase infection risk.",
        "Hemoglobin is the oxygen-carrying protein inside red blood cells. Common adult reference ranges are about 13.5 to 17.5 g/dL for men and 12.0 to 15.5 g/dL for women. Low hemoglobin may indicate anemia, blood loss, iron deficiency, vitamin deficiency, kidney disease, or chronic inflammation.",
        "Platelets are blood cell fragments that help form clots and stop bleeding. A common normal range is 150,000 to 450,000 platelets per microliter. Low platelets can increase bleeding risk, while high platelets may be reactive to inflammation or reflect a bone marrow condition.",
        "LDL cholesterol is often called bad cholesterol because high levels contribute to plaque buildup in arteries. Optimal LDL is below 100 mg/dL, borderline high is 130 to 159 mg/dL, high is 160 to 189 mg/dL, and very high is 190 mg/dL or above. Elevated LDL raises cardiovascular risk and may lead to lifestyle changes or medication discussion.",
        "HDL cholesterol is often called good cholesterol because it helps remove cholesterol from the bloodstream. Higher HDL levels are generally protective, while low HDL can be associated with higher heart disease risk. HDL is interpreted together with LDL, triglycerides, blood pressure, diabetes status, smoking, and family history.",
        "Triglycerides are a type of fat in the blood that rises after meals and can be elevated with excess calories, alcohol, diabetes, obesity, or some medications. Normal fasting triglycerides are below 150 mg/dL, and high triglycerides are 200 mg/dL or above. Very high levels can increase pancreatitis risk and should be addressed promptly.",
        "Total cholesterol measures the combined cholesterol content in the blood, including LDL, HDL, and other lipid particles. A desirable total cholesterol level is below 200 mg/dL. High total cholesterol can indicate increased cardiovascular risk, but interpretation depends on the detailed lipid panel and overall risk profile.",
        "Creatinine is a waste product from muscle metabolism that is filtered by the kidneys. Typical ranges are about 0.7 to 1.3 mg/dL for men and 0.6 to 1.1 mg/dL for women. Elevated creatinine can suggest reduced kidney filtration, dehydration, medication effects, or other kidney-related problems.",
        "Blood urea nitrogen, or BUN, measures urea waste from protein metabolism that is cleared by the kidneys. A common reference range is 7 to 20 mg/dL. High BUN can occur with dehydration, kidney dysfunction, gastrointestinal bleeding, or high protein intake, while low BUN may occur with liver disease or low protein intake.",
        "Estimated glomerular filtration rate, or eGFR, estimates how well the kidneys filter blood. An eGFR above 90 is generally considered normal when no other kidney damage markers are present, while persistent eGFR below 60 can indicate chronic kidney disease. Trends over time are important because a single value can be affected by hydration, age, and lab variation.",
        "Fasting glucose measures blood sugar after not eating for at least eight hours. Normal fasting glucose is 70 to 100 mg/dL, prediabetes is 100 to 125 mg/dL, and diabetes is suggested by 126 mg/dL or higher on confirmatory testing. High fasting glucose can reflect insulin resistance or diabetes and should be interpreted with HbA1c and clinical context.",
        "Random glucose measures blood sugar at any time of day without requiring fasting. A random glucose above 200 mg/dL with classic symptoms such as excessive thirst, frequent urination, or unexplained weight loss can support a diabetes diagnosis. Isolated high readings may need confirmation with fasting glucose, HbA1c, or repeat testing.",
        "Thyroid-stimulating hormone, or TSH, is produced by the pituitary gland and helps regulate thyroid hormone production. A common normal range is about 0.4 to 4.0 mIU/L, with low TSH below 0.4 suggesting possible hyperthyroidism and high TSH above 4.0 suggesting possible hypothyroidism. TSH should be interpreted with free T4, symptoms, pregnancy status, and medications.",
        "T3 and T4 are thyroid hormones that regulate metabolism, temperature, heart rate, and energy use. Free T4 is often paired with TSH to evaluate thyroid function, while T3 can help assess some cases of hyperthyroidism. Abnormal T3 or T4 may indicate thyroid overactivity, underactivity, medication effects, or pituitary-related issues.",
        "Blood pressure measures the force of blood against artery walls and is reported as systolic over diastolic pressure. Normal is below 120/80 mmHg, elevated is 120 to 129 systolic with diastolic below 80, stage 1 hypertension is 130 to 139 or 80 to 89, stage 2 is 140 or higher or 90 or higher, and crisis is above 180 or above 120. Persistently high blood pressure increases risk of stroke, heart disease, kidney disease, and vision problems.",
        "Potassium is an electrolyte that supports heart rhythm, muscle function, and nerve signaling. A common normal range is 3.5 to 5.0 mmol/L. Hyperkalemia above 5.0 can be dangerous, especially at higher levels, because it may trigger abnormal heart rhythms and often requires prompt medical review.",
        "Sodium is a major electrolyte that helps control fluid balance, nerve activity, and muscle function. A common normal range is 136 to 145 mmol/L, and hyponatremia is often defined as sodium below 136 mmol/L. Low sodium can occur with excess water intake, heart failure, kidney disease, liver disease, medications, or hormonal disorders.",
        "Bilirubin is a yellow pigment produced when red blood cells are broken down and processed by the liver. A common total bilirubin range is 0.1 to 1.2 mg/dL. Elevated bilirubin can cause jaundice and may reflect liver disease, bile duct blockage, increased red blood cell breakdown, or inherited conditions such as Gilbert syndrome.",
        "ALT and AST are liver enzymes that can rise when liver cells are irritated or damaged. Elevated ALT or AST can occur with fatty liver disease, viral hepatitis, alcohol-related liver injury, medication toxicity, muscle injury, or bile duct problems. The pattern and degree of elevation help clinicians decide what follow-up testing is needed.",
        "Calcium is a mineral important for bones, muscle contraction, nerve signaling, and heart function. A common blood calcium range is 8.5 to 10.5 mg/dL. Low or high calcium can be related to parathyroid disorders, vitamin D status, kidney disease, certain cancers, medications, or albumin levels.",
        "Uric acid is a waste product from purine breakdown that is filtered by the kidneys. A common range is about 3.5 to 7.2 mg/dL, though reference ranges vary by lab and sex. High uric acid can increase the risk of gout attacks and kidney stones, especially when combined with joint pain or recurrent episodes.",
    ]


def build_and_save_kb(output_dir: str = "data/medical_kb/") -> MedicalRAG:
    """Build the default medical knowledge index and save it under the requested directory."""
    print("Creating MedAI medical knowledge base...")
    documents = create_medical_kb()
    print(f"Loaded {len(documents)} medical knowledge entries.")

    rag = MedicalRAG()
    print("Building FAISS index from knowledge base chunks...")
    rag.build_index(documents)

    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "medical.index")
    chunks_path = os.path.join(output_dir, "chunks.pkl")
    rag.save_index(index_path=index_path, chunks_path=chunks_path)
    print(f"Saved FAISS index to {index_path}")
    print(f"Saved text chunks to {chunks_path}")
    return rag
