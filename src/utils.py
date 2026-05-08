"""Utility helpers for MedAI text processing, display formatting, and sample data."""

import html
import re
from typing import Dict, List

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


ENTITY_COLORS: Dict[str, Dict[str, str]] = {
    "DISEASE": {"bg": "#FECDD3", "text": "#9F1239"},
    "MEDICATION": {"bg": "#FDE68A", "text": "#92400E"},
    "TEST": {"bg": "#BAE6FD", "text": "#075985"},
    "ANATOMY": {"bg": "#A7F3D0", "text": "#065F46"},
    "VALUE": {"bg": "#DDD6FE", "text": "#5B21B6"},
}
DEFAULT_ENTITY_COLOR: Dict[str, str] = {"bg": "#E2E8F0", "text": "#475569"}


def clean_medical_text(text: str) -> str:
    """Clean OCR-heavy medical text while preserving clinically meaningful content."""
    if not text:
        return ""

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
        "\u03bc": "u",
    }
    cleaned = str(text)
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)

    cleaned = "".join(character for character in cleaned if character.isprintable())
    ocr_fixes = {
        r"\bHbA1\s+c\b": "HbA1c",
        r"\bH b A 1 c\b": "HbA1c",
        r"\bW\s+BC\b": "WBC",
        r"\bR\s+BC\b": "RBC",
        r"\bL\s+DL\b": "LDL",
        r"\bH\s+DL\b": "HDL",
        r"\bT\s+SH\b": "TSH",
        r"\be\s+GFR\b": "eGFR",
    }
    for pattern, replacement in ocr_fixes.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:%])", r"\1", cleaned)
    cleaned = re.sub(r"([(<])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)>])", r"\1", cleaned)
    return cleaned.strip()


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from all readable PDF pages and return a cleaned combined string."""
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(file_path)
    except FileNotFoundError:
        return ""

    page_texts: List[str] = []
    for page in reader.pages:
        try:
            extracted = page.extract_text() or ""
        except Exception:
            extracted = ""
        cleaned = clean_medical_text(extracted)
        if cleaned:
            page_texts.append(cleaned)
    return "\n\n".join(page_texts)


def get_entity_color(entity_type: str) -> Dict[str, str]:
    """Return the display background and text color for an entity type."""
    return ENTITY_COLORS.get(str(entity_type).upper(), DEFAULT_ENTITY_COLOR).copy()


def format_entities_for_display(entities: list) -> str:
    """Format model entities as escaped HTML spans with MedAI entity colors."""
    if not entities:
        return ""

    grouped_entities: List[Dict[str, str]] = []
    for entity in entities:
        entity_type = str(entity.get("type", "")).upper()
        entity_text = str(entity.get("text", "")).strip()
        if not entity_text:
            continue
        if grouped_entities and grouped_entities[-1]["type"] == entity_type:
            grouped_entities[-1]["text"] = f"{grouped_entities[-1]['text']} {entity_text}"
        else:
            grouped_entities.append({"type": entity_type, "text": entity_text})

    spans: List[str] = []
    for entity in grouped_entities:
        colors = get_entity_color(entity["type"])
        text = html.escape(entity["text"])
        spans.append(
            f'<span class="entity" style="background: {colors["bg"]}; color: {colors["text"]};">'
            f"{text}</span>"
        )
    return " ".join(spans)


def _format_probability(value: object) -> int:
    """Normalize a probability expressed as either 0-1 or 0-100 into an integer percent."""
    try:
        probability = float(value)
    except (TypeError, ValueError):
        probability = 0.0
    if probability <= 1.0:
        probability *= 100.0
    return int(round(max(0.0, min(100.0, probability))))


def generate_explanation(entities: list, risk_level: str, risk_probs: dict) -> str:
    """Generate a deterministic markdown explanation for model outputs."""
    grouped: Dict[str, Dict[str, object]] = {}
    for entity in entities or []:
        entity_type = str(entity.get("type", "UNKNOWN")).upper()
        entity_text = str(entity.get("text", "")).strip()
        if not entity_text:
            continue
        if entity_type not in grouped:
            grouped[entity_type] = {"texts": [], "count": 0}
        grouped[entity_type]["texts"].append(entity_text)
        grouped[entity_type]["count"] = int(grouped[entity_type]["count"]) + 1

    lines = [f"Your report shows {str(risk_level).upper()} clinical risk.", "", "Key findings:"]
    if grouped:
        for entity_type in sorted(grouped):
            data = grouped[entity_type]
            entity_text = ", ".join(str(text) for text in data["texts"])
            lines.append(f"- {entity_type}: {entity_text} (x{data['count']})")
    else:
        lines.append("- No key medical entities detected.")

    lines.extend(
        [
            "",
            "Risk breakdown:",
            f"- LOW: {_format_probability(risk_probs.get('LOW', 0))}%",
            f"- MEDIUM: {_format_probability(risk_probs.get('MEDIUM', 0))}%",
            f"- HIGH: {_format_probability(risk_probs.get('HIGH', 0))}%",
        ]
    )
    return "\n".join(lines)


def create_sample_reports() -> list:
    """Return synthetic report strings spanning low, medium, and high risk examples."""
    return [
        (
            "Comprehensive Metabolic and CBC Panel: HbA1c 5.4%, WBC 7200/uL, "
            "Hemoglobin 14.2 g/dL, Platelets 245000/uL, Creatinine 0.9 mg/dL. "
            "All listed values are within expected adult reference ranges."
        ),
        (
            "Follow-up diabetes screening: HbA1c 6.1%, WBC 10800/uL, Hemoglobin 12.4 g/dL, "
            "Platelets 410000/uL, Creatinine 1.1 mg/dL. HbA1c is in the prediabetes range "
            "and should be reviewed with a clinician."
        ),
        (
            "Urgent lab review: HbA1c 9.1%, WBC 14000/uL, Hemoglobin 10.8 g/dL, "
            "Platelets 510000/uL, Creatinine 1.8 mg/dL. Results show multiple abnormal "
            "values including high glucose burden, possible inflammation, anemia, and reduced kidney filtration."
        ),
    ]
