"""Unit tests for the MedAI neural model architecture."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src import model as model_module
from src.model import MedicalReportModel, NERHead, RiskHead


class FakeLayer(nn.Module):
    """Tiny trainable layer used to inspect freezing behavior without loading BERT."""

    def __init__(self) -> None:
        """Create a single trainable parameter."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class FakeEncoderStack(nn.Module):
    """Minimal object that mirrors the BERT encoder.layer structure."""

    def __init__(self, num_layers: int = 8) -> None:
        """Create fake transformer layers."""
        super().__init__()
        self.layer = nn.ModuleList([FakeLayer() for _ in range(num_layers)])


class FakeAutoModel(nn.Module):
    """Lightweight encoder returning BERT-shaped hidden states."""

    def __init__(self) -> None:
        """Create fake base, transformer, and pooler parameters."""
        super().__init__()
        self.embeddings = FakeLayer()
        self.encoder = FakeEncoderStack()
        self.pooler = FakeLayer()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        """Return zero hidden states with the expected BiomedBERT shape."""
        batch_size, sequence_length = input_ids.shape
        hidden = torch.zeros(batch_size, sequence_length, 768)
        return SimpleNamespace(last_hidden_state=hidden)


class EncodedInputs(dict):
    """Dictionary-like tokenizer result with a no-op device transfer method."""

    def to(self, device: str) -> "EncodedInputs":
        """Move contained tensors to the requested device."""
        for key, value in self.items():
            if hasattr(value, "to"):
                self[key] = value.to(device)
        return self


class FakeTokenizer:
    """Tokenizer stub that supports the methods used by MedicalReportModel.predict."""

    cls_token = "[CLS]"
    sep_token = "[SEP]"
    pad_token = "[PAD]"

    def __call__(self, text: str, **kwargs) -> EncodedInputs:
        """Return fixed token IDs and attention mask for prediction tests."""
        return EncodedInputs(
            {
                "input_ids": torch.tensor([[101, 2001, 2002, 102]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            }
        )

    def convert_ids_to_tokens(self, token_ids: list) -> list:
        """Map fixed token IDs to WordPiece-style tokens."""
        return ["[CLS]", "hba", "##1c", "[SEP]"]


@pytest.fixture
def fake_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch AutoModel so tests never download external model weights."""
    monkeypatch.setattr(model_module.AutoModel, "from_pretrained", lambda model_name: FakeAutoModel())


def test_ner_head_forward() -> None:
    """NERHead should return per-token logits for all 11 BIO labels."""
    head = NERHead(num_labels=11)
    dummy_hidden = torch.randn(2, 10, 768)
    logits = head(dummy_hidden)
    assert logits.shape == (2, 10, 11)


def test_risk_head_forward() -> None:
    """RiskHead should return 3 probabilities that sum to one for each sample."""
    head = RiskHead(num_labels=3)
    dummy_cls = torch.randn(2, 768)
    probabilities = head(dummy_cls)
    assert probabilities.shape == (2, 3)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_medical_report_model_forward(fake_encoder: None) -> None:
    """MedicalReportModel.forward should expose logits, probabilities, and losses."""
    model = MedicalReportModel()
    input_ids = torch.ones(2, 10, dtype=torch.long)
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    ner_labels = torch.zeros(2, 10, dtype=torch.long)
    risk_labels = torch.zeros(2, dtype=torch.long)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ner_labels=ner_labels,
        risk_labels=risk_labels,
    )

    expected_keys = {"ner_logits", "risk_logits", "risk_probs", "ner_loss", "risk_loss", "loss"}
    assert expected_keys.issubset(outputs.keys())
    assert outputs["ner_logits"].shape == (2, 10, 11)
    assert outputs["risk_logits"].shape == (2, 3)


def test_predict_method(fake_encoder: None) -> None:
    """MedicalReportModel.predict should return the public inference payload keys."""
    model = MedicalReportModel()
    prediction = model.predict("HbA1c 9.1%", FakeTokenizer(), "cpu")

    assert {"entities", "risk_level", "risk_probs", "tokens"}.issubset(prediction.keys())
    assert prediction["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert set(prediction["risk_probs"].keys()) == {"LOW", "MEDIUM", "HIGH"}


def test_freezing_strategy(fake_encoder: None) -> None:
    """The encoder base should freeze while the final 4 layers and pooler stay trainable."""
    model = MedicalReportModel()

    assert all(not parameter.requires_grad for parameter in model.encoder.embeddings.parameters())
    for layer in model.encoder.encoder.layer[:4]:
        assert all(not parameter.requires_grad for parameter in layer.parameters())
    for layer in model.encoder.encoder.layer[-4:]:
        assert all(parameter.requires_grad for parameter in layer.parameters())
    assert all(parameter.requires_grad for parameter in model.encoder.pooler.parameters())
