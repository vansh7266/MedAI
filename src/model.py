"""Core multi-task BiomedBERT model for MedAI report analysis."""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class NERHead(nn.Module):
    """Token-level classifier that predicts BIO medical entity labels for each encoder token."""

    def __init__(self, num_labels: int) -> None:
        """Initialize the NER classification head with the requested number of labels."""
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_labels),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Convert token hidden states into per-token NER logits."""
        return self.classifier(hidden_states)


class RiskHead(nn.Module):
    """Sequence-level classifier that predicts LOW, MEDIUM, or HIGH report risk."""

    def __init__(self, num_labels: int) -> None:
        """Initialize the risk classification head with the requested number of labels."""
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_labels),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward_logits(self, cls_hidden_state: torch.Tensor) -> torch.Tensor:
        """Convert a CLS hidden state into raw risk logits for training loss computation."""
        return self.classifier(cls_hidden_state)

    def forward(self, cls_hidden_state: torch.Tensor) -> torch.Tensor:
        """Convert a CLS hidden state into normalized risk probabilities."""
        return self.softmax(self.forward_logits(cls_hidden_state))


class MedicalReportModel(nn.Module):
    """Shared BiomedBERT encoder with NER and risk heads for medical report analysis."""

    MODEL_NAME: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    NER_LABELS: Tuple[str, ...] = (
        "O",
        "B-DISEASE",
        "I-DISEASE",
        "B-MEDICATION",
        "I-MEDICATION",
        "B-TEST",
        "I-TEST",
        "B-ANATOMY",
        "I-ANATOMY",
        "B-VALUE",
        "I-VALUE",
    )
    RISK_LABELS: Tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")

    def __init__(self) -> None:
        """Load BiomedBERT, attach both task heads, and apply the encoder freezing policy."""
        super().__init__()
        self.encoder = AutoModel.from_pretrained(self.MODEL_NAME)
        self.num_ner_labels = len(self.NER_LABELS)
        self.num_risk_labels = len(self.RISK_LABELS)
        self.ner_head = NERHead(num_labels=self.num_ner_labels)
        self.risk_head = RiskHead(num_labels=self.num_risk_labels)
        self.freeze_encoder()
        self.unfreeze_last_transformer_layers(num_layers=4)
        self.unfreeze_pooler()

    def freeze_encoder(self) -> None:
        """Freeze every encoder parameter before selectively unfreezing trainable layers."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_last_transformer_layers(self, num_layers: int) -> None:
        """Unfreeze the final transformer layers of the BiomedBERT encoder."""
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            for layer in self.encoder.encoder.layer[-num_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    def unfreeze_pooler(self) -> None:
        """Unfreeze the encoder pooler when the underlying model exposes one."""
        pooler = getattr(self.encoder, "pooler", None)
        if pooler is not None:
            for parameter in pooler.parameters():
                parameter.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ner_labels: Optional[torch.Tensor] = None,
        risk_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the shared encoder and return logits, probabilities, and optional task losses."""
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = encoder_outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]

        ner_logits = self.ner_head(sequence_output)
        risk_logits = self.risk_head.forward_logits(cls_output)
        risk_probs = self.risk_head.softmax(risk_logits)

        outputs: Dict[str, torch.Tensor] = {
            "ner_logits": ner_logits,
            "risk_logits": risk_logits,
            "risk_probs": risk_probs,
        }

        if ner_labels is not None:
            ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            outputs["ner_loss"] = ner_loss_fn(
                ner_logits.reshape(-1, self.num_ner_labels),
                ner_labels.reshape(-1),
            )

        if risk_labels is not None:
            risk_loss_fn = nn.CrossEntropyLoss()
            outputs["risk_loss"] = risk_loss_fn(risk_logits, risk_labels)

        if ner_labels is not None and risk_labels is not None:
            outputs["loss"] = outputs["ner_loss"] + 0.5 * outputs["risk_loss"]

        return outputs

    def predict(
        self,
        text: str,
        tokenizer: AutoTokenizer,
        device: str,
        id2ner: Optional[Dict[int, str]] = None,
    ) -> Dict[str, object]:
        """Analyze text and return extracted entities, risk level, risk probabilities, and tokens."""
        ner_mapping = id2ner
        if ner_mapping is None:
            ner_mapping = {index: label for index, label in enumerate(self.NER_LABELS)}

        was_training = self.training
        self.to(device)
        self.eval()

        encoded = tokenizer(
            text,
            max_length=256,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            ner_probabilities = torch.softmax(outputs["ner_logits"], dim=-1)
            ner_confidences, ner_predictions = torch.max(ner_probabilities, dim=-1)
            risk_probabilities = outputs["risk_probs"]

        token_ids = input_ids[0].detach().cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        predicted_label_ids = ner_predictions[0].detach().cpu().tolist()
        confidence_values = ner_confidences[0].detach().cpu().tolist()
        attention_values = attention_mask[0].detach().cpu().tolist()

        special_tokens = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}
        entities = []
        current_entity: Optional[Dict[str, object]] = None
        visible_tokens = []

        for token, label_id, confidence, is_visible in zip(
            tokens,
            predicted_label_ids,
            confidence_values,
            attention_values,
        ):
            if not is_visible or token in special_tokens:
                if current_entity is not None:
                    entities.append(self._finalize_entity(current_entity))
                    current_entity = None
                continue

            is_subword = token.startswith("##")
            clean_token = token[2:] if is_subword else token
            visible_tokens.append(clean_token)
            label = ner_mapping.get(int(label_id), "O")

            if label == "O":
                if current_entity is not None:
                    entities.append(self._finalize_entity(current_entity))
                    current_entity = None
                continue

            label_prefix, entity_type = label.split("-", 1)
            if label_prefix == "B" or current_entity is None or current_entity["type"] != entity_type:
                if current_entity is not None:
                    entities.append(self._finalize_entity(current_entity))
                current_entity = {
                    "type": entity_type,
                    "tokens": [clean_token],
                    "confidences": [float(confidence)],
                }
                continue

            self._append_entity_token(current_entity, clean_token, is_subword, float(confidence))

        if current_entity is not None:
            entities.append(self._finalize_entity(current_entity))

        risk_values = risk_probabilities[0].detach().cpu()
        risk_index = int(torch.argmax(risk_values).item())
        risk_probs = {
            label: float(risk_values[index].item())
            for index, label in enumerate(self.RISK_LABELS)
        }

        if was_training:
            self.train()

        return {
            "entities": entities,
            "risk_level": self.RISK_LABELS[risk_index],
            "risk_probs": risk_probs,
            "tokens": visible_tokens,
        }

    def _append_entity_token(
        self,
        entity: Dict[str, object],
        token: str,
        is_subword: bool,
        confidence: float,
    ) -> None:
        """Append a token fragment to an in-progress BIO entity."""
        tokens = entity["tokens"]
        confidences = entity["confidences"]
        if isinstance(tokens, list) and isinstance(confidences, list):
            if is_subword and tokens:
                tokens[-1] = f"{tokens[-1]}{token}"
            else:
                tokens.append(token)
            confidences.append(confidence)

    def _finalize_entity(self, entity: Dict[str, object]) -> Dict[str, object]:
        """Convert an accumulated BIO entity into the public prediction format."""
        tokens = entity["tokens"]
        confidences = entity["confidences"]
        entity_text = " ".join(tokens) if isinstance(tokens, list) else ""
        confidence = 0.0
        if isinstance(confidences, list) and confidences:
            confidence = float(sum(confidences) / len(confidences))
        return {
            "type": str(entity["type"]),
            "text": entity_text,
            "confidence": confidence,
        }


def get_model_and_tokenizer(
    model_path: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[MedicalReportModel, AutoTokenizer]:
    """Create the MedAI model and tokenizer, optionally loading a saved model state dict."""
    model = MedicalReportModel()
    tokenizer = AutoTokenizer.from_pretrained(MedicalReportModel.MODEL_NAME)

    if model_path is not None and os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(state_dict)

    model.to(device)
    return model, tokenizer
