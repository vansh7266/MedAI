"""Training pipeline for the MedAI multi-task medical report model."""

import json
import os
import random
from typing import Dict, List, Tuple

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.model import MedicalReportModel

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
NER_LABELS: List[str] = [
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
]
RISK_LABELS: List[str] = ["LOW", "MEDIUM", "HIGH"]
LABEL2ID: Dict[str, int] = {label: index for index, label in enumerate(NER_LABELS)}


class MedicalNERDataset(Dataset):
    """Torch dataset wrapper for token-level NER labels and sequence-level risk labels."""

    def __init__(self, samples: List[Dict]) -> None:
        """Store samples containing tokens, NER label IDs, and risk labels."""
        self.samples = samples

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        """Return one training sample by index."""
        return self.samples[index]


def add_entity(tokens: List[str], labels: List[int], phrase: str, entity_type: str) -> None:
    """Append a phrase to token and label lists using BIO tags."""
    words = phrase.split()
    for word_index, word in enumerate(words):
        prefix = "B" if word_index == 0 else "I"
        tokens.append(word)
        labels.append(LABEL2ID[f"{prefix}-{entity_type}"])


def add_plain(tokens: List[str], labels: List[int], phrase: str) -> None:
    """Append non-entity text to token and label lists."""
    for word in phrase.split():
        tokens.append(word)
        labels.append(LABEL2ID["O"])


def count_abnormal_values(tokens: List[str]) -> int:
    """Estimate abnormal numeric values in a tokenized report for risk labeling."""
    abnormal_count = 0
    for index, token in enumerate(tokens):
        lower_token = token.lower()
        context = " ".join(tokens[max(0, index - 4):index]).lower()
        cleaned = lower_token.replace(",", "").replace("mg/dl", "").replace("g/dl", "")
        cleaned = cleaned.replace("mmol/l", "").replace("/ul", "")
        if cleaned.endswith("%"):
            try:
                if float(cleaned[:-1]) >= 6.5:
                    abnormal_count += 1
            except ValueError:
                continue
            continue
        if "/" in cleaned:
            parts = cleaned.split("/")
            try:
                systolic = float(parts[0])
                diastolic = float(parts[1])
                if systolic >= 140 or diastolic >= 90:
                    abnormal_count += 1
            except (ValueError, IndexError):
                continue
            continue
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if "wbc" in context and value > 11000:
            abnormal_count += 1
        elif "hemoglobin" in context and value < 12.0:
            abnormal_count += 1
        elif "ldl" in context and value >= 160:
            abnormal_count += 1
        elif "creatinine" in context and value > 1.3:
            abnormal_count += 1
        elif "tsh" in context and (value < 0.4 or value > 4.0):
            abnormal_count += 1
        elif "potassium" in context and (value < 3.5 or value > 5.0):
            abnormal_count += 1
        elif "glucose" in context and value >= 126:
            abnormal_count += 1
        elif "blood pressure" in context and value >= 140:
            abnormal_count += 1
        elif value >= 200:
            abnormal_count += 1
    return abnormal_count


def risk_from_abnormal_count(abnormal_count: int) -> int:
    """Map an abnormal value count to LOW, MEDIUM, or HIGH risk IDs."""
    if abnormal_count == 0:
        return 0
    if abnormal_count <= 2:
        return 1
    return 2


def make_sample(parts: List[Tuple[str, str]]) -> Dict:
    """Create one sample from labeled phrase parts."""
    tokens: List[str] = []
    labels: List[int] = []
    for phrase, entity_type in parts:
        if entity_type == "O":
            add_plain(tokens, labels, phrase)
        else:
            add_entity(tokens, labels, phrase, entity_type)
    abnormal_count = count_abnormal_values(tokens)
    return {
        "tokens": tokens,
        "ner_labels": labels,
        "risk_label": risk_from_abnormal_count(abnormal_count),
    }


def generate_synthetic_data(num_samples: int = 500) -> List[Dict]:
    """Generate synthetic medical NER sentences with BIO labels and risk labels."""
    diseases = ["diabetes", "hypertension", "anemia", "kidney disease", "thyroid disorder"]
    medications = ["metformin", "atorvastatin", "levothyroxine", "lisinopril", "iron supplement"]
    tests = ["HbA1c", "WBC", "hemoglobin", "LDL cholesterol", "creatinine", "TSH", "potassium"]
    anatomy = ["blood", "kidney", "liver", "thyroid", "heart"]
    normal_values = ["5.4%", "8200/uL", "14.1", "92", "0.9", "2.1", "4.2"]
    abnormal_values = ["9.1%", "14000/uL", "11.2", "185", "1.8", "6.2", "5.8"]

    samples: List[Dict] = []
    for index in range(num_samples):
        disease = diseases[index % len(diseases)]
        medication = medications[(index * 2) % len(medications)]
        test = tests[(index * 3) % len(tests)]
        body_part = anatomy[(index * 5) % len(anatomy)]
        abnormal_count = index % 4
        values = []
        for value_index in range(3):
            if value_index < abnormal_count:
                values.append(abnormal_values[(index + value_index) % len(abnormal_values)])
            else:
                values.append(normal_values[(index + value_index) % len(normal_values)])

        patterns = [
            [
                ("Patient has", "O"),
                (disease, "DISEASE"),
                ("with", "O"),
                (test, "TEST"),
                (values[0], "VALUE"),
                ("and", "O"),
                ("WBC", "TEST"),
                (values[1], "VALUE"),
                ("in", "O"),
                (body_part, "ANATOMY"),
            ],
            [
                ("Report shows", "O"),
                (test, "TEST"),
                (values[0], "VALUE"),
                ("while taking", "O"),
                (medication, "MEDICATION"),
                ("for", "O"),
                (disease, "DISEASE"),
            ],
            [
                ("Clinical history includes", "O"),
                (disease, "DISEASE"),
                ("and", "O"),
                (body_part, "ANATOMY"),
                ("evaluation with", "O"),
                (test, "TEST"),
                (values[0], "VALUE"),
                ("plus glucose", "O"),
                (values[2], "VALUE"),
            ],
        ]
        samples.append(make_sample(patterns[index % len(patterns)]))
    return samples


def normalize_dataset_label(label_name: str) -> str:
    """Map source dataset label names into the MedAI 11-class schema."""
    if label_name == "O":
        return "O"
    prefix = "B" if label_name.startswith("B") else "I"
    lower_name = label_name.lower()
    if "disease" in lower_name:
        return f"{prefix}-DISEASE"
    if "chemical" in lower_name or "drug" in lower_name or "medication" in lower_name:
        return f"{prefix}-MEDICATION"
    return "O"


def load_bc5cdr_samples() -> List[Dict]:
    """Load and normalize the HuggingFace tner/bc5cdr dataset when available."""
    if load_dataset is None:
        raise RuntimeError("datasets is not installed")

    dataset = load_dataset("tner/bc5cdr")
    train_split = dataset["train"]
    token_key = "tokens"
    label_key = "tags" if "tags" in train_split.column_names else "ner_tags"

    label_feature = train_split.features[label_key]
    label_names = label_feature.feature.names if hasattr(label_feature, "feature") else label_feature.names

    samples: List[Dict] = []
    for item in train_split:
        tokens = [str(token) for token in item[token_key]]
        labels = []
        for label_id in item[label_key]:
            source_label = label_names[int(label_id)]
            target_label = normalize_dataset_label(source_label)
            labels.append(LABEL2ID[target_label])
        samples.append(
            {
                "tokens": tokens,
                "ner_labels": labels,
                "risk_label": risk_from_abnormal_count(count_abnormal_values(tokens)),
            }
        )
    return samples


def load_training_samples() -> List[Dict]:
    """Load BC5CDR samples or fall back to synthetic medical NER data."""
    try:
        print("Loading HuggingFace dataset: tner/bc5cdr")
        samples = load_bc5cdr_samples()
        print(f"Loaded {len(samples)} BC5CDR samples.")
        return samples
    except Exception as exc:
        print(f"BC5CDR unavailable, using synthetic data instead: {exc}")
        samples = generate_synthetic_data(num_samples=500)
        print(f"Generated {len(samples)} synthetic samples.")
        return samples


def collate_batch(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """Tokenize a batch and align word-level BIO labels to first WordPiece tokens."""
    token_batch = [item["tokens"] for item in batch]
    encoded = tokenizer(
        token_batch,
        is_split_into_words=True,
        max_length=256,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    aligned_labels: List[List[int]] = []
    for batch_index, item in enumerate(batch):
        word_ids = encoded.word_ids(batch_index=batch_index)
        previous_word_id = None
        labels = item["ner_labels"]
        label_ids: List[int] = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(labels[word_id] if word_id < len(labels) else LABEL2ID["O"])
            else:
                label_ids.append(-100)
            previous_word_id = word_id
        aligned_labels.append(label_ids)

    encoded["ner_labels"] = torch.tensor(aligned_labels, dtype=torch.long)
    encoded["risk_labels"] = torch.tensor([item["risk_label"] for item in batch], dtype=torch.long)
    return encoded


def compute_metrics(
    model: MedicalReportModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute validation loss, NER F1, and risk accuracy."""
    model.eval()
    total_loss = 0.0
    total_ner_loss = 0.0
    total_risk_loss = 0.0
    total_batches = 0
    ner_true: List[int] = []
    ner_pred: List[int] = []
    risk_true: List[int] = []
    risk_pred: List[int] = []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ner_labels = batch["ner_labels"].to(device)
            risk_labels = batch["risk_labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ner_labels=ner_labels,
                    risk_labels=risk_labels,
                )

            total_loss += float(outputs["loss"].item())
            total_ner_loss += float(outputs["ner_loss"].item())
            total_risk_loss += float(outputs["risk_loss"].item())
            total_batches += 1

            predictions = torch.argmax(outputs["ner_logits"], dim=-1)
            active_mask = ner_labels != -100
            ner_true.extend(ner_labels[active_mask].detach().cpu().tolist())
            ner_pred.extend(predictions[active_mask].detach().cpu().tolist())
            risk_true.extend(risk_labels.detach().cpu().tolist())
            risk_pred.extend(torch.argmax(outputs["risk_logits"], dim=-1).detach().cpu().tolist())

    entity_labels = list(range(1, len(NER_LABELS)))
    _, _, ner_f1, _ = precision_recall_fscore_support(
        ner_true,
        ner_pred,
        labels=entity_labels,
        average="micro",
        zero_division=0,
    )
    risk_accuracy = accuracy_score(risk_true, risk_pred) if risk_true else 0.0
    divisor = max(1, total_batches)
    return {
        "combined_loss": total_loss / divisor,
        "ner_loss": total_ner_loss / divisor,
        "risk_loss": total_risk_loss / divisor,
        "ner_f1": float(ner_f1),
        "risk_accuracy": float(risk_accuracy),
    }


def save_training_artifacts(
    model: MedicalReportModel,
    optimizer: AdamW,
    tokenizer: AutoTokenizer,
    epoch: int,
    metrics: Dict[str, float],
    output_dir: str,
) -> None:
    """Save model checkpoint, label maps, and tokenizer artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        os.path.join(output_dir, "best_model.pt"),
    )

    with open(os.path.join(output_dir, "ner_id2label.json"), "w", encoding="utf-8") as label_file:
        json.dump({index: label for index, label in enumerate(NER_LABELS)}, label_file, indent=2)
    with open(os.path.join(output_dir, "risk_id2label.json"), "w", encoding="utf-8") as label_file:
        json.dump({index: label for index, label in enumerate(RISK_LABELS)}, label_file, indent=2)
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))


def main() -> None:
    """Run the complete MedAI training loop."""
    random.seed(42)
    torch.manual_seed(42)
    output_dir = "/kaggle/working/models" if os.path.exists("/kaggle/working") else "models"
    batch_size = 16
    gradient_accumulation_steps = 2
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = load_training_samples()
    dataset = MedicalNERDataset(samples)
    validation_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - validation_size
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, tokenizer),
    )

    model = MedicalReportModel().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    optimizer_steps_per_epoch = max(
        1,
        (len(train_loader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps,
    )
    total_steps = optimizer_steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_ner_f1 = -1.0

    print(
        f"Training {len(train_dataset)} samples, validating {len(validation_dataset)} samples, "
        f"effective batch size {batch_size * gradient_accumulation_steps}."
    )

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_ner_loss = 0.0
        train_risk_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)

        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ner_labels = batch["ner_labels"].to(device)
            risk_labels = batch["risk_labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ner_labels=ner_labels,
                    risk_labels=risk_labels,
                )
                loss = outputs["loss"] / gradient_accumulation_steps

            scaler.scale(loss).backward()
            train_loss += float(outputs["loss"].item())
            train_ner_loss += float(outputs["ner_loss"].item())
            train_risk_loss += float(outputs["risk_loss"].item())

            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(
                {
                    "loss": f"{outputs['loss'].item():.4f}",
                    "ner": f"{outputs['ner_loss'].item():.4f}",
                    "risk": f"{outputs['risk_loss'].item():.4f}",
                }
            )

        validation_metrics = compute_metrics(model, validation_loader, device)
        train_divisor = max(1, len(train_loader))
        epoch_metrics = {
            "train_combined_loss": train_loss / train_divisor,
            "train_ner_loss": train_ner_loss / train_divisor,
            "train_risk_loss": train_risk_loss / train_divisor,
            **{f"val_{key}": value for key, value in validation_metrics.items()},
        }

        print(
            f"Epoch {epoch}: "
            f"train_loss={epoch_metrics['train_combined_loss']:.4f}, "
            f"val_loss={epoch_metrics['val_combined_loss']:.4f}, "
            f"val_ner_f1={epoch_metrics['val_ner_f1']:.4f}, "
            f"val_risk_accuracy={epoch_metrics['val_risk_accuracy']:.4f}"
        )

        if validation_metrics["ner_f1"] > best_ner_f1:
            best_ner_f1 = validation_metrics["ner_f1"]
            save_training_artifacts(model, optimizer, tokenizer, epoch, epoch_metrics, output_dir)
            print(f"Saved new best model to {os.path.join(output_dir, 'best_model.pt')}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Training complete. Best validation NER F1: {best_ner_f1:.4f}")


if __name__ == "__main__":
    main()
