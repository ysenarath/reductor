from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import auto as tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, get_scheduler
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.generic import PaddingStrategy

try:
    from reductor.logging import get_logger
except ModuleNotFoundError:
    from logging import getLogger as get_logger

logger = get_logger(__name__)


def get_device(device: Any = None, return_type: str = "pt") -> Any:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if return_type == "pt" and isinstance(device, str):
        device = torch.device(device)
    elif return_type == "str" and isinstance(device, torch.device):
        device = device.type
    return device


@dataclass
class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


class EarlyStopping:
    def __init__(self, metric: str, patience: int = 3, mode: str = "auto"):
        self.patience = patience
        self.metric = metric
        self.best_score = None
        self.best_epoch = 0
        self.mode = mode

    def _is_better(self, current_score: float) -> bool:
        if self.mode == "auto":
            if "loss" in self.metric:
                mode = "min"
            else:
                mode = "max"
        else:
            mode = self.mode
        if self.best_score is None:
            return True
        if mode == "min":
            return current_score < self.best_score
        return current_score > self.best_score

    def step(self, logs: dict[str, float], epoch: int) -> bool:
        if self.metric not in logs:
            return True
        current_score = logs[self.metric]
        if self._is_better(current_score):
            self.best_score = current_score
            self.best_epoch = epoch
            return True
        elif epoch - self.best_epoch >= self.patience:
            raise StopIteration
        return False


@dataclass
class TrainingArguments:
    num_epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 8
    early_stopping_patience: int = 3
    early_stopping_metric: str = "valid_loss"
    early_stopping_mode: str = "auto"


class Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        collate_fn=None,
        device: str | None = None,
        output_dir: str | None = None,
    ):
        self.args = training_args
        self.model = model
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = get_device(device)
        self.history = []
        self.output_dir = output_dir
        self.best_epoch = None

    @property
    def output_dir(self) -> Path:
        return self._checkpoint_dir

    @output_dir.setter
    def output_dir(self, value: str | Path | None) -> Path:
        if value is None:
            value = "checkpoints"
        value = Path(value)
        if not value.exists():
            value.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = value

    def train(self):
        self.history = []
        device = self.device
        train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.collate_fn,
        )
        eval_dataloader = (
            DataLoader(
                self.eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.collate_fn,
            )
            if self.eval_dataset
            else None
        )
        self.model.to(device)
        num_epochs = self.args.num_epochs
        num_training_steps = num_epochs * len(train_dataloader)
        num_training_steps = num_epochs * len(train_dataloader)
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        self.model.train()
        early_stopping = EarlyStopping(
            metric=self.args.early_stopping_metric,
            patience=self.args.early_stopping_patience,
            mode=self.args.early_stopping_mode,
        )
        progress_bar = tqdm.tqdm(range(num_training_steps), desc="Training")
        batch: dict[str, torch.Tensor]
        for epoch in range(num_epochs):
            loss_sum, loss_count = 0.0, 0
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss: torch.Tensor = outputs.loss
                loss_sum += loss.detach().cpu().numpy()
                loss_count += 1
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            train_metrics = self.evaluate(eval_dataloader, prefix="train_")
            valid_metrics = self.evaluate(eval_dataloader, prefix="valid_")
            metrics = {"epoch": epoch + 1, **train_metrics, **valid_metrics}
            metrics.setdefault("loss", loss_sum / loss_count if loss_count > 0 else 0.0)
            try:
                if early_stopping.step(metrics, epoch):
                    torch.save(
                        self.model.state_dict(),
                        self.output_dir / f"epoch_{early_stopping.best_epoch}.pt",
                    )
            except StopIteration:
                map_location = torch.device("cpu")
                state_dict = torch.load(
                    self.output_dir / f"epoch_{early_stopping.best_epoch}.pt",
                    weights_only=True,
                    map_location=map_location,
                )
                self.model.load_state_dict(state_dict)
                # move model to device
                self.model.to(device)
                self.best_epoch = early_stopping.best_epoch
                break
            if metrics:
                self.history.append(metrics)
            self.model.train()

    def evaluate(self, eval_dataloader: DataLoader | None = None, prefix: str = ""):
        if eval_dataloader is None:
            if self.eval_dataset is None:
                return None
            eval_dataloader = (
                DataLoader(self.eval_dataset, batch_size=self.args.eval_batch_size)
                if self.eval_dataset
                else None
            )
        metric = evaluate.load("accuracy")
        self.model.eval()
        loss_sum, loss_count = 0.0, 0
        for batch in eval_dataloader:
            batch = {k: v.to(self.device) for k, v in dict(batch).items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            loss: torch.Tensor = outputs.loss
            loss_sum += loss.detach().cpu().numpy()
            loss_count += 1
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        results = metric.compute()
        results["loss"] = loss_sum / loss_count if loss_count > 0 else 0.0
        return {f"{prefix}{k}": v for k, v in results.items()}


def example():
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dataset = load_dataset("yelp_review_full")

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def tokenize_function(examples: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(examples["text"], padding=False, truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    )
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-cased", num_labels=5
    )

    training_args = TrainingArguments(
        num_epochs=20,
        train_batch_size=32,
        eval_batch_size=64,
        early_stopping_patience=3,
    )

    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        collate_fn=collate_fn,
    )

    logger.info("Starting training...")

    trainer.train()

    # del model
    # del trainer
    # torch.cuda.empty_cache()

    return trainer
