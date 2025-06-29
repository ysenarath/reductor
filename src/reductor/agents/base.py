from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import BaseLoader, Environment, Template
from smolagents import Tool
from typing_extensions import Literal

import reductor

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """
    Task dataclass representing a classification or tagging task.

    Attributes
    ----------
    input_text : str
        The input text to be processed for the task.
    classes : list of str, optional
        A list of class labels relevant to the task. Defaults to None.
    target_type : {'multiclass_classifier', 'multilabel_classifier', 'binary_classifier', 'boi_tagger'}
        The type of task to be performed. Defaults to 'multiclass_classifier'.
    """

    input_text: str
    classes: list[str] | None = None
    target_type: Literal[
        "multiclass_classifier",
        "multilabel_classifier",
        "binary_classifier",
        "boi_tagger",
    ] = "multiclass_classifier"


class TaskPromptTemplate:
    def __init__(self, task: Task, instruction: str | None = None) -> None:
        self.task = task
        self.template_cache = {}
        self.instruction = instruction

    @property
    def template(self) -> Template:
        if self.task.target_type not in self.template_cache:
            env = Environment(loader=BaseLoader())
            template_path = (
                Path(reductor.__file__).parent
                / "prompts"
                / f"{self.task.target_type}.jinja2"
            )
            with open(template_path, "r") as file:
                template_string = file.read()
            template = env.from_string(template_string)
            self.template_cache[self.task.target_type] = template
        return self.template_cache[self.task.target_type]

    def render(self, input_text: str | None = None, **kwargs) -> str:
        kwargs["input_text"] = input_text or self.task.input_text
        kwargs.setdefault("instruction", self.instruction)
        kwargs.setdefault("classes", self.task.classes)
        return self.template.render(**kwargs)


class TaskFinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {
        "answer": {"type": "any", "description": "The final answer to the problem"}
    }
    output_type = "any"

    def __init__(self, task: Task) -> None:
        super().__init__()
        self.task = task

    def _move_cursor(self, cursor: int, token: str) -> int:
        """Move the cursor based on the token length."""
        text = self.task.input_text
        i = cursor
        while i < len(text) and text[i].isspace():
            i += 1
        end = i + len(token)
        if text[i:end].lower() != token.lower():
            raise ValueError(
                f"Token '{token}' does not match the text at position {i}: '{text[i:end]}'"
            )
        token = text[i:end]
        return token, i, end

    def forward(self, answer: str) -> Any:
        if self.task.target_type == "multilabel_classifier":
            pred_labels = [s.strip() for s in answer.split(",") if s.strip()]
            for pl in pred_labels:
                if pl not in self.task.classes:
                    raise ValueError(
                        f"Label '{pl}' is not in the list of classes: {self.task.classes}"
                    )
            return [float(label in pred_labels) for label in self.task.classes]
        elif self.task.target_type == "multiclass_classifier":
            if answer not in self.task.classes:
                raise ValueError(
                    f"Label '{answer}' is not in the list of classes: {self.task.classes}"
                )
            arr = [0 for _ in self.task.classes]
            arr[self.task.classes.index(answer)] = 1
            return arr
        elif self.task.target_type == "binary_classifier":
            classes_ = self.task.classes or ["no", "yes"]
            if answer.lower() not in set(map(str.lower, classes_)):
                raise ValueError(
                    f"Answer '{answer}' is not in the list of classes: {classes_}"
                )
            return [1.0] if answer.lower() == classes_[1].lower() else [0.0]
        elif self.task.target_type == "boi_tagger":
            errors = []
            lines = answer.strip().split("\n")
            parsed = []
            cursor = 0
            valid_tag_pattern = re.compile(r"^O$|^[BI]-(\w+)$")
            for i, line in enumerate(lines):
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    errors.append(
                        f"Line {i + 1} is malformed: '{line}' (expected 2 tab-separated fields)"
                    )
                    continue
                token, tag = parts
                if not valid_tag_pattern.match(tag):
                    errors.append(f"Line {i + 1} has invalid tag: '{tag}'")
                token, start, end = self._move_cursor(cursor, token)
                assert self.task.input_text[start:end] == token
                parsed.append((token, tag, start, end))
                cursor = end
            if errors:
                error_message = "\n".join(errors)
                raise ValueError(f"Errors in BOI tagging:\n{error_message}")
            return parsed
        else:
            raise ValueError("invalid target type")
