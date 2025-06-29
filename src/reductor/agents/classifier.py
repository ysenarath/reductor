from __future__ import annotations

from typing import Callable

import numpy as np
from smolagents import ToolCallingAgent

from reductor.agents import Task, TaskFinalAnswerTool, TaskPromptTemplate
from reductor.agents.models import Model, model_factory

__all__ = [
    "ClassifierAgent",
]


class ClassifierAgent:
    def __init__(
        self,
        target_type: str | None = None,
        classes: list[str] | None = None,
        max_steps: int = 10,
        temperature: float | None = None,
        seed: int | None = None,
        device: str | None = None,
        model: Callable[[], Model] | dict | None = None,
    ) -> None:
        if isinstance(model, dict):
            model = model_factory(**model)
        elif callable(model):
            self.model = model()
        else:
            self.model = model_factory(
                "llama_cpp",
                kwargs=dict(
                    filename="gemma-3-12b-it-q4_0.gguf",
                    repo_id="google/gemma-3-12b-it-qat-q4_0-gguf",
                    temperature=temperature or 1.0,
                    seed=seed,
                    device=device,
                ),
            )
        self.classes = classes
        self.max_steps = max_steps
        if target_type is None:
            if classes is None:
                target_type = "binary_classifier"
            elif len(classes) > 1:
                target_type = "multiclass_classifier"
        self.target_type = target_type

    def predict(self, x: str | list[str]) -> np.ndarray:
        if isinstance(x, str):
            x = [x]
        outputs = []
        for text in x:
            task = Task(
                input_text=text,
                classes=self.classes,
                target_type=self.target_type,
            )
            final_answer = TaskFinalAnswerTool(task)
            agent = ToolCallingAgent(
                tools=[final_answer],
                model=self.model,
                max_steps=self.max_steps,
            )
            prompt = TaskPromptTemplate(task).render(text)
            outputs += [agent.run(prompt)]
        return np.array(outputs)

    def _predict_proba(self, text: str) -> np.ndarray:
        texts = [text for _ in range(10)]
        outputs = self.predict(texts)
        return np.mean(outputs, axis=0)

    def predict_proba(self, x: str | list[str]) -> np.ndarray:
        if isinstance(x, str):
            x = [x]
        outputs = []
        for text in x:
            outputs += [self._predict_proba(text)]
        return np.array(outputs)
