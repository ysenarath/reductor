from __future__ import annotations

import numpy as np
from smolagents import ToolCallingAgent

from reductor.agents import Task, TaskFinalAnswerTool, TaskPromptTemplate
from reductor.agents.models import LlamaCppModel


class ClassifierAgent:
    def __init__(
        self,
        target_type: str | None = None,
        classes: list[str] | None = None,
        max_steps: int = 10,
        temperature: float | None = None,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        # self.model = OpenAIModel(
        #     model_id="meta-llama/llama-3.2-1b-instruct",
        #     api_base="https://openrouter.ai/api/v1",
        #     api_key=os.environ["OPENROUTER_API_KEY"],
        #     temperature=temperature or 1.0,
        #     seed=seed,
        # )
        # self.model = TransformersModel(
        #     model_id="watt-ai/watt-tool-8B",
        #     temperature=temperature or 1.0,
        #     seed=seed,
        #     device_map=get_device(device),
        # )
        self.model = LlamaCppModel(
            filename="gemma-3-12b-it-q4_0.gguf",
            repo_id="google/gemma-3-12b-it-qat-q4_0-gguf",
            temperature=temperature or 1.0,
            seed=seed,
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
