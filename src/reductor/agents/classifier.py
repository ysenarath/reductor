from __future__ import annotations

import os

from smolagents import OpenAIModel, ToolCallingAgent
from reductor.agents import Task, TaskFinalAnswerTool, TaskPromptTemplate


class LLMClassifier:
    def __init__(
        self,
        target_type: str = "binary_classifier",
        classes: list[str] | None = None,
        max_steps: int = 10,
    ) -> None:
        self.model = OpenAIModel(
            model_id="openai/gpt-4.1-nano",
            api_base="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.classes = classes
        self.target_type = target_type
        self.max_steps = max_steps

    def predict(self, text: str) -> str:
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
        return agent.run(prompt)
