from __future__ import annotations
from typing import Any

from smolagents import ToolCallingAgent, Tool

from reductor.agents.models import get_device, model_factory

device = get_device()
max_steps = 20

model = model_factory(
    "llama_cpp",
    kwargs=dict(
        filename="gemma-3-12b-it-q4_0.gguf",
        repo_id="google/gemma-3-12b-it-qat-q4_0-gguf",
        temperature=1.0,
        seed=42,
        device=device,
    ),
)


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {
        "answer": {"type": "any", "description": "The final answer to the problem"}
    }
    output_type = "any"

    def __init__(self) -> None:
        super().__init__()

    def forward(self, answer: str) -> Any:
        return answer


final_answer = FinalAnswerTool()

agent = ToolCallingAgent(
    tools=[final_answer],
    model=model,
    max_steps=max_steps,
)

agent.run("What is the capital of France?")
