from __future__ import annotations

from typing import Any

from smolagents import Tool, ToolCallingAgent
from tklearn.kb import KnowledgeBase

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

kb = KnowledgeBase("wiktionary")


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Tool to select the final answer for and validate if the selected candidate is valid."
    inputs = {
        "answer": {"type": "integer", "description": "The final answer to the problem"}
    }
    output_type = "any"

    def __init__(self, candidates: dict[int, str]) -> None:
        super().__init__()
        self.candidates = candidates

    def forward(self, answer: int) -> Any:
        answer = int(answer)  # Ensure answer is an integer
        if answer not in self.candidates:
            candidates_str = "Definition -> Sense ID\n"
            for sense_id, definition in self.candidates.items():
                candidates_str += f'"{definition}" -> {sense_id}\n'
            raise ValueError(
                f"Answer {answer} is not a valid candidate ID. "
                "Make sure to select one of the following candidates:\n"
                f"{candidates_str}"
            )
        return answer


kb = KnowledgeBase("wiktionary")

text = "She's a pure Oreo. You know, like the cookie, black outside and white inside."

mention = list(kb.extract_mentions(text))[1]
start, end = mention.span
mention_text = text[start:end]
start_tag = "<m>"
end_tag = "</m>"
text_annotated = text[:start] + start_tag + mention_text + end_tag + text[end:]
candidates_str = "Definition -> Sense ID\n"
candidates = dict()
for candidate in mention.candidates:
    candidates_str += f'"{candidate.definition}" -> {candidate.sense_id}\n'
    candidates[candidate.sense_id] = candidate.definition
final_answer = FinalAnswerTool(candidates=candidates)

agent = ToolCallingAgent(
    tools=[final_answer],
    model=model,
    max_steps=max_steps,
)

agent.run(f"""Select the best matching id of the definition for the meaning of term annotated with <m>...</m> in the text.

Text:
{text_annotated}

Candidates:
{candidates_str}""")
