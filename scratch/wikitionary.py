"""
Note: The LLMs are not able to answer this question correctly as of May 2025 unless
we specifically instruct them to focus on the metaphorical meaning of the term
if applicable. Local model still fails to answer this question correctly
even with this additional instruction due to overthinking.

Note: LLMs are not able to "focus" on the metaphorical meaning of the term "Oreo"
in the context of the sentence and instead focus on the literal meaning
of the term "Oreo" as a cookie.
"""

from __future__ import annotations

import os
from typing import Any

from smolagents import Tool, ToolCallingAgent
from tklearn.kb import KnowledgeBase

from reductor.agents.models import get_device, model_factory


class ValidateAnswerTool(Tool):
    name: str = "validate_answer"
    description: str = (
        "Tool to select the validate your answer. "
        " This is a required step before selecting the final answer."
        " If you don't validate your answer, you won't be able to select the final answer."
        " If you decided to change your answer, before final answer you must revalidate it with this tool before selecting the final answer."
    )
    inputs: dict = {
        "sense_id": {
            "type": "integer",
            "description": "The final answer to the problem",
        },
        "explanation": {
            "type": "string",
            "description": "Explanation of why this is the correct answer.",
        },
    }
    output_type: str = "string"

    def __init__(self, text: str, candidates: dict[int, str]) -> None:
        super().__init__()
        self.text = text
        self.candidates = candidates
        self.validated_answer = None
        self.explanation = None

    def forward(self, sense_id: int, explanation: str) -> str:
        sense_id = int(sense_id)  # Ensure answer is an integer
        if sense_id not in self.candidates:
            candidates_str = "Definition -> Sense ID\n"
            for sense_id, definition in self.candidates.items():
                candidates_str += f'"{definition}" -> {sense_id}\n'
            raise ValueError(
                f"Answer {sense_id} is not a valid candidate ID. "
                "Make sure to select one of the following candidates:\n"
                f"{candidates_str}"
            )
        selected_definition = self.candidates[sense_id]
        self.validated_answer = sense_id
        self.explanation = explanation
        return (
            f'Text: "{self.text}"\n'
            f"Selected Sense ID: {sense_id}.\n"
            f'Selected Definition: "{selected_definition}"\n'
            "If you think this is incorrect answer, you can revalidate another answer "
            "with the 'validate_answer' tool. "
            "If you think this is the correct answer, you can now select the final answer "
            "using the 'final_answer' tool."
        )


class FinalAnswerTool(Tool):
    name: str = "final_answer"
    description: str = (
        "Tool to select the final answer after validate through 'validate_answer' tool."
    )
    inputs: dict = {
        "sense_id": {
            "type": "integer",
            "description": "The final answer to the problem",
        }
    }
    output_type: str = "object"

    def __init__(self, validator: ValidateAnswerTool) -> None:
        super().__init__()
        self.validator = validator

    def forward(self, sense_id: int) -> Any:
        if self.validator.validated_answer is None:
            raise ValueError(
                "You must validate your answer with the validate_answer tool before selecting the final answer."
            )
        if sense_id != self.validator.validated_answer:
            raise ValueError(
                f"Selected sense ID {sense_id} does not match the validated answer {self.validator.validated_answer}."
                f" Please revalidate your answer with the validate_answer tool again before selecting the final answer."
            )
        selected_definition = self.validator.candidates[sense_id]
        return {"sense_id": sense_id, "definition": selected_definition}


def main(max_steps: int = 20, device: Any = get_device()):
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
    # model = model_factory(
    #     "openai",
    #     kwargs=dict(
    #         model_id="openai/gpt-4o-mini",
    #         api_base="https://openrouter.ai/api/v1",
    #         api_key=os.environ["OPENROUTER_API_KEY"],
    #         temperature=1.0,
    #         seed=42,
    #     ),
    # )

    kb = KnowledgeBase("wiktionary")

    text = (
        "She's a pure Oreo. You know, like the cookie, black outside and white inside."
    )

    mention = list(kb.extract_mentions(text))[1]
    start, end = mention.span
    mention_text = text[start:end]
    start_tag = "<m>"
    end_tag = "</m>"
    annotated_text = text[:start] + start_tag + mention_text + end_tag + text[end:]
    # candidates_str = "Definition -> Sense ID\n"
    candidates_str = "Sense ID -> Definition\n"
    candidates = dict()
    for candidate in mention.candidates:
        # candidates_str += f'"{candidate.definition}" -> {candidate.sense_id}\n'
        candidates_str += f'{candidate.sense_id} -> "{candidate.definition}"\n'
        candidates[candidate.sense_id] = candidate.definition

    validate_answer = ValidateAnswerTool(text=annotated_text, candidates=candidates)
    final_answer = FinalAnswerTool(validator=validate_answer)

    agent = ToolCallingAgent(
        tools=[validate_answer, final_answer],
        model=model,
        max_steps=max_steps,
    )

    agent.run(f"""Select the best matching sense id for the meaning of term annotated with <m>...</m> in the text.
              
You must take the decision by thinking of a similar term of the selected sense and replacing that in the provided text and see if it still makes sense. Provide examples and counter examples in your explanation.
            
If the term has a metaphorical meaning, you must focus on that metaphorical meaning and not the literal meaning of the term.

Text:
{annotated_text}

Candidates:
{candidates_str}""")


if __name__ == "__main__":
    main()
