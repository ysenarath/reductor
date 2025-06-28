from __future__ import annotations

import os

from smolagents import OpenAIModel, ToolCallingAgent
from reductor.agents import Task, TaskFinalAnswerTool, TaskPromptTemplate


def multilabel_classifier():
    model = OpenAIModel(
        model_id="openai/gpt-4.1-nano",
        api_base="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    # model = TransformersModel(
    #     model_id="microsoft/phi-4",
    #     device_map="mps",
    #     max_new_tokens=512,
    # )
    task = Task(
        "This is a news article about the latest advancements in AI technology and the relationship between AI and healthcare.",
        classes=[
            "sports",
            "politics",
            "entertainment",
            "technology",
            "health",
        ],
        target_type="multilabel_classifier",
    )
    agent = ToolCallingAgent(
        tools=[
            TaskFinalAnswerTool(task),
        ],
        model=model,
        max_steps=10,
    )
    prompt = TaskPromptTemplate(task).render()
    result = agent.run(prompt)
    print("Result:", result)


def multiclass_classifier():
    model = OpenAIModel(
        model_id="openai/gpt-4.1-nano",
        api_base="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    # model = TransformersModel(
    #     model_id="microsoft/phi-4",
    #     device_map="mps",
    #     max_new_tokens=512,
    # )
    task = Task(
        classes=[
            "sports",
            "politics",
            "entertainment",
            "technology",
            "health",
        ],
        target_type="multiclass_classifier",
    )
    agent = ToolCallingAgent(
        tools=[
            TaskFinalAnswerTool(task),
        ],
        model=model,
        max_steps=10,
    )
    template = TaskPromptTemplate(task)
    prompt = template.render(
        input_text="This is a news article about the latest advancements in AI technology and the relationship between AI and healthcare."
    )
    result = agent.run(prompt)
    print("Result:", result)


def binary_classifier():
    model = OpenAIModel(
        model_id="openai/gpt-4.1-nano",
        api_base="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    # model = TransformersModel(
    #     model_id="microsoft/phi-4",
    #     device_map="mps",
    #     max_new_tokens=512,
    # )
    task = Task(target_type="binary_classifier")
    agent = ToolCallingAgent(
        tools=[
            TaskFinalAnswerTool(task),
        ],
        model=model,
        max_steps=10,
    )
    template = TaskPromptTemplate(
        task,
        instruction="Answer with 'yes' if the input text is positive, otherwise answer with 'no'.",
    )
    prompt = template.render(
        "this is a cool movie, I really liked it! It was great and I enjoyed it a lot."
    )
    result = agent.run(prompt)
    print("Result:", result)
    prompt = template.render(
        "this is a bad movie, I really disliked it! It was terrible and I hated it."
    )
    result = agent.run(prompt)
    print("Result:", result)


def boi_tagger():
    model = OpenAIModel(
        model_id="openai/gpt-4.1-nano",
        api_base="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    # model = TransformersModel(
    #     model_id="microsoft/phi-4",
    #     device_map="mps",
    #     max_new_tokens=512,
    # )
    task = Task(
        "Alice is a software engineer at OpenAI. She works on AI projects.",
        target_type="boi_tagger",
    )
    agent = ToolCallingAgent(
        tools=[
            TaskFinalAnswerTool(task),
        ],
        model=model,
        max_steps=10,
    )
    return agent.run(TaskPromptTemplate(task).render(task.input_text))


def hate_speech_classifier():
    model = OpenAIModel(
        model_id="openai/gpt-4.1-nano",
        api_base="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    task = Task(
        input_text="It's the individual that who did the crime not their whole race.",
        classes=["not_hate_speech", "hate_speech", "counter_speech"],
        target_type="multiclass_classifier",
    )
    agent = ToolCallingAgent(
        tools=[
            TaskFinalAnswerTool(task),
        ],
        model=model,
        max_steps=10,
    )
    print(
        "Result:",
        agent.run(TaskPromptTemplate(task).render(task.input_text)),
    )
