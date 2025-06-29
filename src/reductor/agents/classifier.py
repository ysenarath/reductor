from __future__ import annotations

from typing import Callable

import numpy as np
from smolagents import ToolCallingAgent

from reductor.agents.base import Task, TaskFinalAnswerTool, TaskPromptTemplate
from reductor.agents.models import Model, model_factory

__all__ = [
    "ClassifierAgent",
]


class ClassifierAgent:
    """
    A classification agent that utilizes a model to predict class labels or probabilities
    for given input text. Supports binary and multiclass classification.

    Parameters
    ----------
    target_type : str, optional
        The type of classification task. Can be "binary_classifier" or "multiclass_classifier".
        If not provided, it will be inferred based on the `classes` parameter.
    classes : list of str, optional
        A list of class labels for the classification task. If `None`, defaults to binary classification.
    max_steps : int, optional
        The maximum number of steps the agent can take during inference. Default is 10.
    temperature : float, optional
        The temperature parameter for the model, controlling randomness in predictions. Default is `None`.
    seed : int, optional
        The random seed for reproducibility. Default is `None`.
    device : str, optional
        The device to run the model on (e.g., "cpu", "cuda"). Default is `None`.
    model : Callable or dict, optional
        A callable that returns a model instance or a dictionary of model configuration parameters.
        If `None`, a default model is created using `model_factory`.

    Attributes
    ----------
    model : Model
        The underlying model used for predictions.
    classes : list of str
        The class labels for the classification task.
    max_steps : int
        The maximum number of steps the agent can take during inference.
    target_type : str
        The type of classification task.

    Methods
    -------
    predict(x)
        Predicts class labels for the given input text(s).
    predict_proba(x)
        Predicts class probabilities for the given input text(s).

    Examples
    --------
    >>> agent = ClassifierAgent(classes=["positive", "negative"], max_steps=5)
    >>> predictions = agent.predict(["This is great!", "This is terrible."])
    >>> probabilities = agent.predict_proba(["This is great!", "This is terrible."])
    """

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
        """
        Predicts the output for the given input(s) using a tool-calling agent.

        Parameters
        ----------
        x : str or list of str
            Input text(s) to be classified. Can be a single string or a list of strings.

        Returns
        -------
        np.ndarray
            An array containing the predictions for each input text.

        Notes
        -----
        The method processes each input text by creating a `Task` object with the specified
        classes and target type. It then uses a `ToolCallingAgent` with a `TaskFinalAnswerTool`
        to generate predictions based on the rendered prompt. The predictions are collected
        and returned as a NumPy array.
        """
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
        """
        Predict the probability distribution for a given text.

        This method generates predictions for the input text by creating multiple
        copies of the text, passing them through the `predict` method, and then
        averaging the outputs to produce a probability distribution.

        Parameters
        ----------
        text : str
            The input text for which the probability distribution is to be predicted.

        Returns
        -------
        np.ndarray
            A numpy array representing the averaged probability distribution
            across multiple predictions.
        """
        texts = [text for _ in range(10)]
        outputs = self.predict(texts)
        return np.mean(outputs, axis=0)

    def predict_proba(self, x: str | list[str]) -> np.ndarray:
        """
        Predict the probability distribution for the given input(s).

        Parameters
        ----------
        x : str or list of str
            Input text(s) for which the probability distribution is to be predicted.
            If a single string is provided, it will be converted into a list containing
            that string.

        Returns
        -------
        np.ndarray
            A NumPy array containing the predicted probability distributions for each
            input text. Each element in the array corresponds to the probability
            distribution for a single input text.
        """
        if isinstance(x, str):
            x = [x]
        outputs = []
        for text in x:
            outputs += [self._predict_proba(text)]
        return np.array(outputs)
