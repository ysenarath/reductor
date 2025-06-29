import torch
from smolagents.models import (
    ChatMessage,
    MessageRole,
    Model,
    OpenAIModel,
    TokenUsage,
    TransformersModel,
    parse_json_if_needed,
    remove_stop_sequences,
)
from smolagents.tools import Tool

from reductor.trainer import get_device

try:
    from llama_cpp import Llama, LlamaGrammar
except ModuleNotFoundError:
    Llama = None
    LlamaGrammar = None

try:
    from reductor.logging import get_logger
except ModuleNotFoundError:
    from logging import getLogger as get_logger

logger = get_logger(__name__)


def parse_tool_args_if_needed(message: ChatMessage) -> ChatMessage:
    for tool_call in message.tool_calls:
        tool_call.function.arguments = parse_json_if_needed(
            tool_call.function.arguments
        )
    return message


class LlamaCppModel(Model):
    def __init__(
        self,
        model_path: str | None = None,
        repo_id: str | None = None,
        filename: str | None = None,
        device: str | torch.device | None = None,
        n_ctx: int = 8192,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if Llama is None:
            raise ImportError("`llama-cpp-python` is not installed")
        self.temperature = temperature
        self.seed = seed
        device: torch.device = get_device(device)
        n_gpu_layers = (
            device.index if device.index else -1 if device.type == "cuda" else 0
        )
        if model_path:
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                max_tokens=max_tokens,
                verbose=False,
            )
        elif repo_id and filename:
            self.llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                max_tokens=max_tokens,
                verbose=False,
                **kwargs,
            )
        else:
            raise ValueError("must provide either model_path or repo_id+filename")

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        grammar: str | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """
        Generates a response from the llama.cpp model and integrates tool usage *only if tools are provided*.
        """
        try:
            generation_kwargs = self._prepare_completion_kwargs(
                messages=messages,
                stop_sequences=stop_sequences,
                grammar=grammar,
                tools_to_call_from=tools_to_call_from,
                flatten_messages_as_text=True,
                **kwargs,
            )

            if not tools_to_call_from:
                generation_kwargs.pop("tools", None)
                generation_kwargs.pop("tool_choice", None)

            generation_kwargs.setdefault("temperature", self.temperature)
            generation_kwargs.setdefault("seed", self.seed)

            filtered_kwargs = {
                k: v
                for k, v in generation_kwargs.items()
                if k
                not in [
                    "messages",
                    "stop",
                    "grammar",
                    "max_tokens",
                    "tools_to_call_from",
                    "device_map",
                ]
            }
            max_new_tokens = (
                kwargs.get("max_new_tokens")
                or kwargs.get("max_tokens")
                or self.kwargs.get("max_new_tokens")
                or self.kwargs.get("max_tokens")
            )

            if max_new_tokens:
                generation_kwargs["max_new_tokens"] = max_new_tokens

            generation_kwargs.setdefault("stop", [])

            response = self.llm.create_chat_completion(
                messages=generation_kwargs["messages"],
                stop=generation_kwargs["stop"],
                grammar=LlamaGrammar.from_string(grammar) if grammar else None,
                max_tokens=generation_kwargs.get("max_new_tokens", None),
                **filtered_kwargs,
            )

            content = response["choices"][0]["message"]["content"]

            if stop_sequences:
                content = remove_stop_sequences(content, stop_sequences)

            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                raw={
                    "out": content,
                    "completion_kwargs": {
                        key: value
                        for key, value in generation_kwargs.items()
                        if key != "inputs"
                    },
                },
                token_usage=TokenUsage(
                    input_tokens=response.get("usage", {"prompt_tokens": 0}).get(
                        "prompt_tokens", 0
                    ),
                    output_tokens=response.get("usage", {"completion_tokens": 0}).get(
                        "completion_tokens", 0
                    ),
                ),
            )
        except Exception as e:
            logger.error(f"Model error: {e}")
            return ChatMessage(role="assistant", content=f"Error: {str(e)}")


# model = OpenAIModel(
#     model_id="meta-llama/llama-3.2-1b-instruct",
#     api_base="https://openrouter.ai/api/v1",
#     api_key=os.environ["OPENROUTER_API_KEY"],
#     temperature=temperature or 1.0,
#     seed=seed,
# )

# model = TransformersModel(
#     model_id="watt-ai/watt-tool-8B",
#     temperature=temperature or 1.0,
#     seed=seed,
#     device_map=get_device(device),
# )


def model_factory(client: str, kwargs: dict | None = None):
    if kwargs is None:
        kwargs = {}
    if client == "llama_cpp":
        return LlamaCppModel(**kwargs)
    elif client == "openai":
        return OpenAIModel(**kwargs)
    return TransformersModel(**kwargs)
