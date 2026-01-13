from collections import defaultdict
from typing import Any

import ollama

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class OllamaClient(BaseLM):
    """
    LM Client for running models with Ollama locally.
    Ollama allows running LLMs locally on your machine.
    """

    def __init__(
        self,
        model_name: str | None = None,
        host: str | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.model_name = model_name
        self.host = host or "http://localhost:11434"

        # Initialize clients
        self.client = ollama.Client(host=self.host)
        self.async_client = ollama.AsyncClient(host=self.host)

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)

        # Last call tracking
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0

    def completion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(
            isinstance(item, dict) for item in prompt
        ):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Ollama client.")

        response = self.client.chat(model=model, messages=messages)
        self._track_usage(response, model)
        return response["message"]["content"]

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(
            isinstance(item, dict) for item in prompt
        ):
            messages = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Ollama client.")

        response = await self.async_client.chat(model=model, messages=messages)
        self._track_usage(response, model)
        return response["message"]["content"]

    def _track_usage(self, response: dict, model: str):
        """Track token usage from Ollama response."""
        self.model_call_counts[model] += 1

        # Ollama provides prompt_eval_count and eval_count for token tracking
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        self.model_input_tokens[model] += prompt_tokens
        self.model_output_tokens[model] += completion_tokens
        self.model_total_tokens[model] += total_tokens

        # Track last call for handler to read
        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )

    def list_models(self) -> list[str]:
        """List all available models in Ollama."""
        response = self.client.list()
        return [model.model for model in response.models]

    def pull_model(self, model_name: str) -> None:
        """Pull a model from Ollama registry."""
        self.client.pull(model_name)
