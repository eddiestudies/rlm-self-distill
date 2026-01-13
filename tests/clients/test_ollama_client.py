from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from self_distill.clients.ollama_client import OllamaClient


@pytest.fixture
def mock_ollama():
    with patch("self_distill.clients.ollama_client.ollama") as mock:
        mock.Client.return_value = MagicMock()
        mock.AsyncClient.return_value = AsyncMock()
        yield mock


@pytest.fixture
def client(mock_ollama):
    return OllamaClient(model_name="llama3.2")


class TestOllamaClientInit:
    def test_init_with_model_name(self, mock_ollama):
        client = OllamaClient(model_name="llama3.2")
        assert client.model_name == "llama3.2"
        assert client.host == "http://localhost:11434"

    def test_init_with_custom_host(self, mock_ollama):
        client = OllamaClient(model_name="llama3.2", host="http://custom:11434")
        assert client.host == "http://custom:11434"

    def test_init_creates_clients(self, mock_ollama):
        OllamaClient(model_name="llama3.2")
        mock_ollama.Client.assert_called_once_with(host="http://localhost:11434")
        mock_ollama.AsyncClient.assert_called_once_with(host="http://localhost:11434")


class TestOllamaClientCompletion:
    def test_completion_with_string_prompt(self, client):
        client.client.chat.return_value = {
            "message": {"content": "Hello!"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        result = client.completion("Hi there")

        assert result == "Hello!"
        client.client.chat.assert_called_once_with(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hi there"}],
        )

    def test_completion_with_message_list(self, client):
        client.client.chat.return_value = {
            "message": {"content": "Response"},
            "prompt_eval_count": 15,
            "eval_count": 8,
        }

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = client.completion(messages)

        assert result == "Response"
        client.client.chat.assert_called_once_with(model="llama3.2", messages=messages)

    def test_completion_with_override_model(self, client):
        client.client.chat.return_value = {
            "message": {"content": "Response"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        client.completion("Hello", model="mistral")

        client.client.chat.assert_called_once_with(
            model="mistral",
            messages=[{"role": "user", "content": "Hello"}],
        )

    def test_completion_without_model_raises(self, mock_ollama):
        client = OllamaClient()

        with pytest.raises(ValueError, match="Model name is required"):
            client.completion("Hello")

    def test_completion_invalid_prompt_type(self, client):
        with pytest.raises(ValueError, match="Invalid prompt type"):
            client.completion(12345)


class TestOllamaClientAsyncCompletion:
    @pytest.mark.asyncio
    async def test_acompletion_with_string_prompt(self, client):
        client.async_client.chat.return_value = {
            "message": {"content": "Async hello!"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        result = await client.acompletion("Hi there")

        assert result == "Async hello!"
        client.async_client.chat.assert_called_once_with(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hi there"}],
        )

    @pytest.mark.asyncio
    async def test_acompletion_with_message_list(self, client):
        client.async_client.chat.return_value = {
            "message": {"content": "Async response"},
            "prompt_eval_count": 15,
            "eval_count": 8,
        }

        messages = [{"role": "user", "content": "Hello"}]
        result = await client.acompletion(messages)

        assert result == "Async response"

    @pytest.mark.asyncio
    async def test_acompletion_without_model_raises(self, mock_ollama):
        client = OllamaClient()

        with pytest.raises(ValueError, match="Model name is required"):
            await client.acompletion("Hello")

    @pytest.mark.asyncio
    async def test_acompletion_invalid_prompt_type(self, client):
        with pytest.raises(ValueError, match="Invalid prompt type"):
            await client.acompletion(12345)


class TestOllamaClientUsageTracking:
    def test_tracks_usage_after_completion(self, client):
        client.client.chat.return_value = {
            "message": {"content": "Response"},
            "prompt_eval_count": 100,
            "eval_count": 50,
        }

        client.completion("Hello")

        assert client.model_call_counts["llama3.2"] == 1
        assert client.model_input_tokens["llama3.2"] == 100
        assert client.model_output_tokens["llama3.2"] == 50
        assert client.model_total_tokens["llama3.2"] == 150

    def test_tracks_last_usage(self, client):
        client.client.chat.return_value = {
            "message": {"content": "Response"},
            "prompt_eval_count": 100,
            "eval_count": 50,
        }

        client.completion("Hello")

        assert client.last_prompt_tokens == 100
        assert client.last_completion_tokens == 50

    def test_accumulates_usage_across_calls(self, client):
        client.client.chat.return_value = {
            "message": {"content": "Response"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        client.completion("Hello")
        client.completion("World")

        assert client.model_call_counts["llama3.2"] == 2
        assert client.model_input_tokens["llama3.2"] == 20
        assert client.model_output_tokens["llama3.2"] == 10

    def test_get_usage_summary(self, client):
        client.client.chat.return_value = {
            "message": {"content": "Response"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        client.completion("Hello")
        summary = client.get_usage_summary()

        assert "llama3.2" in summary.model_usage_summaries
        model_summary = summary.model_usage_summaries["llama3.2"]
        assert model_summary.total_calls == 1
        assert model_summary.total_input_tokens == 10
        assert model_summary.total_output_tokens == 5

    def test_get_last_usage(self, client):
        client.client.chat.return_value = {
            "message": {"content": "Response"},
            "prompt_eval_count": 25,
            "eval_count": 15,
        }

        client.completion("Hello")
        last_usage = client.get_last_usage()

        assert last_usage.total_calls == 1
        assert last_usage.total_input_tokens == 25
        assert last_usage.total_output_tokens == 15


class TestOllamaClientModelManagement:
    def test_list_models(self, client):
        # ollama returns objects with .models attribute containing model objects with .model attribute
        mock_model1 = MagicMock()
        mock_model1.model = "llama3.2"
        mock_model2 = MagicMock()
        mock_model2.model = "mistral"

        mock_response = MagicMock()
        mock_response.models = [mock_model1, mock_model2]
        client.client.list.return_value = mock_response

        models = client.list_models()

        assert models == ["llama3.2", "mistral"]
        client.client.list.assert_called_once()

    def test_list_models_empty(self, client):
        mock_response = MagicMock()
        mock_response.models = []
        client.client.list.return_value = mock_response

        models = client.list_models()

        assert models == []

    def test_pull_model(self, client):
        client.pull_model("llama3.2")

        client.client.pull.assert_called_once_with("llama3.2")
