from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from research_assistant.config import Settings
from research_assistant.llm import (
    call_llm,
    llm_call_with_validation,
    parse_json_response,
    retry_with_backoff,
    validate_against_schema,
)


class SimpleModel(BaseModel):
    name: str
    value: int


@pytest.fixture
def settings():
    return Settings(anthropic_api_key="test-key", llm_max_retries=2, llm_backoff_base=0.01)


class TestParseJsonResponse:
    def test_clean_json(self):
        result = parse_json_response('{"name": "test", "value": 1}')
        assert result == {"name": "test", "value": 1}

    def test_fenced_json(self):
        result = parse_json_response('```json\n{"name": "test", "value": 1}\n```')
        assert result == {"name": "test", "value": 1}

    def test_fenced_no_lang(self):
        result = parse_json_response('```\n{"name": "test", "value": 1}\n```')
        assert result == {"name": "test", "value": 1}

    def test_json_array(self):
        result = parse_json_response('[{"a": 1}, {"a": 2}]')
        assert isinstance(result, list)
        assert len(result) == 2

    def test_garbage_input(self):
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            parse_json_response("not json at all")


class TestValidateAgainstSchema:
    def test_valid(self):
        result = validate_against_schema({"name": "test", "value": 42}, SimpleModel)
        assert result.name == "test"
        assert result.value == 42

    def test_invalid(self):
        with pytest.raises(ValueError, match="Schema validation failed"):
            validate_against_schema({"name": "test"}, SimpleModel)


class TestRetryWithBackoff:
    def test_succeeds_first_try(self):
        func = MagicMock(return_value="ok")
        result = retry_with_backoff(func, max_retries=3, base=0.01)
        assert result == "ok"
        assert func.call_count == 1

    def test_retries_then_succeeds(self):
        func = MagicMock(side_effect=[ValueError("bad"), ValueError("bad"), "ok"])
        result = retry_with_backoff(func, max_retries=3, base=0.01)
        assert result == "ok"
        assert func.call_count == 3

    def test_exhausts_retries(self):
        func = MagicMock(side_effect=ValueError("always bad"))
        with pytest.raises(ValueError, match="always bad"):
            retry_with_backoff(func, max_retries=2, base=0.01)
        assert func.call_count == 2


class TestCallLLM:
    @patch("research_assistant.llm.anthropic.Anthropic")
    def test_call_llm(self, mock_anthropic_cls, settings):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"name": "test", "value": 1}')]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_client.messages.create.return_value = mock_response

        result = call_llm("prompt", "system", settings)
        assert result == '{"name": "test", "value": 1}'
        mock_client.messages.create.assert_called_once()


class TestLLMCallWithValidation:
    @patch("research_assistant.llm.call_llm")
    def test_success(self, mock_call, settings):
        mock_call.return_value = '{"name": "test", "value": 42}'
        result = llm_call_with_validation("prompt", "system", SimpleModel, settings)
        assert result.name == "test"
        assert result.value == 42

    @patch("research_assistant.llm.call_llm")
    def test_retries_on_bad_json_then_succeeds(self, mock_call, settings):
        mock_call.side_effect = [
            "not json",
            '{"name": "test", "value": 42}',
        ]
        result = llm_call_with_validation("prompt", "system", SimpleModel, settings)
        assert result.value == 42
        assert mock_call.call_count == 2
