"""
Day 1 — LLM API Foundation
Test suite for student solutions and master keys.

Run from the 02-lab/ folder:
    pytest tests/ -v

All external API calls are mocked — no real API keys or internet connection required.
"""

import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from _support import load_target_module


_m = load_target_module("lab_template")

# Import target functions
call_openai = getattr(_m, "call_openai")
call_gemini = getattr(_m, "call_gemini")
call_anthropic = getattr(_m, "call_anthropic")
compare_models = getattr(_m, "compare_models")
streaming_chatbot = getattr(_m, "streaming_chatbot")
retry_with_backoff = getattr(_m, "retry_with_backoff")
batch_compare = getattr(_m, "batch_compare")
format_comparison_table = getattr(_m, "format_comparison_table")


# ---------------------------------------------------------------------------
# Mock Generators
# ---------------------------------------------------------------------------
def _make_openai_response(text: str = "Hello from OpenAI"):
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    resp.usage = usage
    return resp


def _make_gemini_response(text: str = "Hello from Gemini"):
    resp = MagicMock()
    resp.text = text
    usage = MagicMock()
    usage.prompt_token_count = 12
    usage.candidates_token_count = 25
    resp.usage_metadata = usage
    return resp


def _make_anthropic_response(text: str = "Hello from Anthropic"):
    resp = MagicMock()
    content_part = MagicMock()
    content_part.text = text
    resp.content = [content_part]
    usage = MagicMock()
    usage.input_tokens = 15
    usage.output_tokens = 30
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-proj-synthetic-test"}, clear=True)
class TestCallOpenAI(unittest.TestCase):

    @patch("openai.OpenAI")
    def test_openai_returns_tuple_structure(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("OpenAI Mock Response")

        text, latency, usage = call_openai("Hello")

        self.assertEqual(text, "OpenAI Mock Response")
        self.assertIsInstance(latency, float)
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(usage["input_tokens"], 10)
        self.assertEqual(usage["output_tokens"], 20)


@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
class TestCallGemini(unittest.TestCase):

    @patch("google.genai.Client")
    def test_gemini_returns_tuple_structure_new_sdk(self, MockGenAIClient):
        mock_client = MagicMock()
        MockGenAIClient.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response("Gemini Mock Response")

        text, latency, usage = call_gemini("Hello")

        self.assertEqual(text, "Gemini Mock Response")
        self.assertIsInstance(latency, float)
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(usage["input_tokens"], 12)
        self.assertEqual(usage["output_tokens"], 25)

    @patch("google.genai.Client")
    def test_preserves_original_api_error(self, MockGenAIClient):
        error = RuntimeError("Original Gemini API error")
        MockGenAIClient.return_value.models.generate_content.side_effect = error

        with self.assertRaises(RuntimeError) as raised:
            call_gemini("Hello")

        self.assertIs(raised.exception, error)
        MockGenAIClient.return_value.models.generate_content.assert_called_once()

    @patch("google.genai.Client")
    def test_missing_key_does_not_send_request(self, MockGenAIClient):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                call_gemini("Hello")

        MockGenAIClient.assert_not_called()

    @patch("google.genai.Client")
    def test_rejects_bad_environment_values_before_sending(self, MockGenAIClient):
        for value in (
            "string Trim(Params char[] trimChars), string Trim()",
            "AIzaSy-your-gemini-key-here",
            "test key with spaces",
            '"test-key-with-literal-quotes"',
        ):
            with self.subTest(value=value), patch.dict(os.environ, {"GEMINI_API_KEY": value}):
                with self.assertRaises(ValueError) as raised:
                    call_gemini("Hello")
                self.assertNotIn(value, str(raised.exception))

        MockGenAIClient.assert_not_called()


class TestGeminiKeyConfiguration(unittest.TestCase):

    def test_recovers_from_powershell_method_without_printing_key(self):
        replacement = "AQ.synthetic-replacement-for-tests"
        with patch.dict(os.environ, {"GEMINI_API_KEY": "string Trim()"}, clear=True), \
             patch.object(_m, "getpass", return_value=f"  {replacement}  ") as prompt, \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertTrue(_m._configure_gemini_api_key())
            self.assertEqual(os.environ["GEMINI_API_KEY"], replacement)

        prompt.assert_called_once()
        self.assertIn(".Trim()", output.getvalue())
        self.assertNotIn(replacement, output.getvalue())

    def test_cancelling_key_entry_does_not_configure_a_key(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(_m, "getpass", side_effect=EOFError), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertFalse(_m._configure_gemini_api_key())
            self.assertNotIn("GEMINI_API_KEY", os.environ)

        self.assertIn("cancelled", output.getvalue())


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-synthetic-test"}, clear=True)
class TestCallAnthropic(unittest.TestCase):

    @patch("anthropic.Anthropic")
    def test_anthropic_returns_tuple_structure(self, MockAnthropic):
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _make_anthropic_response("Claude Mock Response")

        text, latency, usage = call_anthropic("Hello")

        self.assertEqual(text, "Claude Mock Response")
        self.assertIsInstance(latency, float)
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(usage["input_tokens"], 15)
        self.assertEqual(usage["output_tokens"], 30)


class TestCompareModels(unittest.TestCase):

    def test_compare_models_evaluates_correct_costs(self):
        with patch.object(_m, "call_openai") as mock_openai, \
             patch.object(_m, "call_gemini") as mock_gemini:

            # Setup mock returns
            mock_openai.side_effect = [
                ("GPT-4o Response", 0.5, {"input_tokens": 10, "output_tokens": 20}),       # 1st call: gpt-4o
                ("GPT-4o-Mini Response", 0.3, {"input_tokens": 10, "output_tokens": 20}),  # 2nd call: gpt-4o-mini
            ]
            mock_gemini.return_value = (
                "Gemini Response",
                0.4,
                {"input_tokens": 10, "output_tokens": 20}
            )

            result = compare_models("Prompt")

            # Verify keys exist
            self.assertIn("gpt4o", result)
            self.assertIn("gpt4o_mini", result)
            self.assertIn("gemini_flash", result)

            # Verify exact cost calculations
            # gpt-4o: (10 * 5.0 + 20 * 20.0) / 1,000,000 = 0.00045
            self.assertAlmostEqual(result["gpt4o"]["cost"], 0.00045, places=8)

            # gpt-4o-mini: (10 * 0.150 + 20 * 0.600) / 1,000,000 = 0.0000135
            self.assertAlmostEqual(result["gpt4o_mini"]["cost"], 0.0000135, places=8)

            # gemini-2.5-flash: (10 * 0.075 + 20 * 0.300) / 1,000,000 = 0.00000675
            self.assertAlmostEqual(result["gemini_flash"]["cost"], 0.00000675, places=8)


class TestRetryWithBackoff(unittest.TestCase):

    def test_succeeds_immediately(self):
        val = retry_with_backoff(lambda: "success")
        self.assertEqual(val, "success")

    def test_succeeds_after_retries(self):
        state = {"attempts": 0}

        def flaky_fn():
            state["attempts"] += 1
            if state["attempts"] < 3:
                raise ValueError("Fail")
            return "ok"

        val = retry_with_backoff(flaky_fn, max_retries=3, base_delay=0.001)
        self.assertEqual(val, "ok")
        self.assertEqual(state["attempts"], 3)

    def test_raises_permanent_exception(self):
        def permanent_fail():
            raise RuntimeError("Permanent")

        with self.assertRaises(RuntimeError):
            retry_with_backoff(permanent_fail, max_retries=2, base_delay=0.001)


class TestBatchCompareAndFormat(unittest.TestCase):

    def test_batch_runs_and_formats_table(self):
        with patch.object(_m, "compare_models") as mock_comp:
            def _get_mock(prompt):
                return {
                    "gpt4o": {"response": "GPT-4o response content here", "latency": 0.5, "cost": 0.00045, "input_tokens": 10, "output_tokens": 20},
                    "gpt4o_mini": {"response": "GPT-4o-Mini response content here", "latency": 0.3, "cost": 0.0000135, "input_tokens": 10, "output_tokens": 20},
                    "gemini_flash": {"response": "Gemini response content here", "latency": 0.4, "cost": 0.00000675, "input_tokens": 10, "output_tokens": 20}
                }
            mock_comp.side_effect = _get_mock

            results = batch_compare(["Q1", "Q2"])
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["prompt"], "Q1")
            self.assertEqual(results[0]["gpt4o"]["response"], "GPT-4o response content here")
            self.assertAlmostEqual(results[1]["gpt4o_mini"]["cost"], 0.0000135)

            table = format_comparison_table(results)
            self.assertIsInstance(table, str)
            self.assertIn("Prompt", table)
            self.assertIn("GPT-4o", table)
            self.assertIn("Gemini-Flash", table)
            self.assertIn("Q1", table)


@patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
class TestStreamingChatbot(unittest.TestCase):

    @patch("builtins.input", side_effect=["quit"])
    @patch("google.genai.Client")
    def test_exits_cleanly(self, MockClient, mock_input):
        # Ensure it quits cleanly without entering infinite loop
        try:
            streaming_chatbot()
        except SystemExit:
            pass

    @patch("builtins.input", side_effect=["hello", "try again", "quit"])
    @patch("google.genai.Client")
    def test_reports_stream_error_and_discards_incomplete_turn(self, MockClient, mock_input):
        def broken_stream():
            yield SimpleNamespace(text="partial reply")
            raise RuntimeError("Original streaming error")

        stream = MockClient.return_value.models.generate_content_stream
        stream.side_effect = [broken_stream(), [SimpleNamespace(text="recovered")]]
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            streaming_chatbot()

        self.assertIn("RuntimeError: Original streaming error", output.getvalue())
        self.assertNotIn("google.generativeai", output.getvalue())
        self.assertEqual(stream.call_count, 2)
        second_contents = stream.call_args_list[1].kwargs["contents"]
        self.assertEqual(len(second_contents), 1)
        self.assertEqual(second_contents[0].parts[0].text, "try again")
        MockClient.return_value.close.assert_called_once()

    @patch("google.genai.Client")
    def test_missing_key_does_not_create_client(self, MockClient):
        with patch.dict(os.environ, {}, clear=True), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            streaming_chatbot()

        self.assertIn("GEMINI_API_KEY", output.getvalue())
        MockClient.assert_not_called()

    @patch("builtins.input", side_effect=["hello", "hello again", "quit"])
    @patch("google.genai.Client")
    def test_replaces_rejected_key_and_uses_new_client(self, MockClient, mock_input):
        original_client = MagicMock()
        replacement_client = MagicMock()
        MockClient.side_effect = [original_client, replacement_client]
        original_client.models.generate_content_stream.side_effect = RuntimeError("400 API_KEY_INVALID")
        replacement_client.models.generate_content_stream.return_value = [SimpleNamespace(text="recovered")]
        replacement_key = "AQ.synthetic-replacement-for-tests"

        with patch.object(_m, "getpass", return_value=replacement_key) as prompt, \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            streaming_chatbot()
            self.assertEqual(os.environ["GEMINI_API_KEY"], replacement_key)

        prompt.assert_called_once()
        self.assertEqual(MockClient.call_args_list[1].kwargs["api_key"], replacement_key)
        original_client.models.generate_content_stream.assert_called_once()
        replacement_client.models.generate_content_stream.assert_called_once()
        self.assertIn("recovered", output.getvalue())
        self.assertNotIn(replacement_key, output.getvalue())
        original_client.close.assert_called_once()
        replacement_client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
