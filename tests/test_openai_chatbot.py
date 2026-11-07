"""Offline coverage for the selected lab's OpenAI chat mode."""

import ast
import io
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


from _support import load_target_module


_m = load_target_module("lab_openai")
TEMPLATE_PATH = Path(_m.__file__)


def response_stream(*events):
    manager = MagicMock()
    manager.__enter__.return_value = iter(events)
    return manager


def completed_reply(text):
    return response_stream(
        SimpleNamespace(type="response.output_text.delta", delta=text),
        SimpleNamespace(type="response.completed"),
    )


class TestOpenAIKeyConfiguration(unittest.TestCase):
    def test_rejects_bad_keys_without_exposing_values(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "synthetic-shared-key"}, clear=True):
            for value in (
                "", "mock-key", "sk-proj-your-openai-key-here", "your_api_key",
                "your-api-key", "AQ.synthetic-google-key", "AIzaSyntheticGoogleKey",
                "synthetic-shared-key", "string Trim()", "key with spaces", '"quoted-key"',
            ):
                with self.subTest(value=value):
                    with self.assertRaises(ValueError) as raised:
                        _m._validate_openai_api_key(value)
                    if value:
                        self.assertNotIn(value, str(raised.exception))
        self.assertEqual(
            _m._validate_openai_api_key("  sk-proj-synthetic-test  "),
            "sk-proj-synthetic-test",
        )

    def test_gemini_key_is_not_used_as_fallback(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "synthetic-google-key"}, clear=True):
            with self.assertRaises(ValueError):
                _m._get_openai_api_key()

    def test_prompts_privately_and_keeps_new_key_in_process(self):
        secret = "sk-proj-synthetic-private-key"
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(_m, "getpass", return_value=f" {secret} ") as prompt, \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertTrue(_m._configure_openai_api_key())
            self.assertEqual(os.environ["OPENAI_API_KEY"], secret)
            self.assertNotIn(secret, output.getvalue())
        self.assertIn("OpenAI", prompt.call_args.args[0])
        self.assertIn("hidden", prompt.call_args.args[0])

    def test_valid_environment_key_does_not_prompt_and_cancel_does_not_save(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-proj-synthetic-test"}, clear=True), \
             patch.object(_m, "getpass") as prompt:
            self.assertTrue(_m._configure_openai_api_key())
            prompt.assert_not_called()
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(_m, "getpass", side_effect=EOFError), \
             patch("sys.stdout", new_callable=io.StringIO):
            self.assertFalse(_m._configure_openai_api_key())
            self.assertNotIn("OPENAI_API_KEY", os.environ)


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-proj-synthetic-test"}, clear=True)
class TestOpenAIStreamingChatbot(unittest.TestCase):
    @patch("openai.OpenAI")
    def test_streams_selected_model_and_keeps_last_three_turns(self, MockOpenAI):
        client = MockOpenAI.return_value
        client.responses.create.side_effect = [completed_reply(f"Reply {n}") for n in range(1, 6)]
        with patch("builtins.input", side_effect=[f"Question {n}" for n in range(1, 6)] + ["quit"]), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            _m.streaming_openai_chatbot(model="synthetic-openai-model")
        requests = client.responses.create.call_args_list
        self.assertEqual(len(requests), 5)
        self.assertEqual(requests[-1].kwargs["input"], [
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Reply 2"},
            {"role": "user", "content": "Question 3"},
            {"role": "assistant", "content": "Reply 3"},
            {"role": "user", "content": "Question 4"},
            {"role": "assistant", "content": "Reply 4"},
            {"role": "user", "content": "Question 5"},
        ])
        self.assertEqual(requests[0].kwargs["model"], "synthetic-openai-model")
        self.assertTrue(requests[0].kwargs["stream"])
        self.assertFalse(requests[0].kwargs["store"])
        self.assertIn("Reply 1", output.getvalue())
        client.close.assert_called_once()

    @patch("openai.OpenAI")
    def test_unsuccessful_streams_do_not_enter_conversation_history(self, MockOpenAI):
        for terminal_events in (
            (),
            (SimpleNamespace(type="response.failed", response=SimpleNamespace(
                error=SimpleNamespace(message="Synthetic response failure"))),),
            (SimpleNamespace(type="response.incomplete", response=SimpleNamespace(
                incomplete_details=SimpleNamespace(reason="max_output_tokens"))),),
            (SimpleNamespace(type="error", message="Synthetic stream failure"),),
        ):
            with self.subTest(events=terminal_events):
                client = MagicMock()
                MockOpenAI.return_value = client
                client.responses.create.side_effect = [
                    response_stream(SimpleNamespace(type="response.output_text.delta", delta="Partial"),
                                    *terminal_events),
                    completed_reply("Recovered"),
                ]
                with patch("builtins.input", side_effect=["First", "Second", "quit"]), \
                     patch("sys.stdout", new_callable=io.StringIO) as output:
                    _m.streaming_openai_chatbot()
                self.assertEqual(client.responses.create.call_count, 2)
                self.assertEqual(client.responses.create.call_args.kwargs["input"], [
                    {"role": "user", "content": "Second"},
                ])
                self.assertIn("Recovered", output.getvalue())
                client.close.assert_called_once()

    @patch("openai.OpenAI")
    def test_unauthorized_key_can_be_replaced_privately(self, MockOpenAI):
        class UnauthorizedError(RuntimeError):
            status_code = 401

        original, replacement = MagicMock(), MagicMock()
        MockOpenAI.side_effect = [original, replacement]
        original.responses.create.side_effect = UnauthorizedError("401 Unauthorized")
        replacement.responses.create.return_value = completed_reply("Recovered with replacement")
        secret = "sk-proj-synthetic-replacement"
        with patch("builtins.input", side_effect=["First", "Second", "quit"]), \
             patch.object(_m, "getpass", return_value=secret) as prompt, \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            _m.streaming_openai_chatbot()
        prompt.assert_called_once()
        self.assertEqual(MockOpenAI.call_args_list[1].kwargs["api_key"], secret)
        self.assertNotIn(secret, output.getvalue())
        self.assertIn("Recovered with replacement", output.getvalue())
        original.close.assert_called_once()
        replacement.close.assert_called_once()

    @patch("openai.OpenAI")
    def test_quota_error_does_not_prompt_for_a_replacement_key(self, MockOpenAI):
        class QuotaError(RuntimeError):
            status_code = 429

        client = MockOpenAI.return_value
        client.responses.create.side_effect = [QuotaError("429 quota exceeded"), completed_reply("Recovered")]
        with patch("builtins.input", side_effect=["First", "Second", "exit"]), \
             patch.object(_m, "_configure_openai_api_key") as configure, \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            _m.streaming_openai_chatbot()
        configure.assert_not_called()
        self.assertIn("429", output.getvalue())
        self.assertIn("Recovered", output.getvalue())
        client.close.assert_called_once()


class TestOpenAIProviderRouting(unittest.TestCase):
    def run_entrypoint(self, arguments):
        tree = ast.parse(TEMPLATE_PATH.read_text(encoding="utf-8"))
        blocks = [node for node in tree.body if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "__name__ == '__main__'"]
        self.assertEqual(len(blocks), 1)
        code = compile(ast.Module(body=blocks, type_ignores=[]), str(TEMPLATE_PATH), "exec")
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(sys, "argv", [str(TEMPLATE_PATH), *arguments]), \
             patch.dict(_m.__dict__, {"__name__": "__main__"}), \
             patch.object(_m, "_configure_openai_api_key", return_value=True) as configure_openai, \
             patch.object(_m, "streaming_openai_chatbot") as openai_chat, \
             patch.object(_m, "_configure_gemini_api_key", return_value=True) as configure_gemini, \
             patch.object(_m, "streaming_chatbot") as gemini_chat, \
             patch.object(_m, "compare_models") as compare, \
             patch("sys.stdout", new_callable=io.StringIO), \
             patch("sys.stderr", new_callable=io.StringIO):
            exec(code, _m.__dict__)
        return configure_openai, openai_chat, configure_gemini, gemini_chat, compare

    def test_openai_provider_needs_no_gemini_key_or_comparison(self):
        configure_openai, openai_chat, configure_gemini, gemini_chat, compare = self.run_entrypoint(
            ["--provider", "openai", "--model", "synthetic-openai-model"]
        )
        configure_openai.assert_called_once()
        self.assertEqual(openai_chat.call_count, 1)
        self.assertIn("synthetic-openai-model", str(openai_chat.call_args))
        configure_gemini.assert_not_called()
        gemini_chat.assert_not_called()
        compare.assert_not_called()

    def test_default_provider_remains_gemini(self):
        configure_openai, openai_chat, configure_gemini, gemini_chat, _ = self.run_entrypoint([])
        configure_openai.assert_not_called()
        openai_chat.assert_not_called()
        configure_gemini.assert_called_once()
        gemini_chat.assert_called_once()

    def test_model_override_requires_openai_provider(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_entrypoint(["--model", "synthetic-openai-model"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
