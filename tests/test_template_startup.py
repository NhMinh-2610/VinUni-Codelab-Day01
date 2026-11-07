"""Offline regressions for the selected lab's CLI and quota handling."""

import ast
import io
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


from _support import load_target_module


_m = load_target_module("lab_startup")
TEMPLATE_PATH = Path(_m.__file__)


class TestTemplateStartup(unittest.TestCase):
    def run_entrypoint(self, arguments, environment, comparison=None):
        # Execute the real entrypoint with the already loaded module's functions
        # patched, so the CLI can be checked without creating any API clients.
        tree = ast.parse(TEMPLATE_PATH.read_text(encoding="utf-8"))
        main_blocks = [
            node for node in tree.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "__name__ == '__main__'"
        ]
        self.assertEqual(len(main_blocks), 1)
        entrypoint = compile(
            ast.Module(body=main_blocks, type_ignores=[]), str(TEMPLATE_PATH), "exec"
        )
        with patch.dict(os.environ, environment, clear=True), \
             patch.object(sys, "argv", [str(TEMPLATE_PATH), *arguments]), \
             patch.dict(_m.__dict__, {"__name__": "__main__"}), \
             patch.object(_m, "compare_models", return_value=comparison or {}) as compare, \
             patch.object(_m, "_configure_gemini_api_key", return_value=False), \
             patch.object(_m, "streaming_chatbot") as chatbot, \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            exec(entrypoint, _m.__dict__)
        return output.getvalue(), compare, chatbot

    def test_default_startup_does_not_compare_with_stale_openai_key(self):
        secret = "AQ.synthetic-stale-google-key"
        output, compare, _ = self.run_entrypoint([], {
            "OPENAI_API_KEY": secret,
            "GEMINI_API_KEY": "synthetic-gemini-test-key",
        })
        compare.assert_not_called()
        self.assertNotIn(secret, output)
        self.assertIn("--compare", output)

    def test_requested_comparison_skips_google_keys_without_printing_them(self):
        for key, google_environment in (
            ("AQ.synthetic-google-key", {"GEMINI_API_KEY": "synthetic-other-key"}),
            ("AIzaSyntheticGoogleKeyForTests", {"GEMINI_API_KEY": "synthetic-other-key"}),
            ("synthetic-shared-key-without-provider-prefix", {
                "GEMINI_API_KEY": "synthetic-shared-key-without-provider-prefix",
            }),
            ("synthetic-shared-google-key", {
                "GOOGLE_API_KEY": "synthetic-shared-google-key",
            }),
        ):
            with self.subTest(key=key):
                output, compare, _ = self.run_entrypoint(["--compare"], {
                    "OPENAI_API_KEY": key,
                    **google_environment,
                })
                compare.assert_not_called()
                self.assertNotIn(key, output)
                self.assertIn("OPENAI_API_KEY", output)

    def test_requested_comparison_skips_missing_or_placeholder_key(self):
        for key in ("", "mock-key", "sk-proj-your-openai-key-here"):
            with self.subTest(key=key):
                output, compare, _ = self.run_entrypoint(["--compare"], {
                    "OPENAI_API_KEY": key,
                    "GEMINI_API_KEY": "synthetic-gemini-test-key",
                })
                compare.assert_not_called()
                self.assertIn("OPENAI_API_KEY", output)

    def test_requested_comparison_uses_distinct_openai_key(self):
        stats = {
            "gpt4o": {
                "latency": 0.1,
                "cost": 0.0001,
                "input_tokens": 1,
                "output_tokens": 2,
                "response": "Mock comparison response",
            },
        }
        output, compare, _ = self.run_entrypoint(["--compare"], {
            "OPENAI_API_KEY": "sk-proj-synthetic-openai-test-key",
            "GEMINI_API_KEY": "synthetic-gemini-test-key",
        }, comparison=stats)
        compare.assert_called_once()
        self.assertIn("Mock comparison response", output)


@patch.dict(os.environ, {"GEMINI_API_KEY": "synthetic-gemini-test-key"}, clear=True)
class TestTemplateQuotaHandling(unittest.TestCase):
    @patch("builtins.input", side_effect=["hello", "quit"])
    @patch("google.genai.Client")
    def test_banner_and_request_use_configured_model(self, MockClient, mock_input):
        stream = MockClient.return_value.models.generate_content_stream
        stream.return_value = [SimpleNamespace(text="Mock reply")]
        with patch.object(_m, "GEMINI_MODEL", "gemini-synthetic-model"), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            _m.streaming_chatbot()
        self.assertIn("Powered by Google gemini-synthetic-model", output.getvalue())
        self.assertEqual(stream.call_args.kwargs["model"], "gemini-synthetic-model")
        MockClient.return_value.close.assert_called_once()

    @patch("google.genai.Client")
    def test_daily_quota_stops_without_retry_or_replacement_prompt(self, MockClient):
        class DailyQuotaError(RuntimeError):
            code = 429

        errors = [
            RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
            DailyQuotaError("RESOURCE_EXHAUSTED: requests per day limit exceeded"),
        ]
        for error in errors:
            with self.subTest(error=str(error)):
                MockClient.reset_mock()
                client = MagicMock()
                MockClient.return_value = client
                stream = client.models.generate_content_stream
                stream.side_effect = error
                with patch("builtins.input", side_effect=["hello", "quit"]) as input_mock, \
                     patch.object(_m, "_configure_gemini_api_key") as configure, \
                     patch("sys.stdout", new_callable=io.StringIO) as output:
                    _m.streaming_chatbot()
                self.assertIn("[Daily quota exhausted]", output.getvalue())
                self.assertIn(str(error), output.getvalue())
                self.assertIn("project", output.getvalue().lower())
                self.assertIn("https://ai.dev/rate-limit", output.getvalue())
                stream.assert_called_once()
                input_mock.assert_called_once()
                configure.assert_not_called()
                client.close.assert_called_once()

    @patch("builtins.input", side_effect=["hello", "try again", "quit"])
    @patch("google.genai.Client")
    def test_minute_quota_allows_manual_retry(self, MockClient, mock_input):
        stream = MockClient.return_value.models.generate_content_stream
        stream.side_effect = [
            RuntimeError("429 RESOURCE_EXHAUSTED requests per minute limit exceeded"),
            [SimpleNamespace(text="Recovered after minute quota")],
        ]
        with patch.object(_m, "_configure_gemini_api_key") as configure, \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            _m.streaming_chatbot()
        self.assertEqual(stream.call_count, 2)
        self.assertIn("Recovered after minute quota", output.getvalue())
        self.assertNotIn("[Daily quota exhausted]", output.getvalue())
        configure.assert_not_called()
        second_contents = stream.call_args_list[1].kwargs["contents"]
        self.assertEqual(len(second_contents), 1)
        self.assertEqual(second_contents[0].parts[0].text, "try again")
        MockClient.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
