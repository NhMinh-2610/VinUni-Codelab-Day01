"""Boundary cases for lab adapters, retries, batches, tables, and chat history."""

import io
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from _support import LAB_DIR, load_target_module, resolve_target_path


_m = load_target_module("lab_edge_cases")
FAKE_KEYS = {
    "OPENAI_API_KEY": "sk-proj-synthetic-lab-test",
    "GEMINI_API_KEY": "synthetic-gemini-lab-test",
    "ANTHROPIC_API_KEY": "sk-ant-synthetic-lab-test",
}


def adapter_cases():
    return (
        ("openai.OpenAI", _m.call_openai, "chat.completions.create"),
        ("google.genai.Client", _m.call_gemini, "models.generate_content"),
        ("anthropic.Anthropic", _m.call_anthropic, "messages.create"),
    )


def get_method(client, dotted_name):
    for part in dotted_name.split("."):
        client = getattr(client, part)
    return client


class TestSolutionSelection(unittest.TestCase):
    def test_relative_override_selects_repository_file(self):
        with patch.dict(os.environ, {"LAB_SOLUTION_PATH": "starter-code/template.py"}):
            self.assertEqual(resolve_target_path(), LAB_DIR / "starter-code" / "template.py")

    def test_override_cannot_escape_repository(self):
        for target in ("../outside.py", str(LAB_DIR.parent / "outside.py")):
            with self.subTest(target=target), \
                 patch.dict(os.environ, {"LAB_SOLUTION_PATH": target}), \
                 self.assertRaises(ValueError):
                resolve_target_path()

    def test_missing_override_is_reported_instead_of_silently_falling_back(self):
        with patch.dict(os.environ, {"LAB_SOLUTION_PATH": "tests/no-such-solution.py"}), \
             self.assertRaises(FileNotFoundError):
            resolve_target_path()

    def test_default_prefers_submission_over_reference_or_starter(self):
        with patch.dict(os.environ, {"LAB_SOLUTION_PATH": ""}), \
             patch.object(Path, "is_file", return_value=True):
            self.assertEqual(resolve_target_path(), LAB_DIR / "solution" / "solution.py")


@patch.dict(os.environ, FAKE_KEYS, clear=True)
class TestAdapterEdgeCases(unittest.TestCase):
    def test_missing_key_never_creates_client(self):
        for sdk_name, adapter, _ in adapter_cases():
            with self.subTest(adapter=adapter.__name__), \
                 patch.dict(os.environ, {}, clear=True), \
                 patch(sdk_name) as factory:
                with self.assertRaises(ValueError):
                    adapter("Synthetic prompt")
                factory.assert_not_called()

    def test_clients_close_and_original_errors_are_preserved(self):
        for sdk_name, adapter, method_name in adapter_cases():
            with self.subTest(adapter=adapter.__name__), patch(sdk_name) as factory:
                original = RuntimeError("Synthetic provider failure")
                get_method(factory.return_value, method_name).side_effect = original
                with self.assertRaises(RuntimeError) as raised:
                    adapter("Synthetic prompt")
                self.assertIs(raised.exception, original)
                factory.return_value.close.assert_called_once()

    def test_latency_is_measured_with_monotonic_performance_counter(self):
        responses = {
            "call_openai": SimpleNamespace(choices=[], usage=None),
            "call_gemini": SimpleNamespace(text="Reply", usage_metadata=None),
            "call_anthropic": SimpleNamespace(content=[], usage=None),
        }
        for sdk_name, adapter, method_name in adapter_cases():
            with self.subTest(adapter=adapter.__name__), patch(sdk_name) as factory, \
                 patch.object(_m.time, "perf_counter", side_effect=[100.0, 100.125]):
                get_method(factory.return_value, method_name).return_value = responses[adapter.__name__]
                _, latency, _ = adapter("Synthetic prompt")
                self.assertEqual(latency, 0.125)

    def test_gemini_google_key_fallback_and_gemini_key_precedence(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "synthetic-fallback-key"}, clear=True):
            self.assertEqual(_m._get_gemini_api_key(), "synthetic-fallback-key")
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "synthetic-primary-key", "GOOGLE_API_KEY": "synthetic-fallback-key",
        }, clear=True):
            self.assertEqual(_m._get_gemini_api_key(), "synthetic-primary-key")

    @patch("openai.OpenAI")
    def test_openai_forwards_arguments_and_handles_empty_choices(self, factory):
        factory.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[], usage=None)
        text, latency, usage = _m.call_openai(
            "Synthetic prompt", model="synthetic-model", temperature=0.2, top_p=0.6, max_tokens=77,
        )
        factory.return_value.chat.completions.create.assert_called_once_with(
            model="synthetic-model", messages=[{"role": "user", "content": "Synthetic prompt"}],
            temperature=0.2, top_p=0.6, max_tokens=77,
        )
        self.assertEqual(text, "")
        self.assertEqual(usage, {"input_tokens": 0, "output_tokens": 0})
        self.assertGreaterEqual(latency, 0.0)
        factory.return_value.close.assert_called_once()

    @patch("openai.OpenAI")
    def test_openai_normalizes_missing_text_and_usage_values(self, factory):
        for usage in (None, SimpleNamespace(prompt_tokens=None, completion_tokens=None)):
            with self.subTest(usage=usage):
                factory.reset_mock()
                factory.return_value.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=None))], usage=usage,
                )
                text, _, tokens = _m.call_openai("Synthetic prompt")
                self.assertEqual(text, "")
                self.assertEqual(tokens, {"input_tokens": 0, "output_tokens": 0})
                factory.return_value.close.assert_called_once()

    @patch("google.genai.Client")
    def test_gemini_forwards_configuration_and_handles_absent_usage(self, factory):
        factory.return_value.models.generate_content.return_value = SimpleNamespace(text=None, usage_metadata=None)
        text, _, usage = _m.call_gemini(
            "Synthetic prompt", model="synthetic-model", temperature=0.3, top_p=0.7, max_tokens=88,
        )
        kwargs = factory.return_value.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "synthetic-model")
        self.assertEqual(kwargs["contents"], "Synthetic prompt")
        self.assertEqual(kwargs["config"].temperature, 0.3)
        self.assertEqual(kwargs["config"].top_p, 0.7)
        self.assertEqual(kwargs["config"].max_output_tokens, 88)
        self.assertEqual(text, "")
        self.assertEqual(usage, {"input_tokens": 0, "output_tokens": 0})
        factory.return_value.close.assert_called_once()

    @patch("google.genai.Client")
    def test_gemini_normalizes_null_token_counts(self, factory):
        factory.return_value.models.generate_content.return_value = SimpleNamespace(
            text="Reply", usage_metadata=SimpleNamespace(prompt_token_count=None, candidates_token_count=None),
        )
        _, _, usage = _m.call_gemini("Synthetic prompt")
        self.assertEqual(usage, {"input_tokens": 0, "output_tokens": 0})

    @patch("anthropic.Anthropic")
    def test_anthropic_forwards_arguments_and_collects_all_text_blocks(self, factory):
        factory.return_value.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking="Synthetic internal block"),
                     SimpleNamespace(type="text", text="First"),
                     SimpleNamespace(type="tool_use", text=123),
                     SimpleNamespace(type="text", text=" second")],
            usage=None,
        )
        text, _, usage = _m.call_anthropic(
            "Synthetic prompt", model="synthetic-model", temperature=0.4, top_p=0.8, max_tokens=99,
        )
        factory.return_value.messages.create.assert_called_once_with(
            model="synthetic-model", messages=[{"role": "user", "content": "Synthetic prompt"}],
            temperature=0.4, top_p=0.8, max_tokens=99,
        )
        self.assertEqual(text, "First second")
        self.assertEqual(usage, {"input_tokens": 0, "output_tokens": 0})
        factory.return_value.close.assert_called_once()

    @patch("anthropic.Anthropic")
    def test_anthropic_handles_empty_content_and_null_token_counts(self, factory):
        for usage in (None, SimpleNamespace(input_tokens=None, output_tokens=None)):
            with self.subTest(usage=usage):
                factory.reset_mock()
                factory.return_value.messages.create.return_value = SimpleNamespace(content=[], usage=usage)
                text, _, tokens = _m.call_anthropic("Synthetic prompt")
                self.assertEqual(text, "")
                self.assertEqual(tokens, {"input_tokens": 0, "output_tokens": 0})
                factory.return_value.close.assert_called_once()


class TestRetryEdgeCases(unittest.TestCase):
    def test_negative_parameters_are_rejected_without_calling_function(self):
        fn = MagicMock()
        for kwargs in ({"max_retries": -1}, {"base_delay": -0.01}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _m.retry_with_backoff(fn, **kwargs)
        fn.assert_not_called()

    def test_zero_retries_calls_once_and_preserves_error(self):
        error = RuntimeError("Synthetic failure")
        fn = MagicMock(side_effect=error)
        with patch.object(_m.time, "sleep") as sleep, self.assertRaises(RuntimeError) as raised:
            _m.retry_with_backoff(fn, max_retries=0)
        self.assertIs(raised.exception, error)
        fn.assert_called_once_with()
        sleep.assert_not_called()

    def test_exhaustion_uses_bounded_attempts_and_exponential_delays(self):
        error = RuntimeError("Synthetic failure")
        fn = MagicMock(side_effect=error)
        with patch.object(_m.time, "sleep") as sleep, self.assertRaises(RuntimeError) as raised:
            _m.retry_with_backoff(fn, max_retries=2, base_delay=0.25)
        self.assertIs(raised.exception, error)
        self.assertEqual(fn.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(0.25), call(0.5)])

    def test_zero_delay_and_falsy_success_value_are_supported(self):
        fn = MagicMock(side_effect=[RuntimeError("Synthetic failure"), 0])
        with patch.object(_m.time, "sleep") as sleep:
            self.assertEqual(_m.retry_with_backoff(fn, max_retries=1, base_delay=0), 0)
        self.assertEqual(fn.call_count, 2)
        sleep.assert_called_once_with(0)

    def test_keyboard_interrupt_is_not_retried(self):
        fn = MagicMock(side_effect=KeyboardInterrupt)
        with patch.object(_m.time, "sleep") as sleep, self.assertRaises(KeyboardInterrupt):
            _m.retry_with_backoff(fn)
        fn.assert_called_once_with()
        sleep.assert_not_called()


class TestBatchAndTableEdgeCases(unittest.TestCase):
    def test_comparison_forwards_the_original_lab_models(self):
        reply = ("Reply", 0.1, {"input_tokens": 1, "output_tokens": 2})
        with patch.object(_m, "call_openai", return_value=reply) as openai, \
             patch.object(_m, "call_gemini", return_value=reply) as gemini:
            _m.compare_models("Synthetic prompt")
        self.assertEqual(openai.call_args_list, [
            call("Synthetic prompt", model="gpt-4o"),
            call("Synthetic prompt", model="gpt-4o-mini"),
        ])
        gemini.assert_called_once_with("Synthetic prompt", model="gemini-2.5-flash")

    def test_comparison_costs_use_each_selected_models_pricing(self):
        reply = ("Reply", 0.1, {"input_tokens": 10, "output_tokens": 20})
        rates = {
            "synthetic-openai": {"input": 1.0, "output": 2.0},
            "synthetic-mini": {"input": 3.0, "output": 4.0},
            "synthetic-gemini": {"input": 5.0, "output": 6.0},
            "gpt-4o": {"input": 90.0, "output": 90.0},
            "gpt-4o-mini": {"input": 90.0, "output": 90.0},
            "gemini-2.5-flash": {"input": 90.0, "output": 90.0},
        }
        with patch.object(_m, "OPENAI_MODEL", "synthetic-openai"), \
             patch.object(_m, "OPENAI_MINI_MODEL", "synthetic-mini"), \
             patch.object(_m, "GEMINI_MODEL", "synthetic-gemini"), \
             patch.object(_m, "PRICING_1M_TOKENS", rates), \
             patch.object(_m, "call_openai", return_value=reply), \
             patch.object(_m, "call_gemini", return_value=reply):
            results = _m.compare_models("Synthetic prompt")
        self.assertAlmostEqual(results["gpt4o"]["cost"], 0.00005)
        self.assertAlmostEqual(results["gpt4o_mini"]["cost"], 0.00011)
        self.assertAlmostEqual(results["gemini_flash"]["cost"], 0.00017)

    def test_comparison_reports_zero_cost_for_zero_usage(self):
        reply = ("Reply", 0.1, {"input_tokens": 0, "output_tokens": 0})
        with patch.object(_m, "call_openai", return_value=reply), \
             patch.object(_m, "call_gemini", return_value=reply):
            results = _m.compare_models("Synthetic prompt")
        for name in ("gpt4o", "gpt4o_mini", "gemini_flash"):
            self.assertEqual(results[name]["cost"], 0.0)

    def test_empty_batch_makes_no_model_calls(self):
        with patch.object(_m, "compare_models") as compare:
            self.assertEqual(_m.batch_compare([]), [])
        compare.assert_not_called()

    def test_failed_batch_item_does_not_prevent_later_success(self):
        success = {
            name: {"response": "Successful reply", "latency": 0.2, "cost": 0.0001,
                   "input_tokens": 2, "output_tokens": 3}
            for name in ("gpt4o", "gpt4o_mini", "gemini_flash")
        }
        with patch.object(_m, "compare_models", side_effect=[RuntimeError("Synthetic error"), success]) as compare:
            results = _m.batch_compare(["First prompt", "Second prompt"])
        self.assertEqual(compare.call_args_list, [call("First prompt"), call("Second prompt")])
        self.assertEqual(results[0]["prompt"], "First prompt")
        for name in success:
            self.assertIn("Synthetic error", results[0][name]["response"])
            self.assertEqual(results[0][name]["latency"], 0.0)
            self.assertEqual(results[0][name]["cost"], 0.0)
            self.assertEqual(results[0][name]["input_tokens"], 0)
            self.assertEqual(results[0][name]["output_tokens"], 0)
            self.assertEqual(results[1][name], success[name])
        self.assertEqual(results[1]["prompt"], "Second prompt")
        self.assertNotIn("prompt", success)

    def test_empty_table_still_contains_column_headers(self):
        table = _m.format_comparison_table([])
        self.assertEqual(len(table.splitlines()), 2)
        self.assertIn("Prompt", table)
        self.assertIn("Tokens (In/Out)", table)

    def test_table_escapes_pipes_backslashes_and_newlines(self):
        table = _m.format_comparison_table([{
            "prompt": "Prompt | value\\path\nnext line",
            "gpt4o": {"response": "Reply | value\\path\nnext line", "latency": 0.125,
                       "cost": 0.000001, "input_tokens": 2, "output_tokens": 4},
        }])
        self.assertEqual(len(table.splitlines()), 3)
        self.assertIn("Prompt \\| value\\\\path next line", table)
        self.assertIn("Reply \\| value\\\\path next line", table)
        self.assertIn("0.12s", table)
        self.assertIn("2/4", table)
        self.assertIn("$0.000001", table)

    def test_long_cells_are_truncated_and_none_response_is_empty(self):
        table = _m.format_comparison_table([{
            "prompt": "p" * 100,
            "gpt4o": {"response": "r" * 100},
            "gpt4o_mini": {"response": None},
        }])
        self.assertIn("p" * 37 + "...", table)
        self.assertNotIn("p" * 38, table)
        self.assertIn("r" * 47 + "...", table)
        self.assertNotIn("r" * 48, table)
        self.assertNotIn("None", table)

    def test_response_truncation_boundary(self):
        for length in (49, 50, 51):
            with self.subTest(length=length):
                table = _m.format_comparison_table([{
                    "prompt": "Synthetic prompt", "gpt4o": {"response": "r" * length},
                }])
                response_cell = table.splitlines()[-1].split("|")[3].strip()
                expected = "r" * length if length <= 50 else "r" * 47 + "..."
                self.assertEqual(response_cell, expected)

    def test_legacy_flat_results_are_supported(self):
        table = _m.format_comparison_table([{
            "prompt": "Two words", "gpt4o_response": "Old reply", "gpt4o_latency": 0.5,
            "gpt4o_cost_estimate": 0.000123,
        }])
        self.assertIn("GPT-4o", table)
        self.assertIn("Old reply", table)
        self.assertIn("0.50s", table)
        self.assertIn("3/3", table)
        self.assertIn("$0.000123", table)


@patch.dict(os.environ, {"GEMINI_API_KEY": FAKE_KEYS["GEMINI_API_KEY"]}, clear=True)
class TestGeminiHistoryEdgeCases(unittest.TestCase):
    @patch("google.genai.Client")
    def test_only_last_three_successful_turns_are_sent(self, factory):
        stream = factory.return_value.models.generate_content_stream
        stream.side_effect = [[SimpleNamespace(text=f"Reply {n}")] for n in range(1, 6)]
        with patch("builtins.input", side_effect=["", "Question 1", "Question 2", "Question 3",
                                                   "Question 4", "Question 5", "exit"]), \
             patch("sys.stdout", new_callable=io.StringIO):
            _m.streaming_chatbot()
        self.assertEqual(stream.call_count, 5)
        last_contents = stream.call_args.kwargs["contents"]
        self.assertEqual([item.role for item in last_contents], [
            "user", "model", "user", "model", "user", "model", "user",
        ])
        self.assertEqual([item.parts[0].text for item in last_contents], [
            "Question 2", "Reply 2", "Question 3", "Reply 3", "Question 4", "Reply 4", "Question 5",
        ])
        factory.return_value.close.assert_called_once()

    @patch("google.genai.Client")
    def test_empty_stream_does_not_record_an_empty_assistant_turn(self, factory):
        stream = factory.return_value.models.generate_content_stream
        stream.side_effect = [[], [SimpleNamespace(text="Recovered")]]
        with patch("builtins.input", side_effect=["First question", "Second question", "quit"]), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            _m.streaming_chatbot()
        self.assertEqual(stream.call_count, 2)
        next_contents = stream.call_args.kwargs["contents"]
        self.assertEqual(len(next_contents), 1)
        self.assertEqual(next_contents[0].parts[0].text, "Second question")
        self.assertIn("Recovered", output.getvalue())
        factory.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
