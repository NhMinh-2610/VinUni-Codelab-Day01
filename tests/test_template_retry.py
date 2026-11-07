"""Offline regression checks for transient Gemini streaming failures."""

import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch


from _support import load_target_module


_m = load_target_module("lab_retry")


class UnavailableError(RuntimeError):
    code = 503


def fail_during_iteration(error, chunks=()):
    yield from chunks
    raise error


class TestGeminiStreamingRetry(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.stream = self.client.models.generate_content_stream
        self.contents = [SimpleNamespace(role="user", text="Synthetic question")]
        self.sleep_patch = patch.object(_m.time, "sleep")
        self.sleep = self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)
        self.output_patch = patch("sys.stdout", new_callable=io.StringIO)
        self.output_patch.start()
        self.addCleanup(self.output_patch.stop)

    def test_retries_failure_while_creating_stream_with_same_request(self):
        reply = SimpleNamespace(text="Recovered reply")
        self.stream.side_effect = [
            UnavailableError("Model is temporarily overloaded"), [reply],
        ]

        self.assertEqual(list(_m._stream_gemini_with_retry(self.client, self.contents)), [reply])

        self.assertEqual(self.stream.call_args_list, [
            call(model=_m.GEMINI_MODEL, contents=self.contents),
            call(model=_m.GEMINI_MODEL, contents=self.contents),
        ])
        self.sleep.assert_called_once_with(2)

    def test_retries_failure_on_first_stream_iteration(self):
        reply = SimpleNamespace(text="Recovered reply")
        self.stream.side_effect = [
            fail_during_iteration(RuntimeError("503 UNAVAILABLE: high demand")),
            [reply],
        ]

        self.assertEqual(list(_m._stream_gemini_with_retry(self.client, self.contents)), [reply])

        self.assertEqual(self.stream.call_count, 2)
        self.sleep.assert_called_once_with(2)

    def test_stops_after_two_retries_and_preserves_error(self):
        error = UnavailableError("Model is temporarily overloaded")
        self.stream.side_effect = error

        with self.assertRaises(UnavailableError) as raised:
            list(_m._stream_gemini_with_retry(self.client, self.contents))

        self.assertIs(raised.exception, error)
        self.assertEqual(self.stream.call_count, 3)
        self.assertEqual(self.sleep.call_args_list, [call(2), call(4)])

    def test_partial_reply_is_not_replayed(self):
        partial = SimpleNamespace(text="Already visible")
        error = UnavailableError("503 UNAVAILABLE: stream interrupted")
        self.stream.return_value = fail_during_iteration(error, [partial])
        received = []

        with self.assertRaises(UnavailableError):
            for chunk in _m._stream_gemini_with_retry(self.client, self.contents):
                received.append(chunk)

        self.assertEqual(received, [partial])
        self.stream.assert_called_once()
        self.sleep.assert_not_called()

    def test_empty_chunk_does_not_prevent_retry(self):
        reply = SimpleNamespace(text="Recovered reply")
        self.stream.side_effect = [
            fail_during_iteration(UnavailableError("503 UNAVAILABLE"), [SimpleNamespace(text="")]),
            [reply],
        ]

        chunks = list(_m._stream_gemini_with_retry(self.client, self.contents))

        self.assertEqual("".join(chunk.text or "" for chunk in chunks), "Recovered reply")
        self.assertEqual(self.stream.call_count, 2)
        self.sleep.assert_called_once_with(2)

    def test_does_not_retry_quota_authentication_or_generic_errors(self):
        for message in (
            "429 RESOURCE_EXHAUSTED: RequestsPerDay",
            "400 API_KEY_INVALID",
            "Unexpected stream failure",
        ):
            with self.subTest(message=message):
                self.stream.reset_mock()
                self.sleep.reset_mock()
                error = RuntimeError(message)
                self.stream.side_effect = error

                with self.assertRaises(RuntimeError) as raised:
                    list(_m._stream_gemini_with_retry(self.client, self.contents))

                self.assertIs(raised.exception, error)
                self.stream.assert_called_once()
                self.sleep.assert_not_called()


@patch.dict(os.environ, {"GEMINI_API_KEY": "synthetic-gemini-test-key"}, clear=True)
class TestChatbotOverloadRecovery(unittest.TestCase):
    @patch("builtins.input", side_effect=["First failed question", "Try again", "quit"])
    @patch("google.genai.Client")
    def test_exhausted_retries_leave_chatbot_usable(self, MockClient, mock_input):
        stream = MockClient.return_value.models.generate_content_stream
        stream.side_effect = [
            UnavailableError("503 UNAVAILABLE: high demand"),
            UnavailableError("503 UNAVAILABLE: high demand"),
            UnavailableError("503 UNAVAILABLE: high demand"),
            [SimpleNamespace(text="Recovered next turn")],
        ]

        with patch.object(_m.time, "sleep") as sleep, \
             patch.object(_m, "_configure_gemini_api_key") as configure, \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            _m.streaming_chatbot()

        self.assertEqual(stream.call_count, 4)
        self.assertEqual(sleep.call_args_list, [call(2), call(4)])
        self.assertIn("overload", output.getvalue().lower())
        self.assertIn("Recovered next turn", output.getvalue())
        self.assertIn("503 UNAVAILABLE", output.getvalue())
        configure.assert_not_called()
        next_contents = stream.call_args_list[-1].kwargs["contents"]
        self.assertEqual(len(next_contents), 1)
        self.assertEqual(next_contents[0].parts[0].text, "Try again")
        MockClient.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
