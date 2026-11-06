"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from getpass import getpass
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


def _validate_gemini_api_key(value: str, source: str = "GEMINI_API_KEY") -> str:
    """Reject common input mistakes without displaying the credential."""
    value = value.strip()
    if not value:
        raise ValueError(f"{source} is empty. Paste the key copied from Google AI Studio.")

    lowered = value.lower()
    if "trim(" in lowered or "psmethod" in lowered:
        raise ValueError(
            f"{source} contains a PowerShell method instead of an API key. "
            "Use .Trim() with parentheses, or enter the key when prompted below."
        )
    if "your-gemini-key-here" in lowered or lowered in {
        "mock-key", "gemini_api_key", "your_api_key", "your-api-key",
    }:
        raise ValueError(f"{source} is an example value. Paste your actual Gemini key.")
    if any(char.isspace() for char in value) or any(char in value for char in "\"'"):
        raise ValueError(f"{source} contains spaces or quote characters. Copy the complete key again.")
    return value


def _get_gemini_api_key() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(name)
        if value:
            return _validate_gemini_api_key(value, source=name)
    raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY before calling Gemini.")


def _configure_gemini_api_key(force_prompt: bool = False) -> bool:
    """Read a replacement key privately and retain it only for this process."""
    if not force_prompt:
        try:
            _get_gemini_api_key()
            return True
        except ValueError as error:
            print(f"[Configuration error] {error}")

    try:
        api_key = _validate_gemini_api_key(getpass("Paste Gemini API key (input hidden): "))
    except (KeyboardInterrupt, EOFError):
        print("\nAPI key entry cancelled.")
        return False
    except ValueError as error:
        print(f"[Configuration error] {error}")
        return False

    os.environ["GEMINI_API_KEY"] = api_key
    return True


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = "mock-key"
        
    client = OpenAI(api_key=api_key)
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time
    
    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    
    return text, latency, usage



# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.
    Uses the google-genai SDK and preserves API errors for diagnosis.
    """
    api_key = _get_gemini_api_key()

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    start_time = time.time()
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start_time

    text = response.text or ""
    metadata = response.usage_metadata
    usage = {
        "input_tokens": (metadata.prompt_token_count or 0) if metadata else 0,
        "output_tokens": (metadata.candidates_token_count or 0) if metadata else 0,
    }
    return text, latency, usage



# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    import anthropic
    
    api_key = os.getenv("ANTHROPIC_API_KEY") or "mock-key"
    client = anthropic.Anthropic(api_key=api_key)
    
    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start_time
    
    text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens": response.usage.input_tokens if response.usage else 0,
        "output_tokens": response.usage.output_tokens if response.usage else 0,
    }
    
    return text, latency, usage



# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.
    """
    # Call GPT-4o
    gpt4o_text, gpt4o_lat, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    gpt4o_cost = (
        gpt4o_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["input"] +
        gpt4o_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["output"]
    ) / 1_000_000
    
    # Call GPT-4o-mini
    mini_text, mini_lat, mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    mini_cost = (
        mini_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["input"] +
        mini_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["output"]
    ) / 1_000_000
    
    # Call Gemini 2.5 Flash
    gemini_text, gemini_lat, gemini_usage = call_gemini(prompt, model=GEMINI_MODEL)
    gemini_cost = (
        gemini_usage["input_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["input"] +
        gemini_usage["output_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["output"]
    ) / 1_000_000
    
    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_lat,
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"]
        },
        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_lat,
            "cost": mini_cost,
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"]
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_lat,
            "cost": gemini_cost,
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"]
        }
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.
    Maintains the last 3 turns of conversation history for context.
    """
    try:
        api_key = _get_gemini_api_key()
    except ValueError as error:
        print(f"[Configuration error] {error}")
        return

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    print("\n\033[94m==============================================")
    print("🤖 Vin Smart Future — Intelligent Chat Assistant")
    print("Powered by Google Gemini 2.5 Flash")
    print("Type 'quit' or 'exit' to end the session.")
    print("==============================================\033[0m\n")
    
    # Store history as a list of dictionaries with 'role' and 'text'
    # Gemini 2.5 structure generally expects 'user' and 'model' roles.
    history = []
    
    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print("\033[93mGoodbye!\033[0m")
                break
                
            # Compile messages from history (trimmed to last 3 turns / 6 messages)
            messages_to_send = []
            for h in history[-6:]:
                messages_to_send.append(h)
            messages_to_send.append({"role": "user", "text": user_input})
            
            print("\033[92mGemini:\033[0m ", end="", flush=True)
            
            formatted_contents = [
                types.Content(
                    role=msg["role"],
                    parts=[types.Part.from_text(text=msg["text"])],
                )
                for msg in messages_to_send
            ]
            response_stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=formatted_contents,
            )
            full_reply = ""
            for chunk in response_stream:
                chunk_text = chunk.text or ""
                print(chunk_text, end="", flush=True)
                full_reply += chunk_text
                    
            print("\n")
            
            # Record the turn in history
            history.append({"role": "user", "text": user_input})
            history.append({"role": "model", "text": full_reply})
            history = history[-6:]
            
        except (KeyboardInterrupt, EOFError):
            print("\n\033[93mSession interrupted. Goodbye!\033[0m")
            break
        except Exception as e:
            print(f"\n\033[91m[Error Calling API] {type(e).__name__}: {e}\033[0m\n")
            if "API_KEY_INVALID" in str(e):
                print("Google rejected this key. Copy a new key from https://aistudio.google.com/apikey.")
                if not _configure_gemini_api_key(force_prompt=True):
                    break
                client.close()
                client = genai.Client(api_key=_get_gemini_api_key())

    client.close()



# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        ValueError: If max_retries or base_delay is negative.
        The last exception raised by fn() after all retries are exhausted.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if base_delay < 0:
        raise ValueError("base_delay must be non-negative")

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
        Failed comparisons contain an error response and zero metrics for
        each model, so subsequent prompts are still processed.
    """
    results = []
    for prompt in prompts:
        try:
            comparison = compare_models(prompt)
        except Exception as exc:
            comparison = {
                model: {
                    "response": f"Error: {exc}",
                    "latency": 0.0,
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
                for model in ("gpt4o", "gpt4o_mini", "gemini_flash")
            }
        results.append({**comparison, "prompt": prompt})
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    def format_cell(text: str, max_length: int) -> str:
        text = " ".join(text.split())
        if len(text) > max_length:
            text = text[:max_length - 3] + "..."
        return text.replace("\\", "\\\\").replace("|", "\\|")

    table_lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    models = [
        ("GPT-4o", "gpt4o", "gpt4o_response", "gpt4o"),
        ("GPT-4o-Mini", "gpt4o_mini", "mini_response", "mini"),
        ("Gemini-Flash", "gemini_flash", "gemini_response", "gemini"),
    ]

    for result in results:
        original_prompt = result["prompt"]
        prompt = format_cell(original_prompt, 40)
        for name, key, legacy_key, legacy_prefix in models:
            data = result.get(key, result.get(legacy_key))
            if data is None:
                continue

            if isinstance(data, dict):
                response = str(data.get("response", "") or "")
                latency = data.get("latency", 0.0)
                input_tokens = data.get("input_tokens", 0)
                output_tokens = data.get("output_tokens", 0)
                cost = data.get("cost", 0.0)
            else:
                # Older results store responses and metrics in separate keys.
                response = str(data)
                latency = result.get(f"{legacy_prefix}_latency", 0.0)
                input_tokens = int(len(original_prompt.split()) * 1.5)
                output_tokens = int(len(response.split()) * 1.5)
                cost = result.get(f"{legacy_prefix}_cost_estimate", 0.0)

            response_cell = format_cell(response, 50)
            table_lines.append(
                f"| {prompt} | {name} | {response_cell} | {latency:.2f}s | "
                f"{input_tokens}/{output_tokens} | ${cost:.6f} |"
            )

    return "\n".join(table_lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_api_key in ("", "mock-key", "sk-proj-your-openai-key-here"):
        print("Skipping model comparison: set a real OPENAI_API_KEY to compare models.")
        print("You can still use the Gemini chatbot below with GEMINI_API_KEY.")
    else:
        try:
            result = compare_models(test_prompt)
            for model_name, stats in result.items():
                print(f"\n[{model_name.upper()}]")
                print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
                print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
                print(f"Response: {stats['response']}")
        except Exception as e:
            print(f"Skipping live API comparison test: {e}")
            print("Set valid API keys to run model comparison.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        if _configure_gemini_api_key():
            streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
