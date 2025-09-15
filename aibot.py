#!/usr/bin/env python3
"""
Ollama CLI – word‑by‑word streaming

Features
--------
  `-list` – List all Ollama models
  `no_md` - Disabled rich Markdown output rendering(Faster raw output, no waits
  `-model` – Choose a model (falls back to env var or first available)
  Prompt can be given on the command line or piped in via stdin
  The answer is streamed word‑by‑word
  `-system` – Pass a system prompt that will be sent before the user prompt
  `-num_ctx` – Number context size (default 8192)
  `-temp` – Sampling temperature (default 0.6)
  `-top_k` - Limit the next‑token choice to the `k` highest‑probability tokens when sampling (default: 40)
  `top_p` - Keep the smallest set of tokens whose cumulative probability ≥ p (default: 1)
  `min_p` - Exclude any token whose probability is below the min_p threshold (default: 0)
  `append_p, -p` - Append prompt, useful when you use stdin for a prompt but you want to add/ask something
"""

import os
import sys
import json
import argparse
import requests
from typing import List, Optional, Iterator

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_URL = "http://192.168.1.11:11434"   # <- change if your server is elsewhere

SYSTEM_MESSAGE = """You are a highly experienced and knowledgeable Linux System Administrator and Shell Scripting Expert. 
Your role is to assist users with all aspects of Linux system administration, including but not limited to:
BUT the IMPORTANT part is the next line:
For example if a version of a program or kernel version isn't provided in the context don't provide that information, same with the linux distribution.

-   **Troubleshooting:** Diagnosing and resolving system issues, network problems, and application errors.
-   **Configuration Management:** Guiding users through configuring services, daemons, and system settings.
-   **User and Permission Management:** Providing instructions for managing users, groups, and file permissions.
-   **Package Management:** Assisting with installing, updating, and removing software packages using various package managers (apt, yum, dnf, pacman, etc.).
-   **Shell Scripting:** Generating, debugging, and explaining shell scripts (Bash, Zsh, etc.) for automation and task execution.
-   **Security:** Offering advice on securing Linux systems, including firewall configuration, user security, and vulnerability mitigation.
-   **Performance Tuning:** Suggesting methods to optimize system performance and resource utilization.
-   **Documentation and Explanation:** Clearly explaining Linux concepts, commands, and best practices.
-   **When unsure, say: "I don’t know about that information" or "This cannot be confirmed."**
-   **Avoid hallucinations:**
-   **Do not fabricate data, names, dates, events, studies, or quotes**
-   **Do not simulate sources or cite imaginary articles**
-   **When asked for evidence, only refer to known inforamtion**

You are proficient in working with various Linux distributions, including Gentoo, Debian, Fedora, Arch Linux, and Red Hat Enterprise Linux.
You have to analyze the context provided and be practical in the response.
Your responses should be accurate, concise, and easy to understand, providing practical solutions and clear instructions.
When generating scripts or commands, prioritize security and best practices.
"""

# Rich helpers
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.markdown import Markdown
except ImportError:
    print("You need the `rich` package (pip install rich).")
    sys.exit(1)

console = Console()

# Ollama utilities
def list_models() -> List[str]:
    """Return a list of available Ollama models (by name)."""
    try:
        resp = requests.get(f"{BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Failed to reach Ollama server at {BASE_URL}") from exc

    data = resp.json()
    tags = data.get("tags") or data.get("models")
    if not isinstance(tags, list):
        raise RuntimeError("Unexpected response format from /api/tags")
    return [t["name"] for t in tags if isinstance(t, dict) and "name" in t]

def pick_model(user_specified: Optional[str]) -> str:
    """Resolve the model name that should be used."""
    if user_specified:
        return user_specified

    env_model = os.getenv("OLLAMA_MODEL")
    if env_model:
        return env_model

    # Fallback: first model returned by the API
    try:
        return list_models()[0]
    except Exception as exc:
        raise RuntimeError("No models available to pick") from exc


# Helpers for streaming
def json_stream(resp: requests.Response) -> Iterator[dict]:
    """Yield JSON objects from a line‑oriented response."""
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        try:
            yield json.loads(raw_line)
        except json.JSONDecodeError:
            continue

def stream_response(
    resp: requests.Response,
    no_md: bool,
    chatmode: bool
) -> str:
    """
    Print response for ollama
    """
    buffer = ""

    if no_md:
        # Plain‑text mode – just echo tokens as they arrive
        for chunk in json_stream(resp):

            if not chatmode:
                buffer = chunk.get("response", "")
            else:
                buffer = chunk.get("message", {}).get("content", "")

            if buffer:
                console.print(buffer, end="", style="green")
    else:
        # Markdown mode – keep updating a Live panel
        with Live(Markdown(""), console=console, refresh_per_second=4) as live:
            for chunk in json_stream(resp):

                if not chatmode:
                    token = chunk.get("response", "")
                else:
                    token = chunk.get("message", {}).get("content", "")
                if token:
                    buffer += token
                    live.update(Markdown(buffer))

    # Trailing newline keeps the console tidy
    console.print()

    eval_count = chunk.get("eval_count", 0)
    eval_duration = chunk.get("eval_duration", 0)
    if eval_duration > 0:
        tokens_per_second = (eval_count / eval_duration) * 1_000_000_000
    console.print(f"[bold cyan on blue]Tokens per second: {tokens_per_second:.2f}[/bold cyan on blue]")
    return buffer

# Token‑by‑token streaming
def stream_generate(
    no_md: bool,
    chatmode: bool,
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    num_ctx: int = 8192,
    temp: float = 0.6,
    top_k: float = 40,
    top_p: float = 1,
    min_p: float = 0,
) -> None:
    """
    Send a prompt to Ollama and stream the answer word‑by‑word,
    printing directly to the terminal (no panel).
    """
    # Show the prompt
    console.print(
        Panel(Text(prompt, style="bold cyan"),
            title=f"📝 Prompt → {model}",
            expand=False)
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temp,
            "num_ctx": num_ctx,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p
        }
    }

    if system_prompt is not None:
        payload["system"] = system_prompt
    else:
        payload["system"] = SYSTEM_MESSAGE

    with requests.post(f"{BASE_URL}/api/generate",
                       json=payload, stream=True) as resp:
        resp.raise_for_status()
        stream_response(resp, no_md, chatmode)


def chat_generate(
    no_md: bool,
    chatmode: bool,
    model: str,
    prompt: str,
    messages: list[dict],
    num_ctx: int = 8192,
    temp: float = 0.6,
    top_k: float = 40,
    top_p: float = 1,
    min_p: float = 0,
) -> str:
    """Send a chat request to Ollama and return the assistant's reply. """

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temp,
            "num_ctx": num_ctx,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p
        }
    }

    with requests.post(f"{BASE_URL}/api/chat",
                       json=payload, stream=True) as resp:
        resp.raise_for_status()
        return stream_response(resp, no_md, chatmode)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ollama CLI – stream answers word‑by‑word",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-list",
                        action="store_true",
                        help="List all Ollama models")
    parser.add_argument("-no_md",
                        action="store_true",
                        help="Disabled rich Markdown output rendering(Faster raw output, no waits)")
    parser.add_argument("-chat", "-c",
                        action="store_true",
                        help="Chat mode with history in RAM")
    parser.add_argument("-model",
                        help="Choose a model (falls back to env var or first available)")
    parser.add_argument("-system",
                        help="Pass a system prompt that will be sent before the user prompt")
    parser.add_argument("-num_ctx",
                        type=int,
                        default=8192,
                        help="Token context size for the model")
    parser.add_argument("-temp",
                        type=float,
                        default=0.6,
                        help="Sampling temperature for the model")
    parser.add_argument("-top_k",
                        type=float,
                        default=40,
                        help="Limit the next‑token choice to the `k` highest‑probability tokens when sampling")
    parser.add_argument("-top_p",
                        type=float,
                        default=1,
                        help="Keep the smallest set of tokens whose cumulative probability ≥ p")
    parser.add_argument("-min_p",
                        type=float,
                        default=0,
                        help="Exclude any token whose probability is below the min_p threshold")
    parser.add_argument("-append_p", "-p",
                        nargs="?",
                        help="Append to prompt, useful when you use stdin for a prompt but you want to add/ask something")
    parser.add_argument("prompt",
                        nargs="?",
                        help="Prompt to send to the model (can also be piped via stdin)")

    args = parser.parse_args()

    if args.list:
        console.print("\n".join(list_models()))
        sys.exit(0)

    # Resolve the model name first – it may be needed by the streaming helpers
    model = pick_model(args.model)
    # If system message is specifided use it else use SYSTEM_MESSAGE
    system_prompt = args.system

    if not args.chat:
        # If the user piped data in, prefer that over the positional prompt(not working for chat mode)
        if args.prompt is None:
            prompt = sys.stdin.read().strip()
            if args.append_p:
                prompt += "\n\n\n" + args.append_p
        else:
            prompt = args.prompt

        if prompt is None:
            # No prompt given at all
            console.print("[red]Error:[/red] No prompt supplied.")
            sys.exit(1)

        assistant_reply = stream_generate(
            args.no_md,
            args.chat,
            model,
            prompt,
            system_prompt,
            args.num_ctx,
            args.temp,
            args.top_k,
            args.top_p,
            args.min_p
        )

    else:
        history: list[dict] = []

        if args.prompt is None:
            # No prompt given at all
            console.print("[red]Error:[/red] No prompt supplied.")
            sys.exit(1)
        else:
            prompt = args.prompt

        console.print(
            Panel(Text(prompt, style="bold cyan"),
                title=f"📝 Prompt → {model}",
                expand=False)
        )

	# System message first and only one is needed
        if system_prompt:
            system = system_prompt
        else:
            system = SYSTEM_MESSAGE

        history.append({"role": "system", "content": system})

        try:
            while True:
                # Add user turn to history
                history.append({"role": "user", "content": prompt})

                console.print("[bold cyan on blue]Sending payload...[/bold cyan on blue]")

                # Ask Ollama for a reply
                assistant_reply = chat_generate(
                    args.no_md,
                    args.chat,
                    model,
                    prompt,
                    history,
                    args.num_ctx,
                    args.temp,
                    args.top_k,
                    args.top_p,
                    args.min_p
                )

                # Show reply and store it
                history.append({"role": "assistant", "content": assistant_reply})

                console.print("[bold cyan on blue]To send the payload enter . in a Newline[/bold cyan on blue]")
                console.print("[bold cyan on blue]Ctrl-C to end chat: [/bold cyan on blue]")

                while True:
                    line = input()
                    if line == '.':
                        break
                    prompt = "\n".join(line)

        except KeyboardInterrupt:
            console.print("\n🛑[bold cyan on blue]Exiting.  (History was only kept in RAM)[bold cyan on blue]")
            sys.exit(0)

if __name__ == "__main__":
    main()
