# ollama-aibot
Python AI chatbot for ollama api for the terminal/console.

- Prompt can be given at the command line or piped-in via stdin
- The answer is streamed Token-by-token
- By default use rich to render markdown, you can  disable for fast and raw output with `-no_md`
- Config for ollama server inside the script(default: `BASE_URL = "http://localhost:11434")`
- Model fallback to the first in ollama list if none is provided by cli or environment variable
- Chat support with history(use a `Newline` with only `.` as `Enter`, This is useful for pasting multiple lines like code etc)
- Chat mode don't support stdin inpui ex: `ps axu | aibot.py -c`
- Non-chat mode support stdin input and you can combine with option `-append_p` to add/ask with more context
- Using Markdown(rich) rendering can cause the console to stall with large responses, wait for the full output to appear(or `-no_md`)

~~~
usage: aibot.py [-h] [-list] [-no_md] [-chat] [-model MODEL] [-system SYSTEM] [-num_ctx NUM_CTX] [-temp TEMP] [-top_k TOP_K] [-top_p TOP_P] [-min_p MIN_P] [-append_p [APPEND_P]] [prompt]

Ollama CLI – stream answers word‑by‑word

positional arguments:
  prompt                Prompt to send to the model (can also be piped via stdin) (default: None)

options:
  -h, --help            show this help message and exit
  -list                 List all Ollama models (default: False)
  -no_md                Disabled rich Markdown output rendering(Faster raw output, no waits) (default: False)
  -chat, -c             Chat mode with history in RAM (default: False)
  -model MODEL          Choose a model (falls back to env var or first available) (default: None)
  -system SYSTEM        Pass a system prompt that will be sent before the user prompt (default: None)
  -num_ctx NUM_CTX      Token context size for the model (default: 8192)
  -temp TEMP            Sampling temperature for the model (default: 0.6)
  -top_k TOP_K          Limit the next‑token choice to the `k` highest‑probability tokens when sampling (default: 40)
  -top_p TOP_P          Keep the smallest set of tokens whose cumulative probability ≥ p (default: 1)
  -min_p MIN_P          Exclude any token whose probability is below the min_p threshold (default: 0)
  -append_p, -p [APPEND_P]
                        Append to prompt, useful when you use stdin for a prompt but you want to add/ask something (default: None)
~~~

Environment variables:

~~~
OLLAMA_MODEL: to specify the model tu use
~~~
