# ollama-aibot
Ai chatbot for ollama to use in the console

- Prompt can be given on the command line or piped in via stdin
- The answer is streamed word‑by‑word
- Config for ollama server inside the script(default: BASE_URL = "http://localhost:11434")
- Fallback to the first model returned by the API if no model is supplyied

~~~
usage: aibot [-h] [-list] [-model MODEL] [-system SYSTEM] [-num_ctx NUM_CTX] [-temp TEMP] [-top_k TOP_K] [-top_p TOP_P] [-min_p MIN_P] [-append_p [APPEND_P]] [prompt]

Ollama CLI – stream answers word‑by‑word

positional arguments:
  prompt                Prompt to send to the model (can also be piped via stdin) (default: None)

options:
  -h, --help            show this help message and exit
  -list                 List all Ollama models (default: False)
  -model MODEL          Choose a model (falls back to env var or first available) (default: None)
  -system SYSTEM        Pass a system prompt that will be sent before the user prompt (default: None)
  -num_ctx NUM_CTX      Token context size for the model (default: 8192) (default: 8192)
  -temp TEMP            Sampling temperature for the model (default: 0.6) (default: 0.6)
  -top_k TOP_K          Limit the next‑token choice to the `k` highest‑probability tokens when sampling (default: 40) (default: 40)
  -top_p TOP_P          Keep the smallest set of tokens whose cumulative probability ≥ p (default: 1) (default: 1)
  -min_p MIN_P          Exclude any token whose probability is below the min_p threshold (default: 0) (default: 0)
  -append_p, -p [APPEND_P]
                        Append to prompt, useful when you use stdin for a prompt but you want to add/ask something (default: None)
~~~

Environment variables:

~~~
OLLAMA_MODEL: to specify the model tu use
~~~
