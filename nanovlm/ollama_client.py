"""Minimal client for the job-private Ollama server (DGX_GUIDE_nanovlm.md
§0 — port 11435, never the ambient CPU-bound daemon on 11434). Shared by
scripts/generate_captions.py (teacher) and scripts/evaluate.py (grader) so
both use identical retry/timeout behavior.
"""

import json
import time
import urllib.error
import urllib.request


def call_ollama(host: str, model: str, prompt: str, timeout: int = 120, temperature: float = 0.7) -> str:
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["response"].strip()


def call_ollama_with_retries(
    host: str, model: str, prompt: str, retries: int = 3, timeout: int = 120, temperature: float = 0.7
) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            return call_ollama(host, model, prompt, timeout=timeout, temperature=temperature)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Ollama call failed after {retries} attempts: {last_err}")
