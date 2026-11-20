"""
Executor: gives the analyst "hands" so it can run the code it writes.

It pulls the Python out of the analyst's Markdown, runs it in the job's
work folder, and reports back what happened: the printed output, or the
error if it crashed. The orchestrator feeds errors back to the analyst
so it can fix its own code. That loop is what makes it an agent.

SAFETY: this runs AI-written code on your computer. Only use it with
brains you trust (mock, your local Ollama, or Claude), keep client data
in the job's work folder, and read the code in runs/ when in doubt.
The timeout stops runaway scripts, but this is not a full sandbox.
"""

import os
import re
import subprocess
import sys

MAX_OUTPUT_CHARS = 8000  # keep error messages short enough to send back to the model


def extract_code(markdown: str) -> str:
    """Return all ```python blocks from the analyst's answer, joined together."""
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", markdown, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks)
    # A block that was opened but never closed (answer cut off): run what's there,
    # so the error points at the cut-off line instead of "no code found".
    unclosed = re.search(r"```(?:python|py)\s*\n(.*)", markdown, flags=re.DOTALL | re.IGNORECASE)
    return unclosed.group(1).strip() if unclosed else ""


def _trim(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    return text[:half] + "\n...[output trimmed]...\n" + text[-half:]


def run_code(code: str, workdir: str, timeout: int = 120, filename: str = "analysis.py") -> dict:
    """Run the code in workdir. Returns {ok, stdout, stderr, files}."""
    if not code.strip():
        return {"ok": False, "stdout": "", "stderr": "No ```python code block was found in your answer.", "files": []}

    os.makedirs(workdir, exist_ok=True)
    script = os.path.join(workdir, filename)
    with open(script, "w", encoding="utf-8") as f:
        f.write(code)

    # Hide secrets from AI-written code: the script gets a copy of the
    # environment with API keys and tokens removed.
    safe_env = {
        k: v for k, v in os.environ.items()
        if not any(word in k.upper() for word in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
    }

    before = set(os.listdir(workdir))
    try:
        result = subprocess.run(
            [sys.executable, filename],
            cwd=workdir,
            env=safe_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        ok = result.returncode == 0
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        ok, stdout, stderr = False, "", f"The script ran longer than {timeout} seconds and was stopped."

    created = sorted(set(os.listdir(workdir)) - before)
    return {"ok": ok, "stdout": _trim(stdout), "stderr": _trim(stderr), "files": created}


def describe_run(run: dict) -> str:
    """A plain-text summary of a code run, for the analyst or QA to read."""
    parts = ["Status: ran successfully" if run["ok"] else "Status: FAILED"]
    if run["stdout"]:
        parts.append(f"Printed output:\n{run['stdout']}")
    if run["stderr"]:
        parts.append(f"Errors:\n{run['stderr']}")
    if run["files"]:
        parts.append("Files created: " + ", ".join(run["files"]))
    return "\n\n".join(parts)
