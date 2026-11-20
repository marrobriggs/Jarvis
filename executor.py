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

MAX_OUTPUT_CHARS = 10000  # keep error messages short enough to send back to the model


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


# Things AI-written analysis code has no business doing on your computer.
BLOCKED_PATTERNS = {
    r"\bpip\b[^\n]*\binstall\b": "installing packages",
    r"\bensurepip\b": "installing packages",
    r"\bsubprocess\b": "running other programs (subprocess)",
    r"\bos\.system\s*\(": "running shell commands (os.system)",
    r"\bos\.popen\s*\(": "running shell commands (os.popen)",
    r"\bmultiprocessing\b": "multiprocessing",
    r"\bshutil\.rmtree\s*\(": "deleting folders (shutil.rmtree)",
}


def check_code(code: str) -> str:
    """Return a description of the first blocked action found, or '' if the code looks safe."""
    for pattern, what in BLOCKED_PATTERNS.items():
        if re.search(pattern, code):
            return what
    return ""


def run_code(code: str, workdir: str, timeout: int = 120, filename: str = "analysis.py") -> dict:
    """Run the code in workdir. Returns {ok, stdout, stderr, files}."""
    if not code.strip():
        return {"ok": False, "stdout": "", "stderr": "No ```python code block was found in your answer.", "files": []}

    blocked = check_code(code)
    if blocked:
        return {"ok": False, "stdout": "", "files": [],
                "stderr": f"BLOCKED BY JARVIS: the script tried {blocked}, which is not allowed. "
                          "Use plain single-process code. If you need a package that isn't "
                          "installed, don't install it in code: put a line "
                          "'REQUIREMENTS: package-name' at the top of your answer instead."}

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


# ---------------------------------------------------------------------------
# Package requests: the analyst asks, Jarvis checks, you approve the unknowns
# ---------------------------------------------------------------------------

# Well-known, widely used packages Jarvis may install without asking.
# Add to this list as you come to trust more packages.
APPROVED_PACKAGES = {
    "pandas", "numpy", "plotly", "kaleido", "matplotlib", "seaborn", "scipy",
    "statsmodels", "openpyxl", "xlsxwriter", "xlrd", "pyarrow", "python-calamine",
    "tabulate", "fpdf2", "reportlab", "python-docx", "jinja2",
}

_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,60}$")


def parse_requirements(markdown: str) -> list:
    """Read a 'REQUIREMENTS: a, b' line from the analyst's answer (names only, no versions)."""
    match = re.search(r"^\s*REQUIREMENTS:\s*(.+)$", markdown, flags=re.MULTILINE | re.IGNORECASE)
    if not match:
        return []
    names = []
    for raw in match.group(1).split(","):
        # Drop version pins and anything after a separator, e.g. "pkg>=1.0; extra junk"
        name = re.split(r"[<>=!~\[;]", raw)[0].strip().strip("`'\"")
        if name and name.lower() not in ("none", "n/a") and _VALID_NAME.match(name):
            names.append(name.lower().replace("_", "-"))
    return sorted(set(names))[:5]  # a handful at most; a long list is a red flag


def is_installed(package: str) -> bool:
    from importlib import metadata
    try:
        metadata.distribution(package)
        return True
    except metadata.PackageNotFoundError:
        return False


def install_package(package: str) -> tuple:
    """Install one package with this Python's pip. Returns (ok, message).

    This is Jarvis running pip on your behalf, after the name has been
    approved. AI-written scripts are still blocked from installing anything.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return False, "pip took longer than 10 minutes and was stopped"
    if result.returncode == 0:
        return True, "installed"
    last = (result.stderr.strip().splitlines() or ["unknown pip error"])[-1]
    return False, last
