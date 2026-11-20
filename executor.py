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


_BLOCK = re.compile(r"```(?:python|py)([^\n]*)\n(.*?)```", re.DOTALL | re.IGNORECASE)
_UNCLOSED = re.compile(r"```(?:python|py)([^\n]*)\n(.*)", re.DOTALL | re.IGNORECASE)
_FILE_TAG = re.compile(r"file\s*[:=]\s*([A-Za-z0-9_\-]+\.py)", re.IGNORECASE)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-]+\.py$")
RESERVED_FILES = {"jarvis_theme.py", "jarvis_deck.py"}  # Jarvis's own files; the analyst can't overwrite them


def extract_files(markdown: str) -> dict:
    """Split the analyst's answer into named Python files.

    A block is named by a first line like '# file: app.py' (or ```python file=app.py).
    Unnamed blocks are joined into analysis.py.
    """
    blocks = _BLOCK.findall(markdown)
    if not blocks:
        last = _UNCLOSED.search(markdown)  # answer cut off mid-block
        blocks = [last.groups()] if last else []

    files, unnamed = {}, []
    for info, body in blocks:
        body = body.strip()
        first = body.splitlines()[0] if body else ""
        tag = _FILE_TAG.search(info) or (_FILE_TAG.search(first) if first.lstrip().startswith("#") else None)
        name = tag.group(1) if tag else ""
        if name and _SAFE_NAME.match(name) and name not in RESERVED_FILES:
            files[name] = (files.get(name, "") + "\n\n" + body).strip()
        else:
            unnamed.append(body)
    if unnamed:
        files["analysis.py"] = (files.get("analysis.py", "") + "\n\n" + "\n\n".join(unnamed)).strip()
    return files


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
    safe_env = _safe_env()

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


def _safe_env() -> dict:
    """A copy of the environment with API keys and other secrets removed."""
    return {k: v for k, v in os.environ.items()
            if not any(word in k.upper() for word in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))}


# Jarvis's own checker: loads app.py in Streamlit's built-in test runner (no
# browser, no server) and reports errors plus what the report contains.
_APP_CHECK = r"""
import json, sys
try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    print(json.dumps({"skipped": "Streamlit is not installed, so the report app could not be checked."}))
    sys.exit(0)
at = AppTest.from_file("app.py", default_timeout=150)
at.run()
def grab(kind, attr="value"):
    try:
        return [str(getattr(x, attr)) for x in getattr(at, kind)]
    except Exception:
        return []
def count(kind):
    try:
        return len(at.get(kind))
    except Exception:
        return 0
errors = []
for e in at.exception:
    errors.append(str(getattr(e, "message", e)) + chr(10) + chr(10).join(getattr(e, "stack_trace", [])[-6:]))
print(json.dumps({
    "errors": errors,
    "title": grab("title"),
    "sections": grab("subheader") + grab("header"),
    "kpis": [f"{l}: {v}" for l, v in zip(grab("metric", "label"), grab("metric", "value"))],
    "charts": count("plotly_chart"),
    "tables": count("arrow_data_frame") or count("dataframe"),
}))
"""


def check_app(workdir: str, timeout: int = 180) -> dict:
    """Load app.py without opening a browser. Returns {ok, summary, error}."""
    import json
    try:
        result = subprocess.run([sys.executable, "-c", _APP_CHECK], cwd=workdir, env=_safe_env(),
                                capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "summary": "", "error": f"The report app took longer than {timeout} seconds to load."}
    lines = [l for l in result.stdout.strip().splitlines() if l.startswith("{")]
    if result.returncode != 0 or not lines:
        return {"ok": False, "summary": "", "error": _trim(result.stderr or result.stdout or "App check failed.")}
    info = json.loads(lines[-1])
    if "skipped" in info:
        return {"ok": True, "summary": "Report app NOT verified: " + info["skipped"], "error": ""}
    summary = (f"Title: {', '.join(info['title']) or '(none)'}\n"
               f"Sections: {' | '.join(info['sections']) or '(none)'}\n"
               f"KPI cards: {' | '.join(info['kpis']) or '(none)'}\n"
               f"Charts: {info['charts']}   Data tables: {info['tables']}")
    if info["errors"]:
        return {"ok": False, "summary": summary, "error": _trim("\n\n".join(info["errors"]))}
    return {"ok": True, "summary": summary, "error": ""}


def describe_decks(workdir: str) -> str:
    """Summarize every .pptx in outputs/ slide by slide, so QA can review a deck it can't see."""
    import glob
    decks = sorted(glob.glob(os.path.join(workdir, "outputs", "*.pptx")))
    if not decks:
        return ""
    try:
        from pptx import Presentation
    except ImportError:
        return "DECK CHECK: python-pptx is not installed, so the deck could not be inspected."
    parts = []
    for path in decks:
        prs = Presentation(path)
        lines = [f"DECK CHECK ({os.path.basename(path)}, {len(prs.slides)} slides):"]
        for i, slide in enumerate(prs.slides, start=1):
            texts, extras = [], []
            for shape in slide.shapes:
                if shape.has_chart:
                    chart = shape.chart
                    kind = str(chart.chart_type).split(".")[-1].split(" ")[0].lower()
                    n = len(chart.plots[0].categories) if chart.plots else 0
                    extras.append(f"{kind} chart, {n} categories, {len(list(chart.series))} series")
                elif shape.has_table:
                    extras.append(f"table, {len(shape.table.rows) - 1} rows")
                elif shape.has_text_frame and shape.text_frame.text.strip():
                    text = shape.text_frame.text.strip().replace("\n", " / ")
                    near_bottom = shape.top is not None and shape.top > prs.slide_height * 0.9
                    if near_bottom or text == "Prepared by Marro Briggs":
                        continue  # footers and page numbers, not content
                    texts.append(text[:90])
            head = texts[0] if texts else "(no text)"
            rest = " | ".join(texts[1:9])
            lines.append(f"  {i}. {head}" + (f" -- {rest}" if rest else "") +
                         (f" [{'; '.join(extras)}]" if extras else ""))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def run_job(markdown: str, workdir: str, timeout: int = 600) -> dict:
    """Run the analyst's files: analysis.py first, then check app.py loads cleanly."""
    files = extract_files(markdown)
    if not files:
        return {"ok": False, "stdout": "", "stderr": "No ```python code block was found in your answer.", "files": []}
    for name, code in files.items():
        blocked = check_code(code)
        if blocked:
            return {"ok": False, "stdout": "", "files": [],
                    "stderr": f"BLOCKED BY JARVIS: {name} tried {blocked}, which is not allowed. "
                              "Use plain single-process code. If you need a package that isn't "
                              "installed, put a line 'REQUIREMENTS: package-name' at the top of your answer."}

    os.makedirs(workdir, exist_ok=True)
    for name, code in files.items():
        with open(os.path.join(workdir, name), "w", encoding="utf-8") as f:
            f.write(code)

    if "analysis.py" in files:
        run = run_code(files["analysis.py"], workdir, timeout=timeout)
        if not run["ok"]:
            run["stderr"] = "analysis.py failed:\n" + run["stderr"]
            return run
    else:
        run = {"ok": True, "stdout": "", "stderr": "", "files": []}

    deck_summary = describe_decks(workdir)
    if deck_summary:
        run["stdout"] = (run["stdout"] + "\n\n" + _trim(deck_summary)).strip()

    if "app.py" in files:
        app = check_app(workdir)
        run["app_ok"] = app["ok"]
        run["stdout"] = (run["stdout"] + "\n\nREPORT APP CHECK (app.py):\n" + app["summary"]).strip()
        if not app["ok"]:
            run["ok"] = False
            run["stderr"] = "app.py failed to load:\n" + app["error"]
    return run


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
    "tabulate", "fpdf2", "reportlab", "python-docx", "jinja2", "streamlit", "python-pptx",
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
