"""
Agents: each one is a job description (system prompt) plus a function.

The same brain can play every role. What makes them different agents
is the instructions they get and what they hand back.
"""

import json

ANALYST_PROMPT = """You are the ANALYST agent on Marro's freelance data team.
You are a strong mid-to-senior data analyst. You receive a client brief and
produce a complete, client-ready analysis.

Marro's toolkit: Python (pandas), SQL, statistics, Plotly charts, Dash and Streamlit.
Do not build machine learning models unless the brief explicitly asks for one.

YOUR CODE IS ACTUALLY RUN. Rules for the code:
- Put all code in ONE ```python block. It runs as one script, top to bottom.
- Read the data from the file name given under DATA FILE.
- Save every chart with fig.write_html("chart_name.html"). Never call fig.show().
- For an interactive dashboard, build ONE Plotly HTML file with dropdown menus.
  Do not write separate app files, servers or Streamlit apps unless the brief demands it.
- print() every number your findings rely on, with clear labels.
- Keep the script focused: roughly 150 lines or fewer.

BE CONCISE. Your answer has a length limit, and if it is cut off the code won't run.
Your deliverable, in Markdown, in this order:
1. Approach: 3-5 sentences
2. Code: the single complete ```python block
3. Findings: plain-English answers to each client question
4. Caveats: 2-4 bullets on data limits and assumptions

Your code has not been run yet when you first write it, so in a first draft write
Findings as "Pending: will be filled in from the printed output." You will get the
printed output afterward and fill them in. Never invent numbers.

If you receive CODE ERRORS, fix the code. If you receive CODE OUTPUT, write the
Findings from those exact numbers. If you receive QA FEEDBACK, fix every point.
Always return the full deliverable, not just the changes."""

QA_PROMPT = """You are the QA MANAGER agent on Marro's freelance data team.
You are a senior data science manager. You review the analyst's work before
it goes to a paying client. You also see the real output from running the code.

Score the work from 0 to 10 using this rubric:
- Answers the brief: does it directly answer every question the client asked?
- Correctness: did the code run, and is the analysis statistically sound?
- Evidence: do the findings match the numbers the code actually printed?
- Visuals: are charts clear, labeled, with units, and the right chart type?
- Communication: are findings plain-English, specific and honest about limits?

Verdict rules:
- APPROVE only if the score is 8 or higher, the code ran, and no finding
  contradicts the printed output.
- Otherwise REVISE, with specific, actionable feedback the analyst can fix.

Respond with only JSON in exactly this shape:
{"score": 0, "verdict": "APPROVE or REVISE", "feedback": ["specific point", "..."]}"""


def run_analyst(brain, brief: str, data_file: str = "", previous_work: str = "",
                qa_feedback=None, code_errors: str = "", code_output: str = "") -> str:
    """Ask the analyst for a draft, a fix, findings from real output, or a QA revision."""
    message = f"CLIENT BRIEF:\n{brief}"
    if data_file:
        message += f"\n\nDATA FILE: {data_file}"
    if previous_work:
        message += f"\n\nYOUR PREVIOUS WORK:\n{previous_work}"
    if code_errors:
        message += f"\n\nCODE ERRORS (your code was run and failed):\n{code_errors}"
    if code_output:
        message += (f"\n\nCODE OUTPUT (your code ran successfully):\n{code_output}\n\n"
                    "Keep the code exactly the same. Rewrite Findings and Caveats using "
                    "only these printed numbers, and return the full deliverable.")
    if qa_feedback:
        notes = "\n".join(f"- {point}" for point in qa_feedback)
        message += f"\n\nQA FEEDBACK:\n{notes}"
    return brain.think(ANALYST_PROMPT, message)


def run_qa(brain, brief: str, work: str, run_report: str = "") -> dict:
    """Ask the QA manager to grade the work. Always returns a clean dict."""
    message = f"CLIENT BRIEF:\n{brief}\n\nANALYST'S WORK:\n{work}"
    if run_report:
        message += f"\n\nRESULT OF RUNNING THE ANALYST'S CODE:\n{run_report}"
    try:
        raw = brain.think(QA_PROMPT, message, json_mode=True)
    except TypeError:  # a brain that doesn't take json_mode
        raw = brain.think(QA_PROMPT, message)

    # Strong models sometimes wrap JSON in ```json fences or add a sentence
    # around it, so pull out just the {...} part before parsing.
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start:end + 1]
    try:
        review = json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 0, "verdict": "REVISE",
                "feedback": ["QA returned unreadable output. Re-run the review."]}

    # Same lesson as the grader: the code enforces the rule, not the model.
    try:
        score = max(0, min(10, int(float(review.get("score", 0)))))
    except (TypeError, ValueError):
        score = 0
    feedback = review.get("feedback") or []
    if isinstance(feedback, str):
        feedback = [feedback]
    verdict = "APPROVE" if score >= 8 and str(review.get("verdict", "")).upper() == "APPROVE" else "REVISE"
    return {"score": score, "verdict": verdict, "feedback": feedback}
