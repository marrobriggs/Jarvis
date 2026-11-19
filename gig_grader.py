"""
Jarvis Gig Grader - version 1 (manual)

Paste a freelance job listing, Jarvis grades it against your skills
using a local Ollama model, then you decide yes or no.
Every decision is logged to graded_gigs.csv for later evaluation.

Setup:
  1. Install Ollama: https://ollama.com
  2. Pull a model:   ollama pull llama3.1
  3. Run:            python gig_grader.py

Uses only the Python standard library, so no pip installs needed.
"""

import csv
import json
import os
import urllib.error
import urllib.request
from datetime import datetime

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "llama3.2"  # small enough for laptops with limited RAM; try "llama3.2:1b" if slow
LOG_FILE = "graded_gigs.csv"

SYSTEM_PROMPT = """You are Jarvis, a gig screening assistant for a freelance data analyst named Marro.
Your job is to read one freelance job listing and judge how well it fits Marro's skills.
You never accept or decline jobs yourself. You give a recommendation and Marro decides.

MARRO'S SKILLS PROFILE

Strong (can deliver confidently):
- SQL querying, including common table expressions (CTEs), joins, aggregations
- Exploratory data analysis in Python with pandas
- Summary statistics and statistical tests (t-tests, chi-square, correlation, etc.)
- Data visualization, preferring Plotly (Matplotlib also fine)
- Interactive dashboards in Dash (Python) and Streamlit
- Pulling data from public APIs (e.g. weather, World Bank) into apps and dashboards
- SQL-based data cleaning and data engineering
- Written reports summarizing findings and visuals

Avoid:
- Machine learning or predictive modeling as the main deliverable
  (classification, regression models, random forest, XGBoost, forecasting models,
  deep learning, NLP models, computer vision)
- R Shiny dashboards

GRADING RULES

1. Identify the core deliverable of the job.
2. Set involves_ml:
   - "core" if the main work is building ML, predictive models, NLP, document or
     data extraction with AI, LLM or AI-agent development, or computer vision
   - "minor" if a simple model is a small, optional add-on to analysis or dashboards
   - "none" otherwise
3. Pros and cons must not contradict each other or the gaps. A skill Marro lacks
   is a gap or a con, never a pro.
4. Score fit from 0 to 10:
   - 9-10: almost entirely strong skills, clear scope
   - 6-8: mostly strong skills, some unknowns or minor gaps
   - 3-5: significant gaps or vague scope
   - 0-2: mostly outside skills or in the avoid list
5. Note red flags: vague scope, unrealistic budget or deadline, requests for
   unpaid test work, asking to communicate or pay off-platform.
6. Be honest. Do not inflate scores to be encouraging.

OUTPUT FORMAT

Respond with only valid JSON in exactly this shape:
{
  "title": "short name of the gig",
  "core_deliverable": "one sentence",
  "matched_skills": ["skills from Marro's profile this job uses"],
  "gaps": ["skills the job needs that Marro lacks or avoids"],
  "involves_ml": "none | minor | core",
  "fit_score": 0,
  "pros": ["reasons to take it"],
  "cons": ["reasons to skip it"],
  "red_flags": ["warning signs, or an empty list if none"],
  "one_line_summary": "what Jarvis would say to Marro out loud"
}
Do not include a recommendation. The program decides that from your answers."""


def grade_gig(listing: str) -> dict:
    """Send a job listing to Ollama and return the parsed JSON verdict."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": listing},
        ],
        "format": "json",          # forces JSON output
        "stream": False,
        "keep_alive": "30m",       # keep the model loaded so later grades are faster
        "options": {
            "temperature": 0.2,    # low = more consistent grading
            "num_ctx": 4096,       # enough room for the prompt plus a long listing
            "num_predict": 700,    # cap the reply length so it can't ramble forever
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    # Bypass any system proxy so the request goes straight to Ollama on this laptop
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=900) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return json.loads(body["message"]["content"])


def enforce_ml_rule(verdict: dict) -> dict:
    """Decide the recommendation in code, not in the model.

    The model only reports facts (score, ML level, red flags). These rules
    turn those facts into TAKE / MAYBE / PASS, so the result can never
    contradict itself, no matter which model is doing the grading.
    """
    try:
        score = max(0, min(10, int(float(verdict.get("fit_score", 0)))))
    except (TypeError, ValueError):
        score = 0
    verdict["fit_score"] = score

    ml = str(verdict.get("involves_ml", "none")).strip().lower()
    red_flags = [f for f in (verdict.get("red_flags") or []) if str(f).strip()]

    if ml == "core":
        rec, reason = "PASS", "ML/AI building is the core of this job"
    elif score >= 7:
        if red_flags:
            rec, reason = "MAYBE", "strong fit, but check the red flags"
        else:
            rec, reason = "TAKE", "strong fit"
    elif score >= 4:
        rec, reason = "MAYBE", "partial fit"
    else:
        rec, reason = "PASS", "weak fit"

    verdict["recommendation"] = rec
    verdict["reason"] = reason
    return verdict


def show(verdict: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  {verdict.get('title', 'Untitled gig')}")
    print("=" * 60)
    print(f"Deliverable:    {verdict.get('core_deliverable', '')}")
    print(f"Fit score:      {verdict.get('fit_score', '?')}/10")
    print(f"ML involved:    {verdict.get('involves_ml', '?')}")
    print(f"Recommendation: {verdict.get('recommendation', '?')} ({verdict.get('reason', '')})")
    for label, key in [("Pros", "pros"), ("Cons", "cons"),
                       ("Gaps", "gaps"), ("Red flags", "red_flags")]:
        items = verdict.get(key) or []
        if items:
            print(f"\n{label}:")
            for item in items:
                print(f"  - {item}")
    print(f"\nJarvis: {verdict.get('one_line_summary', '')}\n")


def log_decision(listing: str, verdict: dict, decision: str) -> None:
    """Append the gig, Jarvis's verdict and your call to a CSV."""
    is_new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "title", "fit_score", "involves_ml",
                             "jarvis_recommendation", "your_decision",
                             "listing", "full_verdict_json"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            verdict.get("title", ""),
            verdict.get("fit_score", ""),
            verdict.get("involves_ml", ""),
            verdict.get("recommendation", ""),
            decision,
            listing,
            json.dumps(verdict),
        ])


def read_listing() -> str:
    print("Paste the job listing, then type END on its own line and press Enter.")
    print("(Type QUIT to exit.)")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "QUIT" and not lines:
            return ""
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main() -> None:
    print("Jarvis Gig Grader is online.\n")
    while True:
        listing = read_listing()
        if not listing:
            print("Shutting down. Good hunting.")
            break

        print("\nGrading... (the first one can take a few minutes on a laptop)")
        start = datetime.now()
        try:
            verdict = enforce_ml_rule(grade_gig(listing))
            print(f"Done in {(datetime.now() - start).seconds} seconds.")
        except urllib.error.HTTPError as e:
            print(f"Ollama returned an error ({e.code}): {e.read().decode('utf-8', 'ignore')}")
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"Couldn't reach Ollama: {e}")
            print("Check http://127.0.0.1:11434 in your browser, or run: ollama serve")
            continue
        except (json.JSONDecodeError, KeyError):
            print("The model didn't return clean JSON. Try again or use a bigger model.")
            continue

        show(verdict)

        decision = ""
        while decision not in ("y", "n"):
            decision = input("Take this gig? (y/n): ").strip().lower()
        log_decision(listing, verdict, "yes" if decision == "y" else "no")
        print(f"Logged to {LOG_FILE}.\n")


if __name__ == "__main__":
    main()
