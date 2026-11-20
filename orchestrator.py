"""
Jarvis orchestrator: replaces you as the courier between agents.

Flow for each round:
  ANALYST writes code -> EXECUTOR runs it -> crashed? analyst fixes it (up to 3 tries)
  -> QA MANAGER reviews the work AND the real output -> approved? done : next round

Usage:
  python orchestrator.py sample_brief.txt --data sample_sales.csv               # fake brain
  python orchestrator.py sample_brief.txt --data sample_sales.csv --brain ollama
  python orchestrator.py sample_brief.txt --data sample_sales.csv --brain claude

Everything is saved in runs/<timestamp>/. Code runs inside runs/<timestamp>/work/.
Needs pandas and plotly installed in the Python you run this with.
"""

import argparse
import json
import os
import shutil
from datetime import datetime

from agents import run_analyst, run_qa
from brains import get_brain
from executor import describe_run, extract_code, run_code

MAX_ROUNDS = 3       # QA review rounds
MAX_FIX_TRIES = 3    # times the analyst may fix crashing code within one round


def save(folder: str, name: str, content: str) -> None:
    with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
        f.write(content)


def run_pipeline(brief: str, brain_name: str = "mock", data_path: str = "") -> dict:
    brain = get_brain(brain_name)
    folder = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    workdir = os.path.join(folder, "work")
    os.makedirs(workdir, exist_ok=True)
    save(folder, "00_brief.txt", brief)

    data_file = ""
    if data_path:
        data_file = os.path.basename(data_path)
        shutil.copy(data_path, os.path.join(workdir, data_file))

    print(f"Jarvis: starting job with the '{brain_name}' brain. Saving to {folder}\n")

    work, review, run = "", None, None
    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"[Round {round_num}] Analyst is working...")
        work = run_analyst(brain, brief, data_file, previous_work=work,
                           qa_feedback=review["feedback"] if review else None)

        # The agent loop: run the code, and let the analyst fix its own errors.
        for attempt in range(1, MAX_FIX_TRIES + 1):
            run = run_code(extract_code(work), workdir)
            tag = f"{round_num:02d}{chr(96 + attempt)}"  # 01a, 01b, 01c...
            save(folder, f"{tag}_analyst.md", work)
            save(folder, f"{tag}_code_run.txt", describe_run(run))
            if run["ok"]:
                print(f"           Code ran successfully (attempt {attempt}).")
                break
            first_error_line = (run["stderr"].splitlines() or ["unknown error"])[-1]
            print(f"           Code failed (attempt {attempt}): {first_error_line}")
            if attempt < MAX_FIX_TRIES:
                errors = describe_run(run)
                if getattr(brain, "last_truncated", False):
                    print("           (The answer was cut off for length. Asking for a shorter one.)")
                    errors = ("YOUR ANSWER WAS CUT OFF because it was too long, so the code was "
                              "incomplete. Shorten the Approach, keep the script focused, and make "
                              "sure the ```python block is closed.\n\n" + errors)
                print("           Analyst is fixing the code...")
                work = run_analyst(brain, brief, data_file, previous_work=work, code_errors=errors)

        # Findings are written from the real printed numbers, not guessed in advance.
        if run["ok"] and brain_name != "mock":
            print(f"[Round {round_num}] Analyst is writing findings from the real output...")
            drafted = run_analyst(brain, brief, data_file, previous_work=work,
                                  code_output=describe_run(run))
            if extract_code(drafted) == extract_code(work):
                work = drafted
            else:  # the code changed, so check that the new version still runs
                rerun = run_code(extract_code(drafted), workdir)
                if rerun["ok"]:
                    work, run = drafted, rerun
            save(folder, f"{round_num:02d}z_with_findings.md", work)

        print(f"[Round {round_num}] QA manager is reviewing the work and its output...")
        review = run_qa(brain, brief, work, describe_run(run))
        if not run["ok"]:
            review["verdict"] = "REVISE"  # code that doesn't run never gets approved
        save(folder, f"{round_num:02d}_qa_review.json", json.dumps(review, indent=2))
        print(f"           Score {review['score']}/10 -> {review['verdict']}")
        for point in review["feedback"]:
            print(f"           - {point}")
        print()

        if review["verdict"] == "APPROVE":
            save(folder, "FINAL_deliverable.md", work)
            print(f"Jarvis: approved in round {round_num}. Final deliverable and charts are in {folder}")
            return {"status": "approved", "rounds": round_num, "folder": folder}

    save(folder, "NEEDS_YOUR_REVIEW.md", work)
    print(f"Jarvis: not approved after {MAX_ROUNDS} rounds. Your call. "
          f"Latest draft: {os.path.join(folder, 'NEEDS_YOUR_REVIEW.md')}")
    return {"status": "needs_review", "rounds": MAX_ROUNDS, "folder": folder}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a client brief through Jarvis's agents.")
    parser.add_argument("brief_file", help="text file containing the client brief")
    parser.add_argument("--data", default="", help="the client's data file (CSV, Excel, etc.)")
    parser.add_argument("--brain", default="mock", choices=["mock", "ollama", "claude"])
    args = parser.parse_args()

    with open(args.brief_file, encoding="utf-8") as f:
        run_pipeline(f.read(), args.brain, args.data)
