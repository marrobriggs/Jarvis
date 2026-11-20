"""
Jarvis orchestrator: replaces you as the courier between agents.

Flow for each round:
  ANALYST writes code -> EXECUTOR runs it -> crashed? analyst fixes it (up to 3 tries)
  -> QA MANAGER reviews the work AND the real output -> approved? done : next round

Usage:
  python orchestrator.py sample_brief.txt --data sample_sales.csv               # fake brain
  python orchestrator.py jobs/my-job/brief.txt --data jobs/my-job --brain claude
      (--data can be one file, a folder of files, or a zip, which gets unzipped)
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

from agents import run_analyst, run_qa, splice_findings
from brains import get_brain
from data_preview import preview_many, stage_data
from executor import (APPROVED_PACKAGES, describe_run, extract_code, install_package,
                      is_installed, parse_requirements, run_job)

MAX_ROUNDS = 3       # QA review rounds
MAX_FIX_TRIES = 3    # times the analyst may fix crashing code within one round
CODE_TIMEOUT = 600   # seconds a script may run; big Excel files need several minutes

# Packages the analyst may use, if they're installed on this computer.
CHECK_PACKAGES = ["pandas", "numpy", "plotly", "streamlit", "pptx", "openpyxl", "matplotlib", "scipy",
                  "statsmodels", "seaborn", "pyarrow", "kaleido", "python_calamine", "xlsxwriter"]


def environment_info() -> str:
    """Describe this computer so the analyst writes code that actually runs here."""
    import importlib.util
    import platform
    import sys

    names = {"pptx": "python-pptx (import pptx)"}
    available = [names.get(p, p) for p in CHECK_PACKAGES if importlib.util.find_spec(p)]
    missing = [names.get(p, p) for p in CHECK_PACKAGES if names.get(p, p) not in available]
    return (f"Operating system: {platform.system()} {platform.release()}\n"
            f"Python version: {platform.python_version()}\n"
            f"Installed packages you may use: {', '.join(available) or 'none detected'}\n"
            f"NOT installed (do not use): {', '.join(missing) or 'none'}\n"
            f"Time limit: the script is stopped after {CODE_TIMEOUT // 60} minutes.\n"
            "Rules: never install packages inside your code (no pip, no subprocess), never use "
            "multiprocessing, and only read and write files inside the working folder. If you "
            "truly need a package that isn't installed, put one line at the very top of your "
            "answer: REQUIREMENTS: name1, name2. Jarvis will install approved packages and ask "
            "Marro about any others.")


def handle_requirements(work: str, decisions: dict) -> str:
    """Install the packages the analyst asked for, asking you about unfamiliar ones.

    decisions remembers your answers for this job, so you're asked once per package.
    Returns a note for the analyst about anything that couldn't be installed.
    """
    problems = []
    for pkg in parse_requirements(work):
        if pkg in decisions:
            if decisions[pkg] != "ok":
                problems.append(f"{pkg}: {decisions[pkg]}")
            continue
        if is_installed(pkg):
            decisions[pkg] = "ok"
            continue

        if pkg in APPROVED_PACKAGES:
            print(f"           Analyst needs '{pkg}' (on the approved list). Installing...")
        else:
            print(f"\n  >>> The analyst wants to install '{pkg}', which is NOT on the approved list.")
            print(f"      Check it first: https://pypi.org/project/{pkg}/")
            answer = ""
            try:
                while answer not in ("y", "n"):
                    answer = input(f"      Allow installing '{pkg}'? (y/n): ").strip().lower()
            except EOFError:  # no one at the keyboard: play it safe
                answer = "n"
            if answer != "y":
                decisions[pkg] = "declined by Marro, so work without it"
                problems.append(f"{pkg}: {decisions[pkg]}")
                print(f"           Declined. The analyst will work without '{pkg}'.")
                continue
            print(f"           Approved. Installing '{pkg}'...")

        ok, message = install_package(pkg)
        decisions[pkg] = "ok" if ok else f"install failed ({message}), so work without it"
        print(f"           {pkg}: {'installed' if ok else 'install failed: ' + message}")
        if not ok:
            problems.append(f"{pkg}: {decisions[pkg]}")
    return "\n".join(problems)


HERE = os.path.dirname(os.path.abspath(__file__))


def install_theme(workdir: str) -> None:
    """Give every job the same look: copy the Jarvis theme and Streamlit settings in."""
    import shutil
    from jarvis_theme import STREAMLIT_CONFIG
    for helper in ("jarvis_theme.py", "jarvis_deck.py"):
        shutil.copy(os.path.join(HERE, helper), os.path.join(workdir, helper))
    os.makedirs(os.path.join(workdir, ".streamlit"), exist_ok=True)
    with open(os.path.join(workdir, ".streamlit", "config.toml"), "w", encoding="utf-8") as f:
        f.write(STREAMLIT_CONFIG)


def write_launcher(workdir: str) -> str:
    """A double-click file that opens the report app (for you, or to hand to a client)."""
    path = os.path.join(workdir, "OPEN_REPORT.bat")
    with open(path, "w", encoding="utf-8") as f:
        f.write('@echo off\r\ncd /d "%~dp0"\r\npython -m streamlit run app.py\r\npause\r\n')
    return path


def optional_next_steps(work: str) -> list:
    """Pull the 'Optional next steps' lines out of the final deliverable, if any."""
    import re
    match = re.search(r"^\s*#{1,4}\s*(?:\d+[.)]\s*)?optional next steps\s*$(.*?)(?=^\s*#{1,4}\s|\Z)",
                      work, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if not match:
        return []
    lines = [l.strip().lstrip("-*• ").strip() for l in match.group(1).splitlines()]
    return [l for l in lines if l]


def save(folder: str, name: str, content: str) -> None:
    with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
        f.write(content)


def build(brain, brain_name, brief, data_file, work, workdir, folder, label, decisions):
    """Run the analyst's code, let it fix its own errors, then write findings from real output.

    Returns (work, run). Used by every review round and by the polish round.
    """
    run = None
    for attempt in range(1, MAX_FIX_TRIES + 1):
        package_note = handle_requirements(work, decisions)
        run = run_job(work, workdir, timeout=CODE_TIMEOUT)
        if package_note and not run["ok"]:
            run["stderr"] = ("PACKAGES NOT AVAILABLE:\n" + package_note + "\n\n" + run["stderr"]).strip()
        tag = f"{label}{chr(96 + attempt)}"  # 01a, 01b, Pa, Pb...
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
                          "sure the ```python blocks are closed.\n\n" + errors)
            print("           Analyst is fixing the code...")
            work = run_analyst(brain, brief, data_file, previous_work=work, code_errors=errors)

    # Findings are written from the real printed numbers, not guessed in advance.
    # The analyst replies with just the text; Jarvis splices it in, so the code is never resent.
    if run["ok"] and brain_name != "mock":
        print("           Analyst is writing findings from the real output...")
        findings = run_analyst(brain, brief, data_file, previous_work=work,
                               code_output=describe_run(run))
        work = splice_findings(work, findings)
        save(folder, f"{label}z_with_findings.md", work)
    return work, run


def polish_round(brain, brain_name, brief, data_file, work, run, review, workdir, folder, decisions):
    """After approval, let the analyst apply QA's suggestions. Keep the result only if it's better.

    The approved work is snapshotted first, so a polish that breaks something or
    scores lower is thrown away and the approved version is restored.
    """
    snapshot = workdir + "_approved"
    if os.path.exists(snapshot):
        shutil.rmtree(snapshot)
    shutil.copytree(workdir, snapshot)

    def restore(reason):
        print(f"           {reason} Keeping the approved version.")
        try:
            shutil.rmtree(workdir)
            shutil.copytree(snapshot, workdir)
            shutil.rmtree(snapshot)
        except OSError as e:
            print(f"           Couldn't restore automatically ({e}). The approved version is in {snapshot}")
        return work, run, review, False

    print("[Polish] Analyst is applying QA's suggestions...")
    suggestions = [f for f in review["feedback"] if not f.strip().upper().startswith("BLOCKING")]
    polished = run_analyst(brain, brief, data_file, previous_work=work,
                           qa_feedback=suggestions, polish=True)
    polished, prun = build(brain, brain_name, brief, data_file, polished, workdir, folder, "P", decisions)
    if not prun["ok"]:
        return restore("The polished code didn't run.")

    print("[Polish] QA manager is checking the polished version...")
    new_review = run_qa(brain, brief, polished, describe_run(prun), previous_review=review)
    save(folder, "P_qa_review.json", json.dumps(new_review, indent=2))
    print(f"           Score {new_review['score']}/10 -> {new_review['verdict']} "
          f"(was {review['score']}/10)")
    if new_review["verdict"] != "APPROVE" or new_review["score"] < review["score"]:
        return restore("The polish didn't improve the score.")

    shutil.rmtree(snapshot, ignore_errors=True)
    for point in new_review["feedback"]:
        print(f"           - {point}")
    return polished, prun, new_review, True


def run_pipeline(brief: str, brain_name: str = "mock", data_path: str = "",
                 brief_path: str = "", polish: bool = True) -> dict:
    brain = get_brain(brain_name)
    folder = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    workdir = os.path.join(folder, "work")
    os.makedirs(workdir, exist_ok=True)
    save(folder, "00_brief.txt", brief)

    data_file = ""
    if data_path:
        if not os.path.exists(data_path):
            raise SystemExit(f"Jarvis: can't find the data at {data_path}")
        # Copy the client's file, folder or zip into the work folder (zips get unzipped).
        files = stage_data(data_path, workdir, exclude=[brief_path])
        if not files:
            raise SystemExit(f"Jarvis: no data files found in {data_path}")
        data_file = ", ".join(files)
        print(f"Jarvis: staged {len(files)} data file(s): {data_file}")
        # Show the agents what's really in the data, so nobody guesses column names.
        profile = preview_many(workdir, files)
        save(folder, "00_data_preview.txt", profile)
        brief = brief + "\n\nDATA PREVIEW (generated automatically by Jarvis):\n" + profile
        print("Jarvis: data preview built and shared with the agents.")

    install_theme(workdir)
    env = environment_info()
    save(folder, "00_environment.txt", env)
    brief = brief + "\n\nRUNTIME ENVIRONMENT (your code runs here):\n" + env

    print(f"Jarvis: starting job with the '{brain_name}' brain. Saving to {folder}\n")

    work, review, run = "", None, None
    package_decisions = {}   # your install answers, remembered for this job
    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"[Round {round_num}] Analyst is working...")
        work = run_analyst(brain, brief, data_file, previous_work=work,
                           qa_feedback=review["feedback"] if review else None)

        work, run = build(brain, brain_name, brief, data_file, work, workdir, folder,
                          f"{round_num:02d}", package_decisions)

        print(f"[Round {round_num}] QA manager is reviewing the work and its output...")
        review = run_qa(brain, brief, work, describe_run(run), previous_review=review)
        if not run["ok"]:
            review["verdict"] = "REVISE"  # code that doesn't run never gets approved
        save(folder, f"{round_num:02d}_qa_review.json", json.dumps(review, indent=2))
        print(f"           Score {review['score']}/10 -> {review['verdict']}"
              f" ({review.get('blocking', 0)} blocking)")
        for point in review["feedback"]:
            print(f"           - {point}")
        print()

        if review["verdict"] == "APPROVE":
            polished = False
            if polish and any(not f.upper().startswith("BLOCKING") for f in review["feedback"]) \
                    and review["score"] < 10:
                work, run, review, polished = polish_round(
                    brain, brain_name, brief, data_file, work, run, review,
                    workdir, folder, package_decisions)
                print()
            save(folder, "FINAL_deliverable.md", work)
            note = " (polished)" if polished else ""
            print(f"Jarvis: approved in round {round_num}{note}. Final deliverable is in {folder}")
            if os.path.exists(os.path.join(workdir, "app.py")):
                write_launcher(workdir)
                print("\nTo open the report app, double-click OPEN_REPORT.bat in the work folder, or run:")
                print(f'  cd "{os.path.abspath(workdir)}"')
                print("  python -m streamlit run app.py")
            extras = optional_next_steps(work)
            if extras:
                print("\nOptional extras to offer the client (not built):")
                for line in extras:
                    print(f"  - {line}")
            outputs = os.path.join(workdir, "outputs")
            if os.path.isdir(outputs):
                client_files = sorted(f for f in os.listdir(outputs)
                                      if os.path.isfile(os.path.join(outputs, f)))
                if client_files:
                    print(f"\nFiles for the client ({os.path.abspath(outputs)}):")
                    for name in client_files:
                        print(f"  - {name}")
                working = os.path.join(outputs, "working")
                if os.path.isdir(working) and os.listdir(working):
                    n = len(os.listdir(working))
                    print(f"  (plus {n} working file{'s' if n != 1 else ''} in outputs\\working, for your reference)")
            return {"status": "approved", "rounds": round_num, "folder": folder}

    save(folder, "NEEDS_YOUR_REVIEW.md", work)
    print(f"Jarvis: not approved after {MAX_ROUNDS} rounds. Your call. "
          f"Latest draft: {os.path.join(folder, 'NEEDS_YOUR_REVIEW.md')}")
    return {"status": "needs_review", "rounds": MAX_ROUNDS, "folder": folder}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a client brief through Jarvis's agents.")
    parser.add_argument("brief_file", help="text file containing the client brief")
    parser.add_argument("--data", default="",
                        help="the client's data: a file, a folder of files, or a zip")
    parser.add_argument("--brain", default="mock", choices=["mock", "ollama", "claude"])
    parser.add_argument("--no-polish", action="store_true",
                        help="skip the polish round after approval (faster, cheaper)")
    args = parser.parse_args()

    with open(args.brief_file, encoding="utf-8") as f:
        run_pipeline(f.read(), args.brain, args.data, brief_path=args.brief_file,
                     polish=not args.no_polish)
