"""
Jarvis orchestrator: replaces you as the courier between agents.

Flow for each round:
  ANALYST writes code -> EXECUTOR runs it -> crashed? analyst fixes it (up to 3 tries)
  -> QA MANAGER reviews the work AND the real output -> approved? done : next round

Usage:
  python orchestrator.py sample_brief.txt --data sample_sales.csv               # fake brain
  ... --ml / --no-ml    force the machine learning hidden test on or off (default: automatic)
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

import re

import ml_check
from agents import plan_ml, run_analyst, run_qa, splice_findings
from brains import get_brain
from data_preview import preview_many, stage_data
from executor import (APPROVED_PACKAGES, describe_run, extract_code, install_package,
                      is_installed, parse_requirements, run_job)

MAX_ROUNDS = 3       # QA review rounds
MAX_FIX_TRIES = 3    # times the analyst may fix crashing code within one round
CODE_TIMEOUT = 600   # seconds a script may run; big Excel files need several minutes

# Packages the analyst may use, if they're installed on this computer.
CHECK_PACKAGES = ["pandas", "numpy", "plotly", "streamlit", "pptx", "openpyxl", "matplotlib", "scipy",
                  "statsmodels", "seaborn", "pyarrow", "kaleido", "python_calamine", "xlsxwriter",
                  "sklearn", "joblib", "xgboost", "lightgbm"]


def environment_info() -> str:
    """Describe this computer so the analyst writes code that actually runs here."""
    import importlib.util
    import platform
    import sys

    names = {"pptx": "python-pptx (import pptx)", "sklearn": "scikit-learn (import sklearn)"}
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


def optional_next_steps(work: str, section: str = "optional next steps") -> list:
    """Pull the lines of a write-up section (default 'Optional next steps') out of the deliverable."""
    import re
    match = re.search(r"^\s*#{1,4}\s*(?:\d+[.)]\s*)?" + re.escape(section) + r"\s*$(.*?)(?=^\s*#{1,4}\s|\Z)",
                      work, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if not match:
        return []
    lines = [l.strip().lstrip("-*• ").strip() for l in match.group(1).splitlines()]
    return [l for l in lines if l]


def save(folder: str, name: str, content: str) -> None:
    with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
        f.write(content)


def hidden_test(run, workdir, holdout):
    """ML jobs: score the analyst's model on the rows it never saw, and add the result to the run."""
    result = ml_check.score_model(workdir, holdout)
    run["ml"] = result
    text = ml_check.report(result, holdout, run["stdout"])
    run["stdout"] = (run["stdout"] + "\n\n" + text).strip()
    if result["ok"]:
        main = result["main_metric"]
        print(f"           Hidden test: {main} {ml_check._fmt(main, result['scores'][main])} vs baseline "
              f"{ml_check._fmt(main, holdout['baseline'][main])} "
              f"({'beats' if result['beats_baseline'] else 'does NOT beat'} it)")
    else:
        run["ok"] = False
        run["stderr"] = (run["stderr"] + "\n\n" + result["error"]).strip()
    return run


def enforce_ml(review, run):
    """Rules in code, not the model: a model that fails Jarvis's hidden test is never approved."""
    result = (run or {}).get("ml")
    if not result:
        return review
    if not result.get("ok"):
        point = "BLOCKING (Jarvis hidden test): model.py could not be scored on the hidden rows."
    elif not result["beats_baseline"]:
        point = ("BLOCKING (Jarvis hidden test): the model does not clearly beat the simple baseline "
                 "on rows it never saw. Improve the features or the model. If the data truly can't "
                 "predict this, say so plainly in the findings; Jarvis will hand it to Marro to decide.")
    else:
        return review
    review["feedback"] = [point] + [f for f in review["feedback"] if f != point]
    review["verdict"] = "REVISE"
    review["blocking"] = review.get("blocking", 0) + 1
    return review


FINAL_RUN_NOTE = ("FINAL RUN ON ALL DATA: QA approved your work. Jarvis put the hidden rows back and "
                  "reran your unchanged code, so the model is now trained on every row and every row is "
                  "scored. Update every count and total to THIS output. Keep reporting the JARVIS HIDDEN "
                  "TEST numbers as the model's expected accuracy on new data: they were measured before "
                  "this retrain, on rows the model hadn't seen, so they are still the honest estimate.")


def final_refit(brain, brain_name, brief, data_file, work, run, review, workdir, folder, holdout):
    """ML jobs, after approval: retrain on ALL rows (hidden ones included) and score everyone.

    The hidden test already did its job (an honest accuracy estimate). Holding those rows back
    from the client's model and scores would only make the delivery worse. If anything goes
    wrong, the approved version is kept.
    """
    snapshot = take_snapshot(workdir, "before_final")

    def keep_approved(reason):
        print(f"[Final] {reason} Keeping the approved version "
              f"(trained on {holdout['train_rows']} of {holdout['full_rows']} rows).")
        restore_snapshot(snapshot, workdir)
        return work, run, review

    print(f"[Final] Final step: retraining on all {holdout['full_rows']} rows and scoring everything...")
    ml_check.restore_full(workdir, holdout)
    final = run_job(work, workdir, timeout=CODE_TIMEOUT)
    save(folder, "F_code_run.txt", describe_run(final))
    if not final["ok"]:
        return keep_approved("The final run on all rows failed.")
    hidden = ml_check.report(run["ml"], holdout, run["stdout"])
    final["stdout"] = (final["stdout"] + "\n\n" + hidden +
                       "\n  (Measured BEFORE the final retrain on all rows.)").strip()
    final["ml"] = run["ml"]

    new_work = work
    if brain_name != "mock":
        print("           Analyst is updating the findings for the full data...")
        findings = run_analyst(brain, brief, data_file, previous_work=work,
                               code_output=FINAL_RUN_NOTE + "\n\n" + describe_run(final))
        new_work = splice_findings(work, findings)
        save(folder, "Fz_with_findings.md", new_work)
        print("[Final] QA manager is checking the final version...")
        new_review = run_qa(brain, brief, new_work, FINAL_RUN_NOTE + "\n\n" + describe_run(final),
                            previous_review=review)
        save(folder, "F_qa_review.json", json.dumps(new_review, indent=2))
        if new_review["verdict"] != "APPROVE":
            for point in new_review["feedback"]:
                if point.upper().startswith("BLOCKING"):
                    print(f"           - {point}")
            return keep_approved("QA didn't approve the final version.")
        review = new_review
    _remove_tree(snapshot)   # leftover copies are harmless if a lock keeps one around
    print(f"[Final] Done: model retrained on all {holdout['full_rows']} rows, every row scored.")
    return new_work, final, review


def build(brain, brain_name, brief, data_file, work, workdir, folder, label, decisions, holdout=None):
    """Run the analyst's code, let it fix its own errors, then write findings from real output.

    Returns (work, run). Used by every review round and by the polish round.
    """
    run = None
    for attempt in range(1, MAX_FIX_TRIES + 1):
        package_note = handle_requirements(work, decisions)
        run = run_job(work, workdir, timeout=CODE_TIMEOUT)
        if run["ok"] and holdout:
            run = hidden_test(run, workdir, holdout)
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


def _remove_tree(path: str) -> bool:
    """Delete a folder, tolerating Windows/OneDrive locks. Returns True if it's gone.

    OneDrive (and antivirus) briefly lock files they're syncing or scanning, and Windows
    then refuses the delete. Clear read-only flags and retry for a few seconds.
    """
    import stat
    import time

    def unlock(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    for _ in range(10):
        if not os.path.exists(path):
            return True
        try:
            shutil.rmtree(path, onerror=unlock)
        except OSError:
            pass
        if not os.path.exists(path):
            return True
        time.sleep(0.5)
    return not os.path.exists(path)


def take_snapshot(workdir: str, tag: str) -> str:
    """Copy the work folder aside before a risky step. Always a fresh folder name."""
    snapshot = f"{workdir}_{tag}_{datetime.now():%H%M%S}"
    shutil.copytree(workdir, snapshot)
    return snapshot


def restore_snapshot(snapshot: str, workdir: str) -> bool:
    """Put the snapshot back as the work folder. Returns True if it worked."""
    try:
        if not _remove_tree(workdir):
            # Couldn't clear it (files locked): copy the approved files back on top instead.
            shutil.copytree(snapshot, workdir, dirs_exist_ok=True)
        else:
            shutil.copytree(snapshot, workdir)
    except OSError as e:
        print(f"           Couldn't restore automatically ({e}). The approved version is in {snapshot}")
        return False
    _remove_tree(snapshot)
    return True


def polish_round(brain, brain_name, brief, data_file, work, run, review, workdir, folder, decisions,
                 holdout=None):
    """After approval, let the analyst apply QA's suggestions. Keep the result only if it's better.

    The approved work is snapshotted first, so a polish that breaks something or
    scores lower is thrown away and the approved version is restored.
    """
    snapshot = take_snapshot(workdir, "approved")

    def restore(reason):
        print(f"           {reason} Keeping the approved version.")
        restore_snapshot(snapshot, workdir)
        return work, run, review, False

    print("[Polish] Analyst is applying QA's suggestions...")
    suggestions = [f for f in review["feedback"] if not f.strip().upper().startswith("BLOCKING")]
    polished = run_analyst(brain, brief, data_file, previous_work=work,
                           qa_feedback=suggestions, polish=True)
    polished, prun = build(brain, brain_name, brief, data_file, polished, workdir, folder, "P", decisions,
                           holdout)
    if not prun["ok"]:
        return restore("The polished code didn't run.")

    print("[Polish] QA manager is checking the polished version...")
    new_review = enforce_ml(run_qa(brain, brief, polished, describe_run(prun), previous_review=review), prun)
    save(folder, "P_qa_review.json", json.dumps(new_review, indent=2))
    print(f"           Score {new_review['score']}/10 -> {new_review['verdict']} "
          f"(was {review['score']}/10)")
    if new_review["verdict"] != "APPROVE" or new_review["score"] < review["score"]:
        return restore("The polish didn't improve the score.")

    _remove_tree(snapshot)   # leftover copies are harmless if a lock keeps one around
    for point in new_review["feedback"]:
        print(f"           - {point}")
    return polished, prun, new_review, True


_ML_LINE = re.compile(r"^\s*ML (?:job|involved)\s*:\s*(core|minor)", re.IGNORECASE | re.MULTILINE)
_ML_WORDS = re.compile(r"\b(predict\w*|forecast\w*|machine learning|classif\w+|regression model|"
                       r"churn model|propensity|scoring model|ML model)\b", re.IGNORECASE)


def ml_wanted(brief: str, ml=None) -> bool:
    """Should Jarvis check for an ML job? --ml / --no-ml win; else the grader's note or the brief's words."""
    if ml is not None:
        return ml
    return bool(_ML_LINE.search(brief) or _ML_WORDS.search(brief))


def setup_hidden_test(brain, brief, profile, workdir, files, folder):
    """Decide what to predict, then hide test rows from the analyst. Returns (holdout, files)."""
    print("Jarvis: this looks like it may need a model. Setting up a hidden test set...")
    try:
        raw_plan = plan_ml(brain, brief + "\n\nDATA PREVIEW:\n" + profile)
        plan, problem = ml_check.validate_plan(raw_plan, workdir, files)
        if not plan:
            print(f"Jarvis: no hidden test set ({problem}).")
            save(folder, "00_ml_plan.json", json.dumps({"plan": raw_plan, "skipped": problem}, indent=2))
            return None, files
        holdout = ml_check.hold_out(workdir, plan)
        if "skipped" in holdout:
            print(f"Jarvis: no hidden test set ({holdout['skipped']}).")
            save(folder, "00_ml_plan.json", json.dumps({"plan": plan, "skipped": holdout["skipped"]}, indent=2))
            return None, files
    except Exception as e:   # never let the ML setup kill a job
        print(f"Jarvis: couldn't set up a hidden test set ({type(e).__name__}: {e}).")
        return None, files
    new_name = holdout["plan"]["file"]
    files = [new_name if f == holdout["original_file"] else f for f in files]
    base = holdout["baseline"]
    main = ml_check.MAIN_METRIC[plan["task"]]
    save(folder, "00_ml_plan.json", json.dumps({
        "plan": holdout["plan"], "train_rows": holdout["train_rows"], "hidden_rows": holdout["test_rows"],
        "split": holdout["split_note"], "baseline": base}, indent=2, default=str))
    print(f"Jarvis: {plan['task']} on '{plan['target']}'. Hid {holdout['test_rows']} of "
          f"{holdout['test_rows'] + holdout['train_rows']} rows ({holdout['split_note']}). "
          f"Baseline {main}: {ml_check._fmt(main, base[main])}.")
    return holdout, files


def learn(memory, brief, work, reviews, status, rounds, brain, brain_name):
    """After a job: save what it taught to memory (never stops the job if it fails)."""
    if memory is None:
        return
    try:
        from memory import remember_job
        distiller = brain if brain_name != "mock" else None   # mock answers are canned, not lessons
        saved = remember_job(memory, brief, work, reviews, status, rounds, distiller)
        print(f"\nJarvis: job saved to memory ({saved} new lesson{'s' if saved != 1 else ''}).")
    except Exception as e:
        print(f"\nJarvis: couldn't save to memory ({type(e).__name__}: {e}).")


def run_pipeline(brief: str, brain_name: str = "mock", data_path: str = "",
                 brief_path: str = "", polish: bool = True, use_memory: bool = True,
                 ml=None) -> dict:
    brain = get_brain(brain_name)
    folder = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    workdir = os.path.join(folder, "work")
    os.makedirs(workdir, exist_ok=True)
    save(folder, "00_brief.txt", brief)

    data_file = ""
    holdout, ml_section = None, ""
    if data_path:
        if not os.path.exists(data_path):
            raise SystemExit(f"Jarvis: can't find the data at {data_path}")
        # Copy the client's file, folder or zip into the work folder (zips get unzipped).
        files = stage_data(data_path, workdir, exclude=[brief_path])
        if not files:
            raise SystemExit(f"Jarvis: no data files found in {data_path}")
        print(f"Jarvis: staged {len(files)} data file(s): {', '.join(files)}")
        # Show the agents what's really in the data, so nobody guesses column names.
        profile = preview_many(workdir, files)

        # ML jobs: hide test rows BEFORE the analyst sees anything (the preview is rebuilt after).
        if ml_wanted(brief, ml):
            holdout, files = setup_hidden_test(brain, brief, profile, workdir, files, folder)
            if holdout:
                profile = preview_many(workdir, files)
                ml_section = ml_check.brief_section(holdout)

        data_file = ", ".join(files)
        save(folder, "00_data_preview.txt", profile)
        brief = brief + "\n\nDATA PREVIEW (generated automatically by Jarvis):\n" + profile
        if ml_section:
            brief = brief + "\n\n" + ml_section
        print("Jarvis: data preview built and shared with the agents.")

    install_theme(workdir)
    env = environment_info()
    save(folder, "00_environment.txt", env)
    brief = brief + "\n\nRUNTIME ENVIRONMENT (your code runs here):\n" + env

    # Memory: recall lessons and similar jobs from the past (never blocks a job).
    memory = None
    if use_memory:
        try:
            from memory import Memory, recall_for_job
            memory = Memory()
            recalled = recall_for_job(memory, brief)
            if recalled:
                brief = brief + "\n\n" + recalled
                save(folder, "00_memory.txt", recalled)
                n_lessons = recalled.split("Similar past jobs:")[0].count("\n- ")
                n_jobs = recalled.split("Similar past jobs:")[1].count("\n- ") if "Similar past jobs:" in recalled else 0
                print(f"Jarvis: recalled {n_lessons} lesson(s) and {n_jobs} similar past job(s) from memory.")
            elif len(memory):
                print("Jarvis: nothing in memory closely matches this job.")
        except Exception as e:
            print(f"Jarvis: memory unavailable this run ({type(e).__name__}: {e}).")
            memory = None

    print(f"Jarvis: starting job with the '{brain_name}' brain. Saving to {folder}\n")

    work, review, run = "", None, None
    reviews = []             # every QA review, for memory
    package_decisions = {}   # your install answers, remembered for this job
    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"[Round {round_num}] Analyst is working...")
        work = run_analyst(brain, brief, data_file, previous_work=work,
                           qa_feedback=review["feedback"] if review else None)

        work, run = build(brain, brain_name, brief, data_file, work, workdir, folder,
                          f"{round_num:02d}", package_decisions, holdout)

        print(f"[Round {round_num}] QA manager is reviewing the work and its output...")
        review = enforce_ml(run_qa(brain, brief, work, describe_run(run), previous_review=review), run)
        reviews.append(review)
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
                approved_review = review
                work, run, review, polished = polish_round(
                    brain, brain_name, brief, data_file, work, run, review,
                    workdir, folder, package_decisions, holdout)
                if review is not approved_review:
                    reviews.append(review)
                print()
            if holdout:
                work, run, review = final_refit(brain, brain_name, brief, data_file, work, run, review,
                                                workdir, folder, holdout)
            save(folder, "FINAL_deliverable.md", work)
            note = " (polished)" if polished else ""
            print(f"Jarvis: approved in round {round_num}{note}. Final deliverable is in {folder}")
            if holdout and (run or {}).get("ml", {}).get("ok"):
                result = run["ml"]
                main = result["main_metric"]
                print(f"\nHidden test ({holdout['test_rows']} rows the analyst never saw): {main} "
                      f"{ml_check._fmt(main, result['scores'][main])} vs baseline "
                      f"{ml_check._fmt(main, holdout['baseline'][main])}.")
                for w in result["warnings"]:
                    print(f"  Warning: {w}")
            if os.path.exists(os.path.join(workdir, "app.py")):
                write_launcher(workdir)
                print("\nTo open the report app, double-click OPEN_REPORT.bat in the work folder, or run:")
                print(f'  cd "{os.path.abspath(workdir)}"')
                print("  python -m streamlit run app.py")
            questions = optional_next_steps(work, "questions for the client")
            if questions:
                print("\nQuestions to include in your message to the client:")
                for line in questions:
                    print(f"  - {line}")
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
            learn(memory, brief, work, reviews, "approved", round_num, brain, brain_name)
            return {"status": "approved", "rounds": round_num, "folder": folder}

    save(folder, "NEEDS_YOUR_REVIEW.md", work)
    print(f"Jarvis: not approved after {MAX_ROUNDS} rounds. Your call. "
          f"Latest draft: {os.path.join(folder, 'NEEDS_YOUR_REVIEW.md')}")
    learn(memory, brief, work, reviews, "not approved", MAX_ROUNDS, brain, brain_name)
    return {"status": "needs_review", "rounds": MAX_ROUNDS, "folder": folder}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a client brief through Jarvis's agents.")
    parser.add_argument("brief_file", help="text file containing the client brief")
    parser.add_argument("--data", default="",
                        help="the client's data: a file, a folder of files, or a zip")
    parser.add_argument("--brain", default="mock", choices=["mock", "ollama", "claude"])
    parser.add_argument("--no-polish", action="store_true",
                        help="skip the polish round after approval (faster, cheaper)")
    parser.add_argument("--no-memory", action="store_true",
                        help="don't recall or save memories for this run")
    parser.add_argument("--ml", dest="ml", action="store_true", default=None,
                        help="this job needs a model: always set up a hidden test set")
    parser.add_argument("--no-ml", dest="ml", action="store_false",
                        help="never set up a hidden test set")
    args = parser.parse_args()

    with open(args.brief_file, encoding="utf-8") as f:
        run_pipeline(f.read(), args.brain, args.data, brief_path=args.brief_file,
                     polish=not args.no_polish, use_memory=not args.no_memory, ml=args.ml)
