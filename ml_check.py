"""
ml_check: Jarvis's independent test for machine learning jobs.

The rule real ML teams use: the people who build a model never get to grade it
on data they've seen. So before the analyst touches the data, Jarvis:

  1. hides a slice of the rows (the "hidden test set"), kept only in memory,
     never written into the analyst's work folder,
  2. scores a simple baseline on it (always guess the average / the most common class),
  3. after the analyst's code runs, calls the analyst's model.py predict(df) on the
     hidden rows WITHOUT the answers, and scores the predictions itself.

QA then judges the work using numbers the analyst couldn't fudge.

Uses only pandas and numpy, so it works even if scikit-learn isn't installed.
"""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

TEST_SHARE = 0.2      # share of rows hidden from the analyst
MIN_ROWS = 100        # below this, a hidden test set would be too small to mean anything
MIN_TEST_ROWS = 20
TASKS = ("regression", "classification", "forecast")
READABLE = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet")

# Higher-is-better metrics; the rest (mae, rmse, mape) are lower-is-better.
HIGHER_IS_BETTER = {"r2", "accuracy", "balanced_accuracy", "macro_f1"}
MAIN_METRIC = {"regression": "mae", "forecast": "mae", "classification": "balanced_accuracy"}


# ---------------------------------------------------------------------------
# Reading and writing the client's table
# ---------------------------------------------------------------------------

def _read(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)            # first sheet
    if ext == ".parquet":
        return pd.read_parquet(path)
    sep = "\t" if ext == ".tsv" else None
    return pd.read_csv(path, sep=sep, engine="python" if sep is None else "c")


def _write(df: pd.DataFrame, path: str) -> str:
    """Write the training rows back. Excel/TSV/TXT come back as CSV (simpler, and one sheet only)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df.to_parquet(path, index=False)
        return path
    if ext == ".csv":
        df.to_csv(path, index=False)
        return path
    new_path = os.path.splitext(path)[0] + ".csv"
    df.to_csv(new_path, index=False)
    os.remove(path)
    return new_path


# ---------------------------------------------------------------------------
# Step 1: check the plan and hide the test rows
# ---------------------------------------------------------------------------

def validate_plan(plan: dict, workdir: str, files: list) -> tuple:
    """Check the ML plan against the real data. Returns (plan, problem)."""
    if not isinstance(plan, dict) or not plan.get("needs_model"):
        return None, "no model is needed for this brief"
    task = str(plan.get("task", "")).lower().strip()
    if task not in TASKS:
        return None, f"unknown task '{task}'"
    name = str(plan.get("file", "")).strip()
    if name not in files:
        matches = [f for f in files if os.path.basename(f) == os.path.basename(name)]
        if not matches:
            return None, f"the plan names a data file that isn't there ('{name}')"
        name = matches[0]
    if os.path.splitext(name)[1].lower() not in READABLE:
        return None, f"{name} isn't a table Jarvis can split"
    df = _read(os.path.join(workdir, name))
    target = str(plan.get("target", "")).strip()
    if target not in df.columns:
        return None, f"target column '{target}' isn't in {name}"
    time_col = str(plan.get("time_column") or "").strip()
    if time_col and time_col not in df.columns:
        time_col = ""
    split = "time" if (task == "forecast" or plan.get("split") == "time") and time_col else "random"
    if task == "forecast" and not time_col:
        return None, "a forecast needs a date column to split on, and none was found"
    clean = {"needs_model": True, "task": task, "file": name, "target": target,
             "time_column": time_col, "split": split, "why": str(plan.get("why", ""))[:300]}
    return clean, ""


def hold_out(workdir: str, plan: dict, seed: int = 42) -> dict:
    """Hide TEST_SHARE of the rows. The work folder keeps only the training rows.

    Returns a holdout dict (kept in memory by the orchestrator), or {"skipped": reason}.
    """
    path = os.path.join(workdir, plan["file"])
    df = _read(path)
    df = df[df[plan["target"]].notna()].reset_index(drop=True)
    if len(df) < MIN_ROWS:
        return {"skipped": f"only {len(df)} rows with a known {plan['target']}, too few for a fair hidden test"}

    n_test = max(MIN_TEST_ROWS, int(round(len(df) * TEST_SHARE)))
    if plan["split"] == "time":
        times = pd.to_datetime(df[plan["time_column"]], errors="coerce")
        if times.isna().mean() > 0.2:
            return {"skipped": f"couldn't read the dates in {plan['time_column']}"}
        order = times.sort_values(kind="stable").index
        cutoff = times.loc[order[len(df) - n_test]]
        test_idx = times.index[times >= cutoff]          # whole periods, never split a day in two
        split_note = f"the most recent rows (from {cutoff.date()} on)"
    elif plan["task"] == "classification":
        rng = np.random.default_rng(seed)
        test_idx = []
        for _, group in df.groupby(plan["target"]):      # stratified: keep class shares the same
            k = int(round(len(group) * n_test / len(df)))
            test_idx.extend(rng.choice(group.index, size=min(k, len(group) - 1), replace=False)
                            if len(group) > 1 else [])
        test_idx = pd.Index(sorted(test_idx))
        split_note = "a random sample, with the same mix of classes"
    else:
        test_idx = df.sample(n=n_test, random_state=seed).index
        split_note = "a random sample"

    test = df.loc[test_idx].reset_index(drop=True)
    train = df.drop(index=test_idx).reset_index(drop=True)
    if len(test) < MIN_TEST_ROWS or len(train) < MIN_TEST_ROWS:
        return {"skipped": "the split left too few rows on one side"}

    new_path = _write(train, path)
    new_name = os.path.relpath(new_path, workdir).replace("\\", "/")
    holdout = {
        "plan": dict(plan, file=new_name),
        "original_file": plan["file"],
        "features": test.drop(columns=[plan["target"]]),
        "answers": test[plan["target"]].reset_index(drop=True),
        "train_rows": len(train), "test_rows": len(test), "split_note": split_note,
    }
    holdout["baseline"] = baseline_scores(train, test, plan)
    return holdout


def baseline_scores(train: pd.DataFrame, test: pd.DataFrame, plan: dict) -> dict:
    """The dumbest reasonable model. A real model has to beat it to be worth anything."""
    y_test = test[plan["target"]]
    if plan["task"] == "classification":
        common = train[plan["target"]].mode().iloc[0]
        guess = pd.Series([common] * len(test))
        return {"name": f"always guess the most common class ({common})", **metrics(y_test, guess, "classification")}
    y_train = pd.to_numeric(train[plan["target"]], errors="coerce")
    if plan["task"] == "forecast":
        times = pd.to_datetime(train[plan["time_column"]], errors="coerce")
        recent = y_train[times >= times.quantile(0.9)]
        value, name = recent.mean(), "repeat the average of the most recent 10% of training rows"
    else:
        value, name = y_train.mean(), "always guess the training average"
    guess = pd.Series([value] * len(test))
    return {"name": name, **metrics(y_test, guess, plan["task"])}


# ---------------------------------------------------------------------------
# Metrics (plain numpy, so Jarvis doesn't depend on scikit-learn)
# ---------------------------------------------------------------------------

def metrics(y_true, y_pred, task: str) -> dict:
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    if task == "classification":
        t, p = y_true.astype(str), y_pred.astype(str)
        classes = sorted(set(t))
        recalls, f1s = [], []
        for c in classes:
            tp = ((t == c) & (p == c)).sum()
            fn = ((t == c) & (p != c)).sum()
            fp = ((t != c) & (p == c)).sum()
            recall = tp / (tp + fn) if tp + fn else 0.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            recalls.append(recall)
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return {"accuracy": float((t == p).mean()), "balanced_accuracy": float(np.mean(recalls)),
                "macro_f1": float(np.mean(f1s))}
    t = pd.to_numeric(y_true, errors="coerce").to_numpy(float)
    p = pd.to_numeric(y_pred, errors="coerce").to_numpy(float)
    ok = ~(np.isnan(t) | np.isnan(p))
    t, p = t[ok], p[ok]
    err = t - p
    ss_tot = ((t - t.mean()) ** 2).sum()
    out = {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
           "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot else 0.0}
    nonzero = t != 0
    if nonzero.mean() > 0.9:
        out["mape"] = float(np.abs(err[nonzero] / t[nonzero]).mean())
    return out


def _better(metric: str, model: float, baseline: float) -> float:
    """How much better the model is than the baseline, as a fraction (+0.25 = 25% better)."""
    if metric in HIGHER_IS_BETTER:
        return (model - baseline) / abs(baseline) if baseline else model
    return (baseline - model) / baseline if baseline else 0.0


# ---------------------------------------------------------------------------
# Step 2: score the analyst's model on the hidden rows
# ---------------------------------------------------------------------------

_SCORER = r"""
import json, sys, traceback
import pandas as pd
try:
    import model
    X = pd.read_pickle("_jarvis_hidden_features.pkl")
    preds = model.predict(X.copy())
    preds = pd.Series(list(preds) if not isinstance(preds, pd.Series) else preds.tolist())
    preds.to_frame("prediction").to_pickle("_jarvis_predictions.pkl")
    print(json.dumps({"ok": True, "n": len(preds)}))
except Exception:
    print(json.dumps({"ok": False, "error": traceback.format_exc()[-3000:]}))
"""


def score_model(workdir: str, holdout: dict, timeout: int = 300) -> dict:
    """Run model.py's predict() on the hidden rows (answers never leave this process)."""
    from executor import _safe_env
    if not os.path.exists(os.path.join(workdir, "model.py")):
        return {"ok": False, "error": "No model.py was found. ML jobs need a '# file: model.py' block "
                                      "with a predict(df) function (see the ML JOB section of the brief)."}
    feats = os.path.join(workdir, "_jarvis_hidden_features.pkl")
    preds_path = os.path.join(workdir, "_jarvis_predictions.pkl")
    holdout["features"].to_pickle(feats)
    try:
        result = subprocess.run([sys.executable, "-c", _SCORER], cwd=workdir, env=_safe_env(),
                                capture_output=True, text=True, timeout=timeout)
        lines = [l for l in result.stdout.splitlines() if l.startswith("{")]
        info = json.loads(lines[-1]) if lines else {"ok": False, "error": result.stderr[-3000:] or "no output"}
        if not info.get("ok"):
            return {"ok": False, "error": "model.py predict(df) failed on the hidden rows:\n" + info["error"]}
        preds = pd.read_pickle(preds_path)["prediction"]
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"model.py predict(df) took longer than {timeout} seconds."}
    finally:
        for p in (feats, preds_path):        # the hidden rows never stay in the work folder
            if os.path.exists(p):
                os.remove(p)

    if len(preds) != holdout["test_rows"]:
        return {"ok": False, "error": f"predict(df) returned {len(preds)} predictions for "
                                      f"{holdout['test_rows']} rows. Return exactly one per row, in order."}
    task = holdout["plan"]["task"]
    if task != "classification" and pd.to_numeric(preds, errors="coerce").isna().mean() > 0.05:
        return {"ok": False, "error": "predict(df) returned values that aren't numbers."}
    scores = metrics(holdout["answers"], preds, task)
    main = MAIN_METRIC[task]
    gain = _better(main, scores[main], holdout["baseline"][main])
    return {"ok": True, "scores": scores, "main_metric": main, "gain_vs_baseline": gain,
            "beats_baseline": gain > 0.02, "warnings": _warnings(scores, holdout, task)}


def _warnings(scores: dict, holdout: dict, task: str) -> list:
    warnings = []
    if task == "classification" and scores["accuracy"] >= 0.99:
        warnings.append("Near-perfect accuracy on unseen rows. Check for leakage: a column that "
                        "gives away the answer (an ID, a status set after the fact, the target in disguise).")
    if task != "classification" and scores["r2"] >= 0.98:
        warnings.append("R² of 0.98+ on unseen rows. Check for leakage: a column that is the target "
                        "in disguise, or information from after the moment of prediction.")
    return warnings


_VALIDATION = re.compile(r"VALIDATION SCORE:\s*([A-Za-z_² ]+?)\s*=\s*(-?[\d.]+)", re.IGNORECASE)
_ALIASES = {"r²": "r2", "r2": "r2", "r_squared": "r2", "mae": "mae", "rmse": "rmse", "mape": "mape",
            "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "macro_f1",
            "f1": "macro_f1", "f1_macro": "macro_f1"}


def validation_gap(stdout: str, result: dict) -> str:
    """Compare the analyst's own validation score with Jarvis's hidden test. A big gap = overfitting."""
    found = {}
    for name, value in _VALIDATION.findall(stdout or ""):
        key = _ALIASES.get(name.strip().lower().replace(" ", "_"))
        if key:
            found[key] = float(value)
    notes = []
    for key, own in found.items():
        hidden = result["scores"].get(key)
        if hidden is None:
            continue
        if key in HIGHER_IS_BETTER:
            gap = own - hidden
            if gap > 0.1:
                notes.append(f"{key}: analyst's validation {own:.3f} vs hidden test {hidden:.3f}. "
                             "The model does much worse on truly new rows (overfitting or leakage).")
        elif hidden > own * 1.3 and hidden - own > 1e-9:
            notes.append(f"{key}: analyst's validation {own:,.3f} vs hidden test {hidden:,.3f}. "
                         "Errors are 30%+ bigger on truly new rows (overfitting or leakage).")
    if not found:
        notes.append("The analyst printed no 'VALIDATION SCORE: metric = value' line, so its own "
                     "validation couldn't be compared with the hidden test.")
    return "\n".join(notes)


# ---------------------------------------------------------------------------
# Reports for the agents
# ---------------------------------------------------------------------------

def _fmt(metric: str, value: float) -> str:
    if metric in ("accuracy", "balanced_accuracy", "macro_f1", "mape"):
        return f"{value * 100:.1f}%"
    if metric == "r2":
        return f"{value:.3f}"
    return f"{value:,.3f}" if abs(value) < 100 else f"{value:,.0f}"


def brief_section(holdout: dict) -> str:
    """What the analyst is told about the ML job (never the hidden rows themselves)."""
    p = holdout["plan"]
    cols = ", ".join(map(str, holdout["features"].columns[:40]))
    renamed = (f" (it was converted from {holdout['original_file']})"
               if holdout["original_file"] != p["file"] else "")
    return f"""ML JOB (set up by Jarvis):
- Task: {p['task']}. Target column: '{p['target']}' in {p['file']}{renamed}.
- Jarvis has HIDDEN {holdout['test_rows']} rows ({holdout['split_note']}) to test your model
  independently. You have {holdout['train_rows']} rows. Do your own validation inside them.
- You MUST write "# file: model.py" with a function predict(df) that takes a DataFrame with
  exactly these columns: {cols}
  (the same layout as {p['file']} without '{p['target']}'), and returns one prediction per row,
  in the same order. It must load the model your analysis.py saved (e.g. with joblib), using a path
  built from os.path.dirname(os.path.abspath(__file__)), and do the same preprocessing. It must not
  read the training data or retrain.
- Print your own validation result in this exact form: VALIDATION SCORE: {MAIN_METRIC[p['task']]} = <number>
- Jarvis's baseline on the hidden rows ({holdout['baseline']['name']}):
  {MAIN_METRIC[p['task']]} = {_fmt(MAIN_METRIC[p['task']], holdout['baseline'][MAIN_METRIC[p['task']]])}. Your model has to beat it clearly."""


def report(result: dict, holdout: dict, stdout: str = "") -> str:
    """The hidden-test result, as text for QA and for the findings."""
    p = holdout["plan"]
    base = holdout["baseline"]
    lines = [f"JARVIS HIDDEN TEST ({p['task']}, target '{p['target']}', {holdout['test_rows']} rows "
             f"the analyst never saw: {holdout['split_note']}):"]
    if not result.get("ok"):
        lines.append("FAILED: " + result.get("error", "unknown error"))
        return "\n".join(lines)
    for metric, value in result["scores"].items():
        lines.append(f"  {metric}: model {_fmt(metric, value)} | baseline {_fmt(metric, base[metric])}")
    main = result["main_metric"]
    verdict = "BEATS" if result["beats_baseline"] else "DOES NOT BEAT"
    lines.append(f"  Main metric ({main}): the model {verdict} the baseline "
                 f"({result['gain_vs_baseline'] * 100:+.1f}%). Baseline = {base['name']}.")
    for w in result["warnings"]:
        lines.append("  LEAKAGE WARNING: " + w)
    gap = validation_gap(stdout, result)
    if gap:
        lines.append("  VALIDATION CHECK: " + gap.replace("\n", "\n  VALIDATION CHECK: "))
    lines.append("  The deliverable must report these hidden-test numbers as the model's expected "
                 "accuracy on new data.")
    return "\n".join(lines)
