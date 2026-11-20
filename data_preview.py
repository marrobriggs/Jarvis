"""
Data preview: shows the agents what's actually in the client's file.

Without this, the analyst only sees the brief and has to guess column
names. With it, the analyst gets the columns, data types, missing values
and a few sample rows before writing a single line of code.
"""

import os
import shutil

PREVIEW_ROWS = 2000  # rows read for the preview; enough to profile, fast on big files


def _profile(df, label: str) -> str:
    import pandas as pd

    lines = [f"{label}: previewed {len(df):,} rows x {len(df.columns)} columns"]
    lines.append("\nColumns (type, missing values in preview, example):")
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        example = series.dropna().iloc[0] if series.notna().any() else "(all empty)"
        lines.append(f"  - {col!r}: {series.dtype}, {missing} missing, e.g. {str(example)[:40]!r}")
    with pd.option_context("display.max_columns", 30, "display.width", 200):
        lines.append("\nFirst 5 rows:\n" + df.head(5).to_string(max_colwidth=30))
    return "\n".join(lines)


def _describe_other(path: str, name: str, ext: str) -> str:
    """For non-table files: show text files' opening lines, or flag binary files."""
    size_kb = os.path.getsize(path) / 1024
    header = f"File: {name} ({ext or 'no extension'}, {size_kb:,.0f} KB). Not a table format."
    with open(path, "rb") as f:
        sample = f.read(4000)
    try:
        text = sample.decode("utf-8")
    except UnicodeDecodeError:
        return (header + " It is a binary file (for example a PDF, image or archive), so it "
                "can't be previewed as text. Inspect it in code, and tell the client if it "
                "needs to be converted into a table first.")
    opening = "\n".join(text.splitlines()[:30])
    hint = " It looks like JSON; consider pandas.json_normalize." if ext == ".json" else ""
    return f"{header}{hint}\nFirst lines of the file:\n{opening}"


def preview(path: str) -> str:
    """Return a plain-text profile of a CSV or Excel file for the agents to read."""
    if not path or not os.path.exists(path):
        return ""
    name = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()
    try:
        import pandas as pd

        if ext in (".xlsx", ".xlsm", ".xls"):
            book = pd.ExcelFile(path)
            parts = [f"File: {name} (Excel). Sheets: {', '.join(book.sheet_names)}"]
            for sheet in book.sheet_names[:3]:
                df = pd.read_excel(book, sheet_name=sheet, nrows=PREVIEW_ROWS)
                parts.append(_profile(df, f"\nSheet {sheet!r}"))
            if len(book.sheet_names) > 3:
                parts.append(f"\n(+{len(book.sheet_names) - 3} more sheets not shown)")
            return "\n".join(parts)

        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            with open(path, "rb") as f:
                total_rows = max(sum(1 for _ in f) - 1, 0)
            try:
                df = pd.read_csv(path, nrows=PREVIEW_ROWS, sep=sep)
            except UnicodeDecodeError:
                df = pd.read_csv(path, nrows=PREVIEW_ROWS, sep=sep, encoding="latin-1")
                name += " (note: not UTF-8, read with encoding='latin-1')"
            return f"File: {name}, about {total_rows:,} data rows in total.\n" + _profile(df, "Preview")

        if ext == ".parquet":
            df = pd.read_parquet(path).head(PREVIEW_ROWS)
            return f"File: {name} (Parquet).\n" + _profile(df, "Preview")

        # Not a table format: describe the file instead of forcing it into a table.
        return _describe_other(path, name, ext)

    except Exception as e:  # never let the preview stop a job
        return f"File: {name}. Automatic preview failed ({type(e).__name__}: {e}). Inspect it in code first."


# ---------------------------------------------------------------------------
# Staging: copy a client's file, folder or zip into the job's work folder
# ---------------------------------------------------------------------------

MAX_FILES_PREVIEWED = 10
SKIP_NAMES = {"brief.txt", "desktop.ini", "thumbs.db", ".ds_store"}


def _safe_extract(zip_path: str, dest: str) -> None:
    """Unzip, refusing any entry that would land outside dest (a 'zip slip')."""
    import zipfile

    dest_root = os.path.realpath(dest)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            target = os.path.realpath(os.path.join(dest, member.filename))
            if not target.startswith(dest_root + os.sep) and target != dest_root:
                raise ValueError(f"Unsafe path in zip: {member.filename}")
            if member.filename.startswith("__MACOSX/"):
                continue  # Mac clutter, not data
            z.extract(member, dest)


def stage_data(data_path: str, workdir: str, exclude=()) -> list:
    """Copy client data into workdir and return the data files' paths, relative to workdir.

    Accepts a single file, a folder of files, or zip files (which are unzipped).
    """
    exclude = {os.path.realpath(p) for p in exclude if p}
    sources = []
    if os.path.isdir(data_path):
        for root, dirs, files in os.walk(data_path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                full = os.path.join(root, f)
                if f.lower() in SKIP_NAMES or f.startswith(".") or os.path.realpath(full) in exclude:
                    continue
                sources.append((full, os.path.relpath(full, data_path)))
    else:
        sources.append((data_path, os.path.basename(data_path)))

    for src, rel in sources:
        dest = os.path.join(workdir, rel)
        os.makedirs(os.path.dirname(dest) or workdir, exist_ok=True)
        if src.lower().endswith(".zip"):
            _safe_extract(src, os.path.dirname(dest) or workdir)
        else:
            shutil.copy(src, dest)

    staged = []
    for root, dirs, files in os.walk(workdir):
        for f in files:
            if f.endswith(".py") or f.startswith("."):
                continue
            staged.append(os.path.relpath(os.path.join(root, f), workdir).replace(os.sep, "/"))
    return sorted(staged)


def preview_many(workdir: str, files: list) -> str:
    """Preview every staged file (dictionaries and readme files first)."""
    def priority(name):
        low = name.lower()
        return 0 if ("dictionary" in low or "readme" in low or "description" in low) else 1

    ordered = sorted(files, key=lambda f: (priority(f), f))
    parts = [f"{len(files)} data file(s) in the working folder: " + ", ".join(files)]
    for rel in ordered[:MAX_FILES_PREVIEWED]:
        parts.append("\n" + "=" * 60 + "\n" + preview(os.path.join(workdir, rel)))
    if len(files) > MAX_FILES_PREVIEWED:
        parts.append(f"\n(+{len(files) - MAX_FILES_PREVIEWED} more files not previewed)")
    return "\n".join(parts)
