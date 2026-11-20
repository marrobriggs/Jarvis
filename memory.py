"""
Jarvis memory: retrieval-augmented generation (RAG) over past jobs.

What it remembers:
  - lesson : a reusable lesson from a past job's QA reviews
             ("KPI cards hold one short number; put comparisons in the note")
  - job    : a past job's brief, the deliverable chosen and how it went
  - gig    : a graded listing and Marro's own yes/no decision

How it works:
  1. Each memory's text is turned into an EMBEDDING: a list of numbers that
     captures its meaning, so similar texts get similar numbers.
  2. Embeddings are stored in a small local VECTOR STORE (memory/ folder).
  3. A new job's brief is embedded the same way, and the closest memories are
     found by COSINE SIMILARITY, then added to the agents' instructions.

This is the same idea a vector database like Chroma implements; the store here
is built by hand so it runs anywhere with just numpy, and its add()/query()
interface mirrors Chroma's so it could be swapped in later.

Embeddings come from the small "nomic-embed-text" model in Ollama
(run: ollama pull nomic-embed-text). If Ollama isn't available, a simple
keyword-based embedding is used instead, so memory never blocks a job.

Command line:
  python memory.py --stats                   what's stored
  python memory.py --search "pizza sales"    try a search
  python memory.py --backfill                learn from existing runs/ and graded_gigs.csv
  python memory.py --backfill --brain claude also distill general lessons with Claude
"""

import hashlib
import json
import os
import re
import urllib.request
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(HERE, "memory")
OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


# ---------------------------------------------------------------------------
# Embedders: text -> vector
# ---------------------------------------------------------------------------

class OllamaEmbedder:
    """Meaning-based embeddings from nomic-embed-text running in Ollama (free, local)."""
    name = f"ollama:{EMBED_MODEL}"
    min_score = 0.55          # similarity below this isn't really "related"

    def embed(self, texts, query=False):
        prefix = "search_query: " if query else "search_document: "   # what nomic expects
        payload = {"model": EMBED_MODEL, "input": [prefix + t[:6000] for t in texts]}
        req = urllib.request.Request(OLLAMA_EMBED_URL, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=120) as resp:
            vectors = json.loads(resp.read().decode("utf-8"))["embeddings"]
        return np.array(vectors, dtype=np.float32)


class KeywordEmbedder:
    """Fallback: hashes words into a fixed-size vector. Matches shared words, not meaning."""
    name = "keyword-hash-512"
    min_score = 0.33          # calibrated: related briefs scored 0.44-0.56, unrelated mostly < 0.3
    dims = 512
    _stop = set("a an and are as at be by for from has have in is it its of on or that the this "
                "to was we were will with you your our their they i want wants need needs owner "
                "company business client looking look quick help please would like".split())

    def embed(self, texts, query=False):
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in self._stop and len(w) > 2]
            for w in words:
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[row, h % self.dims] += 1.0 if (h >> 9) % 2 else -1.0
        return out


def pick_embedder():
    """Use Ollama embeddings when they're available; otherwise the keyword fallback."""
    try:
        OllamaEmbedder().embed(["ping"])
        return OllamaEmbedder()
    except Exception:
        return KeywordEmbedder()


# ---------------------------------------------------------------------------
# The vector store
# ---------------------------------------------------------------------------

def _normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class Memory:
    """A small vector store kept in memory/ as records.json plus vectors.npy."""

    def __init__(self, folder=MEMORY_DIR, embedder=None):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.records_path = os.path.join(folder, "records.json")
        self.vectors_path = os.path.join(folder, "vectors.npy")
        self.embedder = embedder or pick_embedder()
        self.records, self.vectors = [], None
        if os.path.exists(self.records_path):
            with open(self.records_path, encoding="utf-8") as f:
                saved = json.load(f)
            self.records = saved.get("records", [])
            if os.path.exists(self.vectors_path) and self.records:
                self.vectors = np.load(self.vectors_path)
            # Vectors from a different embedder can't be compared: re-embed everything.
            if saved.get("embedder") != self.embedder.name and self.records:
                self._reindex()

    # -- storage ---------------------------------------------------------------
    def _save(self):
        tmp = self.records_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"embedder": self.embedder.name, "records": self.records}, f, indent=1)
        os.replace(tmp, self.records_path)
        if self.vectors is not None:
            np.save(self.vectors_path, self.vectors)

    def _reindex(self):
        texts = [r.get("embed_text") or r["text"] for r in self.records]
        self.vectors = _normalize(self.embedder.embed(texts)) if texts else None
        self._save()

    def __len__(self):
        return len(self.records)

    # -- the two operations every vector store has -------------------------------
    def add(self, kind: str, text: str, meta=None, dedupe: float = 0.95, context: str = "") -> bool:
        """Store one memory. Skips near-duplicates of something already stored.

        context is embedded along with the text but not shown: a general lesson stored
        with the brief it came from can be found again by a similar future brief.
        """
        text = text.strip()
        if not text:
            return False
        embed_text = f"{text}\nContext: {context.strip()}" if context.strip() else text
        # Duplicates are judged on the text alone, so the same lesson from two jobs merges.
        vector = _normalize(self.embedder.embed([embed_text]))[0]
        text_vector = vector if embed_text == text else _normalize(self.embedder.embed([text]))[0]
        if self.vectors is not None and len(self.records):
            same_kind = [i for i, r in enumerate(self.records) if r["kind"] == kind]
            if same_kind:
                plain = _normalize(self.embedder.embed([self.records[i]["text"] for i in same_kind]))
                sims = plain @ text_vector
                best = int(np.argmax(sims))
                if float(sims[best]) >= dedupe:
                    # Seen again: count it, so recurring lessons get recalled on every job.
                    match = self.records[same_kind[best]]
                    match["meta"]["seen"] = match["meta"].get("seen", 1) + 1
                    self._save()
                    return False
        record = {"kind": kind, "text": text, "meta": meta or {},
                  "created": datetime.now().isoformat(timespec="seconds")}
        if embed_text != text:
            record["embed_text"] = embed_text
        self.records.append(record)
        self.vectors = vector[None, :] if self.vectors is None else np.vstack([self.vectors, vector])
        self._save()
        return True

    def recurring(self, kind: str, min_seen: int = 2, k: int = 3) -> list:
        """Memories of this kind that came up again and again, most frequent first."""
        hits = [r for r in self.records if r["kind"] == kind and r["meta"].get("seen", 1) >= min_seen]
        return sorted(hits, key=lambda r: r["meta"].get("seen", 1), reverse=True)[:k]

    def query(self, text: str, kind: str = None, k: int = 5, min_score: float = None) -> list:
        """The k memories most similar to text (optionally only one kind), best first."""
        if self.vectors is None or not self.records:
            return []
        q = _normalize(self.embedder.embed([text], query=True))[0]
        scores = self.vectors @ q                                  # cosine similarity
        floor = self.embedder.min_score if min_score is None else min_score
        hits = [(float(scores[i]), r) for i, r in enumerate(self.records)
                if (kind is None or r["kind"] == kind) and scores[i] >= floor]
        hits.sort(key=lambda h: h[0], reverse=True)
        return [dict(r, score=round(s, 3)) for s, r in hits[:k]]


# ---------------------------------------------------------------------------
# What Jarvis remembers, and how it recalls it
# ---------------------------------------------------------------------------

def _brief_core(brief: str, limit: int = 1200) -> str:
    """The client's own words, without the data preview and environment Jarvis appended."""
    core = re.split(r"\n\s*(DATA PREVIEW|RUNTIME ENVIRONMENT|LESSONS FROM PAST JOBS)", brief)[0]
    return core.strip()[:limit]


def _deliverable_line(work: str) -> str:
    match = re.search(r"Deliverable:\s*([^\n]+)", work or "")
    return match.group(1).strip()[:160] if match else "unknown"


DISTILL_PROMPT = """You are Jarvis's memory keeper. You read a finished data job's QA reviews and
write down what the analyst should remember for FUTURE, DIFFERENT jobs.

Rules:
- 0 to 4 lessons. Only include ones that would apply to other datasets and clients.
- Each lesson is one sentence, an instruction, under 30 words.
- No client names, no numbers or product names from this dataset.
- Good: "On 'least' or 'bottom N' charts, put the smallest item on top."
- Bad: "Chicken Tacos sold 123 units."

Respond with only JSON: {"lessons": ["...", "..."]}"""


def distill_lessons(brain, reviews: list) -> list:
    """Turn job-specific QA feedback into general lessons (one cheap model call)."""
    points = [p for r in reviews for p in r.get("feedback", [])]
    if not points:
        return []
    if brain is not None:
        try:
            raw = brain.think(DISTILL_PROMPT, "QA FEEDBACK FROM THIS JOB:\n" +
                              "\n".join(f"- {p}" for p in points), json_mode=True)
            start, end = raw.find("{"), raw.rfind("}")
            lessons = json.loads(raw[start:end + 1]).get("lessons", [])
            return [str(l).strip() for l in lessons if str(l).strip()][:4]
        except Exception:
            pass
    # No model available: keep the blocking points themselves, shortened.
    blocking = [p for p in points if p.upper().startswith("BLOCKING")]
    return [re.sub(r"^BLOCKING:\s*", "", p, flags=re.I)[:220] for p in blocking[:3]]


def remember_job(memory: Memory, brief: str, work: str, reviews: list, status: str,
                 rounds: int, brain=None) -> int:
    """After a job: store what it was, how it went, and the lessons it taught. Returns lessons saved."""
    core = _brief_core(brief)
    deliverable = _deliverable_line(work)
    outcome = f"{status} in {rounds} round{'s' if rounds != 1 else ''}"
    memory.add("job", f"{core}\n\nDeliverable: {deliverable}\nOutcome: {outcome}",
               meta={"deliverable": deliverable, "outcome": outcome, "summary": core[:140]})
    saved = 0
    for lesson in distill_lessons(brain, reviews):
        if memory.add("lesson", lesson, meta={"from": core[:140]}, dedupe=0.9, context=core[:600]):
            saved += 1
    return saved


def recall_for_job(memory: Memory, brief: str, k_lessons: int = 6, k_jobs: int = 3) -> str:
    """Text to add to a new job's brief: relevant lessons and similar past jobs."""
    core = _brief_core(brief)
    recurring = memory.recurring("lesson", k=3)                       # mistakes that keep coming back
    related = memory.query(core, kind="lesson", k=k_lessons)          # lessons from similar jobs
    seen_texts = {r["text"] for r in recurring}
    lessons = recurring + [m for m in related if m["text"] not in seen_texts][:k_lessons - len(recurring)]
    jobs = memory.query(core, kind="job", k=k_jobs)
    if not lessons and not jobs:
        return ""
    parts = ["LESSONS FROM PAST JOBS (Jarvis memory: apply them where relevant, ignore any that don't fit):"]
    parts += [f"- {m['text']}" + (f" (came up {m['meta']['seen']} times)" if m["meta"].get("seen", 1) > 1 else "")
              for m in lessons]
    if jobs:
        parts.append("Similar past jobs:")
        parts += [f"- \"{m['meta'].get('summary', '')[:90]}...\" -> {m['meta'].get('deliverable', '?')}, "
                  f"{m['meta'].get('outcome', '?')}" for m in jobs]
    return "\n".join(parts)


def remember_gig(memory: Memory, listing: str, verdict: dict, decision: str) -> None:
    """After grading: store the listing and Marro's own call on it."""
    summary = f"{verdict.get('title', '')}: {verdict.get('core_deliverable', '')}".strip(": ")
    memory.add("gig", f"{summary}\n{listing[:1500]}",
               meta={"title": verdict.get("title", "")[:80], "decision": decision,
                     "jarvis": verdict.get("recommendation", ""), "score": verdict.get("fit_score", "")},
               dedupe=0.98)


def recall_for_gig(memory: Memory, listing: str, k: int = 3) -> str:
    """Marro's past decisions on similar listings, for the grader."""
    hits = memory.query(listing[:1500], kind="gig", k=k)
    if not hits:
        return ""
    lines = ["MARRO'S PAST DECISIONS ON SIMILAR LISTINGS (his real preferences; weigh them, "
             "but grade this listing on its own merits):"]
    for m in hits:
        meta = m["meta"]
        lines.append(f"- \"{meta.get('title', '')}\": Jarvis said {meta.get('jarvis', '?')} "
                     f"({meta.get('score', '?')}/10), Marro said {meta.get('decision', '?').upper()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backfill: learn from runs that happened before memory existed
# ---------------------------------------------------------------------------

def backfill(memory: Memory, runs_dir="runs", gigs_csv="graded_gigs.csv", brain=None) -> dict:
    import csv
    import glob
    counts = {"jobs": 0, "lessons": 0, "gigs": 0}
    for folder in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        brief_path = os.path.join(folder, "00_brief.txt")
        if not os.path.isfile(brief_path):
            continue
        with open(brief_path, encoding="utf-8") as f:
            brief = f.read()
        reviews = []
        for path in sorted(glob.glob(os.path.join(folder, "*qa_review.json"))):
            try:
                with open(path, encoding="utf-8") as f:
                    reviews.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                pass
        final = next((os.path.join(folder, n) for n in ("FINAL_deliverable.md", "NEEDS_YOUR_REVIEW.md")
                      if os.path.exists(os.path.join(folder, n))), None)
        if not final:
            continue
        with open(final, encoding="utf-8") as f:
            work = f.read()
        status = "approved" if final.endswith("FINAL_deliverable.md") else "not approved"
        rounds = len([r for r in glob.glob(os.path.join(folder, "[0-9][0-9]_qa_review.json"))])
        before = len([r for r in memory.records if r["kind"] == "job"])
        counts["lessons"] += remember_job(memory, brief, work, reviews, status, max(rounds, 1), brain)
        counts["jobs"] += len([r for r in memory.records if r["kind"] == "job"]) - before
    if os.path.exists(gigs_csv):
        with open(gigs_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                verdict = {"title": row.get("title", ""), "recommendation": row.get("jarvis_recommendation", ""),
                           "fit_score": row.get("fit_score", "")}
                try:
                    verdict.update({k: v for k, v in json.loads(row.get("full_verdict_json") or "{}").items()
                                    if k in ("core_deliverable",)})
                except json.JSONDecodeError:
                    pass
                before = len(memory.records)
                remember_gig(memory, row.get("listing", ""), verdict, row.get("your_decision", "?"))
                counts["gigs"] += len(memory.records) - before
    return counts


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Look inside or build Jarvis's memory.")
    parser.add_argument("--stats", action="store_true", help="show what's stored")
    parser.add_argument("--search", default="", help="find the memories closest to this text")
    parser.add_argument("--backfill", action="store_true", help="learn from existing runs/ and graded_gigs.csv")
    parser.add_argument("--brain", default="", choices=["", "claude", "ollama"],
                        help="with --backfill: use a model to distill general lessons")
    args = parser.parse_args()

    mem = Memory()
    print(f"Jarvis memory: {len(mem)} memories, embeddings by {mem.embedder.name}")
    if isinstance(mem.embedder, KeywordEmbedder):
        print("  (Tip: for meaning-based search, run: ollama pull nomic-embed-text)")
    if args.backfill:
        brain = None
        if args.brain:
            from brains import get_brain
            brain = get_brain(args.brain)
        counts = backfill(mem, brain=brain)
        print(f"Backfill added {counts['jobs']} jobs, {counts['lessons']} lessons, {counts['gigs']} gigs.")
    if args.stats or not (args.search or args.backfill):
        kinds = {}
        for r in mem.records:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        for kind, n in sorted(kinds.items()):
            print(f"  {kind}: {n}")
        for r in [r for r in mem.records if r["kind"] == "lesson"][-8:]:
            print(f"  lesson: {r['text']}")
    if args.search:
        for m in mem.query(args.search, k=6, min_score=0):
            print(f"  [{m['kind']}, similarity {m['score']}] {m['text'][:150].replace(chr(10), ' ')}")
