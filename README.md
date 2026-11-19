# Jarvis: Multi-Agent Freelance Data Assistant

Jarvis is a personal AI assistant that runs a freelance data analysis workflow with a team of AI agents. It screens gigs, does the analysis, reviews its own work, and only hands me results that pass a quality check. I make the final call at each decision point.

## How it works

```
Job listing ──► GIG GRADER ──► Me: take it? (y/n)
                                   │ yes
                                   ▼
Client brief + data ──► ANALYST ──► EXECUTOR runs the code
                           ▲              │
                           │   errors     │ crashed? analyst fixes it (up to 3 tries)
                           └──────────────┘
                                          │ ran
                                          ▼
                                     QA MANAGER reviews the work and the real output
                                          │
                           ┌──────────────┴──────────────┐
                        REVISE                        APPROVE
                  (feedback goes back             final deliverable
                    to the analyst)              + charts saved
```

- **Gig Grader** (`gig_grader.py`) scores a freelance listing against my skills profile and returns a structured verdict (fit score, pros, cons, red flags). Every decision is logged to a CSV to build an evaluation set.
- **Analyst agent** writes a full analysis with runnable Python, Plotly charts and plain-English findings.
- **Executor** (`executor.py`) actually runs the analyst's code and feeds errors back so the agent can fix its own work.
- **QA Manager agent** grades the work against a rubric (answers the brief, correctness, evidence, visuals, communication), using the real code output, not just the code (LLM-as-a-judge).
- **Orchestrator** (`orchestrator.py`) coordinates the loop and saves every draft, code run and review to `runs/` as an audit trail.

## Design choices

- **Swappable model backends.** Every agent talks to a "brain" with the same interface, so the same pipeline runs on a mock model (instant tests), a local open model through [Ollama](https://ollama.com), or Claude through the Anthropic API.
- **Rules live in code, not in the model.** Models report facts (scores, ML involvement, red flags); Python makes the final TAKE / MAYBE / PASS and APPROVE / REVISE decisions, so outputs can't contradict themselves.
- **Human in the loop.** Jarvis recommends; I decide.
- **Mock-first development.** The full agent loop is tested with a deterministic fake model before any real model is plugged in.

## Quick start

Requires Python 3.10+ with `pandas` and `plotly`.

```bash
pip install pandas plotly

# Run the full agent pipeline with the mock brain (instant, free)
python orchestrator.py sample_brief.txt --data sample_sales.csv

# Same pipeline with a local model (install Ollama, then: ollama pull llama3.2)
python orchestrator.py sample_brief.txt --data sample_sales.csv --brain ollama

# Same pipeline with Claude (set the ANTHROPIC_API_KEY environment variable)
python orchestrator.py sample_brief.txt --data sample_sales.csv --brain claude

# Grade freelance job listings interactively with a local model
python gig_grader.py
```

Results land in `runs/<timestamp>/`: each draft, each code run, each QA review, and the final deliverable. Charts are saved as interactive HTML files in the `work/` subfolder.

## Project structure

| File | Purpose |
|---|---|
| `orchestrator.py` | Coordinates the analyst → executor → QA loop |
| `agents.py` | Analyst and QA Manager prompts and logic |
| `brains.py` | Mock, Ollama and Claude model backends |
| `executor.py` | Runs agent-written code and reports results |
| `gig_grader.py` | Grades freelance job listings with human-in-the-loop approval |
| `sample_brief.txt`, `sample_sales.csv` | Example client job for testing |

## Roadmap

- [x] Gig grader with structured JSON output and decision logging
- [x] Multi-agent analyst / QA loop with a swappable model backend
- [x] Code execution with self-correction
- [ ] Connect the grader to the pipeline
- [ ] Reporter agent that packages approved work into a client-ready report
- [ ] Retrieval (RAG) over past gigs and deliverables
- [ ] Voice interface

## Note on safety

The executor runs AI-written code on the local machine with a timeout, but it is not a full sandbox. Review generated code before running it on anything important, and only use client data with the client's permission.
