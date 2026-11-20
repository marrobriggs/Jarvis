"""
Brains: the swappable "thinking" part of Jarvis.

Every brain has the same method:  think(system_prompt, user_message) -> str
The agents and the orchestrator never care which brain is plugged in,
so you can move from fake -> Ollama -> Claude API by changing one setting.
"""

import json
import os
import urllib.request


class MockBrain:
    """A fake model for testing the plumbing. Instant, free, zero RAM.

    It returns canned answers based on which agent is calling. The QA
    manager rejects the first draft and approves the second, so you can
    watch the revise loop work.
    """

    def __init__(self):
        self.qa_calls = 0
        self.analyst_calls = 0
        self.last_truncated = False

    def think(self, system_prompt: str, user_message: str) -> str:
        if "QA MANAGER" in system_prompt:
            self.qa_calls += 1
            if self.qa_calls == 1:
                return json.dumps({
                    "score": 6,
                    "verdict": "REVISE",
                    "feedback": [
                        "Findings point to the printed output instead of stating the numbers in plain English.",
                        "The brief asks for an interactive dashboard, which is not delivered yet.",
                    ],
                })
            return json.dumps({
                "score": 9,
                "verdict": "APPROVE",
                "feedback": ["Clear, accurate and ready for the client."],
            })

        if "ANALYST" in system_prompt:
            self.analyst_calls += 1
            data_file = "data.csv"
            for line in user_message.splitlines():
                if line.startswith("DATA FILE:"):
                    data_file = line.split(":", 1)[1].strip()

            if self.analyst_calls == 1:
                # First draft has a deliberate bug (wrong column name),
                # so you can watch the analyst read the error and fix it.
                return (
                    "## Analysis (first draft)\n\n"
                    "```python\nimport pandas as pd\n"
                    f"df = pd.read_csv('{data_file}', parse_dates=['date'])\n"
                    "print(df['sales'].sum())\n```\n"
                )

            stage = "after QA feedback" if "QA FEEDBACK" in user_message else "fixed code"
            return (
                f"## Analysis ({stage})\n\n"
                "### Approach\nAggregate revenue by month, then compare growth by region "
                "and category between the first and second year.\n\n"
                "### Code\n```python\n"
                "import pandas as pd\nimport plotly.express as px\n\n"
                f"df = pd.read_csv('{data_file}', parse_dates=['date'])\n"
                "df['year'] = df['date'].dt.year\n\n"
                "monthly = df.groupby('date', as_index=False)['revenue'].sum()\n"
                "fig = px.line(monthly, x='date', y='revenue', title='Monthly revenue',\n"
                "              labels={'date': 'Month', 'revenue': 'Revenue (USD)'})\n"
                "fig.write_html('monthly_revenue.html')\n\n"
                "by_month = df.groupby(df['date'].dt.month)['revenue'].mean()\n"
                "print('Strongest month:', by_month.idxmax(), '| weakest month:', by_month.idxmin())\n\n"
                "for col in ['region', 'product_category']:\n"
                "    pivot = df.pivot_table(index=col, columns='year', values='revenue', aggfunc='sum')\n"
                "    pivot['growth_%'] = (pivot.iloc[:, 1] / pivot.iloc[:, 0] - 1) * 100\n"
                "    print(f'\\nYear-over-year growth by {col}:')\n"
                "    print(pivot['growth_%'].round(1).sort_values(ascending=False).to_string())\n"
                "```\n\n"
                "### Findings\nSee the printed output: seasonality peaks late in the year, "
                "and growth by region and category is ranked above.\n"
            )

        return "Mock brain: no canned answer for this agent."


class OllamaBrain:
    """A local open model running in Ollama on this laptop. Free, slow."""

    def __init__(self, model: str = "llama3.2", url: str = "http://127.0.0.1:11434/api/chat"):
        self.model = model
        self.url = url
        self.last_truncated = False

    def think(self, system_prompt: str, user_message: str, json_mode: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 1200},
        }
        if json_mode:
            payload["format"] = "json"
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # skip proxies
        with opener.open(req, timeout=1200) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        self.last_truncated = body.get("done_reason") == "length"
        return body["message"]["content"]


class ClaudeBrain:
    """Claude through the Anthropic API. Fast and sharp, pay per use.

    Needs an API key from console.anthropic.com saved in an environment
    variable named ANTHROPIC_API_KEY. API usage is billed separately
    from a Claude Pro subscription.
    """

    def __init__(self, model: str = "claude-sonnet-5-5", max_tokens: int = 12000):
        self.model = model
        self.max_tokens = max_tokens   # room for a full analysis script plus write-up
        self.last_truncated = False    # True if the last answer hit the length cap
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable first.")

    def think(self, system_prompt: str, user_message: str, json_mode: bool = False) -> str:
        if json_mode:
            user_message += "\n\nRespond with only valid JSON, no other text."
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        self.last_truncated = body.get("stop_reason") == "max_tokens"
        return "".join(block.get("text", "") for block in body["content"])


def get_brain(name: str):
    """Pick a brain by name: 'mock', 'ollama' or 'claude'."""
    name = name.lower()
    if name == "mock":
        return MockBrain()
    if name == "ollama":
        return OllamaBrain()
    if name == "claude":
        return ClaudeBrain()
    raise ValueError(f"Unknown brain '{name}'. Use mock, ollama or claude.")
