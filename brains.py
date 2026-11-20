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
                if line.startswith("DATA FILES"):
                    data_file = line.split(":", 1)[1].split(",")[0].strip()

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
            analysis = (
                "# file: analysis.py\n"
                "import os\nimport pandas as pd\n\n"
                f"df = pd.read_csv('{data_file}', parse_dates=['date'])\n"
                "df['year'] = df['date'].dt.year\n"
                "os.makedirs('outputs/working', exist_ok=True)\n\n"
                "monthly = df.groupby('date', as_index=False)['revenue'].sum()\n"
                "monthly.to_csv('outputs/working/monthly_revenue.csv', index=False)\n"
                "by_month = df.groupby(df['date'].dt.month)['revenue'].mean()\n"
                "print('Strongest month:', by_month.idxmax(), '| weakest month:', by_month.idxmin())\n\n"
                "pivot = df.pivot_table(index='region', columns='year', values='revenue', aggfunc='sum')\n"
                "growth = ((pivot.iloc[:, 1] / pivot.iloc[:, 0] - 1) * 100).round(1)\n"
                "growth = growth.rename('growth_pct').reset_index().sort_values('growth_pct', ascending=False)\n"
                "growth.to_csv('outputs/working/region_growth.csv', index=False)\n"
                "print('Total revenue:', round(df['revenue'].sum()))\n"
                "print('Year-over-year growth by region (%):')\n"
                "print(growth.to_string(index=False))\n"
            )
            app = (
                "# file: app.py\n"
                "import pandas as pd\nimport plotly.express as px\nimport jarvis_theme as jt\n\n"
                "monthly = pd.read_csv('outputs/working/monthly_revenue.csv', parse_dates=['date'])\n"
                "growth = pd.read_csv('outputs/working/region_growth.csv')\n\n"
                "jt.page('Sales Review', 'Sample retailer, two years of monthly sales')\n"
                "jt.kpis([('Total revenue', jt.money(monthly['revenue'].sum())),\n"
                "         ('Months covered', jt.num(len(monthly))),\n"
                "         ('Fastest-growing region', growth.iloc[0]['region'])])\n\n"
                "jt.section('How has revenue trended?')\n"
                "fig = px.line(monthly, x='date', y='revenue', title='Revenue peaks every December',\n"
                "              labels={'date': 'Month', 'revenue': 'Revenue ($)'})\n"
                "jt.chart(fig, 'Sales climb through the fall and peak in December.', table=monthly)\n\n"
                "jt.section('Which regions are driving growth?')\n"
                "fig = px.bar(growth, x='region', y='growth_pct', color='region',\n"
                "             color_discrete_map=jt.highlight(growth['region'], growth.iloc[0]['region']),\n"
                "             title=f\"{growth.iloc[0]['region']} grew fastest\",\n"
                "             labels={'region': 'Region', 'growth_pct': 'Year-over-year growth (%)'})\n"
                "fig.update_layout(showlegend=False)\n"
                "jt.chart(fig, f\"{growth.iloc[0]['region']} grew {growth.iloc[0]['growth_pct']}% year over year.\", table=growth)\n\n"
                "jt.bullets('Recommendations', ['Stock up ahead of the December peak.',\n"
                "                               'Study what is working in the fastest-growing region.'])\n"
                "jt.footer('Source: sample_sales.csv', ['Two years of data, so seasonality is tentative.'])\n"
            )
            return (
                f"## Analysis ({stage})\n\n"
                "### Approach\nAggregate revenue by month, compare regional growth, "
                "and present it as a short report app.\n\n"
                f"```python\n{analysis}```\n\n```python\n{app}```\n\n"
                "### Findings\nSee the printed output: seasonality peaks late in the year, "
                "and growth by region is ranked above.\n"
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

    def __init__(self, model: str = "claude-sonnet-5-5", max_tokens: int = 16000):
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
        with urllib.request.urlopen(req, timeout=900) as resp:
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
