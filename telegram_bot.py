"""
Jarvis on Telegram: run your freelance pipeline from your phone.

Jarvis keeps running on your laptop; Telegram is just the remote control.
The bot asks Telegram for new messages ("long polling"), so no server or
public web address is needed. Only YOUR Telegram account can use it.

The flow:
  1. Paste a job listing            -> Jarvis grades it and asks "take it?"
  2. Reply yes/no (yeah, nah, ...)  -> yes saves the job; no logs the pass
  3. Send the client's data         -> CSV, Excel or zip, up to 20 MB each
  4. Say "go"                       -> the agents run; progress comes to the chat
  5. Jarvis sends back the files, where they are in OneDrive, and a
     ready-to-paste message for the client

Commands: /help  /status  /cancel  /lessons

Setup (once):
  1. In Telegram, message @BotFather, send /newbot, and follow the prompts.
     Copy the token it gives you.
  2. setx TELEGRAM_BOT_TOKEN "the-token"
  3. Close and reopen the terminal, run: python telegram_bot.py
  4. Message your bot. It replies with your Telegram user ID; save it:
     setx TELEGRAM_ALLOWED_USER "your-id"
  5. Reopen the terminal and run python telegram_bot.py again. Done.

Keep your laptop awake while you're away (Windows Settings > System >
Power > "When plugged in, put my device to sleep after": Never).
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
# The bot's state lives outside OneDrive (in %LOCALAPPDATA%\Jarvis), because OneDrive can lock
# the file mid-sync and Windows then refuses the save ("PermissionError: Access is denied").
STATE_DIR = os.path.join(os.environ.get("LOCALAPPDATA") or HERE, "Jarvis")
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "telegram_state.json")
_OLD_STATE = os.path.join(HERE, "telegram_state.json")
if not os.path.exists(STATE_FILE) and os.path.exists(_OLD_STATE) and _OLD_STATE != STATE_FILE:
    import shutil
    shutil.copy(_OLD_STATE, STATE_FILE)   # one-time move: keep the bot's memory of open jobs
BRAIN = os.environ.get("JARVIS_BOT_BRAIN", "claude")   # "mock" for free test runs
MAX_DOWNLOAD = 20 * 1024 * 1024                        # Telegram bots can download up to 20 MB
MAX_UPLOAD = 50 * 1024 * 1024                          # ...and send up to 50 MB

YES = {"y", "yes", "yeah", "yea", "yep", "yup", "ya", "sure", "ok", "okay", "take", "take it",
       "lets go", "let's go", "do it", "absolutely", "definitely", "sounds good", "go for it", "bet"}
NO = {"n", "no", "nah", "nope", "pass", "skip", "no thanks", "not this one", "hard pass", "naw"}
GO = {"go", "start", "run", "done", "ready", "that's all", "thats all", "go ahead", "begin", "lets go",
      "let's go", "start it", "run it", "go for it", "yes", "yep", "yeah", "ok", "okay", "thats it",
      "that's it", "all set", "send it"}


def is_go(text: str) -> bool:
    t = re.sub(r"[^a-z' ]", "", text.lower()).strip()
    return t in GO or t.startswith(("go ", "start ", "run ", "lets go", "let's go"))


def interpret(text: str):
    """'yes', 'no', or None for an answer like 'yeah!!' / 'nah' / 'hmm'."""
    t = re.sub(r"[^a-z' ]", "", text.lower()).strip()
    if t in YES or any(t.startswith(w + " ") for w in ("yes", "yeah", "yep", "sure", "ok")):
        return "yes"
    if t in NO or any(t.startswith(w + " ") for w in ("no", "nah", "nope", "pass")):
        return "no"
    return None


# ---------------------------------------------------------------------------
# Telegram's Bot API, using only the standard library
# ---------------------------------------------------------------------------

class TelegramAPI:
    def __init__(self, token: str):
        self.base = f"https://api.telegram.org/bot{token}/"
        self.file_base = f"https://api.telegram.org/file/bot{token}/"

    def _call(self, method, params=None, timeout=70):
        data = json.dumps(params or {}).encode("utf-8")
        req = urllib.request.Request(self.base + method, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if not body.get("ok"):
            raise RuntimeError(f"Telegram {method} failed: {body.get('description')}")
        return body["result"]

    def get_updates(self, offset):
        return self._call("getUpdates", {"offset": offset, "timeout": 50,
                                         "allowed_updates": ["message"]})

    def send(self, chat_id, text):
        for start in range(0, max(len(text), 1), 4000):      # Telegram's limit is 4096 characters
            self._call("sendMessage", {"chat_id": chat_id, "text": text[start:start + 4000],
                                       "disable_web_page_preview": True})

    def send_file(self, chat_id, path, caption=""):
        boundary = uuid.uuid4().hex
        name = os.path.basename(path)
        with open(path, "rb") as f:
            content = f.read()
        parts = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="chat_id"\r\n\r\n{chat_id}\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="caption"\r\n\r\n{caption[:1000]}\r\n'.encode(),
            (f'--{boundary}\r\nContent-Disposition: form-data; name="document"; filename="{name}"\r\n'
             f'Content-Type: application/octet-stream\r\n\r\n').encode() + content + b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
        req = urllib.request.Request(self.base + "sendDocument", data=b"".join(parts),
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            json.loads(resp.read().decode("utf-8"))

    def download(self, file_id, dest):
        info = self._call("getFile", {"file_id": file_id})
        with urllib.request.urlopen(self.file_base + info["file_path"], timeout=300) as resp, \
                open(dest, "wb") as f:
            f.write(resp.read())


# ---------------------------------------------------------------------------
# The bot
# ---------------------------------------------------------------------------

class JarvisBot:
    def __init__(self, api, allowed_user: str = ""):
        self.api = api
        self.allowed = str(allowed_user).strip()
        self.state = self._load()
        self.lock = threading.Lock()
        self.job_thread = None
        self.job_process = None
        self.cancelled = False           # True when you stopped the running job with /cancel
        self.pending_package = None      # a package install waiting for your yes/no
        self.package_timer = None
        self.approval_timeout = 15 * 60  # seconds before an unanswered request is declined
        self.memory = None
        try:
            from memory import Memory
            self.memory = Memory()
        except Exception:
            pass

    # -- saved state (survives restarts) ------------------------------------
    def _load(self):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            state = {}
        state.setdefault("offset", 0)
        state.setdefault("stage", "idle")
        if state["stage"] == "running":          # the laptop restarted mid-job
            state["stage"] = "awaiting_data"
            state["note"] = "interrupted"
        return state

    def _save(self):
        """Save state safely. If Windows (or a sync/antivirus program) briefly locks the file, retry."""
        tmp = STATE_FILE + ".tmp"
        for _ in range(10):
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.state, f, indent=1)
                os.replace(tmp, STATE_FILE)
                return
            except PermissionError:
                time.sleep(0.2)
        try:   # last resort: write the file directly
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=1)
        except OSError as e:
            print(f"Couldn't save bot state ({e}). Carrying on; it will try again next time.")

    def set(self, **changes):
        with self.lock:
            self.state.update(changes)
            self._save()

    # -- main loop ----------------------------------------------------------
    def run_forever(self):
        print("Jarvis is listening on Telegram. Press Ctrl+C to stop.")
        while True:
            try:
                for update in self.api.get_updates(self.state["offset"]):
                    self.set(offset=update["update_id"] + 1)
                    self.handle(update.get("message") or {})
            except KeyboardInterrupt:
                print("Stopped.")
                return
            except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as e:
                print(f"Connection problem ({e}); retrying in 10 seconds...")
                time.sleep(10)

    def handle(self, msg):
        chat_id = (msg.get("chat") or {}).get("id")
        user_id = str((msg.get("from") or {}).get("id", ""))
        if not chat_id:
            return
        if not self.allowed:
            self.api.send(chat_id, f"Hi! Your Telegram user ID is {user_id}.\n\nTo make this bot yours, "
                                   f"run this on your laptop, then restart the bot:\n"
                                   f"setx TELEGRAM_ALLOWED_USER \"{user_id}\"")
            return
        if user_id != self.allowed:
            print(f"Ignored a message from an unknown Telegram user ({user_id}).")
            return
        self.set(chat_id=chat_id)
        try:
            if msg.get("document"):
                self.on_file(chat_id, msg["document"])
            elif msg.get("photo"):
                self.api.send(chat_id, "That came through as a photo. Send data as a file (📎 > File).")
            elif msg.get("text"):
                self.on_text(chat_id, msg["text"].strip())
        except Exception as e:  # never let one message crash the bot
            print(f"Error handling a message: {type(e).__name__}: {e}")
            self.api.send(chat_id, f"Something went wrong on my side ({type(e).__name__}: {e}). "
                                   "Try again, or check the laptop terminal.")

    # -- text messages ------------------------------------------------------
    def on_text(self, chat_id, text):
        command = text.split()[0].lower().split("@")[0] if text.startswith("/") else ""
        stage = self.state["stage"]
        if command in ("/start", "/help"):
            return self.api.send(chat_id, HELP)
        if command == "/status":
            return self.api.send(chat_id, self.status_text())
        if command == "/lessons":
            return self.api.send(chat_id, self.lessons_text())
        if command == "/cancel":
            return self.cancel(chat_id)
        if command == "/jobs":
            return self.api.send(chat_id, self.jobs_text())
        if command == "/resume":
            return self.resume(chat_id, text[len("/resume"):].strip())

        if stage == "running":
            if self.pending_package:
                answer = interpret(text)
                if answer:
                    return self.answer_package(chat_id, answer)
                return self.api.send(chat_id, f"Should I install '{self.pending_package}'? Yes or no.")
            return self.api.send(chat_id, "I'm working on the current job. Send /status to check on it.")
        if stage == "awaiting_decision":
            answer = interpret(text)
            if answer:
                return self.decide(chat_id, answer)
            if len(text) < 200:
                return self.api.send(chat_id, "Sorry, was that a yes or a no? (Or paste a different listing.)")
        if stage == "awaiting_data":
            if is_go(text):
                return self.start_job(chat_id)
            if len(text) < 200:
                return self.api.send(chat_id, "Send the client's data as a file (📎 > File), then say \"go\". "
                                              "Or /cancel to drop this job.")
        if is_go(text) and self.recent_jobs():
            return self.api.send(chat_id, "There's no job open right now. To rerun a saved one, "
                                          "send /jobs and then /resume with its number.")
        if len(text) < 80:
            return self.api.send(chat_id, "Paste a full job listing and I'll grade it. /help for more.")
        self.grade(chat_id, text)

    # -- step 1: grade a listing -------------------------------------------
    def grade(self, chat_id, listing):
        import gig_grader
        self.api.send(chat_id, "Grading...")
        if BRAIN == "mock":
            verdict = gig_grader.mock_grade(listing)
        else:
            context = ""
            if self.memory is not None:
                try:
                    from memory import recall_for_gig
                    context = recall_for_gig(self.memory, listing)
                except Exception:
                    pass
            to_grade = listing + ("\n\n" + context if context else "")
            verdict = (gig_grader.grade_with_claude(to_grade) if BRAIN == "claude"
                       else gig_grader.grade_gig(to_grade))
        verdict = gig_grader.enforce_ml_rule(verdict)
        verdict["graded_by"] = BRAIN
        self.set(stage="awaiting_decision", listing=listing, verdict=verdict)
        self.api.send(chat_id, format_verdict(verdict) + "\n\nWant to take this gig?")

    # -- step 2: yes or no --------------------------------------------------
    def decide(self, chat_id, answer):
        import gig_grader
        listing, verdict = self.state.get("listing", ""), self.state.get("verdict", {})
        if BRAIN != "mock":
            gig_grader.log_decision(listing, verdict, answer)
            if self.memory is not None:
                try:
                    from memory import remember_gig
                    remember_gig(self.memory, listing, verdict, answer)
                except Exception:
                    pass
        if answer == "no":
            self.set(stage="idle", listing="", verdict={})
            return self.api.send(chat_id, "Passed and logged. Paste another listing anytime.")
        job = gig_grader.save_job(listing, verdict)
        self.set(stage="awaiting_data", job=job, files=[], note="")
        self.api.send(chat_id, f"Saved as job \"{os.path.basename(job)}\".\n\nWhen the client sends their data, "
                               "send it to me here as a file (CSV, Excel or zip, up to 20 MB each). "
                               "Send as many files as needed, then say \"go\".")

    # -- step 3: receive the data ---------------------------------------------
    def on_file(self, chat_id, doc):
        if self.state["stage"] != "awaiting_data":
            return self.api.send(chat_id, "I wasn't expecting a file yet. Paste a listing first, and say yes to it.")
        name = re.sub(r"[^\w.\- ()+]", "_", doc.get("file_name") or "data").strip() or "data"
        if name.lower() == "brief.txt":
            name = "client_brief.txt"
        if doc.get("file_size", 0) > MAX_DOWNLOAD:
            return self.api.send(chat_id, f"{name} is over 20 MB, which is Telegram's limit for bots. "
                                          f"Put it in the job folder on your laptop ({self.state.get('job')}), "
                                          "then say \"go\" here.")
        dest = os.path.join(HERE, self.state["job"], name)
        self.api.download(doc["file_id"], dest)
        files = self.state.get("files", []) + [name]
        self.set(files=files)
        self.api.send(chat_id, f"Got {name}. Send more files, or say \"go\" to start.")

    # -- step 4: run the agents -----------------------------------------------
    def start_job(self, chat_id):
        job = self.state.get("job")
        data_files = [f for f in os.listdir(os.path.join(HERE, job)) if f != "brief.txt"] if job else []
        if not data_files:
            return self.api.send(chat_id, "I don't have any data for this job yet. Send the file first.")
        self.set(stage="running", started=time.time())
        self.api.send(chat_id, f"Starting the agents on {len(data_files)} file(s). I'll keep you posted.")
        self.job_thread = threading.Thread(target=self._run_job, args=(chat_id, job), daemon=True)
        self.job_thread.start()

    def _command(self, job):
        brief = os.path.join(job, "brief.txt")
        return [sys.executable, "-u", "orchestrator.py", brief, "--data", job, "--brain", BRAIN]

    def _run_job(self, chat_id, job):
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        folder, lines, pending, last_sent = None, [], [], time.time()
        try:
            # stdin is a pipe so the bot can type your answer to Jarvis's questions.
            self.job_process = subprocess.Popen(self._command(job), cwd=HERE, env=env, stdin=subprocess.PIPE,
                                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                text=True, encoding="utf-8", errors="replace")
            for line in self.job_process.stdout:
                line = line.rstrip()
                lines.append(line)
                asked = re.search(r"wants to install '([^']+)', which is NOT on the approved list", line)
                if asked:
                    if pending:
                        self.api.send(chat_id, "\n".join(pending))
                        pending = []
                    self.ask_package(chat_id, asked.group(1))
                    continue
                found = re.search(r"(runs[\\/][\w\-]+)", line)
                if found and folder is None:
                    folder = found.group(1)
                if is_progress(line):
                    pending.append(line.strip())
                if pending and time.time() - last_sent > 8:          # batch updates, don't spam
                    self.api.send(chat_id, "\n".join(pending))
                    pending, last_sent = [], time.time()
            self.job_process.wait()
            if pending:
                self.api.send(chat_id, "\n".join(pending))
            run = os.path.join(HERE, folder) if folder else ""
            finished = run and (os.path.exists(os.path.join(run, "FINAL_deliverable.md")) or
                                os.path.exists(os.path.join(run, "NEEDS_YOUR_REVIEW.md")))
            if self.cancelled:
                outcome = "cancelled"
                self.api.send(chat_id, "Stopped. The data is still saved: say \"go\" to run it again, "
                                       "or /cancel again to drop the job.")
            elif self.job_process.returncode == 0 and finished:
                outcome = "finished"
                self.deliver(chat_id, folder, "\n".join(lines))
            else:
                outcome = "crashed"
                self.report_crash(chat_id, folder, lines)
        except Exception as e:
            outcome = "crashed"
            self.api.send(chat_id, f"The job hit a problem: {type(e).__name__}: {e}")
        finally:
            self._clear_package()
            self.job_process = None
            self.cancelled = False
            if outcome == "finished":
                self.set(stage="idle", listing="", verdict={}, files=[])
            else:   # keep the job and its data so "go" can try again
                self.set(stage="awaiting_data", note="")

    def report_crash(self, chat_id, folder, lines):
        """A crash is not a QA rejection: say what happened and keep the job ready to retry."""
        # The error itself (e.g. "RuntimeError: Gave up after 6 tries: ..."), not the code around it.
        errors = [l.strip() for l in lines if re.match(r"^\s*[\w.]*(Error|Exception)\b.*:", l)]
        detail = errors[-1] if errors else "\n".join(lines[-6:])
        detail = detail[-1500:]
        hint = ""
        if any(k in detail for k in ("Connection to Claude", "Gave up", "URLError", "timed out", "10054")):
            hint = ("\n\nThis looks like the laptop lost its internet connection. Check that it's awake and "
                    "online (screen-off power saving can cut Wi-Fi), then say \"go\" to retry.")
        elif "credit balance" in detail.lower():
            hint = "\n\nYour Claude API credit looks used up. Add credit in the Claude Console, then say \"go\"."
        elif "401" in detail or "api-key" in detail.lower():
            hint = "\n\nThe Claude API key was rejected. Check ANTHROPIC_API_KEY on the laptop."
        where = (f"\nPartial work is in Projects/{os.path.basename(HERE)}/{folder.replace(os.sep, '/')}"
                 if folder else "")
        self.api.send(chat_id, "❌ The job stopped before finishing (this was a crash, not a QA rejection)."
                               f"\n\nLast error:\n{detail or 'no details'}{hint}{where}"
                               "\n\nYour data is still saved: say \"go\" to try again, or /cancel to drop it.")

    # -- package approvals, relayed to your phone ------------------------------
    def ask_package(self, chat_id, package):
        self._clear_package()
        self.pending_package = package
        self.api.send(chat_id, f"📦 The analyst wants to install '{package}', which isn't on the approved list.\n"
                               f"Check it: https://pypi.org/project/{package}/\n\n"
                               f"Allow it? (yes/no). I'll decline automatically in "
                               f"{self.approval_timeout // 60} minutes.")
        self.package_timer = threading.Timer(self.approval_timeout, self._package_timeout, args=(chat_id,))
        self.package_timer.daemon = True
        self.package_timer.start()

    def answer_package(self, chat_id, answer, automatic=False):
        package, process = self.pending_package, self.job_process
        self._clear_package()
        if not package or process is None or process.stdin is None:
            return
        try:
            process.stdin.write(("y" if answer == "yes" else "n") + "\n")
            process.stdin.flush()
        except (OSError, ValueError):
            return self.api.send(chat_id, "The job ended before I could pass that on.")
        if automatic:
            self.api.send(chat_id, f"No answer on '{package}', so I declined it. The analyst will work without it.")
        elif answer == "yes":
            self.api.send(chat_id, f"Installing '{package}'...")
        else:
            self.api.send(chat_id, f"Declined '{package}'. The analyst will work without it.")

    def _package_timeout(self, chat_id):
        if self.pending_package:
            self.answer_package(chat_id, "no", automatic=True)

    def _clear_package(self):
        if self.package_timer is not None:
            self.package_timer.cancel()
        self.package_timer = None
        self.pending_package = None

    # -- step 5: send the results ---------------------------------------------
    def deliver(self, chat_id, folder, log):
        if not folder or not os.path.isdir(os.path.join(HERE, folder)):
            return self.api.send(chat_id, "The job finished, but I couldn't find its run folder. Check the laptop.")
        run = os.path.join(HERE, folder)
        work = os.path.join(run, "work")
        approved = os.path.exists(os.path.join(run, "FINAL_deliverable.md"))
        writeup = read(os.path.join(run, "FINAL_deliverable.md" if approved else "NEEDS_YOUR_REVIEW.md"))
        outputs = os.path.join(work, "outputs")
        files = sorted(os.path.join(outputs, f) for f in os.listdir(outputs)
                       if os.path.isfile(os.path.join(outputs, f))) if os.path.isdir(outputs) else []

        if not approved:
            self.api.send(chat_id, "⚠️ The job finished, but QA didn't approve it after 3 rounds, so it needs "
                                   "your review before it goes anywhere. Files below are drafts.")
        sent, too_big = [], []
        for path in files:
            if os.path.getsize(path) > MAX_UPLOAD:
                too_big.append(os.path.basename(path))
                continue
            self.api.send_file(chat_id, path)
            sent.append(os.path.basename(path))
        if os.path.exists(os.path.join(work, "app.py")):
            app_zip = zip_report_app(work, os.path.join(run, "report_app.zip"))
            if os.path.getsize(app_zip) <= MAX_UPLOAD:
                self.api.send_file(chat_id, app_zip, caption="Report app: unzip on a laptop and double-click "
                                                             "OPEN_REPORT.bat (needs Python + Streamlit).")
                sent.append("report_app.zip")
        target = outputs if os.path.isdir(outputs) else run
        onedrive = os.path.relpath(target, os.path.dirname(HERE))      # e.g. Jarvis/runs/.../work/outputs
        summary = f"✅ Done. Sent: {', '.join(sent) or 'no files'}."
        if too_big:
            summary += f"\nToo big for Telegram (over 50 MB): {', '.join(too_big)}."
        summary += f"\nEverything is also in OneDrive: Projects/{onedrive.replace(os.sep, '/')}"
        self.api.send(chat_id, summary)
        if approved:
            self.api.send(chat_id, "📋 Message for the client (edit as you like):\n\n" +
                          client_message(writeup, sent))

    # -- commands -------------------------------------------------------------
    def status_text(self):
        stage = self.state["stage"]
        if stage == "running":
            mins = int((time.time() - self.state.get("started", time.time())) / 60)
            waiting = (f" Waiting for your yes/no on installing '{self.pending_package}'."
                       if self.pending_package else "")
            return f"Running job \"{os.path.basename(self.state.get('job', ''))}\" for {mins} min.{waiting}"
        if stage == "awaiting_decision":
            return f"Waiting for your yes/no on \"{self.state.get('verdict', {}).get('title', 'a listing')}\"."
        if stage == "awaiting_data":
            files = self.state.get("files", [])
            note = " (the last run was interrupted; say \"go\" to restart it)" if self.state.get("note") else ""
            return (f"Waiting for data for \"{os.path.basename(self.state.get('job', ''))}\"{note}. "
                    f"Files so far: {', '.join(files) or 'none'}.")
        return "Idle. Paste a listing to grade it."

    def recent_jobs(self, n=8):
        """Job folders, newest first, with the data files each one holds."""
        base = os.path.join(HERE, "jobs")
        if not os.path.isdir(base):
            return []
        folders = [f for f in os.listdir(base) if os.path.isdir(os.path.join(base, f))]
        folders.sort(key=lambda f: os.path.getmtime(os.path.join(base, f)), reverse=True)
        return [(f, [d for d in os.listdir(os.path.join(base, f)) if d != "brief.txt"]) for f in folders[:n]]

    def jobs_text(self):
        jobs = self.recent_jobs()
        if not jobs:
            return "No saved jobs yet."
        lines = ["Your recent jobs (newest first):"]
        for i, (name, files) in enumerate(jobs, start=1):
            lines.append(f"{i}. {name}  ({len(files)} data file{'s' if len(files) != 1 else ''})")
        lines.append("\nTo pick one back up: /resume 1  (or /resume rail)")
        return "\n".join(lines)

    def resume(self, chat_id, which):
        if self.state["stage"] == "running":
            return self.api.send(chat_id, "A job is running right now. /cancel it first, or wait for it to finish.")
        jobs = self.recent_jobs(50)
        if not which:
            return self.api.send(chat_id, self.jobs_text())
        if which.isdigit() and 1 <= int(which) <= len(jobs):
            name, files = jobs[int(which) - 1]
        else:
            matches = [(n, f) for n, f in jobs if which.lower() in n.lower()]
            if len(matches) != 1:
                return self.api.send(chat_id, ("No job matches that." if not matches else
                                               "More than one job matches; use its number from /jobs."))
            name, files = matches[0]
        self.set(stage="awaiting_data", job=os.path.join("jobs", name), files=files, note="")
        have = f"It has: {', '.join(files)}. Say \"go\" to run it, or send more files." if files else \
               "It has no data yet. Send the file(s), then say \"go\"."
        self.api.send(chat_id, f"Resumed \"{name}\". {have}")

    def lessons_text(self):
        if self.memory is None or not len(self.memory):
            return "My memory is empty so far."
        lessons = sorted((r for r in self.memory.records if r["kind"] == "lesson"),
                         key=lambda r: r["meta"].get("seen", 1), reverse=True)[:8]
        return "What I've learned most often:\n" + "\n".join(
            f"• {r['text']} ({r['meta'].get('seen', 1)}x)" for r in lessons)

    def cancel(self, chat_id):
        if self.job_process is not None:
            self.cancelled = True
            self.job_process.terminate()
            self.api.send(chat_id, "Stopping the running job...")
            return
        self.set(stage="idle", listing="", verdict={}, files=[])
        self.api.send(chat_id, "Cleared. Paste a listing whenever you're ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HELP = """I'm Jarvis. Here's how this works:

1. Paste a job listing and I'll grade it.
2. Tell me yes or no (yeah, nah, yep... all fine).
3. If yes, send me the client's data as a file (CSV, Excel or zip), then say "go".
4. My agents do the work and I send back the files plus a message you can paste to the client.

/status  what I'm doing
/jobs    your recent jobs
/resume  pick a saved job back up (e.g. /resume 1)
/lessons what I've learned from past jobs
/cancel  drop the current listing or stop a running job"""


def format_verdict(v: dict) -> str:
    icon = {"TAKE": "🟢", "MAYBE": "🟡", "PASS": "🔴"}.get(v.get("recommendation", ""), "⚪")
    lines = [f"{icon} {v.get('recommendation', '?')}: {v.get('title', 'Untitled gig')}",
             f"Fit {v.get('fit_score', '?')}/10 ({v.get('reason', '')}). ML: {v.get('involves_ml', '?')}.",
             "", v.get("one_line_summary", "")]
    for label, key in (("Pros", "pros"), ("Cons", "cons"), ("Red flags", "red_flags")):
        items = [i for i in (v.get(key) or []) if str(i).strip()][:3]
        if items:
            lines += ["", f"{label}:"] + [f"• {i}" for i in items]
    return "\n".join(lines).strip()


def is_progress(line: str) -> bool:
    keys = ("Round", "Score", "approved", "Polish", "Code failed", "Code ran", "recalled", "not approved",
            "BLOCKED", "on the approved list). Installing", ": installed", "install failed",
            "retrying in", "Gave up",
            "Hidden test", "hidden test set", "on '", "Warning:", "[Final]")   # ML jobs: Jarvis's own model test
    skip = ("NOT on the approved list", "Check it first", "Allow installing", "Approved. Installing", "Declined.")
    return (any(k in line for k in keys) and not any(k in line for k in skip)
            and not line.strip().startswith("-"))


def read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def section(markdown: str, name: str) -> str:
    match = re.search(r"^\s*#{1,4}\s*(?:\d+[.)]\s*)?" + re.escape(name) + r"\s*$(.*?)(?=^\s*#{1,4}\s|\Z)",
                      markdown, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def plain(text: str) -> str:
    """Strip Markdown so a client sees 'Trend: ...' and not '**Trend:** ...'."""
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"(?<!\w)\*(?!\s)([^*]+)\*(?!\w)", r"\1", text)    # *italics*
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)            # [link](url) -> link
    return re.sub(r"\s+", " ", text).strip()


def _bullets(markdown: str) -> list:
    """Top-level points, with label-only bullets ('Categories:') merged into the point below."""
    points = []
    for line in markdown.splitlines():
        if not line.strip().startswith(("-", "*", "•")) or line.strip().startswith("**") and not line.strip("* ").strip():
            continue
        text = plain(line.strip().lstrip("-*• "))
        if not text:
            continue
        if points and points[-1].endswith(":"):
            points[-1] = points[-1] + " " + text       # "Categories:" + "Computers lead with..."
        else:
            points.append(text)
    return points


def _shorten(text: str, limit: int = 240) -> str:
    """Keep a highlight to a sentence or two, cut at a sentence boundary."""
    if len(text) <= limit:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = ""
    for s_ in sentences:
        if kept and len(kept) + len(s_) + 1 > limit:
            break
        kept = (kept + " " + s_).strip()
    return kept if len(kept) <= limit + 80 else text[:limit].rsplit(" ", 1)[0] + "..."


def client_message(writeup: str, files: list) -> str:
    """A short delivery message for the client, built from the approved write-up."""
    findings = section(writeup, "Findings")
    # Keep the first few points: bullets if there are any, otherwise the first sentences.
    bullets = _bullets(findings)
    if bullets:
        highlights = [_shorten(b) for b in bullets[:4]]
    else:
        sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", plain(findings)))
        highlights = [s for s in sentences if s][:3]
    questions = [plain(l.strip().lstrip("-*• ")) for l in section(writeup, "Questions for the client").splitlines() if l.strip()]
    extras = [plain(l.strip().lstrip("-*• ")) for l in section(writeup, "Optional next steps").splitlines() if l.strip()]
    deliverables = [f for f in files if not f.endswith(".zip")] or files

    parts = ["Hi! Your analysis is ready. I've attached " +
             (", ".join(deliverables) if deliverables else "the results") + "."]
    if highlights:
        parts.append("A few highlights:\n" + "\n".join(f"• {h}" for h in highlights))
    if questions:
        parts.append("A couple of questions so I can sharpen this:\n" + "\n".join(f"• {q}" for q in questions))
    if extras:
        parts.append(" ".join(extras))
    parts.append("Happy to walk you through any of it.\n\nMarro")
    return "\n\n".join(parts)


def zip_report_app(work: str, dest: str) -> str:
    """Zip what the report app needs (not the raw client data) so it opens on any laptop."""
    keep = ("app.py", "jarvis_theme.py", "OPEN_REPORT.bat")
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for name in keep:
            if os.path.exists(os.path.join(work, name)):
                z.write(os.path.join(work, name), name)
        for sub in (".streamlit", os.path.join("outputs", "working")):
            base = os.path.join(work, sub)
            for root, _, files in os.walk(base):
                for f in files:
                    full = os.path.join(root, f)
                    z.write(full, os.path.relpath(full, work))
    return dest


def main():
    os.chdir(HERE)   # jobs/, runs/ and the gig log live next to this file
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("Set your bot token first:  setx TELEGRAM_BOT_TOKEN \"<token from @BotFather>\"\n"
              "then close and reopen the terminal.")
        return
    allowed = os.environ.get("TELEGRAM_ALLOWED_USER", "").strip()
    if not allowed:
        print("No TELEGRAM_ALLOWED_USER set yet: message your bot and it will tell you your ID.")
    print(f"Grading and analysis brain: {BRAIN}")
    JarvisBot(TelegramAPI(token), allowed).run_forever()


if __name__ == "__main__":
    main()
