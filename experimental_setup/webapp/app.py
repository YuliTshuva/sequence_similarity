"""
Yuli Tshuva
Labeling site for the human-alignment study.

Participants register, log in, and work through a shuffled list of trials. Each
trial shows a query curve and two candidates; they pick the candidate that looks
more similar, or declare a tie.

The stimuli come from a trials file (see TRIALS_PATH) that carries only curves
and trial groupings -- no algorithm ever appears in what reaches the browser, so
the labels stay measure-agnostic and can be scored against any distance later.

Run locally:
    python app.py                 -> http://localhost:8000/

Deploy:
    gunicorn --bind 0.0.0.0:8000 --workers 3 app:app
    (on PythonAnywhere, point the WSGI config at `from app import app as application`)

Environment:
    SECRET_KEY       session signing key   (required in production)
    ADMIN_PASSWORD   unlocks /admin        (default "changeme")
    TRIALS_PATH      path to trials.json
    DB_PATH          path to the SQLite file
"""

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from functools import wraps

from flask import (Flask, flash, g, jsonify, redirect, render_template, request,
                   session, url_for, Response)
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRIALS_PATH = os.environ.get("TRIALS_PATH", os.path.join(BASE_DIR, "trials.json"))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "labels.db"))
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme")

MIN_USERNAME = 3
MAX_USERNAME = 32
MIN_PASSWORD = 6
MAX_RESPONSE_MS = 24 * 60 * 60 * 1000     # anything longer is a stale tab, not a judgement
CHOICES = {"a", "b", "tie"}

MAX_SIGNIN_FAILURES = 8                   # per username, within the window
SIGNIN_WINDOW_S = 300
_signin_failures = {}                     # username -> [timestamps]

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key-change-me")


# ---------------------------------------------------------------------------
# Stimuli
# ---------------------------------------------------------------------------
def load_trials():
    """
    Read the trials file once and cache it on the app.

    Expected shape:
        {"sequences": {"<seq_id>": [float, ...], ...},
         "trials": [{"trial_id": str, "query": <seq_id>,
                     "candidate_a": <seq_id>, "candidate_b": <seq_id>}, ...]}
    """
    if not hasattr(app, "_trials"):
        if not os.path.exists(TRIALS_PATH):
            raise SystemExit(
                f"No trials file at {TRIALS_PATH}. "
                "Build one first:  python build_trials.py --triplets ... --sequences ...")
        with open(TRIALS_PATH, encoding="utf-8") as f:
            app._trials = json.load(f)
    return app._trials


def trial_index():
    """trial_id -> trial dict."""
    if not hasattr(app, "_trial_index"):
        app._trial_index = {t["trial_id"]: t for t in load_trials()["trials"]}
    return app._trial_index


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS responses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id),
    trial_id    TEXT NOT NULL,
    choice      TEXT NOT NULL,
    shown_left  TEXT NOT NULL,
    shown_right TEXT NOT NULL,
    response_ms INTEGER,
    created_at  TEXT NOT NULL,
    UNIQUE (user_id, trial_id)
);
"""


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login"))
        return view(*args, **kwargs)
    return wrapped


@app.get("/login")
def login():
    if "user_id" in session:
        return redirect(url_for("label"))
    return render_template("login.html")


@app.post("/register")
def register():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""

    if (not MIN_USERNAME <= len(username) <= MAX_USERNAME
            or not username.replace("_", "").replace("-", "").isalnum()):
        flash(f"Username must be {MIN_USERNAME}-{MAX_USERNAME} characters, "
              "letters/digits/_/- only.", "error")
        return redirect(url_for("login"))
    if len(password) < MIN_PASSWORD:
        flash(f"Password must be at least {MIN_PASSWORD} characters.", "error")
        return redirect(url_for("login"))

    try:
        cur = db().execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), now()),
        )
        db().commit()
    except sqlite3.IntegrityError:
        flash("That username is taken. Pick another, or sign in.", "error")
        return redirect(url_for("login"))

    session.clear()
    session["user_id"] = cur.lastrowid
    session["username"] = username
    return redirect(url_for("label"))


def _throttled(username):
    """
    True when this username has failed too often lately.

    A study site sits on the open web with self-registered passwords, so
    unlimited guessing should not be free. Deliberately in-memory and simple --
    it slows down guessing without adding a dependency.
    """
    import time
    cutoff = time.time() - SIGNIN_WINDOW_S
    recent = [t for t in _signin_failures.get(username, []) if t > cutoff]
    _signin_failures[username] = recent
    return len(recent) >= MAX_SIGNIN_FAILURES


def _record_failure(username):
    import time
    _signin_failures.setdefault(username, []).append(time.time())


@app.post("/signin")
def signin():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""

    if _throttled(username):
        flash("Too many failed attempts. Wait a few minutes and try again.", "error")
        return redirect(url_for("login"))

    row = db().execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    if row is None or not check_password_hash(row["password_hash"], password):
        _record_failure(username)
        flash("Wrong username or password.", "error")
        return redirect(url_for("login"))

    _signin_failures.pop(username, None)

    session.clear()
    session["user_id"] = row["id"]
    session["username"] = row["username"]
    return redirect(url_for("label"))


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------
def user_order(user_id):
    """
    The trial order for one participant.

    Derived by hashing the user id against each trial id rather than stored, so
    the order is stable across sessions and devices without another table, and
    two participants see the trials in different orders.
    """
    trials = load_trials()["trials"]
    return sorted(trials, key=lambda t: hashlib.sha256(
        f"{user_id}:{t['trial_id']}".encode()).hexdigest())


def sides_for(user_id, trial):
    """
    Which candidate is drawn on the left, decided by a hash so it is stable on
    refresh and balanced across participants. Side bias is a real effect in
    two-alternative tasks, so it must not correlate with anything.
    """
    digest = hashlib.sha256(f"side:{user_id}:{trial['trial_id']}".encode()).digest()
    if digest[0] % 2:
        return "b", "a"
    return "a", "b"


def answered(user_id):
    rows = db().execute("SELECT trial_id FROM responses WHERE user_id = ?", (user_id,))
    return {r["trial_id"] for r in rows}


@app.get("/")
def home():
    return redirect(url_for("label") if "user_id" in session else url_for("login"))


@app.get("/label")
@login_required
def label():
    user_id = session["user_id"]
    order = user_order(user_id)
    done = answered(user_id)
    remaining = [t for t in order if t["trial_id"] not in done]

    if not remaining:
        return render_template("done.html", total=len(order))

    trial = remaining[0]
    left, right = sides_for(user_id, trial)
    sequences = load_trials()["sequences"]

    return render_template(
        "label.html",
        trial_id=trial["trial_id"],
        query=sequences[trial["query"]],
        left_curve=sequences[trial[f"candidate_{left}"]],
        right_curve=sequences[trial[f"candidate_{right}"]],
        left_role=left,
        right_role=right,
        completed=len(done),
        total=len(order),
    )


def clean_response_ms(value):
    """
    Coerce the client-supplied timing to a sane integer or None.

    SQLite will happily store the string "abc" in an INTEGER column, so an
    unvalidated value silently poisons the timing column -- which only shows up
    much later, as an analysis that cannot compute a mean.
    """
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return None
    return ms if 0 <= ms <= MAX_RESPONSE_MS else None


def next_trial_id(user_id):
    """The trial this participant is currently being shown, if any."""
    done = answered(user_id)
    for trial in user_order(user_id):
        if trial["trial_id"] not in done:
            return trial["trial_id"]
    return None


@app.post("/api/label")
@login_required
def save_label():
    payload = request.get_json(silent=True) or {}
    trial_id = payload.get("trial_id")
    choice = payload.get("choice")

    if trial_id not in trial_index():
        return jsonify({"error": "unknown trial"}), 400
    if choice not in CHOICES:
        return jsonify({"error": "invalid choice"}), 400
    # Only the trial currently on screen may be answered, so the stored order
    # really is the order the participant saw.
    if trial_id != next_trial_id(session["user_id"]):
        return jsonify({"error": "not the current trial"}), 409

    left, right = sides_for(session["user_id"], trial_index()[trial_id])
    try:
        db().execute(
            "INSERT INTO responses (user_id, trial_id, choice, shown_left, shown_right,"
            " response_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session["user_id"], trial_id, choice, left, right,
             clean_response_ms(payload.get("response_ms")), now()),
        )
        db().commit()
    except sqlite3.IntegrityError:
        pass  # already answered; a double-submit is not an error
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------
@app.get("/admin/login")
def admin_login():
    return render_template("admin_login.html")


@app.post("/admin/login")
def admin_signin():
    if request.form.get("password") == ADMIN_PASSWORD:
        session["is_admin"] = True
        return redirect(url_for("admin"))
    flash("Wrong password.", "error")
    return redirect(url_for("admin_login"))


@app.get("/admin")
@admin_required
def admin():
    total_trials = len(load_trials()["trials"])
    rows = db().execute("""
        SELECT u.username, u.created_at,
               COUNT(r.id) AS n,
               SUM(CASE WHEN r.choice = 'tie' THEN 1 ELSE 0 END) AS ties,
               AVG(r.response_ms) AS mean_ms
        FROM users u LEFT JOIN responses r ON r.user_id = u.id
        GROUP BY u.id ORDER BY n DESC, u.username
    """).fetchall()

    participants = [{
        "username": r["username"],
        "joined": r["created_at"][:10],
        "n": r["n"],
        "pct": round(100 * r["n"] / total_trials) if total_trials else 0,
        "ties": r["ties"] or 0,
        "mean_s": round((r["mean_ms"] or 0) / 1000, 1),
    } for r in rows]

    covered = db().execute(
        "SELECT COUNT(DISTINCT trial_id) AS c FROM responses").fetchone()["c"]

    return render_template("admin.html", participants=participants,
                           total_trials=total_trials, covered=covered,
                           total_responses=sum(p["n"] for p in participants))


@app.get("/admin/export.json")
@admin_required
def export_json():
    rows = db().execute("""
        SELECT u.username, r.trial_id, r.choice, r.shown_left, r.shown_right,
               r.response_ms, r.created_at
        FROM responses r JOIN users u ON u.id = r.user_id
        ORDER BY u.username, r.created_at
    """).fetchall()
    body = json.dumps([dict(r) for r in rows], indent=2)
    return Response(body, mimetype="application/json", headers={
        "Content-Disposition": 'attachment; filename="labels.json"'})


@app.get("/admin/export.csv")
@admin_required
def export_csv():
    rows = db().execute("""
        SELECT u.username, r.trial_id, r.choice, r.shown_left, r.shown_right,
               r.response_ms, r.created_at
        FROM responses r JOIN users u ON u.id = r.user_id
        ORDER BY u.username, r.created_at
    """).fetchall()
    header = "username,trial_id,choice,shown_left,shown_right,response_ms,created_at"
    lines = [header] + [",".join("" if r[k] is None else str(r[k]) for k in r.keys())
                        for r in rows]
    return Response("\n".join(lines), mimetype="text/csv", headers={
        "Content-Disposition": 'attachment; filename="labels.csv"'})


@app.get("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin_login"))


init_db()

if __name__ == "__main__":
    # Pick up template edits without a restart when running the dev server.
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True
    app.run(host="0.0.0.0", port=8000, debug=False)
