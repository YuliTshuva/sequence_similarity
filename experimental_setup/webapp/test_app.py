"""
Yuli Tshuva
Tests for the labeling site -- participant side and admin side.

Run:  python test_app.py

Uses a throwaway database so it never touches real study data. The checks that
matter most are the ones about *which* answer gets stored: if screen position
leaked into the label, or if a participant's trial order were not a permutation,
the collected data would be quietly wrong rather than visibly broken.
"""

import json
import os
import re
import sqlite3
import sys
import tempfile

TMP_DB = os.path.join(tempfile.mkdtemp(), "test_labels.db")
os.environ["DB_PATH"] = TMP_DB
os.environ["ADMIN_PASSWORD"] = "adminpw"
os.environ["SECRET_KEY"] = "test-secret"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import app, load_trials, sides_for, user_order  # noqa: E402

app.config["TESTING"] = True
TRIALS = load_trials()["trials"]
SEQUENCES = load_trials()["sequences"]

PASSES, FAILURES = 0, []


def check(name, condition, detail=""):
    global PASSES
    if condition:
        PASSES += 1
        print(f"  PASS  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}  {detail}")


def new_user(username, password="testpass1"):
    c = app.test_client()
    c.post("/register", data={"username": username, "password": password})
    return c


def current_trial(client):
    """Parse the served page for the trial id and the two curves as drawn."""
    html = client.get("/label").data.decode()
    if "That&rsquo;s all of them" in html or "all of them" in html:
        return None
    trial_id = re.search(r'const TRIAL_ID = "(.*?)"', html).group(1)
    left = json.loads(re.search(r"const LEFT = (\[.*?\]);", html).group(1))
    right = json.loads(re.search(r"const RIGHT = (\[.*?\]);", html).group(1))
    query = json.loads(re.search(r"const QUERY = (\[.*?\]);", html).group(1))
    return {"trial_id": trial_id, "left": left, "right": right, "query": query}


def db_rows(sql, args=()):
    conn = sqlite3.connect(TMP_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, args).fetchall()
    conn.close()
    return rows


# ===========================================================================
print("\n--- Participant: accounts ---")

anon = app.test_client()
check("anonymous /label redirects to login", anon.get("/label").status_code == 302)
check("anonymous /api/label is refused",
      anon.post("/api/label", json={"trial_id": TRIALS[0]["trial_id"], "choice": "a"}).status_code == 302)
check("anonymous /admin redirects", anon.get("/admin").status_code == 302)

alice = new_user("alice")
check("registration lands on the labeling page",
      b"Reference curve" in alice.get("/label").data)

for label, data, expected in (
        ("duplicate username rejected", {"username": "alice", "password": "testpass1"}, b"taken"),
        ("short username rejected", {"username": "ab", "password": "testpass1"}, b"3-32 characters"),
        ("over-long username rejected", {"username": "u" * 500, "password": "testpass1"}, b"3-32 characters"),
        ("short password rejected", {"username": "bobby", "password": "12"}, b"at least 6"),
        ("odd characters rejected", {"username": "a b!c", "password": "testpass1"}, b"letters/digits")):
    r = app.test_client().post("/register", data=data, follow_redirects=True)
    check(label, expected in r.data)

check("wrong password refused",
      b"Wrong username" in app.test_client().post(
          "/signin", data={"username": "alice", "password": "nope"},
          follow_redirects=True).data)
check("unknown user refused",
      b"Wrong username" in app.test_client().post(
          "/signin", data={"username": "ghost", "password": "testpass1"},
          follow_redirects=True).data)

# ===========================================================================
print("\n--- Participant: the trial sequence ---")

order_alice = [t["trial_id"] for t in user_order(1)]
check("order is a permutation of all trials",
      sorted(order_alice) == sorted(t["trial_id"] for t in TRIALS),
      f"{len(order_alice)} vs {len(TRIALS)}")
check("order is stable across calls", order_alice == [t["trial_id"] for t in user_order(1)])
check("two participants get different orders",
      order_alice != [t["trial_id"] for t in user_order(2)])

first = current_trial(alice)
check("refresh keeps the same trial", current_trial(alice)["trial_id"] == first["trial_id"])
check("refresh keeps the same left/right", current_trial(alice)["left"] == first["left"])

# ===========================================================================
print("\n--- Participant: what actually gets stored ---")

# The critical one: click the LEFT card and confirm the stored answer is the
# candidate that was on the left, not the literal string "left".
trial = current_trial(alice)
spec = next(t for t in TRIALS if t["trial_id"] == trial["trial_id"])
left_is = "a" if trial["left"] == SEQUENCES[spec["candidate_a"]] else "b"
alice.post("/api/label", json={"trial_id": trial["trial_id"], "choice": left_is,
                               "response_ms": 1234})
row = db_rows("SELECT * FROM responses WHERE trial_id = ?", (trial["trial_id"],))[0]
check("clicking the left card stores that candidate's identity",
      row["choice"] == left_is, f"stored {row['choice']}, left showed {left_is}")
check("the side shown is recorded", row["shown_left"] == left_is)
check("response time is recorded", row["response_ms"] == 1234)

check("query curve matches the trial spec", trial["query"] == SEQUENCES[spec["query"]])
check("the two candidates differ", trial["left"] != trial["right"])

sides = [sides_for(uid, TRIALS[0])[0] for uid in range(1, 201)]
balance = sides.count("a") / len(sides)
check("left/right assignment is roughly balanced", 0.4 < balance < 0.6, f"{balance:.2f} 'a' on left")

check("advanced to the next trial", current_trial(alice)["trial_id"] != trial["trial_id"])

# Re-submitting the same trial must not create a second row.
before = len(db_rows("SELECT * FROM responses"))
alice.post("/api/label", json={"trial_id": trial["trial_id"], "choice": "b"})
check("double submit does not duplicate", len(db_rows("SELECT * FROM responses")) == before)
check("double submit does not overwrite the first answer",
      db_rows("SELECT * FROM responses WHERE trial_id = ?", (trial["trial_id"],))[0]["choice"] == left_is)

check("bad choice rejected",
      alice.post("/api/label", json={"trial_id": TRIALS[5]["trial_id"], "choice": "x"}).status_code == 400)
check("unknown trial rejected",
      alice.post("/api/label", json={"trial_id": "no_such_trial", "choice": "a"}).status_code == 400)
check("empty body rejected", alice.post("/api/label", json={}).status_code == 400)

# ===========================================================================
print("\n--- Participant: sessions and progress ---")

html = alice.get("/label").data.decode()
check("progress shows trial 2 of N", f"Trial 2 of {len(TRIALS)}" in html, html[:0])

alice.get("/logout")
check("after logout /label redirects", alice.get("/label").status_code == 302)
alice.post("/signin", data={"username": "alice", "password": "testpass1"})
check("signing back in resumes at trial 2",
      f"Trial 2 of {len(TRIALS)}" in alice.get("/label").data.decode())

bob = new_user("bob")
check("a second participant starts at trial 1",
      f"Trial 1 of {len(TRIALS)}" in bob.get("/label").data.decode())
check("participants do not share answers",
      len(db_rows("SELECT * FROM responses WHERE user_id = 2")) == 0)

# Finish every trial for bob and confirm the completion page.
answered = 0
while True:
    t = current_trial(bob)
    if t is None:
        break
    bob.post("/api/label", json={"trial_id": t["trial_id"], "choice": "tie", "response_ms": 900})
    answered += 1
    if answered > len(TRIALS) + 5:
        break
check("working through every trial ends on the done page", answered == len(TRIALS),
      f"answered {answered} of {len(TRIALS)}")
check("done page is shown", b"all of them" in bob.get("/label").data)

# ===========================================================================
print("\n--- Admin ---")

admin = app.test_client()
check("admin gate redirects when locked", admin.get("/admin").status_code == 302)
check("wrong admin password refused",
      b"Wrong password" in admin.post("/admin/login", data={"password": "guess"},
                                      follow_redirects=True).data)
check("exports refused when locked", admin.get("/admin/export.json").status_code == 302)

admin.post("/admin/login", data={"password": "adminpw"})
page = admin.get("/admin").data.decode()
check("admin page loads", "Participants" in page)
check("both participants listed", "alice" in page and "bob" in page)

total_responses = len(db_rows("SELECT * FROM responses"))
check("response total is right", f">{total_responses}<" in page,
      f"expected {total_responses} in page")
check("bob shows as complete", f"{len(TRIALS)} / {len(TRIALS)}" in page)

exported = json.loads(admin.get("/admin/export.json").data)
check("json export row count matches the database", len(exported) == total_responses,
      f"{len(exported)} vs {total_responses}")
check("json export carries the needed fields",
      set(exported[0]) == {"username", "trial_id", "choice", "shown_left",
                           "shown_right", "response_ms", "created_at"})
check("json export never contains 'left'/'right' as a choice",
      all(r["choice"] in {"a", "b", "tie"} for r in exported))

csv_lines = admin.get("/admin/export.csv").data.decode().splitlines()
check("csv export row count matches", len(csv_lines) == total_responses + 1)
check("csv header is right",
      csv_lines[0] == "username,trial_id,choice,shown_left,shown_right,response_ms,created_at")
check("csv rows have the right column count",
      all(len(l.split(",")) == 7 for l in csv_lines[1:]))

admin.get("/admin/logout")
check("admin logout re-locks", admin.get("/admin").status_code == 302)

check("a participant session cannot reach admin", alice.get("/admin").status_code == 302)

# ===========================================================================
print("")
print("--- Hardening: what a hostile or broken client can send ---")

carol = new_user("carol")

# response_ms is client-supplied, and SQLite stores a string in an INTEGER
# column without complaint, so it has to be validated on the way in.
for _label, _value in (("non-numeric", "abc"), ("negative", -999), ("absurd", 10 ** 12)):
    _cur = current_trial(carol)
    carol.post("/api/label", json={"trial_id": _cur["trial_id"], "choice": "a",
                                   "response_ms": _value})
    _got = db_rows("SELECT r.response_ms FROM responses r JOIN users u ON u.id = r.user_id"
                   " WHERE u.username = 'carol' AND r.trial_id = ?",
                   (_cur["trial_id"],))[0]["response_ms"]
    check(f"{_label} response_ms is stored as NULL", _got is None, f"got {_got!r}")

check("every stored response_ms is an integer or NULL",
      all(r["response_ms"] is None or isinstance(r["response_ms"], int)
          for r in db_rows("SELECT response_ms FROM responses")))

# A participant must not be able to answer a trial that is not on screen.
served = current_trial(carol)["trial_id"]
other = next(t["trial_id"] for t in TRIALS if t["trial_id"] != served)
before_n = len(db_rows("SELECT * FROM responses"))
resp = carol.post("/api/label", json={"trial_id": other, "choice": "a"})
check("answering a trial that is not on screen is refused", resp.status_code == 409,
      f"status {resp.status_code}")
check("the refused answer was not stored",
      len(db_rows("SELECT * FROM responses")) == before_n)
check("the served trial is unchanged after a refused submit",
      current_trial(carol)["trial_id"] == served)

# Sign-in throttling.
for _ in range(10):
    app.test_client().post("/signin", data={"username": "carol", "password": "wrong"})
r = app.test_client().post("/signin", data={"username": "carol", "password": "testpass1"},
                           follow_redirects=True)
check("repeated wrong passwords lock that account briefly",
      b"Too many failed attempts" in r.data)
check("throttling is per-username, not global",
      b"Too many failed attempts" not in app.test_client().post(
          "/signin", data={"username": "alice", "password": "testpass1"},
          follow_redirects=True).data)

# CSV must leave a null cell empty rather than writing the string "None".
admin2 = app.test_client()
admin2.post("/admin/login", data={"password": "adminpw"})
csv_body = admin2.get("/admin/export.csv").data.decode()
check("csv writes empty cells for NULL, never the text 'None'", ",None" not in csv_body)
check("csv still has one row per response",
      len(csv_body.splitlines()) == len(db_rows("SELECT * FROM responses")) + 1)

# ===========================================================================
print(f"\n{PASSES} passed, {len(FAILURES)} failed")
if FAILURES:
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
