#!/usr/bin/env python3
"""Generates the mock observability dataset used by the agent's tools.

Scenario: a deploy of `payments` at 14:31 UTC on the incident day lowers its DB
connection pool timeout too aggressively. Payments starts timing out talking
to payments-db under normal load, which cascades into checkout (which calls
payments synchronously) and then into frontend (which calls checkout).
inventory / inventory-db are unaffected — a control group so the agent has
to actually localize the fault instead of assuming "everything is broken".

The incident day defaults to *yesterday (UTC)* relative to when you run the
script, so a question like "why did checkout error rate spike yesterday
afternoon?" makes sense against freshly generated data. Pin a specific day
with --date for a reproducible dataset (e.g. for evals).

Run: python3 scripts/gen_data.py [--date YYYY-MM-DD]
Writes JSON files into data/.
"""
import argparse
import json
import os
import random
from datetime import date, datetime, timedelta, timezone

random.seed(7)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--date", help="incident day, YYYY-MM-DD (UTC). Default: yesterday.")
args = parser.parse_args()

if args.date:
    INCIDENT_DAY = date.fromisoformat(args.date)
else:
    INCIDENT_DAY = datetime.now(timezone.utc).date() - timedelta(days=1)


def at(hour, minute, days=0):
    """A UTC timestamp on the incident day (or `days` before/after it)."""
    d = INCIDENT_DAY + timedelta(days=days)
    return datetime(d.year, d.month, d.day, hour, minute, 0, tzinfo=timezone.utc)


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

START = at(13, 0)
END = at(16, 0)
DEPLOY_TS = at(14, 31)
RECOVER_TS = at(15, 45)  # rollback happens


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


SERVICES = [
    {
        "name": "frontend",
        "description": "Public web BFF that renders the storefront and checkout UI.",
        "depends_on": ["checkout"],
    },
    {
        "name": "checkout",
        "description": "Orchestrates an order: reserves inventory, then charges payment.",
        "depends_on": ["payments", "inventory"],
    },
    {
        "name": "payments",
        "description": "Processes card charges against the payments database and an external gateway.",
        "depends_on": ["payments-db"],
    },
    {
        "name": "inventory",
        "description": "Reserves and decrements stock levels.",
        "depends_on": ["inventory-db"],
    },
    {
        "name": "payments-db",
        "description": "Postgres primary backing the payments service.",
        "depends_on": [],
    },
    {
        "name": "inventory-db",
        "description": "Postgres primary backing the inventory service.",
        "depends_on": [],
    },
]

DEPLOYS = [
    {
        "service": "checkout",
        "version": "v4.2.0",
        "timestamp": iso(at(8, 15)),
        "author": "bob",
        "notes": "Add gift-card redemption step to checkout flow.",
    },
    {
        "service": "payments",
        "version": "v2.13.2",
        "timestamp": iso(at(9, 0, days=-1)),
        "author": "alice",
        "notes": "Bump gateway SDK to 3.1.0.",
    },
    {
        "service": "payments",
        "version": "v2.14.0",
        "timestamp": iso(DEPLOY_TS),
        "author": "carol",
        "notes": "Reduce payments-db connection pool timeout from 5000ms to 500ms and tighten retry backoff, to cut tail latency.",
    },
    {
        "service": "payments",
        "version": "v2.14.1",
        "timestamp": iso(RECOVER_TS),
        "author": "carol",
        "notes": "Rollback: restore connection pool timeout to 5000ms after incident INC-1042.",
    },
]

with open(os.path.join(OUT_DIR, "services.json"), "w") as f:
    json.dump(SERVICES, f, indent=2)

with open(os.path.join(OUT_DIR, "deploys.json"), "w") as f:
    json.dump(DEPLOYS, f, indent=2)

# ---- metrics ----
# 1-minute resolution error_rate (%) and p99_latency_ms per service.
BASELINE = {
    "frontend": {"error_rate": 0.3, "p99_latency_ms": 180},
    "checkout": {"error_rate": 0.4, "p99_latency_ms": 220},
    "payments": {"error_rate": 0.2, "p99_latency_ms": 90},
    "inventory": {"error_rate": 0.3, "p99_latency_ms": 60},
    "payments-db": {"error_rate": 0.1, "p99_latency_ms": 8},
    "inventory-db": {"error_rate": 0.1, "p99_latency_ms": 7},
}

# How much each service is impacted during the incident window (multipliers).
IMPACT = {
    "frontend": {"error_rate": 18, "p99_latency_ms": 5},
    "checkout": {"error_rate": 35, "p99_latency_ms": 9},
    "payments": {"error_rate": 55, "p99_latency_ms": 14},
    "inventory": {"error_rate": 1.0, "p99_latency_ms": 1.0},  # unaffected control group
    "payments-db": {"error_rate": 1.0, "p99_latency_ms": 1.2},  # DB itself is fine; pool is the problem
    "inventory-db": {"error_rate": 1.0, "p99_latency_ms": 1.0},
}

metrics = {name: {"error_rate": [], "p99_latency_ms": []} for name in BASELINE}

t = START
while t <= END:
    in_incident = DEPLOY_TS <= t < RECOVER_TS
    # ramp up over ~3 min after deploy, ramp down over ~2 min after rollback
    ramp = 1.0
    if in_incident:
        minutes_in = (t - DEPLOY_TS).total_seconds() / 60
        ramp = min(1.0, minutes_in / 3.0)
    for name, base in BASELINE.items():
        for metric in ("error_rate", "p99_latency_ms"):
            noise = random.uniform(0.85, 1.15)
            value = base[metric] * noise
            if in_incident:
                value = base[metric] * (1 + (IMPACT[name][metric] - 1) * ramp) * noise
            if metric == "error_rate":
                value = round(min(value, 100.0), 2)
            else:
                value = round(value, 1)
            metrics[name][metric].append({"ts": iso(t), "value": value})
    t += timedelta(minutes=1)

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ---- logs ----
logs = []


def add(ts, service, level, message, trace_id=None):
    entry = {"ts": iso(ts), "service": service, "level": level, "message": message}
    if trace_id:
        entry["trace_id"] = trace_id
    logs.append(entry)


# sparse baseline INFO noise
t = START
while t <= END:
    if random.random() < 0.15:
        svc = random.choice(list(BASELINE.keys()))
        add(t, svc, "INFO", f"{svc}: request handled ok", trace_id=f"trc-{random.randint(100000,999999)}")
    t += timedelta(seconds=random.randint(20, 90))

# the deploy event itself, logged by the deploy pipeline
add(DEPLOY_TS, "payments", "INFO", "deploy: rolling out payments v2.14.0 (connection pool timeout 5000ms -> 500ms)")

# incident-window error bursts, cascading payments -> checkout -> frontend
t = DEPLOY_TS + timedelta(seconds=45)
while t < RECOVER_TS:
    trace_id = f"trc-{random.randint(100000,999999)}"
    if random.random() < 0.8:
        add(t, "payments", "ERROR",
            "payments-db: context deadline exceeded acquiring connection from pool (timeout=500ms, pool_size=20, in_use=20)",
            trace_id)
        add(t + timedelta(milliseconds=520), "checkout", "ERROR",
            "payments.Charge failed: rpc error: code = DeadlineExceeded desc = context deadline exceeded",
            trace_id)
        if random.random() < 0.6:
            add(t + timedelta(milliseconds=560), "frontend", "ERROR",
                "checkout service returned 503 Service Unavailable", trace_id)
    else:
        add(t, "checkout", "WARN", "retrying payments.Charge (attempt 2/3)", trace_id)
    t += timedelta(seconds=random.randint(3, 8))

add(RECOVER_TS, "payments", "INFO", "deploy: rolling out payments v2.14.1 (rollback, connection pool timeout restored to 5000ms)")
add(RECOVER_TS + timedelta(seconds=30), "payments", "INFO", "payments-db: connection pool healthy, in_use=4/20")

logs.sort(key=lambda e: e["ts"])
with open(os.path.join(OUT_DIR, "logs.json"), "w") as f:
    json.dump(logs, f, indent=2)

print(f"incident day: {INCIDENT_DAY.isoformat()} (payments deploy at {iso(DEPLOY_TS)})")
print(f"services: {len(SERVICES)}")
print(f"deploys: {len(DEPLOYS)}")
print(f"metrics points: {sum(len(m['error_rate']) for m in metrics.values())}")
print(f"logs: {len(logs)}")
