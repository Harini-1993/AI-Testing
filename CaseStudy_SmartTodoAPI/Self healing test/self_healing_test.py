"""
Phase 3 — Self-Healing Test Script
=====================================
This test script intelligently handles API field changes.

Problem:  API returns { "priority": "High" }
Developer changes it to: { "level": "High" }
Normal tests break immediately.

This script:
  1. Detects if the expected field is missing
  2. Tries alternative field names automatically
  3. Logs a WARNING if a fallback is used
  4. Continues testing if a valid field is found
  5. Fails properly only if NO matching field is found

Run with:
  python self_healing_test.py

Requirements:
  pip install requests
"""

import requests
import json
import sys
from datetime import datetime

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
BASE_URL      = "http://localhost:5000"
VALID_LABELS  = {"High", "Medium", "Low"}

# Field priority order — the script tries each in order
PRIORITY_FIELD_CANDIDATES = [
    "priority",   # original / expected
    "level",      # common rename
    "urgency",    # alternative name
    "rank",       # another alternative
    "importance", # fallback
]

CONFIDENCE_FIELD_CANDIDATES = [
    "confidence",
    "score",
    "probability",
    "certainty",
]

# Test tasks
TEST_TASKS = [
    "Fix production bug",
    "Prepare quarterly report",
    "Read design newsletter",
    "Server is down immediately",
    "Update project documentation",
    "Critical security vulnerability found",
    "Watch tutorial on new framework",
]

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def log(level: str, message: str):
    """Structured logger."""
    icons = {"INFO": "ℹ️ ", "WARN": "⚠️ ", "PASS": "✅", "FAIL": "❌", "HEAD": "🔍"}
    icon = icons.get(level, "  ")
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {icon} [{level}] {message}")


def find_field(body: dict, candidates: list) -> tuple:
    """
    Searches `body` for the first matching field from `candidates`.

    Returns:
        (field_name, value, is_fallback)
        - field_name : the key that was found
        - value      : the value at that key
        - is_fallback: True if the primary field (candidates[0]) was NOT used
    """
    primary = candidates[0]

    for i, field in enumerate(candidates):
        if field in body:
            return field, body[field], (i > 0)  # is_fallback = True if not primary

    return None, None, False  # Nothing found


# ─────────────────────────────────────────
# Core Test Function
# ─────────────────────────────────────────

def run_predict_test(task: str) -> bool:
    """
    Sends one prediction request and validates the response.
    Returns True if the test passed (even with a fallback field).
    """
    log("INFO", f"Testing task: '{task}'")

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"task": task},
            timeout=5
        )
    except requests.exceptions.ConnectionError:
        log("FAIL", "Cannot connect to API — is the Flask server running?")
        return False
    except requests.exceptions.Timeout:
        log("FAIL", "Request timed out after 5 seconds.")
        return False

    # ── Status Code Check ────────────────
    if response.status_code != 200:
        log("FAIL", f"Unexpected HTTP status: {response.status_code}")
        return False

    # ── Parse Body ───────────────────────
    try:
        body = response.json()
    except json.JSONDecodeError:
        log("FAIL", "Response is not valid JSON.")
        return False

    # ── Self-Healing: Find Priority Field ─
    p_field, p_value, p_is_fallback = find_field(body, PRIORITY_FIELD_CANDIDATES)

    if p_field is None:
        log("FAIL", f"No priority-like field found. Available keys: {list(body.keys())}")
        log("FAIL", f"Tried fields: {PRIORITY_FIELD_CANDIDATES}")
        return False

    if p_is_fallback:
        log("WARN", f"Primary field 'priority' missing. Using fallback field: '{p_field}'")

    # ── Validate Priority Value ───────────
    if p_value not in VALID_LABELS:
        log("FAIL", f"Invalid priority value '{p_value}'. Expected one of: {VALID_LABELS}")
        return False

    # ── Self-Healing: Find Confidence Field ──
    c_field, c_value, c_is_fallback = find_field(body, CONFIDENCE_FIELD_CANDIDATES)

    if c_field is None:
        log("WARN", f"No confidence-like field found. Available keys: {list(body.keys())}. Skipping confidence check.")
    else:
        if c_is_fallback:
            log("WARN", f"Primary field 'confidence' missing. Using fallback field: '{c_field}'")
        try:
            confidence_float = float(c_value)
            if not (0.0 <= confidence_float <= 1.0):
                log("WARN", f"Confidence value {confidence_float} is outside [0.0, 1.0] range.")
        except (TypeError, ValueError):
            log("WARN", f"Confidence value '{c_value}' is not a number.")

    log("PASS", f"Task='{task}' → {p_field}='{p_value}'" +
        (f", {c_field}={c_value}" if c_field else "") +
        (" [FALLBACK FIELDS USED]" if (p_is_fallback or c_is_fallback) else ""))
    return True


def run_health_test() -> bool:
    """Checks the /health endpoint."""
    log("HEAD", "Checking /health endpoint...")
    try:
        res = requests.get(f"{BASE_URL}/health", timeout=5)
        body = res.json()
        if res.status_code == 200 and body.get("status") == "ok":
            log("PASS", "/health returned status: ok")
            return True
        else:
            log("FAIL", f"/health check failed: {res.status_code} — {body}")
            return False
    except Exception as e:
        log("FAIL", f"/health check exception: {e}")
        return False


# ─────────────────────────────────────────
# Main Test Runner
# ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 3 — Self-Healing Test Suite")
    print(f"  Target: {BASE_URL}")
    print(f"  Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = []

    # Health check first
    results.append(run_health_test())
    print()

    # Run prediction tests
    log("HEAD", f"Running {len(TEST_TASKS)} prediction tests...\n")
    for task in TEST_TASKS:
        passed = run_predict_test(task)
        results.append(passed)
        print()

    # ── Summary ──────────────────────────
    total  = len(results)
    passed = sum(results)
    failed = total - passed

    print("=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  🎉 All tests passed!")
    else:
        print(f"  ❌ {failed} test(s) failed.")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
