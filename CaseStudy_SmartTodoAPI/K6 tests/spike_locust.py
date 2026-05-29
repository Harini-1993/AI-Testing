"""
Phase 2 - Test 2: Spike Load Test (Locust version)
====================================================
Suddenly jumps from 5 → 100 users to simulate a traffic burst.

Run with:
  locust -f spike_locust.py --headless -u 100 -r 95 --run-time 3m --host http://localhost:5000

  -u 100    = peak users
  -r 95     = spawn 95 users per second (instant spike!)
  --run-time = how long to run
"""

from locust import HttpUser, task, between, events
import random
import time

TASKS = [
    "Fix production bug",
    "Server is down immediately",
    "Prepare quarterly report",
    "Schedule weekly team meeting",
    "Read design newsletter",
    "Update project documentation",
    "Critical security vulnerability found",
    "Watch tutorial on new framework",
    "Deploy hotfix to live servers",
    "Improve search functionality",
    "Database crash in production",
    "Review sprint backlog",
    "Write unit tests for new feature",
    "Browse industry articles",
    "Fix broken CI/CD pipeline",
]

class SpikeUser(HttpUser):
    # Shorter wait time to maintain spike intensity
    wait_time = between(0.5, 1)

    @task
    def predict_task(self):
        """Hits /predict rapidly during the spike."""
        task_text = random.choice(TASKS)
        with self.client.post(
            "/predict",
            json={"task": task_text},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "priority" not in data:
                    response.failure("Missing 'priority' field")
                else:
                    response.success()
            elif response.status_code == 429:
                response.failure("Rate limited (429)")
            elif response.status_code >= 500:
                response.failure(f"Server error: {response.status_code}")
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task
    def health_check(self):
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
