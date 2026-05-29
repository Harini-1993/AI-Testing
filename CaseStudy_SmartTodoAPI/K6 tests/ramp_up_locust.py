"""
Phase 2 - Test 1: Ramp-Up Load Test (Locust version)
======================================================
Gradually increases users: 1 → 10 → 50
to find when the API becomes slow.

Run with:
  locust -f ramp_up_locust.py --headless -u 50 -r 1 --run-time 4m --host http://localhost:5000
  
  -u 50     = total users (peak)
  -r 1      = spawn 1 new user per second (gradual ramp)
  --run-time = how long to run
"""

from locust import HttpUser, task, between
import random

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
]

class TaskPriorityUser(HttpUser):
    # Each user waits 1-2 seconds between requests (realistic think time)
    wait_time = between(1, 2)

    @task(5)
    def predict_task(self):
        """Hits /predict — weighted 5x more than health check."""
        task_text = random.choice(TASKS)
        with self.client.post(
            "/predict",
            json={"task": task_text},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "priority" not in data:
                    response.failure("Missing 'priority' field in response")
                elif data["priority"] not in ["High", "Medium", "Low"]:
                    response.failure(f"Invalid priority value: {data['priority']}")
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def health_check(self):
        """Hits /health — lightweight check."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200 and response.json().get("status") == "ok":
                response.success()
            else:
                response.failure("Health check failed")
