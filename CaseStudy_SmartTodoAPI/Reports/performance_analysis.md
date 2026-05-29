# Performance Test Analysis Report

**Project:** AI Task Priority API  
**Phase:** 2 — Load Testing with Locust  
**Date:** 29 May 2026  
**Tester:** Lakshmipathi H

---

## Test Environment

| Item | Details |
|------|---------|
| API URL | http://localhost:5000 |
| Model | scikit-learn LogisticRegression |
| Machine | Windows 11, Python 3.x, Virtual Environment (.venv) |
| Tool Used | Locust 2.x (used as K6 alternative) |

---

## Test 1: Ramp-Up Load Test

### Configuration
- Stage 1: 1 user (warm-up), spawning 1 user/sec
- Stage 2: Ramp to 50 users over 4 minutes
- Endpoints tested: POST /predict and GET /health

### Results

| Metric | Value |
|--------|-------|
| Total Requests | 7,103 |
| Avg Response Time | 10.84 ms |
| p95 Response Time | 21 ms |
| Max Response Time | 54 ms |
| Error Rate | 0% |
| Requests/sec (peak) | 29.74 |

### Breakdown by Endpoint

| Endpoint | Requests | Avg (ms) | p95 (ms) | Max (ms) | Failures |
|----------|----------|----------|----------|----------|----------|
| POST /predict | 5,887 | 11.24 ms | 21 ms | 54 ms | 0 |
| GET /health | 1,216 | 8.95 ms | 17 ms | 38 ms | 0 |

### Analysis

**Q1: At what point does response time increase?**

> Response time remained stable throughout the entire ramp-up. The average stayed at ~10.84ms and the p95 was just 21ms even at peak load of 50 concurrent users. There was no noticeable degradation at any user level, indicating the API handled the gradual ramp comfortably within this range.

**Q2: Does the server start failing?**

> No failures were observed across all 7,103 requests. The error rate held at 0% throughout the entire test, from 1 user all the way up to 50 concurrent users. The Flask dev server was able to serve all requests successfully under gradual load.

**Key Finding:**  
The API performed exceptionally well under ramp-up conditions, with a p95 response time of just 21ms at 50 users — well within acceptable limits. The 0% error rate and stable throughput of ~29.74 requests/sec confirm that the model inference is fast enough to handle gradual traffic growth without any degradation. The single-threaded Flask dev server was sufficient for this load level.

---

## Test 2: Spike Load Test

### Configuration
- Baseline: 5 users
- Spike: Instantly jump to 100 users (spawn rate: 95 users/sec)
- Run time: 3 minutes
- Endpoints tested: POST /predict and GET /health

### Results

| Metric | Value |
|--------|-------|
| Total Requests | 23,136 |
| Avg Response Time | 20.05 ms |
| p95 Response Time | 51 ms |
| Max Response Time | 260 ms |
| Error Rate | 0% |
| Peak VUs | 100 |

### Breakdown by Endpoint

| Endpoint | Requests | Avg (ms) | p95 (ms) | Max (ms) | Failures |
|----------|----------|----------|----------|----------|----------|
| POST /predict | 11,588 | 21.35 ms | 52 ms | 255 ms | 0 |
| GET /health | 11,548 | 18.74 ms | 48 ms | 256 ms | 0 |

### Analysis

**Q1: Does the server crash during the spike?**

> The server did not crash. Even under an instant spike to 100 concurrent users, the API continued serving all requests successfully with 0 failures. The max response time reached 260ms during peak load — a noticeable jump from the ramp-up baseline of 54ms — indicating the Flask dev server was under pressure but did not break.

**Q2: Does it recover after the spike?**

> The server demonstrated resilience throughout the spike test. Since no failures occurred and throughput reached 129.16 requests/sec at peak, the API absorbed the sudden burst effectively. Response times stayed within acceptable limits (p95 = 51ms), suggesting the model inference is lightweight enough to handle high concurrency on this hardware.

**Key Finding:**  
The spike test revealed that the API can handle a sudden jump to 100 users without crashing or producing any errors. The p95 response time of 51ms under full spike load is impressive for a Flask dev server. However, the max response time of 260ms (vs 54ms in ramp-up) shows that extreme concurrency does introduce tail latency, which would need to be addressed with a production WSGI server like Gunicorn for real-world deployment.

---


## Conclusion

Both load tests confirmed that the AI Task Priority API performs reliably under the tested conditions. The ramp-up test showed zero failures and a p95 response time of just 21ms across 7,103 requests, while the spike test handled a sudden burst to 100 users with a p95 of 51ms and still zero failures across 23,136 requests. The most significant observation is that max response time jumped from 54ms (ramp-up) to 260ms (spike), indicating tail latency under extreme concurrency. If this were a real production system, the next steps would be to replace the Flask dev server with Gunicorn, add response caching for repeated task inputs, and set up autoscaling to handle sustained traffic beyond 100 concurrent users.

---

