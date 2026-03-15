# Scalability Write-up

**Architectural Challenge:** This agent currently works for one class of 60 students. AIValytics serves 50+ colleges with 500–1,000 students each. How would you architect this system to handle 50,000 concurrent student profiles — including real-time state updates after every quiz submission, personalized recommendation generation, and WhatsApp delivery — while keeping infrastructure costs low for a bootstrapped EdTech startup?

---

## Architecture Design: Low-Cost, High-Scale Learning Agent

The core constraint is a **bootstrapped budget** — this rules out expensive real-time ML pipelines. The right mental model is *async-first, compute-lazy*: do the heavy work only when a student actually needs it, and batch everything you can.

### 1. Data Pipeline

After every quiz submission, a lightweight event (student_id, quiz_id, answers, timestamp) is pushed to a **message queue** — either AWS SQS free tier or a self-hosted RabbitMQ on a single $6/month VPS. This decouples the quiz submission HTTP response (instant) from all downstream processing (async). The queue absorbs bursts — 50 colleges finishing class simultaneously — without dropping events.

A **feature computation worker** (a simple Python process) consumes from the queue, updates a Redis hash per student with their rolling 4-week feature vector (weekly_accuracy, trend, gap, etc.), and writes the refreshed vector to a PostgreSQL table (Supabase free tier: 500MB, sufficient for 50K student records at ~500 bytes each). Redis acts as the low-latency read cache; Postgres is the durable store.

### 2. Model Serving Strategy

The classification model (XGBoost, ~1MB serialised) is **not deployed as a microservice** — that would require an always-on GPU instance. Instead, it is loaded into the feature worker process itself. State prediction is a microsecond CPU operation; no separate inference server is needed. For 50,000 students with ~5 quiz events per day, peak load is ~250 predictions/second, easily handled by 2–3 worker processes on a $20/month VPS.

Model retraining is scheduled **weekly via a cron job** on the same VPS. The retrained model is swapped in atomically via a file rename — no downtime, no Kubernetes, no ECS. This is the startup trade-off: slightly stale models vs. infrastructure complexity.

### 3. Recommendation Generation & Caching

Generating a personalized recommendation (rule-based + optional LLM message) is triggered only when the **student's state changes** or when they open their portal. Results are cached in Redis with a 24-hour TTL. This means the LLM API (Groq free tier: 6,000 requests/day) is called at most once per student per day rather than on every quiz event. For 50,000 students, 24-hour caching reduces LLM calls from potentially millions to a few thousand — staying within free-tier limits.

For the MCQ generation component, a pre-built MCQ bank per topic is indexed in Redis. The recommendation engine picks from this bank deterministically, so no inference is needed for MCQ delivery.

### 4. WhatsApp Delivery

WhatsApp messages (5 MCQs post-class, weekly report) are pushed via the **WhatsApp Business API** (Meta free tier covers 1,000 user-initiated conversations/month; for outbound scale, Twilio WhatsApp starts at $0.005/message). The delivery worker reads from a second queue (SQS FIFO), applies per-college rate limiting (Meta enforces message-per-second limits), and sends messages with retry logic. A $5/month Twilio plan handles ~1,000 messages/day comfortably for the early scale.

### 5. Cost Summary at 50K Students

| Component | Service | Monthly Cost |
|---|---|---|
| Queue | AWS SQS (1M requests free) | $0 |
| Feature store | Supabase (free tier) | $0 |
| Cache | Upstash Redis (10K req/day free) | $0 → $10 at scale |
| Compute (workers + model) | Hetzner VPS (2 vCPU, 4GB) | $6 |
| LLM (recommendations) | Groq free tier | $0 |
| WhatsApp delivery | Twilio | ~$15 |
| **Total** | | **~$31/month** |

### 6. Scaling Beyond This

When revenue allows, the next inflection points are: (a) **read replicas** on Postgres when the Supabase free tier is exceeded, (b) moving model serving to a **FastAPI + gunicorn** container on Render (free tier has cold starts, paid is $7/month), and (c) replacing the cron-trained model with **online learning** (River library) that updates per batch of events — eliminating weekly retraining lag. College-level sharding (one Redis namespace per college) enables horizontal partitioning without any architectural change.

The fundamental principle throughout is: **defer complexity until revenue justifies it**, but design interfaces (queue-based decoupling, separate read/write paths) that allow components to be upgraded independently without rewriting the system.
