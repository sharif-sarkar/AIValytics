# 🎓 AIValytics — Personalized Learning Agent

**Candidate:** Sharif Hossain Sarkar | IIT Kharagpur (B.Tech + M.Tech Dual Degree)  
**Role:** AI/ML Intern — Data & Learning Intelligence | AIValytics

---

## Project Overview

A Personalized Learning Agent that classifies each student into one of four learning states — **STRUGGLING**, **PLATEAUING**, **PROGRESSING**, or **MASTERED** — and generates structured, actionable Next Action Cards. Built as a two-view Streamlit dashboard.

```
personalized-learning-agent/
├── app.py                       ← Streamlit dashboard (Student + Faculty views)
├── train_model.py               ← EDA + model training (standalone)
├── data/
│   ├── generate_dataset.py      ← Synthetic dataset generator
│   └── student_data.csv         ← Auto-generated on first run
├── models/
│   ├── rf_model.pkl             ← Trained Random Forest
│   ├── xgb_model.pkl            ← Trained XGBoost
│   ├── label_encoder.pkl        ← Sklearn LabelEncoder
│   └── plots/                   ← EDA + confusion matrix plots
├── recommendation/
│   └── engine.py                ← Rule-based + optional LLM recommendation engine
├── feature_docs.md              ← Engineered feature documentation
├── scalability_writeup.md       ← 500-word architectural write-up
└── requirements.txt
```

---

## How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Pre-generate data and train models
```bash
python data/generate_dataset.py   # saves data/student_data.csv
python train_model.py             # saves models/*.pkl and EDA plots
```
*If you skip this step, `app.py` will do it automatically on first launch.*

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## Dataset Generation Logic

`data/generate_dataset.py` generates 500 synthetic student records.

Each student is first assigned a **latent learning profile** (STRUGGLING / PLATEAUING / PROGRESSING / MASTERED) drawn from a realistic class distribution (25 / 25 / 30 / 20 %). Profile-specific parameter ranges then control the statistical distributions for each feature:

| Feature | STRUGGLING | PLATEAUING | PROGRESSING | MASTERED |
|---|---|---|---|---|
| base accuracy | 20–50% | 50–70% | 55–82% | 85–99% |
| accuracy trend | –0.08–0 | –0.02–+0.02 | +0.02–+0.08 | 0–+0.04 |
| response time | 55–90 s | 35–60 s | 25–50 s | 10–30 s |
| concept gap | 55–85% | 35–55% | 20–40% | 5–20% |

Gaussian noise (σ ≈ 0.02) is added to each feature to prevent perfect linear separability, ensuring the model learns generalised decision boundaries rather than memorising thresholds.

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Random Forest, LabelEncoder, train/test split, metrics |
| `xgboost` | XGBoost classifier |
| `streamlit` | Dashboard framework |
| `plotly` | Interactive charts |
| `matplotlib`, `seaborn` | EDA plots (train_model.py) |

---

## Feature Documentation

See [`feature_docs.md`](feature_docs.md) for detailed rationale on all three engineered features (`accuracy_trend`, `concept_gap_score`, `engagement_consistency`).

---

## Scalability Write-up

See [`scalability_writeup.md`](scalability_writeup.md) for the full architectural response to scaling this system to 50,000 concurrent student profiles.

---

## Recommendation Engine Design Choice

The recommendation engine uses a **rule-based core with optional LLM enhancement**:

- **Why not pure LLM?** API costs, latency, and rate limits make full LLM generation impractical at 50K+ students. Rules are deterministic, testable, and free.
- **Why not pure rule-based?** Rule-based messages are generic. An LLM (Groq free tier) can generate a personalised 2-sentence message given the student's specific context.
- **Hybrid design:** Rules handle content format selection (deterministic, reproducible) and MCQ retrieval (from a verified bank). LLM is called optionally and only for the motivational message, with silent fallback to rule-based output if the API is unavailable.

To enable LLM enhancement, set the `GROQ_API_KEY` environment variable (free at console.groq.com) and pass `use_llm=True` to `get_recommendation()`.
