"""
generate_dataset.py
-------------------
Generates a synthetic dataset of 500 students across 4 weeks of quiz activity.
Each student is first assigned a latent learning profile, which controls the
statistical distributions used to sample their features. This ensures a
realistic correlation structure (e.g., struggling students have both low
accuracy AND high response times).

Run standalone:
    python data/generate_dataset.py          → saves data/student_data.csv
"""

import numpy as np
import pandas as pd
import os


TOPICS = [
    "Python Basics", "Data Structures", "OOP", "SQL",
    "Statistics", "ML Fundamentals", "Deep Learning",
    "NLP", "Computer Vision", "System Design",
    "Algorithms", "Probability", "Linear Algebra", "APIs",
]

PROFILES = ["STRUGGLING", "PLATEAUING", "PROGRESSING", "MASTERED"]

# (base_acc, trend_range, resp_time_range, attend_range, n_topics_range, login_std_range, concept_gap_range)
PROFILE_PARAMS = {
    "STRUGGLING":  dict(base_acc=(0.20, 0.50), trend=(-0.08, 0.00), resp=(55, 90),
                        attend=(0.40, 0.65), topics=(2, 6),  std=(2.5, 4.0), gap=(0.55, 0.85)),
    "PLATEAUING":  dict(base_acc=(0.50, 0.70), trend=(-0.02, 0.02), resp=(35, 60),
                        attend=(0.60, 0.80), topics=(5, 9),  std=(1.5, 3.0), gap=(0.35, 0.55)),
    "PROGRESSING": dict(base_acc=(0.55, 0.82), trend=(0.02, 0.08),  resp=(25, 50),
                        attend=(0.70, 0.90), topics=(6, 11), std=(0.8, 2.0), gap=(0.20, 0.40)),
    "MASTERED":    dict(base_acc=(0.85, 0.99), trend=(0.00, 0.04),  resp=(10, 30),
                        attend=(0.85, 1.00), topics=(9, 14), std=(0.2, 1.2), gap=(0.05, 0.20)),
}


def generate_student_data(n_students: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per student.
    Columns include raw features, 3 engineered features, per-week accuracies,
    topic lists, and the ground-truth learning_state label.
    """
    np.random.seed(seed)

    # Balanced-ish class distribution
    weights = [0.25, 0.25, 0.30, 0.20]
    profiles = np.random.choice(PROFILES, size=n_students, p=weights)

    rows = []
    for i, profile in enumerate(profiles):
        p = PROFILE_PARAMS[profile]

        base_acc     = np.random.uniform(*p["base_acc"])
        trend        = np.random.uniform(*p["trend"])
        resp_time    = np.random.uniform(*p["resp"])   + np.random.normal(0, 3)
        attendance   = np.random.uniform(*p["attend"]) + np.random.normal(0, 0.02)
        n_topics     = np.random.randint(*p["topics"])
        login_std    = np.random.uniform(*p["std"])    + np.random.normal(0, 0.15)
        concept_gap  = np.random.uniform(*p["gap"])    + np.random.normal(0, 0.025)

        # Build week-by-week accuracy (random walk with profile-driven drift)
        weekly_acc = []
        acc = base_acc
        for _ in range(4):
            acc = np.clip(acc + trend + np.random.normal(0, 0.025), 0.05, 1.0)
            weekly_acc.append(round(float(acc), 4))

        # Topics and weak-topic list
        topic_list   = list(np.random.choice(TOPICS, size=n_topics, replace=False))
        n_weak       = max(1, int(n_topics * concept_gap))
        weak_topics  = list(np.random.choice(topic_list, size=min(n_weak, n_topics), replace=False))

        rows.append({
            # Identifiers
            "student_id":           f"STU{i + 1:04d}",
            "cgpa":                 round(np.random.uniform(5.5, 9.8), 2),

            # ── Raw features ────────────────────────────────────────────────
            "weekly_accuracy":      round(float(np.mean(weekly_acc)), 4),
            "week1_accuracy":       weekly_acc[0],
            "week2_accuracy":       weekly_acc[1],
            "week3_accuracy":       weekly_acc[2],
            "week4_accuracy":       weekly_acc[3],
            "avg_response_time_sec":round(float(np.clip(resp_time, 5, 120)), 2),
            "topics_attempted":     n_topics,
            "attendance_rate":      round(float(np.clip(attendance, 0.20, 1.0)), 4),

            # ── Engineered features ──────────────────────────────────────────
            # accuracy_trend: week-over-week linear slope (positive = improving)
            "accuracy_trend":       round(float(trend + np.random.normal(0, 0.008)), 4),
            # concept_gap_score: fraction of attempted topics where student is weak
            "concept_gap_score":    round(float(np.clip(concept_gap, 0.0, 1.0)), 4),
            # engagement_consistency: std of daily login frequency; lower = more consistent
            "engagement_consistency": round(float(np.clip(login_std, 0.1, 5.0)), 4),

            # Topic metadata
            "topics_list":          "|".join(topic_list),
            "weak_topics":          "|".join(weak_topics),

            # Ground-truth label
            "learning_state":       profile,
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_student_data(500)
    out = "data/student_data.csv"
    df.to_csv(out, index=False)
    print(f"✅  Saved {len(df)} rows → {out}")
    print(df["learning_state"].value_counts())
