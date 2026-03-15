"""
train_model.py
--------------
Trains and compares Random Forest vs XGBoost classifiers on the synthetic
student dataset.  Also performs EDA with 4 visualisations saved to
models/plots/.

Run standalone:
    python train_model.py    → saves models/rf_model.pkl, models/xgb_model.pkl
                               + models/label_encoder.pkl, models/plots/*.png
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import (classification_report, confusion_matrix,
                                        ConfusionMatrixDisplay)
from sklearn.preprocessing     import LabelEncoder
from xgboost                   import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/student_data.csv"
MODEL_DIR  = "models"
PLOT_DIR   = os.path.join(MODEL_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

FEATURE_COLS = [
    "weekly_accuracy", "avg_response_time_sec", "topics_attempted",
    "attendance_rate", "accuracy_trend", "concept_gap_score",
    "engagement_consistency",
]
TARGET_COL = "learning_state"
STATE_ORDER = ["STRUGGLING", "PLATEAUING", "PROGRESSING", "MASTERED"]
PALETTE     = {
    "STRUGGLING":  "#E74C3C",
    "PLATEAUING":  "#F39C12",
    "PROGRESSING": "#2ECC71",
    "MASTERED":    "#3498DB",
}


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Data loading / generation
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print("Dataset not found — generating …")
        sys.path.insert(0, "data")
        from generate_dataset import generate_student_data
        df = generate_student_data(500)
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  EDA — four meaningful visualisations
# ─────────────────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    # ── Plot 1: Class distribution ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df[TARGET_COL].value_counts().reindex(STATE_ORDER)
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[s] for s in counts.index], edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Learning State Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Learning State"); ax.set_ylabel("Number of Students")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "01_class_distribution.png"), dpi=150)
    plt.close()
    print("  [OK]  Plot 1 saved: class distribution")

    # ── Plot 2: Accuracy vs Response Time (scatter) ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for state in STATE_ORDER:
        sub = df[df[TARGET_COL] == state]
        ax.scatter(sub["avg_response_time_sec"], sub["weekly_accuracy"],
                   label=state, color=PALETTE[state], alpha=0.55, s=25, edgecolors="none")
    ax.set_xlabel("Avg Response Time (sec)")
    ax.set_ylabel("Weekly Accuracy")
    ax.set_title("Accuracy vs Response Time by Learning State", fontweight="bold")
    ax.legend(title="State", framealpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "02_accuracy_vs_response_time.png"), dpi=150)
    plt.close()
    print("  [OK]  Plot 2 saved: accuracy vs response time")

    # ── Plot 3: Feature box-plots per state ──────────────────────────────────
    plot_features = ["weekly_accuracy", "concept_gap_score",
                     "attendance_rate", "engagement_consistency"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()
    for ax, feat in zip(axes, plot_features):
        data_by_state = [df[df[TARGET_COL] == s][feat].values for s in STATE_ORDER]
        bp = ax.boxplot(data_by_state, patch_artist=True, notch=False,
                        medianprops=dict(color="white", linewidth=2))
        for patch, state in zip(bp["boxes"], STATE_ORDER):
            patch.set_facecolor(PALETTE[state])
            patch.set_alpha(0.75)
        ax.set_xticklabels(STATE_ORDER, rotation=12, fontsize=9)
        ax.set_title(feat.replace("_", " ").title(), fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Feature Distributions Across Learning States", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "03_feature_boxplots.png"), dpi=150)
    plt.close()
    print("  [OK]  Plot 3 saved: feature box-plots")

    # ── Plot 4: Correlation heat-map of engineered features ──────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_df = df[FEATURE_COLS].corr()
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "04_correlation_heatmap.png"), dpi=150)
    plt.close()
    print("  [OK]  Plot 4 saved: correlation heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Model training & evaluation
# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame):
    le = LabelEncoder()
    le.fit(STATE_ORDER)                    # deterministic class order
    y = le.transform(df[TARGET_COL])
    X = df[FEATURE_COLS].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Random Forest ────────────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_cv  = cross_val_score(rf, X, y, cv=5, scoring="f1_macro").mean()
    rf_rep = classification_report(y_test, rf.predict(X_test),
                                   target_names=le.classes_, output_dict=True)

    # ── XGBoost ─────────────────────────────────────────────────────────────
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                        eval_metric="mlogloss",
                        random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_cv  = cross_val_score(xgb, X, y, cv=5, scoring="f1_macro").mean()
    xgb_rep = classification_report(y_test, xgb.predict(X_test),
                                    target_names=le.classes_, output_dict=True)

    # ── Print comparison -----------------------------------------------------
    print("\n" + "=" * 55)
    print(f"  Random Forest  - CV macro-F1: {rf_cv:.4f}")
    print(f"  XGBoost        - CV macro-F1: {xgb_cv:.4f}")
    print("=" * 55)
    print("\n-- Random Forest classification report --")
    print(classification_report(y_test, rf.predict(X_test), target_names=le.classes_))
    print("\n-- XGBoost classification report --")
    print(classification_report(y_test, xgb.predict(X_test), target_names=le.classes_))

    # ── Confusion matrices ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model, name in [(axes[0], rf, "Random Forest"), (axes[1], xgb, "XGBoost")]:
        cm = confusion_matrix(y_test, model.predict(X_test))
        disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nCV macro-F1: {rf_cv if name == 'Random Forest' else xgb_cv:.3f}",
                     fontweight="bold")
        ax.set_xticklabels(le.classes_, rotation=25, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "05_confusion_matrices.png"), dpi=150)
    plt.close()
    print("  [OK]  Plot 5 saved: confusion matrices")

    # ── Feature importance (RF) ──────────────────────────────────────────────
    fi = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    fi.plot(kind="barh", ax=ax, color="#3498DB", edgecolor="white")
    ax.set_title("Random Forest Feature Importances", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "06_feature_importance.png"), dpi=150)
    plt.close()
    print("  [OK]  Plot 6 saved: feature importance")

    return rf, xgb, le


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Persist artifacts
# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(rf, xgb, le) -> None:
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"),  "wb") as f: pickle.dump(rf,  f)
    with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "wb") as f: pickle.dump(xgb, f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f: pickle.dump(le, f)
    print(f"\n[DONE] Models saved -> {MODEL_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("-- Loading data --")
    df = load_data()
    print(f"   {len(df)} rows  |  {df[TARGET_COL].value_counts().to_dict()}\n")

    print("-- Running EDA --")
    run_eda(df)

    print("\n-- Training models --")
    rf, xgb, le = train_and_evaluate(df)

    save_artifacts(rf, xgb, le)
