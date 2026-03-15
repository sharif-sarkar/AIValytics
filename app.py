"""
app.py  —  AIValytics Personalized Learning Agent
==================================================
Two views:
  • Student View  — Individual learning state, weak topics, Next Action Card
  • Faculty View  — Class-level table, STRUGGLING alerts, aggregate trends
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── Make sibling imports work regardless of working directory ─────────────────
sys.path.insert(0, os.path.dirname(__file__))
from recommendation.engine import get_recommendation

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIValytics — Learning Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global fonts & background ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1200px; }

/* ── State badge ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: 24px;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.badge:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.1);
}
.badge-STRUGGLING  { background: linear-gradient(135deg, #FFF0F0 0%, #FDECEA 100%); color:#C0392B; border:1px solid rgba(231, 76, 60, 0.3); }
.badge-PLATEAUING  { background: linear-gradient(135deg, #FFFAF0 0%, #FEF9E7 100%); color:#B7770D; border:1px solid rgba(243, 156, 18, 0.3); }
.badge-PROGRESSING { background: linear-gradient(135deg, #F0FFF4 0%, #EAFAF1 100%); color:#1B8043; border:1px solid rgba(46, 204, 113, 0.3); }
.badge-MASTERED    { background: linear-gradient(135deg, #F0F8FF 0%, #EBF5FB 100%); color:#1A5276; border:1px solid rgba(52, 152, 219, 0.3); }

/* ── Next Action Card ── */
.action-card {
    background: linear-gradient(145deg, #1A1A2E 0%, #16213E 100%);
    border-radius: 16px;
    padding: 24px 28px;
    color: #ffffff;
    margin-top: 16px;
    box-shadow: 0 12px 32px rgba(22, 33, 62, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.05);
    transition: transform 0.3s ease;
}
.action-card:hover { text-shadow: none; transform: translateY(-3px); }
.action-card h4 { margin: 0 0 8px 0; font-size: 0.8rem; opacity: 0.7; letter-spacing: 1.5px; text-transform: uppercase; font-weight: 600; }
.action-card p  { margin: 0 0 16px 0; font-size: 0.95rem; line-height: 1.5; }
.action-card .topic-pill {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 6px 14px;
    margin: 4px 4px 8px 0;
    font-size: 0.85rem;
    font-weight: 500;
    backdrop-filter: blur(4px);
}
.action-card .message-box {
    background: rgba(255,255,255,0.06);
    border-left: 4px solid #3498DB;
    border-radius: 8px;
    padding: 14px 18px;
    font-style: italic;
    font-size: 0.95rem;
    margin-top: 12px;
    line-height: 1.5;
}

/* ── Alert row ── */
.alert-row { 
    background: linear-gradient(90deg, #FFF5F5 0%, #FFFFFF 100%); 
    border-left: 4px solid #E74C3C; 
    border-radius: 8px; 
    padding: 12px 16px; 
    margin-bottom: 8px;
    box-shadow: 0 2px 8px rgba(231,76,60,0.08);
    transition: transform 0.2s ease;
}
.alert-row:hover { transform: translateX(4px); }

/* ── Metric cards ── */
.metric-box {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(0,0,0,0.04);
    box-shadow: 0 4px 16px rgba(0,0,0,0.03);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-box:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
}
.metric-box .val { font-size: 2rem; font-weight: 700; background-clip: text; -webkit-background-clip: text; }
.metric-box .lbl { font-size: 0.8rem; color: #718096; margin-top: 6px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Typography Enhancements ── */
h2, h3, h4 { letter-spacing: -0.5px; }
hr { opacity: 0.4; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "weekly_accuracy", "avg_response_time_sec", "topics_attempted",
    "attendance_rate", "accuracy_trend", "concept_gap_score",
    "engagement_consistency",
]
STATE_ORDER  = ["STRUGGLING", "PLATEAUING", "PROGRESSING", "MASTERED"]
STATE_COLORS = {
    "STRUGGLING":  "#E74C3C",
    "PLATEAUING":  "#F39C12",
    "PROGRESSING": "#2ECC71",
    "MASTERED":    "#3498DB",
}
STATE_ICONS = {"STRUGGLING": "⚠️", "PLATEAUING": "⏸️", "PROGRESSING": "📈", "MASTERED": "🏆"}


# ─────────────────────────────────────────────────────────────────────────────
# Cached data + model loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating dataset…")
def load_data() -> pd.DataFrame:
    data_path = "data/student_data.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    from data.generate_dataset import generate_student_data
    df = generate_student_data(500)
    os.makedirs("data", exist_ok=True)
    df.to_csv(data_path, index=False)
    return df


@st.cache_resource(show_spinner="Training models (first run only)…")
def load_models():
    """Load persisted models if available, else train fresh."""
    rf_path  = "models/rf_model.pkl"
    xgb_path = "models/xgb_model.pkl"
    le_path  = "models/label_encoder.pkl"

    le = None; rf = None; xgb = None

    if os.path.exists(rf_path) and os.path.exists(le_path):
        with open(rf_path,  "rb") as f: rf = pickle.load(f)
        with open(le_path,  "rb") as f: le = pickle.load(f)
    if XGBOOST_AVAILABLE and os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f: xgb = pickle.load(f)

    if rf is not None and le is not None:
        return rf, xgb, le

    # Train inline
    from sklearn.model_selection import train_test_split
    df = load_data()
    le = LabelEncoder()
    le.fit(STATE_ORDER)
    y = le.transform(df["learning_state"])
    X = df[FEATURE_COLS].values
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    xgb = None
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                            eval_metric="mlogloss",
                            random_state=42, n_jobs=-1)
        xgb.fit(X_tr, y_tr)

    os.makedirs("models", exist_ok=True)
    with open(rf_path, "wb") as f: pickle.dump(rf, f)
    with open(le_path, "wb") as f: pickle.dump(le, f)
    if xgb:
        with open(xgb_path, "wb") as f: pickle.dump(xgb, f)

    return rf, xgb, le


def predict_state(features: list, model_choice: str, rf, xgb, le) -> tuple[str, np.ndarray]:
    if model_choice == "XGBoost" and xgb is not None:
        model = xgb
    else:
        model = rf
    X = np.array(features).reshape(1, -1)
    pred_idx  = model.predict(X)[0]
    pred_prob = model.predict_proba(X)[0]
    return le.inverse_transform([pred_idx])[0], pred_prob


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def state_badge(state: str) -> str:
    icon = STATE_ICONS.get(state, "")
    return f'<span class="badge badge-{state}">{icon} {state}</span>'


def render_action_card(card: dict) -> None:
    topics_html = " ".join(
        f'<span class="topic-pill">📌 {t}</span>' for t in card["top_weak_topics"]
    )
    mcq_lines = ""
    for i, q in enumerate(card["mcqs"], 1):
        opts = "  |  ".join(q["opts"])
        mcq_lines += f"<p style='margin:4px 0'><b>Q{i}:</b> {q['q']}<br><small style='opacity:0.7'>Options: {opts}</small><br><small style='color:#2ECC71'>✔ {q['ans']}</small></p>"

    st.markdown(f"""
    <div class="action-card">
      <h4>Next Action Card</h4>
      <p><b>🎯 Priority Topics to Revise</b></p>
      <p>{topics_html}</p>
      <p><b>📚 Recommended Format</b></p>
      <p>{card['content_format']}</p>
      <p><b>📝 Auto-Generated MCQs — {card['top_weak_topics'][0]}</b></p>
      {mcq_lines}
      <div class="message-box">💬 {card['message']}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# View A — Student View
# ─────────────────────────────────────────────────────────────────────────────
def student_view(df: pd.DataFrame, rf, xgb, le) -> None:
    st.markdown("## 🎓 Student Dashboard")
    st.caption("Enter a Student ID to look up their profile, or fill in a custom profile below.")

    col_left, col_right = st.columns([1, 1.8], gap="large")

    with col_left:
        st.markdown("### 🔍 Profile Input")
        input_mode = st.radio("Input mode", ["Lookup by Student ID", "Custom profile"], horizontal=True)

        if input_mode == "Lookup by Student ID":
            sid = st.selectbox("Student ID", df["student_id"].tolist())
            row = df[df["student_id"] == sid].iloc[0]
            weekly_acc = float(row["weekly_accuracy"])
            resp_time  = float(row["avg_response_time_sec"])
            topics     = int(row["topics_attempted"])
            attend     = float(row["attendance_rate"])
            trend      = float(row["accuracy_trend"])
            gap        = float(row["concept_gap_score"])
            eng_cons   = float(row["engagement_consistency"])
            weak_topics= row["weak_topics"].split("|") if pd.notna(row["weak_topics"]) else []
            all_topics = row["topics_list"].split("|") if pd.notna(row["topics_list"]) else []
            cgpa       = float(row["cgpa"])
        else:
            weekly_acc  = st.slider("Weekly Accuracy (%)",     0,   100,  65)  / 100
            resp_time   = st.slider("Avg Response Time (sec)", 10,  120,  40)
            topics      = st.slider("Topics Attempted",        1,   14,   7)
            attend      = st.slider("Attendance Rate (%)",     20,  100,  80)  / 100
            trend       = st.slider("Accuracy Trend (delta)",  -0.10, 0.10, 0.02, 0.01)
            gap         = st.slider("Concept Gap Score",       0.0, 1.0, 0.35, 0.05)
            eng_cons    = st.slider("Engagement Consistency (std)", 0.1, 5.0, 1.5, 0.1)
            weak_topics = ["Python Basics", "Statistics"]
            all_topics  = weak_topics + ["ML Fundamentals", "SQL"]

        model_options = ["Random Forest", "XGBoost"] if XGBOOST_AVAILABLE else ["Random Forest"]
        model_choice = st.selectbox("Classification Model", model_options)
        st.markdown("---")

    with col_right:
        features = [weekly_acc, resp_time, topics, attend, trend, gap, eng_cons]
        predicted_state, probs = predict_state(features, model_choice, rf, xgb, le)
        profile = {
            "avg_response_time_sec": resp_time,
            "engagement_consistency": eng_cons,
            "attendance_rate": attend,
        }
        card = get_recommendation(predicted_state, weak_topics, profile)

        # ── State badge ──
        st.markdown(f"### Learning State &nbsp; {state_badge(predicted_state)}",
                    unsafe_allow_html=True)

        # ── Confidence bar ──
        prob_df = pd.DataFrame({"State": le.classes_, "Confidence": probs})
        fig_prob = px.bar(prob_df, x="Confidence", y="State", orientation="h",
                          color="State",
                          color_discrete_map={s: STATE_COLORS[s] for s in STATE_ORDER},
                          title="Model Confidence", height=200)
        fig_prob.update_layout(showlegend=False, margin=dict(l=0, r=0, t=35, b=0),
                               xaxis_range=[0, 1], yaxis={"categoryorder": "array",
                               "categoryarray": STATE_ORDER})
        fig_prob.update_traces(texttemplate="%{x:.0%}", textposition="outside")
        st.plotly_chart(fig_prob, use_container_width=True)

        # ── Weak / strong topics ──
        st.markdown("#### 📊 Topic Breakdown")
        strong_topics = [t for t in all_topics if t not in weak_topics]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🔴 Weak Topics**")
            for t in weak_topics[:5]:
                st.markdown(f"- {t}")
        with c2:
            st.markdown("**🟢 Strong Topics**")
            for t in (strong_topics or ["—"])[:5]:
                st.markdown(f"- {t}")

        # ── Accuracy trend chart ──
        if input_mode == "Lookup by Student ID":
            weeks = [row[f"week{w}_accuracy"] for w in range(1, 5)]
            fig_trend = go.Figure()
            def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
                h = hex_color.lstrip("#")
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                return f"rgba({r},{g},{b},{alpha})"

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=["Week 1", "Week 2", "Week 3", "Week 4"], y=[w * 100 for w in weeks],
                mode="lines+markers+text",
                line=dict(color=STATE_COLORS[predicted_state], width=3),
                marker=dict(size=9),
                text=[f"{w*100:.0f}%" for w in weeks], textposition="top center",
                fill="tozeroy", fillcolor=hex_to_rgba(STATE_COLORS[predicted_state]),
            ))
            fig_trend.update_layout(
                title="Accuracy Trend (4 Weeks)", yaxis_range=[0, 105],
                yaxis_title="Accuracy (%)", height=220,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        render_action_card(card)


# ─────────────────────────────────────────────────────────────────────────────
# View B — Faculty / Admin View
# ─────────────────────────────────────────────────────────────────────────────
def faculty_view(df: pd.DataFrame, rf, xgb, le) -> None:
    st.markdown("## 👩‍🏫 Faculty / Admin Dashboard")

    # Predict all students in batch
    X_all = df[FEATURE_COLS].values
    preds = le.inverse_transform(rf.predict(X_all))
    df = df.copy()
    df["predicted_state"] = preds

    # ── Summary metrics ──────────────────────────────────────────────────────
    st.markdown("### 📊 Class Summary")
    total = len(df)
    counts = df["predicted_state"].value_counts()

    m_cols = st.columns(5)
    metrics = [
        ("Total Students", str(total), "#1A252F"),
        ("⚠️ Struggling",  str(counts.get("STRUGGLING",  0)), "#E74C3C"),
        ("⏸️ Plateauing",  str(counts.get("PLATEAUING",  0)), "#F39C12"),
        ("📈 Progressing",  str(counts.get("PROGRESSING", 0)), "#2ECC71"),
        ("🏆 Mastered",     str(counts.get("MASTERED",    0)), "#3498DB"),
    ]
    for col, (label, val, color) in zip(m_cols, metrics):
        col.markdown(
            f'<div class="metric-box"><div class="val" style="color:{color}">{val}</div>'
            f'<div class="lbl">{label}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        # ── State distribution donut ──────────────────────────────────────────
        pie_df = df["predicted_state"].value_counts().reset_index()
        pie_df.columns = ["State", "Count"]
        fig_pie = px.pie(pie_df, names="State", values="Count",
                         color="State",
                         color_discrete_map=STATE_COLORS,
                         hole=0.5, title="Learning State Distribution")
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, height=320, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # ── Class accuracy trend (avg per week) ──────────────────────────────
        week_avgs = [df[f"week{w}_accuracy"].mean() * 100 for w in range(1, 5)]
        fig_cls = go.Figure()
        fig_cls.add_trace(go.Scatter(
            x=["Week 1", "Week 2", "Week 3", "Week 4"], y=week_avgs,
            mode="lines+markers+text",
            line=dict(color="#3498DB", width=3),
            marker=dict(size=9),
            text=[f"{v:.1f}%" for v in week_avgs], textposition="top center",
            fill="tozeroy", fillcolor="rgba(52,152,219,0.12)",
        ))
        fig_cls.update_layout(title="Class Average Accuracy Trend",
                              yaxis_title="Accuracy (%)", yaxis_range=[0, 100],
                              height=320, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_cls, use_container_width=True)

    # ── Struggling alerts ────────────────────────────────────────────────────
    struggling = df[df["predicted_state"] == "STRUGGLING"].sort_values("weekly_accuracy")

    if len(struggling):
        st.markdown(f"### 🚨 Students Requiring Immediate Attention ({len(struggling)})")
        for _, row in struggling.head(10).iterrows():
            weak = row["weak_topics"].split("|")[:2] if pd.notna(row["weak_topics"]) else ["N/A"]
            st.markdown(
                f'<div class="alert-row">⚠️ <b>{row["student_id"]}</b> — '
                f'Accuracy: <b>{row["weekly_accuracy"]*100:.0f}%</b> | '
                f'Attendance: <b>{row["attendance_rate"]*100:.0f}%</b> | '
                f'Weak: <b>{", ".join(weak)}</b></div>',
                unsafe_allow_html=True
            )
        if len(struggling) > 10:
            st.caption(f"… and {len(struggling) - 10} more. Download full table below.")

    st.markdown("---")

    # ── Full student table ────────────────────────────────────────────────────
    st.markdown("### 📋 Full Student Table")
    filter_state = st.multiselect(
        "Filter by State", STATE_ORDER, default=STATE_ORDER, key="filter"
    )
    display_cols = ["student_id", "cgpa", "weekly_accuracy", "avg_response_time_sec",
                    "attendance_rate", "concept_gap_score", "predicted_state"]
    filtered = df[df["predicted_state"].isin(filter_state)][display_cols].copy()
    filtered["weekly_accuracy"] = (filtered["weekly_accuracy"] * 100).round(1).astype(str) + "%"
    filtered["attendance_rate"] = (filtered["attendance_rate"] * 100).round(1).astype(str) + "%"
    filtered["concept_gap_score"] = filtered["concept_gap_score"].round(3)
    filtered["avg_response_time_sec"] = filtered["avg_response_time_sec"].round(1)

    def colour_state(val):
        c = {"STRUGGLING": "#FDECEA", "PLATEAUING": "#FEF9E7",
             "PROGRESSING": "#EAFAF1", "MASTERED": "#EBF5FB"}.get(val, "")
        return f"background-color: {c}; font-weight: 600"

    st.dataframe(
        filtered.style.map(colour_state, subset=["predicted_state"]),
        use_container_width=True, height=420
    )
    csv = filtered.to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, "student_states.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar & routing
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    df     = load_data()
    rf, xgb, le = load_models()

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
        st.title("AIValytics")
        st.caption("Personalized Learning Agent")
        st.markdown("---")
        view = st.radio("Navigation", ["🎓 Student View", "👩‍🏫 Faculty / Admin View"])
        st.markdown("---")
        st.markdown(f"**Dataset:** {len(df)} students")
        st.markdown(f"**Model:** Random Forest + XGBoost")
        st.caption("Built by Sharif Hossain Sarkar | IIT Kharagpur")

    if view == "🎓 Student View":
        student_view(df, rf, xgb, le)
    else:
        faculty_view(df, rf, xgb, le)


if __name__ == "__main__":
    main()
