"""
recommendation/engine.py
------------------------
Generates a structured Next Action Card for a given learning state and
student profile.

Design choice: Rule-based core + optional LLM enhancement (Groq/OpenAI).
Rationale documented in feature_docs.md.

Usage:
    from recommendation.engine import get_recommendation
    card = get_recommendation(state="STRUGGLING", weak_topics=["SQL","OOP"],
                              profile={"avg_response_time_sec": 72, ...})
"""

import os
import random
from typing import Optional

# ── MCQ bank (topic → 3 MCQs) ────────────────────────────────────────────────
MCQ_BANK: dict[str, list[dict]] = {
    "Python Basics": [
        {"q": "Which of the following is a mutable data type in Python?",
         "opts": ["tuple", "str", "list", "frozenset"], "ans": "list"},
        {"q": "What does `//` operator do in Python?",
         "opts": ["Float division", "Floor division", "Modulo", "Power"], "ans": "Floor division"},
        {"q": "What is the output of `bool([])` in Python?",
         "opts": ["True", "False", "None", "Error"], "ans": "False"},
    ],
    "Data Structures": [
        {"q": "What is the time complexity of searching in a Hash Table on average?",
         "opts": ["O(n)", "O(log n)", "O(1)", "O(n²)"], "ans": "O(1)"},
        {"q": "Which data structure follows LIFO?",
         "opts": ["Queue", "Stack", "Deque", "Heap"], "ans": "Stack"},
        {"q": "A complete binary tree with n nodes has height:",
         "opts": ["O(n)", "O(log n)", "O(n log n)", "O(1)"], "ans": "O(log n)"},
    ],
    "SQL": [
        {"q": "Which SQL clause is used to filter grouped results?",
         "opts": ["WHERE", "HAVING", "GROUP BY", "ORDER BY"], "ans": "HAVING"},
        {"q": "What type of JOIN returns all rows from both tables?",
         "opts": ["INNER JOIN", "LEFT JOIN", "FULL OUTER JOIN", "CROSS JOIN"], "ans": "FULL OUTER JOIN"},
        {"q": "Which keyword removes duplicate rows from a SELECT result?",
         "opts": ["UNIQUE", "DISTINCT", "NODUP", "FILTER"], "ans": "DISTINCT"},
    ],
    "Statistics": [
        {"q": "The median is resistant to outliers because:",
         "opts": ["It uses all data points", "It is based on rank, not value",
                  "It equals the mean", "It uses standard deviation"], "ans": "It is based on rank, not value"},
        {"q": "A p-value < 0.05 means:",
         "opts": ["H₀ is true", "H₁ is false",
                  "Result is statistically significant", "Effect size is large"], "ans": "Result is statistically significant"},
        {"q": "Standard deviation is the ___ of variance.",
         "opts": ["square", "square root", "log", "reciprocal"], "ans": "square root"},
    ],
    "ML Fundamentals": [
        {"q": "Overfitting is best addressed by:",
         "opts": ["Adding more features", "Reducing regularisation",
                  "Adding more training data / regularisation", "Using a simpler loss function"], "ans": "Adding more training data / regularisation"},
        {"q": "Which metric is most appropriate for imbalanced classification?",
         "opts": ["Accuracy", "F1 Score", "MSE", "R²"], "ans": "F1 Score"},
        {"q": "Cross-validation is primarily used to:",
         "opts": ["Speed up training", "Estimate generalisation performance",
                  "Increase model complexity", "Reduce training data size"], "ans": "Estimate generalisation performance"},
    ],
    "Algorithms": [
        {"q": "Which sorting algorithm has worst-case O(n log n)?",
         "opts": ["Bubble Sort", "Insertion Sort", "Merge Sort", "Quick Sort"], "ans": "Merge Sort"},
        {"q": "Dijkstra's algorithm finds:",
         "opts": ["Minimum spanning tree", "Shortest path",
                  "Maximum flow", "Topological order"], "ans": "Shortest path"},
        {"q": "Dynamic Programming is characterised by:",
         "opts": ["Greedy choices", "Divide and conquer without overlapping sub-problems",
                  "Overlapping sub-problems + optimal substructure", "Random sampling"], "ans": "Overlapping sub-problems + optimal substructure"},
    ],
    "OOP": [
        {"q": "Which OOP principle hides internal implementation details?",
         "opts": ["Inheritance", "Polymorphism", "Encapsulation", "Abstraction"], "ans": "Encapsulation"},
        {"q": "Method overriding is an example of:",
         "opts": ["Compile-time polymorphism", "Runtime polymorphism",
                  "Encapsulation", "Association"], "ans": "Runtime polymorphism"},
        {"q": "A class that cannot be instantiated is called:",
         "opts": ["Static class", "Abstract class", "Interface", "Singleton"], "ans": "Abstract class"},
    ],
    "Deep Learning": [
        {"q": "What does the ReLU activation function output for negative inputs?",
         "opts": ["The input value", "1", "0", "-1"], "ans": "0"},
        {"q": "Batch Normalisation is applied to:",
         "opts": ["Only the output layer", "Activations to stabilise training",
                  "Only the input layer", "Loss function"], "ans": "Activations to stabilise training"},
        {"q": "Dropout regularisation works by:",
         "opts": ["Removing features", "Randomly zeroing neurons during training",
                  "Reducing learning rate", "Adding noise to labels"], "ans": "Randomly zeroing neurons during training"},
    ],
    "NLP": [
        {"q": "TF-IDF penalises terms that appear:",
         "opts": ["Rarely in a document", "Frequently across many documents",
                  "Only in short documents", "In the title"], "ans": "Frequently across many documents"},
        {"q": "In transformer models, attention is computed using:",
         "opts": ["RNNs", "CNNs", "Query, Key, Value matrices", "Bayesian inference"], "ans": "Query, Key, Value matrices"},
        {"q": "Word embeddings capture:",
         "opts": ["Exact spelling", "Semantic similarity between words",
                  "Grammatical rules", "Sentence length"], "ans": "Semantic similarity between words"},
    ],
    "Computer Vision": [
        {"q": "A convolutional layer's primary purpose is to:",
         "opts": ["Flatten inputs", "Detect local spatial features",
                  "Classify outputs", "Normalise pixel values"], "ans": "Detect local spatial features"},
        {"q": "Max pooling reduces spatial dimensions by:",
         "opts": ["Averaging values", "Taking the maximum in a window",
                  "Padding zeros", "Transposing the matrix"], "ans": "Taking the maximum in a window"},
        {"q": "Transfer learning in CV typically fine-tunes:",
         "opts": ["Only the input layer", "Only the classification head / top layers",
                  "The entire network from scratch", "The loss function"], "ans": "Only the classification head / top layers"},
    ],
    "Linear Algebra": [
        {"q": "The rank of a matrix is:",
         "opts": ["Number of rows", "Number of non-zero eigenvalues",
                  "Dimension of the column space", "Number of columns"], "ans": "Dimension of the column space"},
        {"q": "PCA finds directions of:",
         "opts": ["Minimum variance", "Maximum variance",
                  "Zero variance", "Mean-centred variance"], "ans": "Maximum variance"},
        {"q": "A matrix is invertible if and only if its determinant is:",
         "opts": ["Zero", "One", "Non-zero", "Greater than 1"], "ans": "Non-zero"},
    ],
    "Probability": [
        {"q": "Bayes' theorem relates:",
         "opts": ["Prior and likelihood to posterior",
                  "Mean and variance", "PDF and CDF", "Sample and population"], "ans": "Prior and likelihood to posterior"},
        {"q": "For independent events A and B, P(A ∩ B) =",
         "opts": ["P(A) + P(B)", "P(A) × P(B)", "P(A) − P(B)", "P(A) / P(B)"], "ans": "P(A) × P(B)"},
        {"q": "The Central Limit Theorem states that the sample mean is approximately:",
         "opts": ["Uniformly distributed", "Exponentially distributed",
                  "Normally distributed for large n", "Chi-squared distributed"], "ans": "Normally distributed for large n"},
    ],
    "System Design": [
        {"q": "Horizontal scaling means:",
         "opts": ["Adding more CPU/RAM to one server",
                  "Adding more servers to a pool",
                  "Increasing database storage",
                  "Reducing latency via caching"], "ans": "Adding more servers to a pool"},
        {"q": "A message queue decouples systems by:",
         "opts": ["Synchronising requests", "Buffering messages asynchronously",
                  "Reducing database load", "Encrypting traffic"], "ans": "Buffering messages asynchronously"},
        {"q": "CAP theorem states a distributed system can guarantee at most:",
         "opts": ["3 of 3 properties", "2 of 3 properties (CA, CP, or AP)",
                  "1 of 3 properties", "Depends on network speed"], "ans": "2 of 3 properties (CA, CP, or AP)"},
    ],
    "APIs": [
        {"q": "REST APIs are stateless because:",
         "opts": ["They use TCP", "Each request contains all required information",
                  "They always use JSON", "They require authentication"], "ans": "Each request contains all required information"},
        {"q": "HTTP 401 means:",
         "opts": ["Not Found", "Server Error", "Unauthorised", "Bad Request"], "ans": "Unauthorised"},
        {"q": "GraphQL differs from REST in that:",
         "opts": ["It uses XML", "Clients specify exact data shape in the query",
                  "It is always faster", "It doesn't use HTTP"], "ans": "Clients specify exact data shape in the query"},
    ],
}

# Fallback generic MCQs if topic not in bank
_FALLBACK_MCQS = [
    {"q": "Which principle is most important for maintainable code?",
     "opts": ["Speed", "Single Responsibility", "File size", "Comments only"], "ans": "Single Responsibility"},
    {"q": "Version control systems like Git primarily help with:",
     "opts": ["Compiling code", "Tracking changes and collaboration",
              "Running tests", "Deploying servers"], "ans": "Tracking changes and collaboration"},
    {"q": "Big-O notation describes:",
     "opts": ["Exact runtime", "Algorithm complexity as input grows",
              "Memory usage only", "Number of lines of code"], "ans": "Algorithm complexity as input grows"},
]

# ── Messages per state ────────────────────────────────────────────────────────
MESSAGES: dict[str, list[str]] = {
    "STRUGGLING": [
        "You're facing real challenges right now, but every expert was once a beginner. "
        "Let's break this into small, achievable goals — one topic at a time.",
        "Struggling now means you're at the edge of your learning zone. "
        "That's exactly where growth happens. Your faculty is here to help.",
        "Don't give up. Review the fundamentals on your two weakest topics today "
        "and retake the quiz. Consistency beats intensity.",
    ],
    "PLATEAUING": [
        "You've built a solid foundation. Now it's time to challenge yourself "
        "with harder problems to break through the plateau.",
        "Stability is good, but growth is better. Try tackling a new topic "
        "or a timed quiz to rekindle momentum.",
        "Your knowledge base is strong. Introducing spaced repetition on your "
        "weaker topics will push your accuracy to the next level.",
    ],
    "PROGRESSING": [
        "Excellent momentum! You're on the right track — keep the consistency "
        "and push to master the topics you've started.",
        "Your week-over-week improvement is impressive. Keep it up and you'll "
        "reach Mastered status soon!",
        "Great work! Challenge yourself with advanced questions on your stronger "
        "topics while reinforcing the weaker ones.",
    ],
    "MASTERED": [
        "Outstanding performance! You're ready for advanced challenges — "
        "consider peer tutoring or competitive problem-solving.",
        "You've mastered this material. Use your edge to explore adjacent topics "
        "and deepen your specialisation.",
        "Top of the class! Start exploring project-based learning or "
        "industry case studies to put your knowledge to work.",
    ],
}

# ── Content format rules ──────────────────────────────────────────────────────
def _suggest_content_format(avg_response_time_sec: float,
                             engagement_consistency: float) -> str:
    """
    Heuristic: slow responders → prefer structured notes (reduces cognitive load).
    Inconsistent engagement → short video clips (easier to fit in irregular schedules).
    Fast, consistent learners → direct quizzes / practice problems.
    """
    if avg_response_time_sec > 55:
        return "📄 Detailed Notes + Worked Examples"
    elif engagement_consistency > 2.5:
        return "🎬 Short Video Clips (5–10 min each)"
    elif avg_response_time_sec < 30:
        return "📝 Timed Practice Quizzes"
    else:
        return "🎥 Video + Follow-up Quiz (blended)"


def _get_mcqs(topic: str) -> list[dict]:
    return MCQ_BANK.get(topic, _FALLBACK_MCQS)


# ── Public API ────────────────────────────────────────────────────────────────
def get_recommendation(
    state: str,
    weak_topics: list[str],
    profile: dict,
    use_llm: bool = False,
    llm_api_key: Optional[str] = None,
) -> dict:
    """
    Returns a Next Action Card dict with keys:
        state, top_weak_topics, content_format, mcqs, message
    """
    if not weak_topics:
        weak_topics = ["Python Basics", "Statistics"]

    top2 = weak_topics[:2]
    primary_topic = top2[0]

    content_fmt = _suggest_content_format(
        profile.get("avg_response_time_sec", 45),
        profile.get("engagement_consistency", 1.5),
    )

    # Message: optionally enhanced by LLM, fallback to rule-based
    message = _rule_based_message(state)
    if use_llm and llm_api_key:
        try:
            message = _llm_message(state, top2, profile, llm_api_key) or message
        except Exception:
            pass  # silent fallback to rule-based

    return {
        "state":            state,
        "top_weak_topics":  top2,
        "content_format":   content_fmt,
        "mcqs":             _get_mcqs(primary_topic),
        "message":          message,
    }


def _rule_based_message(state: str) -> str:
    pool = MESSAGES.get(state, MESSAGES["PROGRESSING"])
    return random.choice(pool)


def _llm_message(state: str, topics: list[str], profile: dict,
                 api_key: str) -> Optional[str]:
    """
    Calls Groq (llama-3-8b-instant, free tier) to generate a 2-sentence
    personalised message.  Falls back silently if the call fails.
    """
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = (
            f"A student is in the '{state}' learning state. "
            f"Their two weakest topics are: {', '.join(topics)}. "
            f"Attendance rate: {profile.get('attendance_rate', 0.7):.0%}. "
            "Write a 2-sentence motivational/corrective message tailored to their situation. "
            "Be specific, empathetic, and actionable. No more than 60 words."
        )
        resp = client.chat.completions.create(
            model="llama-3-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None
