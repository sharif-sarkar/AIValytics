# Feature Documentation

## Raw Features

| Feature | Type | Description |
|---|---|---|
| `weekly_accuracy` | Raw | Percentage of MCQs answered correctly across the 4-week window. The primary signal of knowledge retention. |
| `avg_response_time_sec` | Raw | Average seconds taken per MCQ response. Slow responders often indicate confusion or low confidence; very fast responders may indicate guessing. |
| `topics_attempted` | Raw | Count of distinct topics the student has been quizzed on. Breadth of exposure is a key indicator of engagement. |
| `attendance_rate` | Raw | Fraction of scheduled sessions attended. Lower attendance directly reduces learning opportunities and correlates with weaker states. |

## Engineered Features

### `accuracy_trend`
**Definition:** Week-over-week linear slope in accuracy (positive = improving, negative = declining).  
**Rationale:** A student with 60% accuracy who is trending *upward* (+0.05/week) is in a fundamentally different state than one trending *downward* (–0.05/week) at the same accuracy. This feature separates *PROGRESSING* from *PLATEAUING* and catches early deterioration before absolute accuracy falls below a threshold. Computed as the linear regression coefficient over the four weekly accuracy values.

### `concept_gap_score`
**Definition:** Ratio of weak topics to total topics attempted. Range [0, 1]. Higher = more gaps.  
**Rationale:** Two students may both attempt 8 topics, but if one is weak in 6 of them and the other in only 1, their intervention needs differ enormously. This feature captures *breadth of weakness relative to breadth of exposure*, which is more informative than either metric alone. Weak topics are defined as topics where the student's per-topic accuracy falls below 50%.

### `engagement_consistency`
**Definition:** Standard deviation of daily login frequency over the 4-week period. Lower = more consistent engagement.  
**Rationale:** Irregular study patterns (high std) predict spaced forgetting and disengagement even in students with decent accuracy. A student who logs in 5 days one week and 0 the next is at higher risk of state deterioration than one who logs in 2–3 days every week. This complements attendance_rate (which measures class presence) with *self-directed study behaviour*.

## Additional Feature Notes

- All features are kept on their natural scales. No normalisation is applied before tree-based models (Random Forest, XGBoost), which are scale-invariant.  
- For future linear/neural models, a `StandardScaler` pipeline should be added.  
- The `week1_accuracy` through `week4_accuracy` columns are stored for visualisation but excluded from model features to avoid data leakage into the trend signal.

## Labeling Logic

Ground-truth labels were assigned during synthetic data generation using the following rule system, designed to reflect the definitions in the assignment brief:

| State | Rule |
|---|---|
| STRUGGLING | base_acc ∈ [0.20, 0.50), trend ∈ [–0.08, 0.00), concept_gap ∈ [0.55, 0.85) |
| PLATEAUING | base_acc ∈ [0.50, 0.70), trend ∈ [–0.02, +0.02), concept_gap ∈ [0.35, 0.55) |
| PROGRESSING | base_acc ∈ [0.55, 0.82), trend ∈ [+0.02, +0.08), concept_gap ∈ [0.20, 0.40) |
| MASTERED | base_acc ∈ [0.85, 0.99), trend ∈ [0.00, +0.04), concept_gap ∈ [0.05, 0.20) |

This produces a *realistic correlation structure*: struggling students will have both lower accuracy AND higher response times AND higher concept gaps, matching what a real educator would observe. Gaussian noise (σ = 0.02–0.03) is added to each feature to prevent perfect separability and force the model to learn robust decision boundaries.
