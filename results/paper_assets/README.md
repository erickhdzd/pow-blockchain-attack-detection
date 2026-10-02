# Paper assets

This directory was generated automatically from the frozen full-training outputs.
No model was retrained and no metric was entered manually.

## Validated experiment

- Runs: 100
- Tasks: general, early
- Models: 10
- Seeds: 42, 123, 2026, 2027, 2028
- Train endpoints: 9540
- Validation endpoints: 1980
- Test endpoints: 1980
- Evaluation sample SHA-256: `63809aac07dc7a634d227508dd96f12fdf2e1c373c971794c8b267e43b845ead`
- General Macro-F1 winner: GRU
- Early Macro-F1 winner: CNN-LSTM

## Statistical interpretation

Wilcoxon signed-rank tests are paired by seed and Holm-adjusted within each comparison family.
Because there are only five seeds, these tests are exploratory and should not be treated as strong inferential evidence.
Effect magnitudes, seed stability, class-level metrics, and confidence limitations should be discussed alongside p-values.

## Files

- `tables/model_comparison_paper.csv`: model-level mean and standard deviation metrics.
- `tables/top_models_per_class.csv`: class-level metrics for the best Macro-F1 model in each task.
- `tables/general_vs_early.csv`: paired General/Early comparison.
- `tables/efficiency.csv`: performance and efficiency summary.
- `statistics/best_vs_others_macro_f1.csv`: exploratory paired Wilcoxon tests against each task winner.
- `figures/macro_f1_general_vs_early.png`: main aligned-endpoint performance figure.
- `figures/macro_f1_seed_stability.png`: five-seed stability figure.
- `figures/top_models_per_class_f1.png`: per-class F1 for the two task winners.
- `figures/performance_efficiency.png`: performance/latency/parameter trade-off.
- `figures/general__gru__confusion.png` and `figures/early__cnn_lstm__confusion.png`: selected aggregate confusion matrices from training outputs.
- `manifest.json`: source and generated-file hashes.
