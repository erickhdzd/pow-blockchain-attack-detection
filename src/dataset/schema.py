"""Stable column schemas for raw, model, target, and metadata outputs."""

OBSERVABLE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "canonical_height",
    "mean_height",
    "min_height",
    "max_height",
    "height_std",
    "height_range",
    "unique_tip_count",
    "consensus_ratio",
    "reference_height",
    "tip_age_seconds",
    "latest_block_interval_seconds",
    "mean_recent_block_interval_seconds",
    "std_recent_block_interval_seconds",
    "unique_miner_count",
    "dominant_miner_ratio",
    "miner_hhi",
    "blocks_added",
    "blocks_removed",
    "common_ancestor_height",
    "reorganization_detected",
    "reorganization_depth",
    "tip_changed",
)

MODEL_EXCLUDED_OBSERVABLE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "canonical_height",
    "reference_height",
)

MODEL_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column
    for column in OBSERVABLE_COLUMNS
    if column not in MODEL_EXCLUDED_OBSERVABLE_COLUMNS
)

EARLY_DETECTION_EXCLUDED_COLUMNS: tuple[str, ...] = (
    "blocks_removed",
    "common_ancestor_height",
    "reorganization_detected",
    "reorganization_depth",
)

EARLY_DETECTION_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column
    for column in MODEL_FEATURE_COLUMNS
    if column not in EARLY_DETECTION_EXCLUDED_COLUMNS
)

METADATA_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "simulation_id",
    "scenario_id",
    "seed",
    "step",
    "configured_attack_type",
    "attack_active",
    "attack_successful",
    "attack_phase",
    "attack_start_step",
    "attack_end_step",
    "attacker_hash_share",
    "gamma",
    "network_profile",
    "num_nodes",
    "num_miners",
    "base_latency_ms",
    "latency_std_ms",
    "packet_loss_probability",
    "miner_concentration",
    "split",
)

TARGET_COLUMNS: tuple[str, ...] = ("attack_label",)

FORBIDDEN_MODEL_COLUMNS: frozenset[str] = frozenset(
    {
        *METADATA_COLUMNS,
        *MODEL_EXCLUDED_OBSERVABLE_COLUMNS,
        *TARGET_COLUMNS,
    }
)
