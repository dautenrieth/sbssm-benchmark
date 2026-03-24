"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Defines standardised metric groupings for evaluating Search-Based Scenario
    Selection Methods (SBSSMs). Each set covers a distinct evaluation objective
    (general statistics, extremum search, surrogate model fidelity, boundary
    detection) and maps to a category key used by the Metrics suite.

    - key role: Single source of truth for which metrics belong to which
                evaluation category; controls Metrics.run_metrics_suite().
    - dependency: metrics.py (consumes ALL_METRICS_BY_CATEGORY)
    - output: Four frozenset-compatible sets + ALL_METRICS_BY_CATEGORY dict.

Categories:
    GENERAL_METRICS            — coverage and distributional statistics
    EXTREMUM_SEARCH_METRICS    — convergence to extreme criticality values
    MODEL_RECONSTRUCTION_METRICS — surrogate model accuracy (R², MAE, F1)
    BOUNDARY_DETECTION_METRICS — precision/recall of criticality boundaries

Usage:
    from metric_categories import ALL_METRICS_BY_CATEGORY
    # Pass category keys to Metrics.run_metrics_suite():
    space.metrics.run_metrics_suite(method_categories=["general", "boundary_detection"])
"""

GENERAL_METRICS = {
    "average_criticality",
    "min_criticality",
    "max_criticality",
    "average_criticality_selected",
    "min_criticality_selected",
    "max_criticality_selected",
    "spatial_entropy",
    "discrepancy",
}

EXTREMUM_SEARCH_METRICS = {"convergence_rate", "extremum_gap"}

MODEL_RECONSTRUCTION_METRICS = {"model_approximation_error", "model_r2_score", "f1_coverage"}

BOUNDARY_DETECTION_METRICS = {
    "boundary_precision",
    "boundary_effectiveness",
    "boundary_coverage",
}

ALL_METRICS_BY_CATEGORY = {
    "general": GENERAL_METRICS,
    "extremum_search": EXTREMUM_SEARCH_METRICS,
    "model_reconstruction": MODEL_RECONSTRUCTION_METRICS,
    "boundary_detection": BOUNDARY_DETECTION_METRICS,
}
