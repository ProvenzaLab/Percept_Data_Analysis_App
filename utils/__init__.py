"""Helper utilities for the Percept Data Analysis App.

Modules
-------
- utils: General helpers (z-scoring, outlier filling, contiguous-run labeling).
- json_utils: Parsers for Medtronic Percept BrainSense Timeline JSON files.
- state_utils: Patient clinical-state labeling (Pre-DBS, Responder, etc.).
- model_utils: Autoregressive model fitting and lag selection.
- plotting_utils: Plotly figure generation for the GUI and headless reports.
- gui_utils: Qt / Plotly helpers for export, dialogs, and HTML rendering.
"""

__all__ = [
    "utils",
    "json_utils",
    "state_utils",
    "model_utils",
    "plotting_utils",
    "gui_utils",
]
