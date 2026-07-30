"""Core analysis pipeline for the Percept Data Analysis App.

Modules
-------
- generate_raw: Read Percept BrainSense Timeline JSON files into pandas DataFrames.
- process_data: Clean, normalize, z-score, and correct overvoltage artifacts.
- model_data: Fit AR(1)/AR(k) sliding-window models and compute R² metrics.
- pipeline: Orchestrate the three stages across multiple patients.
- ui: PySide6 Qt widgets for the desktop app (screens, dialogs, plots).
"""

__all__ = ["generate_raw", "process_data", "model_data", "pipeline", "ui"]
