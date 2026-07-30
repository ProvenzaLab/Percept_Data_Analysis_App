"""PySide6 GUI screens and widgets for the Percept Data Analysis App.

Modules
-------
- main_window: Top-level ``MainWindow`` and its ``LoadingScreen``.
- plots: ``Plots`` widget that renders the Plotly figure and side panels.
- opening: ``OpeningScreen``, ``HelpMenu``, ``DocMenu``.
- settings: ``SettingsMenu`` (model + window-size + AR(k) configuration).
- patient: ``PatientMenu`` (add / delete / list patients from the JSON db).
"""

from .main_window import MainWindow, LoadingScreen
from .plots import Plots
from .opening import OpeningScreen, HelpMenu, DocMenu
from .settings import SettingsMenu
from .patient import PatientMenu

__all__ = [
    "MainWindow",
    "LoadingScreen",
    "Plots",
    "OpeningScreen",
    "HelpMenu",
    "DocMenu",
    "SettingsMenu",
    "PatientMenu",
]
