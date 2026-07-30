"""Top-level ``MainWindow`` for the Percept Data Analysis App.

Hosts the stack of Qt screens (Opening / Loading / Patient / Help /
Settings / Doc / Plots), kicks off the worker process that runs the
analysis pipeline, and routes results to the ``Plots`` widget when
complete.
"""

import multiprocessing

from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QProgressBar,
    QMessageBox,
)
from PySide6.QtCore import Qt, QTimer

from src.pipeline import run_pipeline
from src.ui.opening import OpeningScreen, HelpMenu, DocMenu
from src.ui.settings import SettingsMenu
from src.ui.patient import PatientMenu
from src.ui.plots import Plots


WINDOW_WIDTH = 900
WINDOW_HEIGHT = 640
MIN_WINDOW_WIDTH = 720
MIN_WINDOW_HEIGHT = 520
PLOTS_WINDOW_WIDTH = 1280
PLOTS_WINDOW_HEIGHT = 820
LOADING_SCREEN_INTERVAL = 100  # in milliseconds


class MainWindow(QWidget):
    """Stack-based shell that swaps screens and manages the pipeline worker."""

    def __init__(self):
        super().__init__()
        self.initUI()
        self.result_queue = multiprocessing.Queue()
        self.worker_process = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_for_results)

    def initUI(self):
        self.setWindowTitle("Percept Data App")
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        self.stack = QWidget(self)
        self.layout = QVBoxLayout(self.stack)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.opening_screen = OpeningScreen(self)
        self.layout.addWidget(self.opening_screen)

        self.loading_screen = LoadingScreen(self)
        self.layout.addWidget(self.loading_screen)
        self.loading_screen.hide()

        self.patient_menu = PatientMenu(self)
        self.layout.addWidget(self.patient_menu)
        self.patient_menu.hide()

        self.help_menu = HelpMenu(self)
        self.layout.addWidget(self.help_menu)
        self.help_menu.hide()

        self.settings_menu = SettingsMenu(self)
        self.layout.addWidget(self.settings_menu)
        self.settings_menu.hide()

        self.doc_menu = DocMenu(self)
        self.layout.addWidget(self.doc_menu)
        self.doc_menu.hide()

    def show_loading_screen(self, patient_dict):
        self.loading_screen.show()
        self.loading_screen.progress_bar.setRange(0, 0)
        self.opening_screen.hide()

        self.patient_dict = patient_dict
        self.worker_process = multiprocessing.Process(
            target=run_pipeline, args=(self.patient_dict, self.result_queue)
        )
        self.worker_process.start()
        self.timer.start(LOADING_SCREEN_INTERVAL)

    def check_for_results(self):
        try:
            if not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                if result is None:
                    self.on_script_finished(None, None)
                else:
                    df_final, pt_changes_df = result
                    self.worker_process.terminate()
                    self.on_script_finished(df_final, pt_changes_df)
        except Exception as e:
            print(f"Error checking results: {e}")

    def on_script_finished(self, df_final=None, pt_changes_df=None):
        self.loading_screen.hide()
        if df_final is not None and not df_final.empty:
            self.show_plots(df_final, pt_changes_df)
        else:
            QMessageBox.warning(
                self, "Error", "Failed to process the data. Please try again."
            )
            self.show_opening_screen()

    def show_plots(self, df_final, pt_changes_df):
        self.resize(PLOTS_WINDOW_WIDTH, PLOTS_WINDOW_HEIGHT)
        self.plots = Plots(self, df_final, pt_changes_df)
        self.layout.addWidget(self.plots)
        self.plots.show()

    def show_opening_screen(self):
        self.opening_screen.show()
        self.loading_screen.hide()
        self.hide_all_menus()

    def show_help_menu(self):
        self.hide_all_menus()
        self.opening_screen.hide()
        self.help_menu.show()

    def show_settings_menu(self):
        self.hide_all_menus()
        self.opening_screen.hide()
        self.settings_menu.show()

    def show_patient_menu(self):
        self.hide_all_menus()
        self.opening_screen.hide()
        self.patient_menu.show()

    def hide_all_menus(self):
        self.help_menu.hide()
        self.settings_menu.hide()
        self.patient_menu.hide()

    def show_doc_menu(self):
        self.hide_all_menus()
        self.opening_screen.hide()
        self.doc_menu.show()


class LoadingScreen(QWidget):
    """Indeterminate progress bar shown while the pipeline worker runs."""

    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.layout.addStretch()

        self.label = QLabel(
            "The application is processing your data.\n"
            "Please wait a moment, this may take a couple of minutes.\n"
            "Do not close or restart the application.",
            self,
        )
        self.label.setObjectName("loadingLabel")
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(360)
        self.layout.addWidget(self.progress_bar, alignment=Qt.AlignHCenter)

        self.layout.addStretch()

        self.setLayout(self.layout)
