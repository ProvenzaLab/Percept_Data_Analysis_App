"""Top-level ``MainWindow`` for the Percept Data Analysis App.

Hosts the stack of Qt screens (Opening / Loading / Patient / Help /
Settings / Doc / Plots), kicks off the worker process that runs the
analysis pipeline, and routes results to the ``Plots`` widget when
complete.
"""

import json
import multiprocessing

import pandas as pd
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
from utils.utils import get_data_path


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

        # Session-lifetime cache of already-processed patients, so a later
        # run only needs to process patients not already in here. Cleared
        # whenever data/param.json changes, since cached results reflect
        # whatever settings were active when they were produced. Set before
        # initUI(): PatientMenu reads processed_patient_ids while building
        # its table during construction.
        self.processed_df_final = pd.DataFrame()
        self.processed_pt_changes_df = pd.DataFrame()
        self.processed_patient_ids = set()
        self.cached_param_snapshot = None

        self.pending_total = 0
        self.pending_completed = 0
        self.pending_failed = []

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

    def show_loading_screen(self, selected_patient_dict):
        """Process the selected patients not already cached this session.

        Patients already in ``self.processed_patient_ids`` are skipped
        entirely (their cached results are reused), unless the processing
        settings have changed since they were produced, in which case the
        whole cache is invalidated first.
        """
        with open(get_data_path("data\\param.json"), "r") as f:
            param_dict = json.load(f)

        if (
            self.cached_param_snapshot is not None
            and param_dict != self.cached_param_snapshot
        ):
            self.processed_df_final = pd.DataFrame()
            self.processed_pt_changes_df = pd.DataFrame()
            self.processed_patient_ids = set()
        self.cached_param_snapshot = param_dict

        to_process = {
            pt: info
            for pt, info in selected_patient_dict.items()
            if pt not in self.processed_patient_ids
        }

        self.opening_screen.hide()
        self.hide_all_menus()

        if not to_process:
            self.show_plots(self.processed_df_final, self.processed_pt_changes_df)
            return

        self.pending_total = len(to_process)
        self.pending_completed = 0
        self.pending_failed = []

        self.loading_screen.show()
        self.loading_screen.set_total(self.pending_total)

        self.worker_process = multiprocessing.Process(
            target=run_pipeline, args=(to_process, self.result_queue)
        )
        self.worker_process.start()
        self.timer.start(LOADING_SCREEN_INTERVAL)

    def check_for_results(self):
        try:
            while not self.result_queue.empty():
                msg = self.result_queue.get_nowait()
                msg_type = msg.get("type")
                if msg_type == "started":
                    self.pending_total = msg["total"]
                    self.loading_screen.set_total(self.pending_total)
                elif msg_type == "progress":
                    self.pending_completed += 1
                    if not msg["success"]:
                        self.pending_failed.append((msg["pt_id"], msg["error"]))
                    self.loading_screen.update_progress(
                        self.pending_completed,
                        self.pending_total,
                        len(self.pending_failed),
                    )
                elif msg_type == "done":
                    self._finish_processing(msg)
                elif msg_type == "error":
                    self._fail_processing(msg["message"])
        except Exception as e:
            print(f"Error checking results: {e}")

    def _stop_worker(self):
        self.timer.stop()
        if self.worker_process is not None:
            self.worker_process.terminate()
            self.worker_process = None
        self.loading_screen.hide()

    def _finish_processing(self, msg):
        self._stop_worker()

        df_final = msg["df_final"]
        pt_changes_df = msg["pt_changes_df"]
        if not df_final.empty:
            self.processed_df_final = pd.concat(
                [self.processed_df_final, df_final], ignore_index=True
            )
            self.processed_pt_changes_df = pd.concat(
                [self.processed_pt_changes_df, pt_changes_df], ignore_index=True
            )
            self.processed_patient_ids |= set(df_final["pt_id"].unique())

        failed = msg["failed"]
        if failed:
            failed_lines = "\n".join(f"• {pt}: {err}" for pt, err in failed)
            QMessageBox.warning(
                self,
                "Some patients failed to process",
                f"{len(failed)} patient(s) could not be processed:\n\n{failed_lines}",
            )

        if self.processed_df_final.empty:
            QMessageBox.warning(
                self, "Error", "Failed to process the data. Please try again."
            )
            self.show_opening_screen()
        else:
            self.show_plots(self.processed_df_final, self.processed_pt_changes_df)

    def _fail_processing(self, message):
        self._stop_worker()
        QMessageBox.warning(self, "Error", f"Failed to process the data: {message}")
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
    """Determinate progress bar + status label shown while patients process."""

    STATUS_SUFFIX = "\nDo not close or restart the application."

    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.layout.addStretch()

        self.label = QLabel(
            "Preparing to process patient data..." + self.STATUS_SUFFIX,
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

    def set_total(self, total):
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(0)
        self.label.setText(f"Processing 0 of {total} patient(s)..." + self.STATUS_SUFFIX)

    def update_progress(self, completed, total, failed_count):
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(completed)
        suffix = f" ({failed_count} failed)" if failed_count else ""
        self.label.setText(
            f"Processing {completed} of {total} patient(s) complete{suffix}"
            + self.STATUS_SUFFIX
        )
