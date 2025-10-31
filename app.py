import sys
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QProgressBar,
    QMessageBox,
    QCheckBox,
    QComboBox,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsTextItem,
    QGraphicsEllipseItem,
    QGraphicsRectItem,
)
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt, QUrl, QTimer
from PySide6.QtGui import QPen, QBrush, QFont
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
import src.generate_raw as generate_raw
import src.process_data as process_data
import src.model_data as model_data
from utils.utils import get_data_path
import utils.plotting_utils as plots
import utils.gui_utils as gui_utils
import multiprocessing
import sys
import json
import pandas as pd
import numpy as np
from src.opening_windows import OpeningScreen, HelpMenu, SettingsMenu, DocMenu
from src.patient_menu import PatientMenu

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
LOADING_SCREEN_INTERVAL = 100  # in milliseconds

def worker_function(patient_dict, result_queue):
    try:
        df_final = pd.DataFrame()
        pt_changes_df = pd.DataFrame()

        with open(get_data_path("data\\param.json"), "r") as f:
            param_dict = json.load(f)

        for pt in patient_dict.keys():
            try:
                raw_df, param_changes = generate_raw.generate_raw(pt, patient_dict[pt])

            except TypeError or ValueError as e:
                print(f"Unable to retrieve data for pateint {pt}")
                continue

            processed_data = process_data.process_data(
                pt,
                raw_df,
                patient_dict[pt],
                ark=param_dict["ark"],
                max_lag=param_dict["lags"] if param_dict["ark"] else 1,
            )

            df_w_preds = model_data.model_data(
                processed_data,
                use_constant=False if not param_dict["ark"] else True,
                ark=param_dict["ark"],
                max_lag=param_dict["lags"] if param_dict["ark"] else 1,
            )

            pt_changes_df = pd.concat([pt_changes_df, param_changes], ignore_index=True)

            df_final = pd.concat([df_final, df_w_preds], ignore_index=True)

        result_queue.put((df_final, pt_changes_df))

    except Exception as e:
        print(f"Error in worker_function: {e}")
        result_queue.put(None)


class MainWindow(QWidget):
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

        self.stack = QWidget(self)
        self.layout = QVBoxLayout(self.stack)
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

        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet(
            """
            QWidget {
                background-color: #2d2d2d;
                color: #f5f5f5;
                font-family: Arial, sans-serif;
            }
            QLabel {
                color: #f5f5f5;
                font-size: 14px;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #3d3d3d;
                color: #f5f5f5;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #1e90ff;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1c86ee;
            }
            QComboBox QAbstractItemView {
                background-color: #3d3d3d;
                color: #f5f5f5;
                selection-background-color: #1e90ff;
            }
        """
        )

    def show_loading_screen(self, patient_dict):
        self.loading_screen.show()
        self.loading_screen.progress_bar.setRange(0, 0)
        self.opening_screen.hide()

        self.patient_dict = patient_dict
        self.worker_process = multiprocessing.Process(
            target=worker_function, args=(self.patient_dict, self.result_queue)
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
        if not df_final.empty:
            self.show_plots(df_final, pt_changes_df)
        else:
            QMessageBox.warning(
                self, "Error", "Failed to process the data. Please try again."
            )
            self.show_opening_screen()

    def show_plots(self, df_final, pt_changes_df):
        self.setGeometry(100, 100, 1200, 800)
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
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.label = QLabel(
            "The application is processing your data.\nPlease wait a moment, this may take a couple of minutes.\nDo not close or restart the application.",
            self,
        )
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(
            """
            QLabel {
                font-size: 16px;
                font-family: 'Arial', sans-serif;
                color: #ffffff;
                padding: 20px;
            }
        """
        )
        self.layout.addWidget(self.label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background-color: #1e90ff;
                width: 20px;
            }
        """
        )
        self.layout.addWidget(self.progress_bar)

        self.setStyleSheet(
            """
            background-color: #2d2d2d;
        """
        )

        self.setLayout(self.layout)


class Plots(QWidget):
    def __init__(self, parent, df_final, pt_changes_df):
        super().__init__(parent)
        self.parent = parent
        self.df_final = df_final
        self.pt_changes_df = pt_changes_df

        # Load static resources only once
        if not hasattr(Plots, "param_dict"):
            with open(get_data_path("data\\param.json"), "r") as f:
                self.param_dict = json.load(f)
        if not hasattr(Plots, "patient_dict"):
            with open(get_data_path("data\\patient_info.json")) as f:
                self.patient_dict = json.load(f)

        self.curr_pt = list(self.patient_dict.keys())[0]
        self.hemisphere = "left"
        self.current_plot = None
        self.web_view = QWebEngineView(self)
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.content_layout = QHBoxLayout()

        self.init_json_frame()
        self.init_plot_frame()

        self.layout.addLayout(self.content_layout)
        self.init_bottom_buttons()
        self.setLayout(self.layout)

        self.refresh_patient_view()

    # -------------------------
    # JSON panel
    # -------------------------
    def init_json_frame(self, index=0):
        self.json_fields_frame = QWidget(self)
        self.legend_frame = QWidget(self)
        self.json_layout = QVBoxLayout(self.json_fields_frame)

        # Patient selector
        self.patient_selector = QComboBox(self)
        self.patient_selector.addItems(self.patient_dict.keys())
        self.patient_selector.setCurrentIndex(index)
        self.patient_selector.currentIndexChanged.connect(self.patient_change)
        self.json_layout.addWidget(self.patient_selector)

        # Hemisphere selector
        self.hemisphere_selector = QComboBox(self)
        self.hemisphere_selector.addItems(["Left Hemisphere", "Right Hemisphere"])
        self.hemisphere_selector.setCurrentIndex(index)
        self.hemisphere_selector.currentIndexChanged.connect(self.on_hemisphere_change)
        self.json_layout.addWidget(self.hemisphere_selector)

        # JSON display
        self.json_text = QTextEdit(self.json_fields_frame)
        self.json_text.setReadOnly(True)
        self.json_text.setMinimumHeight(200)
        self.json_text.setStyleSheet(
            "background:#4d4d4d; color:#f5f5f5; border:1px solid #555; padding:10px;"
        )
        self.json_layout.addWidget(self.json_text)

        # Legend
        self.legend = QGraphicsScene()
        self.legend.setBackgroundBrush(QBrush("#FFFFFF"))
        self.legend_view = QGraphicsView(self.legend)
        self.legend_view.setSceneRect(0, 0, 200, 200)
        self.json_layout.addWidget(self.legend_view, alignment=Qt.AlignCenter)

        # Controls
        self.changes_checkbox = QCheckBox("Show Parameter Changes", self)
        self.changes_checkbox.stateChanged.connect(
            lambda _: self.refresh_patient_view()
        )
        self.json_layout.addWidget(self.changes_checkbox, alignment=Qt.AlignCenter)

        self.export_button = QPushButton("Export LinAR R² feature", self)
        self.export_button.clicked.connect(self.export_data)
        self.json_layout.addWidget(self.export_button, alignment=Qt.AlignCenter)

        self.content_layout.addWidget(self.json_fields_frame, 2)

    def update_json_fields(self, patient):
        pt_df = self.df_final.query("pt_id == @patient")
        self.json_text.clear()
        self.json_text.append(f"Subject_name: {patient}\n")
        self.json_text.append(
            f"Initial DBS programming: {self.patient_dict[patient]['dbs_date']}\n"
        )
        self.json_text.append(f"Total samples: {len(pt_df)}\n")
        self.json_text.append(f"Total days: {pt_df['days_since_dbs'].nunique()}\n")
        if self.patient_dict[patient]["response_status"] == 1:
            self.json_text.append(
                f"Responder on {self.patient_dict[patient]['response_date']}\n"
            )
        else:
            self.json_text.append("Non-responder\n")

    # -------------------------
    # Plot panel
    # -------------------------
    def init_plot_frame(self):
        self.web_view.setMinimumSize(800, 600)
        self.configure_web_view()

        self.plot_layout = QVBoxLayout()
        self.plot_layout.addWidget(self.web_view)

        self.content_layout.addLayout(self.plot_layout, 8)

    def configure_web_view(self):
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)

    # -------------------------
    # Controls
    # -------------------------
    def init_bottom_buttons(self):
        self.button_layout = QHBoxLayout()

        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back)
        self.button_layout.addWidget(self.back_button, alignment=Qt.AlignLeft)

        self.data_export_button = QPushButton("Export Raw Data", self)
        self.data_export_button.clicked.connect(self.export_raw)
        self.button_layout.addWidget(self.data_export_button, alignment=Qt.AlignCenter)

        self.download_button = QPushButton("Download plot", self)
        self.download_button.clicked.connect(self.download_image)
        self.button_layout.addWidget(self.download_button, alignment=Qt.AlignRight)

        self.layout.addLayout(self.button_layout)

    # -------------------------
    # Actions
    # -------------------------
    def refresh_patient_view(self):
        self.update_json_fields(self.curr_pt)
        self.update_plot(
            self.curr_pt, self.hemisphere, self.changes_checkbox.isChecked()
        )

    def patient_change(self, index):
        self.curr_pt = list(self.patient_dict.keys())[index]
        self.refresh_patient_view()

    def on_hemisphere_change(self, index):
        self.hemisphere = "left" if index == 0 else "right"
        self.refresh_patient_view()

    def update_plot(self, patient, hemisphere="left", show_changes=False):
        fig, tval, pval = plots.plot_metrics(
            df=self.df_final,
            patient=patient,
            hemisphere=hemisphere,
            changes_df=self.pt_changes_df,
            show_changes=show_changes,
            patients_dict=self.patient_dict,
            param_dict=self.param_dict,
        )
        self.current_plot = fig

        temp_file_path = gui_utils.create_temp_plot(fig)
        self.web_view.setUrl(QUrl.fromLocalFile(temp_file_path))

        self.create_legend()

        self.json_text.append(
            f"Pre-DBS vs. Post-DBS t-test stats:\nt = {np.round(tval, 4)}\np = {np.round(pval, 4) if pval > 0.0001 else 'p < 10⁻⁴'}"
        )

    def create_legend(self):
        self.legend.clear()
        labels = {
            "Raw LFP (z-scored)": QGraphicsEllipseItem(5, 5, 10, 10),
            "AR predicted LFP (z-scored)": QGraphicsEllipseItem(5, 20, 10, 10),
            "DBS On": QGraphicsRectItem(5, 35, 10, 10),
            "Pre-DBS": QGraphicsEllipseItem(5, 50, 10, 10),
        }
        colors = {
            "Raw LFP (z-scored)": "#808080",
            "AR predicted LFP (z-scored)": "#33a02c",
            "DBS On": "#eb6bde",
            "Pre-DBS": "#ffe900",
        }
        pt_params = self.patient_dict[self.curr_pt]
        if pt_params["response_status"] == 1:
            labels["Response"] = QGraphicsEllipseItem(5, 65, 10, 10)
            colors["Response"] = "#0000ff"
        else:
            labels["Non-Response"] = QGraphicsEllipseItem(5, 65, 10, 10)
            colors["Non-Response"] = "#ffb900"
        if self.changes_checkbox.isChecked():
            labels["Parameter Change"] = QGraphicsRectItem(5, 80, 10, 10)
            colors["Parameter Change"] = "#000000"

        y_pos = 5
        offset = 15

        for label, item in labels.items():
            item.setPos(5, y_pos)
            item.setPen(QPen(colors[label]))
            item.setBrush(QBrush(colors[label]))
            self.legend.addItem(item)

            text_item = QGraphicsTextItem(label)
            text_item.setFont(QFont("Arial", 8))

            text_y = item.pos().y() * 2

            text_item.setPos(item.pos().x() + 20, text_y)
            self.legend.addItem(text_item)

            y_pos += offset

        # Adjust scene rect to fit items
        self.legend.setSceneRect(self.legend.itemsBoundingRect())

    def go_back(self):
        self.hide()
        self.parent.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.parent.show_opening_screen()

    def download_image(self):
        file_path = gui_utils.open_save_dialog(self, "Save Image", "")
        if file_path and self.current_plot:
            gui_utils.save_plot(self.current_plot, file_path)
        else:
            QMessageBox.warning(self, "Error", "No plot is available to save.")

    def export_raw(self):
        file_path = gui_utils.open_save_dialog(self, "Save Raw Data", "")
        if file_path:
            data = self.df_final.query("pt_id == @self.curr_pt")
            gui_utils.save_raw_data(data, file_path)

    def export_data(self):
        file_path = gui_utils.open_save_dialog(self, "Save Data", "")
        if file_path:
            data = self.df_final.query("pt_id == @self.curr_pt")
            gui_utils.save_lin_ar_feature(data, file_path, self.param_dict)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    multiprocessing.freeze_support()
    # basedir = os.path.dirname(__file__)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icons/Icon.ico"))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
