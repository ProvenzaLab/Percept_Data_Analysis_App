"""Plotly-backed results viewer for the GUI.

Renders the per-patient LFP / R² / violin plots inside a
``QWebEngineView`` and provides side-panel JSON summary plus the export
buttons used by the desktop app.
"""

from PySide6.QtWidgets import (
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QMessageBox,
    QCheckBox,
    QComboBox,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsTextItem,
    QGraphicsEllipseItem,
    QGraphicsRectItem,
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QPen, QBrush, QFont
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
import json
import numpy as np
from utils.utils import get_data_path
import utils.plotting_utils as plots
import utils.gui_utils as gui_utils
from src.ui import theme


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


class Plots(QWidget):
    """Side-by-side viewer: left = patient summary, right = Plotly figure."""

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
        self.layout.setContentsMargins(16, 16, 16, 16)
        self.content_layout = QHBoxLayout()
        self.content_layout.setSpacing(16)

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
        self.json_text.setObjectName("summaryPanel")
        self.json_text.setReadOnly(True)
        self.json_text.setMinimumHeight(200)
        self.json_layout.addWidget(self.json_text)

        # Legend. Deliberately a light "card" so it reads as part of the
        # (light-backgrounded) Plotly figure it annotates rather than as a
        # stray panel floating in the dark chrome.
        self.legend = QGraphicsScene()
        self.legend.setBackgroundBrush(QBrush(theme.LEGEND_CARD_BG))
        self.legend_view = QGraphicsView(self.legend)
        self.legend_view.setObjectName("legendView")
        self.legend_view.setFrameShape(QGraphicsView.NoFrame)
        self.legend_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.legend_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.legend_view.setFixedHeight(130)
        self.json_layout.addWidget(self.legend_view)

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
        self.button_layout.setContentsMargins(0, 8, 0, 0)

        self.back_button = QPushButton("Back", self)
        self.back_button.setObjectName("secondaryButton")
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
            text_item.setFont(QFont("Segoe UI", 8))
            text_item.setDefaultTextColor(theme.LEGEND_CARD_TEXT)

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
