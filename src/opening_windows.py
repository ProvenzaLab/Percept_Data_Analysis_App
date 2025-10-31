from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QCheckBox,
    QToolBar,
    QButtonGroup,
    QGroupBox,
    QFormLayout,
)
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt, QUrl, QSize
from PySide6.QtGui import QDesktopServices
from PySide6.QtWebEngineWidgets import QWebEngineView
import os
import json
from utils.utils import get_data_path, resource_path


class OpeningScreen(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)

        self.welcome_label = QLabel("Welcome to the Percept Data Analysis App", self)
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setStyleSheet(
            """
            QLabel {
                font-size: 25px;
                font-family: 'Arial', sans-serif;
                color: #ffffff;
                padding: 10px;
            }
        """
        )
        self.layout.addWidget(self.welcome_label)

        self.description_label = QLabel(
            "This application helps you process and analyze Medtronic percept data.<br>"
            "Please proceed to start the data processing.<br><br>"
            "Developed by the Provenza Lab",
            self,
        )
        self.description_label.setAlignment(Qt.AlignCenter)
        self.description_label.setStyleSheet(
            """
            QLabel {
                font-size: 16px;
                font-family: 'Arial', sans-serif;
                color: #ffffff;
                padding: 20px;
            }
        """
        )
        self.layout.addWidget(self.description_label)

        self.proceed_button = QPushButton("Start Data Processing", self)
        self.proceed_button.setStyleSheet(
            """
            QPushButton {
                background-color: #1e90ff;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #1c86ee;
            }
        """
        )
        self.proceed_button.clicked.connect(self.proceed)
        self.layout.addWidget(self.proceed_button, alignment=(Qt.AlignHCenter))

        self.patient_menu_button = QPushButton("Add Patients", self)
        self.patient_menu_button.setStyleSheet(
            """
            QPushButton {
                background-color: #1e90ff;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #1c86ee;
            }
        """
        )

        self.patient_menu_button.clicked.connect(self.parent.show_patient_menu)
        self.layout.addWidget(self.patient_menu_button, alignment=(Qt.AlignHCenter))

        def open_url(url):
            """Opens the given URL in the default web browser."""
            qurl = QUrl(url)
            QDesktopServices.openUrl(qurl)

        toolbar = QToolBar("Main Window Toolbar")
        toolbar.setIconSize(QSize(30, 30))
        self.layout.addWidget(toolbar)

        doc_button = QAction(
            QIcon(resource_path("icons/doc_icon.ico")),
            "See GitHub documentation of the app",
            self,
        )
        doc_button.setStatusTip("See GitHub Documentation of the app")
        doc_button.triggered.connect(self.parent.show_doc_menu)
        toolbar.addAction(doc_button)

        help_button = QAction(
            QIcon(resource_path("icons/help_icon.ico")), "How to use the app", self
        )
        help_button.setStatusTip("For a step-by-step guide to use the app")
        help_button.triggered.connect(self.parent.show_help_menu)
        toolbar.addAction(help_button)

        settings_button = QAction(
            QIcon(resource_path("icons/settings_icon.ico")), "Processing settings", self
        )
        settings_button.setStatusTip("Processing settings")
        settings_button.triggered.connect(self.parent.show_settings_menu)
        toolbar.addAction(settings_button)

    def proceed(self):
        if not os.path.exists(get_data_path("data\\patient_info.json")):
            return WindowsError
        with open(get_data_path("data\\patient_info.json"), "r") as f:
            try:
                patient_dict = json.load(f)
            except json.JSONDecodeError:
                QMessageBox.warning(
                    self, "Validation Error", "No patient data is stored"
                )
                return
        if len(patient_dict) == 0:
            QMessageBox.warning(self, "Validation Error", "No patient data is stored")
            return
        self.parent.show_loading_screen(patient_dict)


class HelpMenu(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)
        help_view = QWebEngineView()
        help_view.load(QUrl("https://github.com/ProvenzaLab/Percept_Data_Analysis_App/blob/main/help.md"))
        self.layout.addWidget(help_view)

        self.init_bottom_buttons()
        self.setLayout(self.layout)

    def go_back(self):
        self.hide()
        self.window().show_opening_screen()

    def init_bottom_buttons(self):
        self.button_layout = QHBoxLayout()

        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back)
        self.button_layout.addWidget(
            self.back_button, alignment=Qt.AlignLeft | Qt.AlignBottom
        )

        self.layout.addLayout(self.button_layout)

class DocMenu(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)
        doc_view = QWebEngineView()
        doc_view.load(QUrl("https://github.com/ProvenzaLab/Percept_Data_Analysis_App/blob/main/README.md"))
        self.layout.addWidget(doc_view)
        
        self.init_bottom_buttons()
        self.setLayout(self.layout)
    
    def go_back(self):
        self.hide()
        self.window().show_opening_screen()

    def init_bottom_buttons(self):
        self.button_layout = QHBoxLayout()

        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back)
        self.button_layout.addWidget(
            self.back_button, alignment=Qt.AlignLeft | Qt.AlignBottom
        )

        self.layout.addLayout(self.button_layout)

class SettingsMenu(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.field_order = ["Window size"]
        self.fields = {"Window size": 3}
        self.tooltips = self.get_tooltips()

        self.entries = {}
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Model type
        layout.addWidget(self.create_model_type_group())

        # Fields (window size, etc.)
        layout.addWidget(self.create_fields_group())

        # Delta normalize
        layout.addWidget(self.create_delta_group())

        # AR(k) model
        layout.addWidget(self.create_ark_group())

        # Buttons
        layout.addLayout(self.init_bottom_buttons())

        self.setLayout(layout)

    # ----------------------------
    # Group creation methods
    # ----------------------------

    def create_fields_group(self):
        group = QGroupBox("General Settings")
        form = QFormLayout(group)

        for key in self.field_order:
            value = self.fields[key]
            entry = QLineEdit(str(value), self)
            entry.setToolTip(self.tooltips[key])
            form.addRow(QLabel(key + ":"), entry)
            self.entries[key] = entry

        return group

    def create_model_type_group(self):
        group = QGroupBox("Model Type")
        hbox = QHBoxLayout(group)

        self.naive_checkbox = QCheckBox("Threshold", self)
        self.threshold_checkbox = QCheckBox("Threshold + Interpolation", self)
        self.overage_checkbox = QCheckBox("Overage Correction", self)

        for cb, tt in [
            (self.naive_checkbox, "Threshold"),
            (self.threshold_checkbox, "Threshold + Interpolation"),
            (self.overage_checkbox, "Overage Correction"),
        ]:
            cb.setToolTip(self.tooltips[tt])

        # Exclusive selection
        checkbox_group = QButtonGroup(self)
        checkbox_group.setExclusive(True)
        for cb in [self.naive_checkbox, self.threshold_checkbox, self.overage_checkbox]:
            checkbox_group.addButton(cb)

        self.overage_checkbox.setChecked(True)

        hbox.addWidget(self.naive_checkbox)
        hbox.addWidget(self.threshold_checkbox)
        hbox.addWidget(self.overage_checkbox)
        hbox.addStretch()

        return group

    def create_delta_group(self):
        group = QGroupBox("Delta Normalization")
        hbox = QHBoxLayout(group)

        self.delta_checkbox = QCheckBox("Delta normalize R² with pre-DBS average")
        self.delta_checkbox.setToolTip(
            "Normalize R² value with pre-DBS average. Will revert to original R² values if no pre-DBS data is available."
        )
        hbox.addWidget(self.delta_checkbox)
        return group

    def create_ark_group(self):
        group = QGroupBox("AR(k) Model")
        form = QFormLayout(group)

        self.ark_checkbox = QCheckBox("Enable AR(k) model")
        self.ark_checkbox.setToolTip(
            "Use an AR(k) model to predict LFP data. Default model is AR(1)."
        )
        self.ark_checkbox.stateChanged.connect(self.toggle_lags)

        self.lag_entry = QLineEdit("72", self)
        self.lag_label = QLabel("Lags:")
        form.addRow(self.ark_checkbox)
        form.addRow(self.lag_label, self.lag_entry)

        # Hide lag inputs by default
        self.lag_label.hide()
        self.lag_entry.hide()
        return group

    def init_bottom_buttons(self):
        button_layout = QHBoxLayout()

        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back)

        self.default_button = QPushButton("Reset to Default", self)
        self.default_button.clicked.connect(self.set_default_settings)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_settings)

        button_layout.addWidget(self.back_button, alignment=Qt.AlignLeft)
        button_layout.addStretch()
        button_layout.addWidget(self.default_button)
        button_layout.addStretch()
        button_layout.addWidget(self.save_button, alignment=Qt.AlignRight)

        return button_layout

    def toggle_lags(self):
        if self.ark_checkbox.isChecked():
            self.lag_label.show()
            self.lag_entry.show()
        else:
            self.lag_label.hide()
            self.lag_entry.hide()

    def get_tooltips(self):
        return {
            "Threshold": "Identifies and removes all values that include at least one overvolage reading",
            "Threshold + Interpolation": "Interpolate thresholded data using PCHIP for windows < 12 samples",
            "Overage Correction": "Overage event correction and recalculation for overvoltage events (Recommended)",
            "Window size": "Window size to train the autoregressive model on",
        }

    def validate_fields(self):
        if not self.entries["Window size"].text():
            QMessageBox.warning(self, "Invalid Input", "Window size must be filled in")
            return False
        try:
            tmp = int(self.entries["Window size"].text())
        except Exception or tmp <= 0:
            QMessageBox.warning(
                self, "Invalid Input", "Window size must be an integer > 0"
            )
            return False
        if (
            not self.naive_checkbox.isChecked()
            and not self.threshold_checkbox.isChecked()
            and not self.overage_checkbox.isChecked()
        ):
            QMessageBox.warning(
                self, "Invalid Input", "No overage handling method is checked"
            )
            return False
        if self.ark_checkbox.isChecked():
            try:
                tmp = int(self.lag_entry.text())
            except Exception or tmp <= 0:
                QMessageBox.warning(
                    self, "Invalid Input", "Lags must an integer greater than 0"
                )
                return
        return True

    def go_back(self):
        self.hide()
        self.window().show_opening_screen()

    def set_default_settings(self):
        self.entries["Window size"].setText("3")

        self.naive_checkbox.setChecked(False)
        self.threshold_checkbox.setChecked(False)
        self.overage_checkbox.setChecked(True)

        self.delta_checkbox.setChecked(False)
        self.ark_checkbox.setChecked(False)

    def save_settings(self):
        if not self.validate_fields():
            return

        param_dict = {}
        param_dict["hemisphere"] = "left"
        for key, entry in self.entries.items():
            if key == "Window size":
                param_dict[key] = int(entry.text())
            param_dict[key] = entry.text()

        if self.naive_checkbox.isChecked():
            param_dict["model"] = "naive"

        elif self.threshold_checkbox.isChecked():
            param_dict["model"] = "SLOvER+"

        elif self.overage_checkbox.isChecked():
            param_dict["model"] = "OvER"

        param_dict["delta"] = 1 if self.delta_checkbox.isChecked() else 0

        param_dict["ark"] = 1 if self.ark_checkbox.isChecked() else 0

        param_dict["lags"] = (
            int(self.lag_entry.text()) if self.ark_checkbox.isChecked() else False
        )

        try:
            with open(get_data_path("data\\param.json"), "w") as f:
                json.dump(param_dict, f, indent=4)
                f.close()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save settings: {e}")
            return

        self.hide()
        self.window().show_opening_screen()
