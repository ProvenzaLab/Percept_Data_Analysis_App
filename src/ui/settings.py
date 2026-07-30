"""Settings menu for choosing model type, window size, AR(k), and delta."""

from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QCheckBox,
    QButtonGroup,
    QGroupBox,
    QFormLayout,
)
from PySide6.QtCore import Qt
import json
from utils.utils import get_data_path


class SettingsMenu(QWidget):
    """Editor for ``data/param.json``.

    Supports three outlier correction methods (Threshold, Threshold +
    Interpolation, Overage Correction), a positive integer window size,
    an optional delta normalization, and an optional AR(k) model with a
    configurable number of lags.
    """

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
            "Threshold": "Identifies and removes all values that include at least one overvoltage reading",
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
        except (TypeError, ValueError):
            QMessageBox.warning(
                self, "Invalid Input", "Window size must be an integer > 0"
            )
            return False
        if tmp <= 0:
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
            except (TypeError, ValueError):
                QMessageBox.warning(
                    self, "Invalid Input", "Lags must be an integer greater than 0"
                )
                return False
            if tmp <= 0:
                QMessageBox.warning(
                    self, "Invalid Input", "Lags must be an integer greater than 0"
                )
                return False
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
            else:
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
        except OSError as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save settings: {e}")
            return

        self.hide()
        self.window().show_opening_screen()
