from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDialog,
    QButtonGroup,
)

from PySide6.QtCore import Qt
import os
import json
from utils.utils import get_data_path
import utils.gui_utils as gui_utils
from pathlib import Path


class PatientMenu(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.field_order = [
            "Patient ID",
            "directory",
            "dbs_date",
            "response_status",
            "response_date"
            #"disinhibited_dates",
        ]
        self.tooltips = self.get_tooltips()
        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout(self)

        self.table_layout = QVBoxLayout()
        self.main_layout.addLayout(self.table_layout)

        self.load_patients_table()
        self.init_bottom_buttons()

        self.setLayout(self.main_layout)

    def load_patients_table(self):
        self.table = QTableWidget(self)

        patients = self.load_patient_data()
        self.table.setRowCount(len(patients))
        if len(patients) == 0:
            return

        display_fields = ["Patient ID", "Directory", "Response Status"]
        display_keys = {"Directory": "directory", "Response Status": "response_status"}
        display_response = {0: "Non-responder", 1: "Responder"}
        self.table.setColumnCount(len(display_fields))
        self.table.setHorizontalHeaderLabels(display_fields)

        for row, patient in enumerate(patients.keys()):
            for col, key in enumerate(display_fields):
                if key == "Patient ID":
                    self.table.setItem(row, col, QTableWidgetItem(patient))
                elif key == "Response Status":
                    response_status = display_response[
                        patients[patient][display_keys[key]]
                    ]
                    self.table.setItem(row, col, QTableWidgetItem(response_status))
                else:
                    self.table.setItem(
                        row, col, QTableWidgetItem(patients[patient][display_keys[key]])
                    )

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_layout.addWidget(self.table)

    def load_patient_data(self):
        if not os.path.exists(get_data_path("data\\patient_info.json")):
            return {}
        with open(get_data_path("data\\patient_info.json"), "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    def go_back(self):
        self.hide()
        self.window().show_opening_screen()

    def refresh_table(self):
        # Remove existing table from layout
        if hasattr(self, "table"):
            self.table_layout.removeWidget(self.table)
            self.table.deleteLater()
            self.table = None

        self.load_patients_table()

    def add_patient(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Patient")
        layout = QVBoxLayout(dialog)

        form_entries = {}

        key_labels = {
            "Patient ID": "Patient ID",
            "directory": "Directory",
            "dbs_date": "DBS Date",
            "response_status": "Response Status",
            "response_date": "Response Date",
            "disinhibited_dates": "Disinhibited Dates",
        }

        for key in self.field_order:
            if key == "response_status":
                hbox = QHBoxLayout()
                label = QLabel(key_labels[key])
                response_checkbox = QCheckBox("Responder")
                non_response_checkbox = QCheckBox("Non-responder")

                checkbox_group = QButtonGroup(dialog)
                checkbox_group.setExclusive(True)
                checkbox_group.addButton(response_checkbox)
                checkbox_group.addButton(non_response_checkbox)

                hbox.addWidget(label)
                hbox.addWidget(response_checkbox)
                hbox.addWidget(non_response_checkbox)
                layout.addLayout(hbox)
            elif key == "response_date":
                response_date_layout = QHBoxLayout()
                response_date_label = QLabel(key_labels[key])
                response_date_entry = QLineEdit()
                response_date_layout.addWidget(response_date_label)
                response_date_layout.addWidget(response_date_entry)
                layout.addLayout(response_date_layout)
                # Hide initially
                response_date_label.hide()
                response_date_entry.hide()
                response_date_entry.setToolTip(self.tooltips[key])
            elif key == "directory":
                continue
            else:
                hbox = QHBoxLayout()
                label = QLabel(key_labels[key])
                entry = QLineEdit()
                hbox.addWidget(label)
                hbox.addWidget(entry)
                layout.addLayout(hbox)
                form_entries[key] = entry
                entry.setToolTip(self.tooltips[key])

        def toggle_response_checkbox():
            if response_checkbox.isChecked():
                response_date_label.show()
                response_date_entry.show()

            if non_response_checkbox.isChecked():
                response_date_label.hide()
                response_date_entry.hide()

        def save_and_close():
            pt_dict = {}
            patient = form_entries["Patient ID"].text()
            pt_dict[patient] = {}
            for key in self.field_order:
                if key == "Patient ID":
                    continue
                if key == "response_status":
                    pt_dict[patient][key] = 1 if response_checkbox.isChecked() else 0
                elif key == "response_date":
                    if response_checkbox.isChecked():
                        pt_dict[patient][key] = response_date_entry.text()
                elif key == "directory":
                    pt_dict[patient][key] = gui_utils.select_folder()
                else:
                    if form_entries[key].text() == "":
                        continue
                    pt_dict[patient][key] = form_entries[key].text()

            if not pt_dict[patient] or "directory" not in pt_dict[patient].keys():
                QMessageBox.warning(
                    dialog, "Validation Error", "Patient ID and Directory are required."
                )
                return

            if not Path(pt_dict[patient]["directory"]).is_dir():
                QMessageBox.warning(
                    dialog, "Validation Error", "Path is not a valid directory"
                )
                return

            patients = self.load_patient_data()

            if len(patients) > 0 and patient in patients.keys():
                QMessageBox.warning(
                    dialog,
                    "Validation Error",
                    "Patient ID is already in the app database",
                )
                return

            if not gui_utils.validate_date(pt_dict[patient]["dbs_date"]):
                QMessageBox.warning(
                    dialog,
                    "Validation Error",
                    "DBS activation date is required in YYYY-MM-DD format.",
                )
                return

            if response_checkbox.isChecked() and not gui_utils.validate_date(
                pt_dict[patient]["response_date"]
            ):
                try:
                    pt_dict[patient]["response_date"] = int(
                        pt_dict[patient]["response_date"]
                    )
                except Exception:
                    QMessageBox.warning(
                        dialog,
                        "Validation Error",
                        "Response date is required in YYYY-MM-DD format if patient is a responder.",
                    )
                    return

            patients.update(pt_dict)

            with open(get_data_path("data\\patient_info.json"), "w") as f:
                json.dump(patients, f, indent=4)

            dialog.accept()
            self.refresh_table()
            dialog.hide()

        response_checkbox.stateChanged.connect(toggle_response_checkbox)
        non_response_checkbox.stateChanged.connect(toggle_response_checkbox)

        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(save_and_close)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)

        dialog.exec()

    def delete_patient(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Delete Patient")
        layout = QVBoxLayout(dialog)

        if len(self.load_patient_data()) == 0:
            QMessageBox.warning(
                dialog, "Validation Error", "No patients in the database to delete."
            )
            return

        layout.addWidget(QLabel("Enter Patient ID to delete:"))
        patient_id_entry = QLineEdit()
        layout.addWidget(patient_id_entry)

        def delete_and_close():
            patient_id = patient_id_entry.text().strip()
            if not patient_id:
                QMessageBox.warning(dialog, "Input Error", "Please enter a Patient ID.")
                return

            patients = self.load_patient_data()

            if patient_id not in patients.keys():
                QMessageBox.warning(
                    dialog, "Not Found", f"No patient found with ID: {patient_id}"
                )
                return

            del patients[patient_id]
            with open(get_data_path("data\\patient_info.json"), "w") as f:
                json.dump(patients, f, indent=4)

            dialog.accept()
            self.refresh_table()

        button_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(delete_and_close)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(delete_button)

        layout.addLayout(button_layout)

        dialog.exec()

    def init_bottom_buttons(self):
        button_layout = QHBoxLayout()

        back_button = QPushButton("Back", self)
        back_button.clicked.connect(self.go_back)
        button_layout.addWidget(back_button, alignment=Qt.AlignLeft)

        delete_button = QPushButton("Delete patient", self)
        delete_button.clicked.connect(self.delete_patient)
        button_layout.addWidget(delete_button, alignment=Qt.AlignCenter)

        add_button = QPushButton("Add patient", self)
        add_button.clicked.connect(self.add_patient)
        button_layout.addWidget(add_button, alignment=Qt.AlignRight)

        self.main_layout.addLayout(button_layout)

    def get_tooltips(self):
        return {
            "Patient ID": "Unique patient identifier.",
            "directory": "Directory where patient data is stored wrapped in quotes (Tip: Use CTRL-SHIFT-C on a highlighted folder to copy the path to your clipboard).",
            "dbs_date": "Initial DBS programming date (YYYY-MM-DD format).",
            "response_status": "Responder status (Yes/No).",
            "response_date": "Date the patient became a responder (Enter YYYY-MM-DD format or # of days post-DBS patient achieved response).",
            "disinhibited_dates": "Dates the patient was disinhibited in [start date, end date] format (Enter dates in YYYY-MM-DD format or post-DBS day range patient was disinhibited).",
        }
