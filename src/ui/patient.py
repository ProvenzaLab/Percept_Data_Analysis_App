"""Patient database UI (add / delete / list patients)."""

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
    QComboBox,
)

from PySide6.QtCore import Qt
import os
import json
from utils.utils import get_data_path
import utils.gui_utils as gui_utils
from pathlib import Path


class PatientMenu(QWidget):
    """Side panel listing patients stored in ``data/patient_info.json``.

    Provides Add, Edit (double-click a row), and Delete dialogs, plus
    per-patient checkboxes and a "Process Selected" button used to choose
    which patients to run the analysis pipeline on.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.field_order = [
            "Patient ID",
            "directory",
            "dbs_date",
            "response_status",
            "response_date",
        ]
        self.tooltips = self.get_tooltips()
        self.select_checkboxes = {}
        self.row_to_patient = []
        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(14)

        self.heading = QLabel("Patients", self)
        self.heading.setObjectName("titleLabel")
        self.main_layout.addWidget(self.heading)

        self.table_layout = QVBoxLayout()
        self.main_layout.addLayout(self.table_layout)

        self.load_patients_table()
        self.init_bottom_buttons()

        self.setLayout(self.main_layout)

    def load_patients_table(self):
        self.table = QTableWidget(self)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.cellDoubleClicked.connect(self.on_row_double_clicked)

        self.select_checkboxes = {}
        self.row_to_patient = []

        patients = self.load_patient_data()
        self.table.setRowCount(len(patients))
        if len(patients) == 0:
            self.table.setColumnCount(1)
            self.table.setHorizontalHeaderLabels(["Patient ID"])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.table_layout.addWidget(self.table)
            return

        processed_ids = self.parent.processed_patient_ids

        display_fields = ["Select", "Patient ID", "Directory", "Response Status", "Status"]
        display_keys = {"Directory": "directory", "Response Status": "response_status"}
        display_response = {0: "Non-responder", 1: "Responder"}
        self.table.setColumnCount(len(display_fields))
        self.table.setHorizontalHeaderLabels(display_fields)
        self.row_to_patient = list(patients.keys())

        for row, patient in enumerate(patients.keys()):
            for col, key in enumerate(display_fields):
                if key == "Select":
                    checkbox = QCheckBox()
                    checkbox.setChecked(True)
                    self.select_checkboxes[patient] = checkbox
                    cell_widget = QWidget()
                    cell_layout = QHBoxLayout(cell_widget)
                    cell_layout.addWidget(checkbox)
                    cell_layout.setAlignment(Qt.AlignCenter)
                    cell_layout.setContentsMargins(0, 0, 0, 0)
                    self.table.setCellWidget(row, col, cell_widget)
                elif key == "Patient ID":
                    self.table.setItem(row, col, QTableWidgetItem(patient))
                elif key == "Response Status":
                    response_status = display_response[
                        patients[patient][display_keys[key]]
                    ]
                    self.table.setItem(row, col, QTableWidgetItem(response_status))
                elif key == "Status":
                    status = "Processed" if patient in processed_ids else "Not yet processed"
                    self.table.setItem(row, col, QTableWidgetItem(status))
                else:
                    self.table.setItem(
                        row, col, QTableWidgetItem(patients[patient][display_keys[key]])
                    )

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(
            display_fields.index("Status"), QHeaderView.ResizeToContents
        )
        self.table.setColumnWidth(0, 56)
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
        self._open_patient_dialog()

    def on_row_double_clicked(self, row, column):
        if row < 0 or row >= len(self.row_to_patient):
            return
        self._open_patient_dialog(existing_id=self.row_to_patient[row])

    def _open_patient_dialog(self, existing_id=None):
        """Shared Add/Edit Patient dialog.

        With ``existing_id=None`` this behaves as "Add Patient" (empty
        fields, editable Patient ID). With ``existing_id`` set, fields are
        pre-filled from that patient's stored data, the Patient ID is
        locked (renaming a patient's key is out of scope), and saving
        overwrites that patient's entry instead of requiring a new,
        unused ID.
        """
        editing = existing_id is not None
        existing = self.load_patient_data().get(existing_id, {}) if editing else {}

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Patient" if editing else "Add Patient")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        form_entries = {}
        directory_value = {"path": existing.get("directory", "")}

        key_labels = {
            "Patient ID": "Patient ID",
            "directory": "Directory",
            "dbs_date": "DBS Date",
            "response_status": "Response Status",
            "response_date": "Response Date",
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

                if existing.get("response_status") == 1:
                    response_checkbox.setChecked(True)
                else:
                    non_response_checkbox.setChecked(True)
            elif key == "response_date":
                response_date_layout = QHBoxLayout()
                response_date_label = QLabel(key_labels[key])
                response_date_entry = QLineEdit()
                response_date_layout.addWidget(response_date_label)
                response_date_layout.addWidget(response_date_entry)
                layout.addLayout(response_date_layout)
                if existing.get("response_status") == 1:
                    response_date_entry.setText(str(existing.get("response_date", "")))
                else:
                    response_date_label.hide()
                    response_date_entry.hide()
                response_date_entry.setToolTip(self.tooltips[key])
            elif key == "directory":
                hbox = QHBoxLayout()
                label = QLabel(key_labels[key])
                directory_entry = QLineEdit(directory_value["path"])
                directory_entry.setReadOnly(True)
                directory_entry.setToolTip(self.tooltips[key])
                browse_button = QPushButton("Browse...")
                browse_button.setObjectName("secondaryButton")

                def browse_for_directory():
                    folder = gui_utils.select_folder()
                    if folder:
                        directory_value["path"] = folder
                        directory_entry.setText(folder)

                browse_button.clicked.connect(browse_for_directory)
                hbox.addWidget(label)
                hbox.addWidget(directory_entry)
                hbox.addWidget(browse_button)
                layout.addLayout(hbox)
            elif key == "Patient ID":
                hbox = QHBoxLayout()
                label = QLabel(key_labels[key])
                entry = QLineEdit(existing_id if editing else "")
                entry.setReadOnly(editing)
                hbox.addWidget(label)
                hbox.addWidget(entry)
                layout.addLayout(hbox)
                form_entries[key] = entry
                entry.setToolTip(self.tooltips[key])
            else:
                hbox = QHBoxLayout()
                label = QLabel(key_labels[key])
                entry = QLineEdit(str(existing.get(key, "")))
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
            patient = existing_id if editing else form_entries["Patient ID"].text()
            if not patient:
                QMessageBox.warning(dialog, "Validation Error", "Patient ID is required.")
                return

            pt_dict = {patient: {}}
            for key in self.field_order:
                if key == "Patient ID":
                    continue
                if key == "response_status":
                    pt_dict[patient][key] = 1 if response_checkbox.isChecked() else 0
                elif key == "response_date":
                    if response_checkbox.isChecked():
                        pt_dict[patient][key] = response_date_entry.text()
                elif key == "directory":
                    if directory_value["path"]:
                        pt_dict[patient][key] = directory_value["path"]
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

            if not editing and patient in patients.keys():
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
                except (TypeError, ValueError):
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
        cancel_button = QPushButton("Cancel")
        cancel_button.setObjectName("secondaryButton")
        cancel_button.clicked.connect(dialog.reject)
        save_button = QPushButton("Save")
        save_button.clicked.connect(save_and_close)
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)

        dialog.exec()

    def delete_patient(self):
        patients = self.load_patient_data()
        if len(patients) == 0:
            QMessageBox.warning(
                self, "Validation Error", "No patients in the database to delete."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Delete Patient")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Select patient to delete:"))
        patient_combo = QComboBox()
        patient_combo.addItems(list(patients.keys()))
        # Pre-select the row the user had highlighted in the table, if any.
        selected_rows = self.table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            if 0 <= row < len(self.row_to_patient):
                idx = patient_combo.findText(self.row_to_patient[row])
                if idx >= 0:
                    patient_combo.setCurrentIndex(idx)
        layout.addWidget(patient_combo)

        def delete_and_close():
            patient_id = patient_combo.currentText()
            if not patient_id:
                QMessageBox.warning(dialog, "Input Error", "Please select a Patient ID.")
                return

            confirm = QMessageBox.question(
                dialog,
                "Confirm Delete",
                f"Delete patient {patient_id}? This cannot be undone.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
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
        cancel_button.setObjectName("secondaryButton")
        cancel_button.clicked.connect(dialog.reject)
        delete_button = QPushButton("Delete")
        delete_button.setObjectName("dangerButton")
        delete_button.clicked.connect(delete_and_close)
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(delete_button)

        layout.addLayout(button_layout)

        dialog.exec()

    def init_bottom_buttons(self):
        selection_layout = QHBoxLayout()

        select_all_button = QPushButton("Select All", self)
        select_all_button.setObjectName("secondaryButton")
        select_all_button.clicked.connect(self.select_all)
        selection_layout.addWidget(select_all_button, alignment=Qt.AlignLeft)

        select_none_button = QPushButton("Select None", self)
        select_none_button.setObjectName("secondaryButton")
        select_none_button.clicked.connect(self.select_none)
        selection_layout.addWidget(select_none_button, alignment=Qt.AlignLeft)

        selection_layout.addStretch()

        process_button = QPushButton("Process Selected", self)
        process_button.clicked.connect(self.process_selected)
        selection_layout.addWidget(process_button, alignment=Qt.AlignRight)

        self.main_layout.addLayout(selection_layout)

        button_layout = QHBoxLayout()

        back_button = QPushButton("Back", self)
        back_button.setObjectName("secondaryButton")
        back_button.clicked.connect(self.go_back)
        button_layout.addWidget(back_button, alignment=Qt.AlignLeft)

        button_layout.addStretch()

        delete_button = QPushButton("Delete patient", self)
        delete_button.setObjectName("dangerButton")
        delete_button.clicked.connect(self.delete_patient)
        button_layout.addWidget(delete_button)

        add_button = QPushButton("Add patient", self)
        add_button.clicked.connect(self.add_patient)
        button_layout.addWidget(add_button)

        self.main_layout.addLayout(button_layout)

    def select_all(self):
        for checkbox in self.select_checkboxes.values():
            checkbox.setChecked(True)

    def select_none(self):
        for checkbox in self.select_checkboxes.values():
            checkbox.setChecked(False)

    def process_selected(self):
        patients = self.load_patient_data()
        selected = {
            pt: patients[pt]
            for pt, checkbox in self.select_checkboxes.items()
            if checkbox.isChecked() and pt in patients
        }
        if not selected:
            QMessageBox.warning(
                self, "No Patients Selected", "Select at least one patient to process."
            )
            return
        self.parent.show_loading_screen(selected)

    def get_tooltips(self):
        return {
            "Patient ID": "Unique patient identifier.",
            "directory": "Directory where patient data is stored wrapped in quotes (Tip: Use CTRL-SHIFT-C on a highlighted folder to copy the path to your clipboard).",
            "dbs_date": "Initial DBS programming date (YYYY-MM-DD format).",
            "response_status": "Responder status (Yes/No).",
            "response_date": "Date the patient became a responder (Enter YYYY-MM-DD format or # of days post-DBS patient achieved response).",
        }
