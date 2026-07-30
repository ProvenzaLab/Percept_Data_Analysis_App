"""Opening, Help, and Doc screens.

The original ``src/opening_windows.py`` grouped these three classes
together. Splitting them into per-screen modules keeps each file focused
on a single Qt widget and makes it easier to evolve one screen at a time.
"""

from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QToolBar,
)
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt, QUrl, QSize
from PySide6.QtWebEngineWidgets import QWebEngineView
import os
import json
from utils.utils import get_data_path, resource_path


class OpeningScreen(QWidget):
    """Landing screen with primary actions (Start / Add Patients) and toolbar."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(40, 0, 40, 0)
        self.layout.setSpacing(12)

        self.layout.addStretch()

        self.welcome_label = QLabel("Welcome to the Percept Data Analysis App", self)
        self.welcome_label.setObjectName("titleLabel")
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setWordWrap(True)
        self.layout.addWidget(self.welcome_label)

        self.description_label = QLabel(
            "This application helps you process and analyze Medtronic Percept data.<br>"
            "Please proceed to start the data processing.<br><br>"
            "Developed by the Provenza Lab",
            self,
        )
        self.description_label.setObjectName("subtitleLabel")
        self.description_label.setAlignment(Qt.AlignCenter)
        self.description_label.setWordWrap(True)
        self.layout.addWidget(self.description_label)

        self.proceed_button = QPushButton("Start Data Processing", self)
        self.proceed_button.setObjectName("ctaButton")
        self.proceed_button.clicked.connect(self.proceed)
        self.layout.addWidget(self.proceed_button, alignment=(Qt.AlignHCenter))

        self.patient_menu_button = QPushButton("Add Patients", self)
        self.patient_menu_button.setObjectName("ctaButton")
        self.patient_menu_button.clicked.connect(self.parent.show_patient_menu)
        self.layout.addWidget(self.patient_menu_button, alignment=(Qt.AlignHCenter))

        self.layout.addStretch()

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
            return False
        with open(get_data_path("data\\patient_info.json"), "r") as f:
            try:
                patient_dict = json.load(f)
            except json.JSONDecodeError:
                QMessageBox.warning(
                    self, "Validation Error", "No patient data is stored"
                )
                return False
        if len(patient_dict) == 0:
            QMessageBox.warning(self, "Validation Error", "No patient data is stored")
            return False
        # Patient selection (which patients to actually process) happens in
        # the Patient menu, via its checkboxes + "Process Selected" button.
        self.parent.show_patient_menu()
        return True


class HelpMenu(QWidget):
    """Embedded view of the GitHub-hosted help.md guide."""

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
        self.button_layout.setContentsMargins(12, 8, 12, 12)

        self.back_button = QPushButton("Back", self)
        self.back_button.setObjectName("secondaryButton")
        self.back_button.clicked.connect(self.go_back)
        self.button_layout.addWidget(
            self.back_button, alignment=Qt.AlignLeft | Qt.AlignBottom
        )

        self.layout.addLayout(self.button_layout)


class DocMenu(QWidget):
    """Embedded view of the GitHub-hosted README.md."""

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
        self.button_layout.setContentsMargins(12, 8, 12, 12)

        self.back_button = QPushButton("Back", self)
        self.back_button.setObjectName("secondaryButton")
        self.back_button.clicked.connect(self.go_back)
        self.button_layout.addWidget(
            self.back_button, alignment=Qt.AlignLeft | Qt.AlignBottom
        )

        self.layout.addLayout(self.button_layout)
