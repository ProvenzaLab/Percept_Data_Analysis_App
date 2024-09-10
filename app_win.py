import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QTextEdit, QProgressBar, QMessageBox, QCheckBox, QComboBox
)
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt, QUrl, QTimer
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
import generate_data
import calc_circadian
import plotting_utils as plots
import gui_utils
import multiprocessing
import os


try:
    from ctypes import windll
    myappid = 'Provenza_Labs.Percept_Data_Analysis App.v1.0'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
LOADING_SCREEN_INTERVAL = 100  # in milliseconds

def worker_function(param_dict, file_list, result_queue):
    try:
        percept_data, zone_index = generate_data.generate_data(
            subject_name=param_dict['subject_name'],
            param=param_dict,
            file_list=file_list
        )

        percept_data = calc_circadian.calc_circadian(
            percept_data=percept_data,
            zone_index=zone_index,
            cosinor_window_left=int(param_dict['cosinor_window_left']),
            cosinor_window_right=int(param_dict['cosinor_window_right']),
            include_nonlinear=param_dict['include_nonlinear']
        )
        result_queue.put((percept_data, zone_index))

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

        self.frame1 = Frame1(self)
        self.layout.addWidget(self.frame1)
        self.frame1.hide()

        self.loading_screen = LoadingScreen(self)
        self.layout.addWidget(self.loading_screen)
        self.loading_screen.hide()

        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet("""
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
        """)

    def show_frame1(self):
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.opening_screen.hide()
        self.frame1.show()

    def show_loading_screen(self, param_dict):
        self.loading_screen.show()
        self.loading_screen.progress_bar.setRange(0, 0)
        self.frame1.hide()

        file_list = gui_utils.open_file_dialog(self)

        if not file_list:
            self.on_script_finished(None, None)
            return

        self.param_dict = gui_utils.translate_param_dict(param_dict)
        self.worker_process = multiprocessing.Process(
            target=worker_function,
            args=(self.param_dict, file_list, self.result_queue)
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
                    percept_data, zone_index = result
                    self.worker_process.terminate()
                    self.on_script_finished(percept_data, zone_index)
        except Exception as e:
            print(f"Error checking results: {e}")

    def on_script_finished(self, percept_data=None, zone_index=None):
        self.loading_screen.hide()
        if percept_data and zone_index:
            self.show_frame2(percept_data, zone_index)
        else:
            QMessageBox.warning(self, "Error", "Failed to process the data. Please try again.")
            self.show_frame1()

    def show_frame2(self, percept_data, zone_index):
        self.setGeometry(100, 100, 1200, 800)
        self.frame2 = Frame2(self, self.param_dict, percept_data, zone_index)
        self.layout.addWidget(self.frame2)
        self.frame1.hide()
        self.frame2.show()

class OpeningScreen(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)

        self.welcome_label = QLabel("Welcome to the Percept Data Analysis App", self)
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setStyleSheet("""
            QLabel {
                font-size: 25px;
                font-family: 'Arial', sans-serif;
                color: #ffffff;
                padding: 10px;
            }
        """)
        self.layout.addWidget(self.welcome_label)

        self.description_label = QLabel(
            'This application helps you process and analyze Medtronic percept data.<br>'
            'Please proceed to start the data processing.<br><br>'
            '<a href="https://github.com/ProvenzaLab/Percept_Data_Analysis_App/blob/main/README.md#user-manual" style="color: #1e90ff;">Click this link for documentation on the app</a><br><br>'
            'Developed by the Provenza Lab', self)
        self.description_label.setAlignment(Qt.AlignCenter)
        self.description_label.setOpenExternalLinks(True)
        self.description_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-family: 'Arial', sans-serif;
                color: #ffffff;
                padding: 20px;
            }
        """)
        self.layout.addWidget(self.description_label)

        self.proceed_button = QPushButton("Start Data Processing", self)
        self.proceed_button.setStyleSheet("""
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
        """)
        self.proceed_button.clicked.connect(self.proceed)
        self.layout.addWidget(self.proceed_button, alignment=Qt.AlignCenter)

    def proceed(self):
        self.parent.show_frame1()

class LoadingScreen(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.label = QLabel(
            "The application is processing your data.\nPlease wait a moment, this may take a couple of minutes.\nDo not close or restart the application.", 
            self
        )
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-family: 'Arial', sans-serif;
                color: #ffffff;
                padding: 20px;
            }
        """)
        self.layout.addWidget(self.label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background-color: #1e90ff;
                width: 20px;
            }
        """)
        self.layout.addWidget(self.progress_bar)

        self.setStyleSheet("""
            background-color: #2d2d2d;
        """)

        self.setLayout(self.layout)

class Frame2(QWidget):
    def __init__(self, parent, param_dict, percept_data, zone_index):
        super().__init__(parent)
        self.parent = parent
        self.param_dict = param_dict
        self.percept_data = percept_data
        self.zone_index = zone_index
        self.current_plot = None
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.content_layout = QHBoxLayout()

        self.init_json_frame()
        self.init_plot_frame()

        self.layout.addLayout(self.content_layout)
        self.init_bottom_buttons()
        self.setLayout(self.layout)
        self.update_plot()

    def init_json_frame(self):
        self.json_fields_frame = QWidget(self)
        self.json_layout = QVBoxLayout(self.json_fields_frame)
        self.json_text = QTextEdit(self.json_fields_frame)
        self.json_text.setReadOnly(True)
        self.json_text.setStyleSheet("""
            background-color: #4d4d4d;
            color: #f5f5f5;
            border: 1px solid #555;
            border-radius: 5px;
            padding: 10px;
            font-size: 14px;
            font-family: 'Roboto', sans-serif;
        """)
        self.json_layout.addWidget(self.json_text)

        self.populate_json_fields()

        self.export_button = QPushButton("Export LinAR R² feature", self)
        self.export_button.clicked.connect(self.export_data)
        self.json_layout.addWidget(self.export_button, alignment=Qt.AlignCenter | Qt.AlignBottom)

        self.json_fields_frame.setLayout(self.json_layout)
        self.content_layout.addWidget(self.json_fields_frame, 2)

    def populate_json_fields(self):
        self.json_text.append(f"Subject_name: {self.param_dict['subject_name']}\n")
        self.json_text.append(f"Initial_DBS_programming_date: {self.param_dict['dbs_date']}\n")
        self.json_text.append(f"Pre_DBS_example_days: {self.param_dict['pre_DBS_example_days']}\n")
        self.json_text.append(f"Post_DBS_example_days: {self.param_dict['post_DBS_example_days']}\n")
        if(len(self.param_dict['responder_zone_idx']) > 0):
            self.json_text.append(f"Responder_date: {self.param_dict['responder_date']}\n")
        else:
            self.json_text.append(f"Responder: {False}\n")

    def init_plot_frame(self):
        self.web_view = QWebEngineView(self)
        self.web_view.setFixedSize(900, 650)
        self.configure_web_view()

        self.hemisphere_selector_layout = QHBoxLayout()
        self.hemisphere_selector_layout.addStretch()
        self.init_hemisphere_selector()
        self.hemisphere_selector_layout.addWidget(self.hemisphere_selector)
        self.hemisphere_selector_layout.addSpacing(10)

        self.plot_layout = QVBoxLayout()
        self.plot_layout.addLayout(self.hemisphere_selector_layout)
        self.plot_layout.addWidget(self.web_view)

        self.content_layout.addLayout(self.plot_layout, 8)

    def configure_web_view(self):
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)

    def init_hemisphere_selector(self):
        self.hemisphere_selector = QComboBox(self)
        self.hemisphere_selector.addItems(["Left Hemisphere", "Right Hemisphere"])
        self.hemisphere_selector.setCurrentIndex(self.param_dict['hemisphere'])
        self.hemisphere_selector.currentIndexChanged.connect(self.on_hemisphere_change)

    def init_bottom_buttons(self):
        self.button_layout = QHBoxLayout()

        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back)
        self.button_layout.addWidget(self.back_button, alignment=Qt.AlignLeft | Qt.AlignBottom)

        self.download_button = QPushButton("Download plot", self)
        self.download_button.clicked.connect(self.download_image)
        self.button_layout.addWidget(self.download_button, alignment=Qt.AlignRight | Qt.AlignBottom)

        self.layout.addLayout(self.button_layout)

    def on_hemisphere_change(self, index):
        self.param_dict['hemisphere'] = index
        self.update_plot()

    def update_plot(self):
        fig = plots.plot_metrics(
            percept_data=self.percept_data,
            subject=self.param_dict['subject_name'],
            hemisphere=self.param_dict['hemisphere'],
            pre_DBS_bounds=self.param_dict['pre_DBS_example_days'],
            post_DBS_bounds=self.param_dict['post_DBS_example_days'],
            zone_index=self.zone_index
        )

        self.current_plot = fig

        temp_file_path = gui_utils.create_temp_plot(fig)

        self.web_view.setUrl(QUrl.fromLocalFile(temp_file_path))

    def go_back(self):
        self.hide()
        self.parent.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.parent.frame1.show()

    def download_image(self):
        file_path = gui_utils.open_save_dialog(self, "Save Image", "")
        if file_path and self.current_plot:
            gui_utils.save_plot(self.current_plot, file_path)
        else:
            QMessageBox.warning(self, "Error", "No plot is available to save.")

    def export_data(self):
        file_path = gui_utils.open_save_dialog(self, "Save Data", "")
        if file_path:
            lin_ar_df = gui_utils.prepare_export_data(self.percept_data, self.param_dict)
            gui_utils.save_lin_ar_feature(lin_ar_df, file_path)

class Frame1(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.fields = self.get_initial_fields()
        self.field_order = [
            "subject_name",
            "Initial_DBS_programming_date",
            "pre_DBS_example_days",
            "post_DBS_example_days"
        ]
        self.tooltips = self.get_tooltips()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(5)
        self.entries = {}

        self.create_field_entries()
        self.create_responder_checkbox()
        self.create_responder_date_entry()

        self.save_button = QPushButton("Save and Continue", self)
        self.layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_data)

    def get_initial_fields(self):
        return {
            "subject_name": "009",
            "Initial_DBS_programming_date": "12-22-2023",
            "pre_DBS_example_days": ["12-10-2023", "12-13-2023"],
            "post_DBS_example_days": ["03-16-2024", "03-18-2024"],
            "responder": False,
            "responder_date": ""
        }

    def get_tooltips(self):
        return {
            "subject_name": "Enter the subject's name or ID.",
            "Initial_DBS_programming_date": "Enter the date of initial DBS programming in MM-DD-YYYY format.",
            "pre_DBS_example_days": "Enter two example days before DBS in MM-DD-YYYY format.",
            "post_DBS_example_days": "Enter two example days after DBS in MM-DD-YYYY format.",
            "responder": "Select if the subject is a responder to the treatment.",
            "responder_date": "Enter the date when the subject became a responder in MM-DD-YYYY format."
        }

    def create_field_entries(self):
        for key in self.field_order:
            value = self.fields[key]
            hbox = QHBoxLayout()
            label = QLabel(self.format_field_name(key), self)
            hbox.addWidget(label)

            if isinstance(value, list):
                self.create_list_field_entries(key, value, hbox)
            else:
                entry = QLineEdit(self)
                entry.setText(str(value))
                entry.setToolTip(self.tooltips[key])
                hbox.addWidget(entry)
                self.entries[key] = entry

            self.layout.addLayout(hbox)

    def create_list_field_entries(self, key, value, layout):
        entry1 = QLineEdit(self)
        entry1.setText(str(value[0]))
        entry1.setToolTip(self.tooltips[key])
        layout.addWidget(entry1)
        entry2 = QLineEdit(self)
        entry2.setText(str(value[1]))
        entry2.setToolTip(self.tooltips[key])
        layout.addWidget(entry2)
        self.entries[key] = (entry1, entry2)

    def create_responder_checkbox(self):
        self.responder_label = QLabel("Responder ", self)
        self.responder_yes_checkbox = QCheckBox("Yes", self)
        self.responder_no_checkbox = QCheckBox("No", self)
        self.responder_yes_checkbox.setToolTip(self.tooltips["responder"])
        self.responder_no_checkbox.setToolTip(self.tooltips["responder"])

        responder_layout = QHBoxLayout()
        responder_layout.addWidget(self.responder_label)
        responder_layout.addWidget(self.responder_yes_checkbox)
        responder_layout.addSpacing(15)
        responder_layout.addWidget(self.responder_no_checkbox)
        responder_layout.addStretch()

        self.layout.addLayout(responder_layout)

        self.responder_yes_checkbox.stateChanged.connect(self.on_responder_checkbox_changed)
        self.responder_no_checkbox.stateChanged.connect(self.on_responder_checkbox_changed)

    def create_responder_date_entry(self):
        self.responder_date_label = QLabel("Responder Date ", self)
        self.responder_date_entry = QLineEdit(self)
        self.responder_date_entry.setToolTip(self.tooltips["responder_date"])

        self.responder_date_layout = QHBoxLayout()
        self.responder_date_layout.addWidget(self.responder_date_label)
        self.responder_date_layout.addWidget(self.responder_date_entry)

        self.layout.addLayout(self.responder_date_layout)

        self.responder_date_label.hide()
        self.responder_date_entry.hide()

    def format_field_name(self, field_name):
        words = field_name.split('_')
        formatted_words = [word.capitalize() if word.lower() != 'dbs' else 'DBS' for word in words]
        return ' '.join(formatted_words)

    def on_responder_checkbox_changed(self):
        if self.sender() == self.responder_yes_checkbox:
            if self.responder_yes_checkbox.isChecked():
                self.responder_no_checkbox.setChecked(False)
                self.responder_date_label.show()
                self.responder_date_entry.show()
            else:
                self.responder_date_label.hide()
                self.responder_date_entry.hide()

        elif self.sender() == self.responder_no_checkbox:
            if self.responder_no_checkbox.isChecked():
                self.responder_yes_checkbox.setChecked(False)
                self.responder_date_label.hide()
                self.responder_date_entry.hide()

        if not self.responder_yes_checkbox.isChecked() and not self.responder_no_checkbox.isChecked():
            self.responder_date_label.hide()
            self.responder_date_entry.hide()

    def validate_fields(self):
        if not gui_utils.validate_date(self.entries['Initial_DBS_programming_date'].text()):
            QMessageBox.warning(self, "Invalid Input", "Initial DBS programming date must be in the format MM-DD-YYYY")
            return False

        if not self.entries['subject_name'].text():
            QMessageBox.warning(self, "Invalid Input", "Subject name must be filled in")
            return False

        pre_DBS_example_days = [entry.text() for entry in self.entries['pre_DBS_example_days']]
        post_DBS_example_days = [entry.text() for entry in self.entries['post_DBS_example_days']]
        if not all(gui_utils.validate_date(date) for date in pre_DBS_example_days + post_DBS_example_days):
            QMessageBox.warning(self, "Invalid Input", "Example days must be in the format MM-DD-YYYY")
            return False

        if self.responder_yes_checkbox.isChecked() and not gui_utils.validate_date(self.responder_date_entry.text()):
            QMessageBox.warning(self, "Invalid Input", "Responder date must be in the format MM-DD-YYYY")
            return False

        if not (self.responder_yes_checkbox.isChecked() or self.responder_no_checkbox.isChecked()):
            QMessageBox.warning(self, "Invalid Input", "Responder status must be checked. Select 'Yes' for responder or 'No' for non-responder.")
            return False

        return True

    def save_data(self):
        if not self.validate_fields():
            return

        param_dict = {}
        for key, entry in self.entries.items():
            if isinstance(entry, tuple):
                param_dict[key] = [e.text() for e in entry]
            else:
                param_dict[key] = entry.text()

        param_dict["responder"] = self.responder_yes_checkbox.isChecked()
        if self.responder_yes_checkbox.isChecked():
            param_dict["responder_date"] = self.responder_date_entry.text()
        else:
            param_dict["responder_date"] = ""

        self.parent.show_loading_screen(param_dict)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    multiprocessing.freeze_support()
    #basedir = os.path.dirname(__file__)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('Icon.ico'))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
