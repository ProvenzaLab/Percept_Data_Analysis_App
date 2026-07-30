"""Application entry point.

Launches the ``MainWindow`` on top of a ``QApplication``. On macOS and
in dev-mode on any platform, run this directly with ``python app.py``.
On Windows, prefer ``app_win.py`` so the OS associates this process
with the correct AppUserModelID.
"""

import sys
import multiprocessing

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from src.ui.main_window import MainWindow


def main():
    multiprocessing.set_start_method("spawn")
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icons/Icon.ico"))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
