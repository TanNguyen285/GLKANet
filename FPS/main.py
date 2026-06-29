"""
Entry point — chạy: python main.py
"""
import sys

# Import core.deps TRƯỚC PyQt5 để tránh DLL conflict trên Windows
from data.core.deps import ensure   # noqa: F401 (side-effects: cài psutil + torch)

from PyQt5.QtWidgets import QApplication

from data.ui.theme       import STYLESHEET, apply_dark_palette
from data.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Model Profiler")
    apply_dark_palette(app)
    app.setStyleSheet(STYLESHEET)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
