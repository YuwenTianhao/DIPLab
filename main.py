import sys
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMainWindow
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from functools import partial
from Lab1_lib.Lab1 import lab1
from Lab2_lib.Lab2 import lab2
from Lab3_lib.Lab3 import lab3
from Lab4_lib.Lab4 import lab4
from Lab5_lib.test import test

class ExperimentWindow(QWidget):
    def __init__(self, experiment_number):
        super().__init__()
        self.lab_func(experiment_number)
    def lab_func(self, experiment_number):
        if experiment_number == 1:
            lab1()
        elif experiment_number == 2:
            lab2()
        elif experiment_number == 3:
            lab3()
        elif experiment_number == 4:
            lab4()
        elif experiment_number == 5:
            test()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数字图像处理实验程序")
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        main_layout = QVBoxLayout()

        # Top label
        title_label = QLabel("数字图像处理实验程序")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Central widget and layout
        central_widget = QWidget()
        central_layout = QHBoxLayout()

        # Left side: Image
        image_label = QLabel()
        pixmap = QPixmap("imgs\lena.png")
        image_label.setPixmap(pixmap)
        central_layout.addWidget(image_label)

        # Right side: Buttons
        buttons_layout = QVBoxLayout()
        self.buttons = []
        for i in range(1, 6):
            button = QPushButton(f"实验 {i}")
            button.clicked.connect(partial(self.open_experiment,i))
            buttons_layout.addWidget(button)
            self.buttons.append(button)
        central_layout.addLayout(buttons_layout)

        central_widget.setLayout(central_layout)
        main_layout.addWidget(central_widget)

        # Set main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def open_experiment(self, experiment_number):
        self.experiment_window = ExperimentWindow(experiment_number)
        self.experiment_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create and show main window
    main_window = MainWindow()
    main_window.show()

    # Run the application event loop
    sys.exit(app.exec())
