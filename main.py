from Controllers.Controller import Controller
from PyQt6.QtWidgets import QApplication
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    con = Controller()
    con.show()
    sys.exit(app.exec())
