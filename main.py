"""
Prediction of compensation values of legal judgments frmo JEC.

@author Thiago Raulino Dal Pont
@date Aug 06, 2021
"""

import logging

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout, QApplication


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename='jec_prediction.log',
                        level=logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def main():
    setup_logging()

    logging.debug("-" * 50)
    logging.debug("PREDICTION OF COMPENSATION VALUES FROM JEC")

    logging.debug("Starting UI")


if __name__ == "__main__":
    main()
