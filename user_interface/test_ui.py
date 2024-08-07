import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, \
    QTextEdit
from PyQt5.QtCore import pyqtSignal, QObject, QThread
import requests


class LogFetcher(QObject):
    new_log = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.process = None
        self.running = True

    def run(self):
        try:
            self.process = subprocess.Popen(
                ['docker', 'logs', '-f', 'ec-scheduler'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            while self.running:
                stdout_line = self.process.stdout.readline()
                stderr_line = self.process.stderr.readline()

                if stdout_line:
                    self.new_log.emit(stdout_line.strip())
                if stderr_line:
                    self.new_log.emit(f"Error: {stderr_line.strip()}")
        except Exception as e:
            self.new_log.emit(f"Exception: {e}")

    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.wait()


class DataSenderWorker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, population_size, number_of_islands, number_of_generations, date):
        super().__init__()
        self.population_size = population_size
        self.number_of_islands = number_of_islands
        self.number_of_generations = number_of_generations
        self.date = date

    def run(self):
        try:
            response = requests.post(
                'http://127.0.0.1:8072/run',
                json={
                    "populationSize": [int(self.population_size)],
                    "numberOfIslands": [int(self.number_of_islands)],
                    "numberrOfGeneration": [int(self.number_of_generations)],
                    "date": self.date
                }
            )
            response_json = response.json()
            log_message = f'Status Code: {response.status_code}\nResponse JSON: {response_json}'
            self.finished.emit(log_message)
        except Exception as e:
            log_message = f'Error: {e}'
            self.error.emit(log_message)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Sender and Log Viewer")
        self.setGeometry(100, 100, 600, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.form_layout = QFormLayout()
        self.population_size_input = QLineEdit()
        self.number_of_islands_input = QLineEdit()
        self.number_of_generations_input = QLineEdit()
        self.date_input = QLineEdit()

        self.form_layout.addRow('Population Size:', self.population_size_input)
        self.form_layout.addRow('Number of Islands:', self.number_of_islands_input)
        self.form_layout.addRow('Number of Generations:', self.number_of_generations_input)
        self.form_layout.addRow('Date:', self.date_input)

        self.send_button = QPushButton('Send Data')
        self.send_button.clicked.connect(self.handle_send_data)

        self.layout.addLayout(self.form_layout)
        self.layout.addWidget(self.send_button)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.layout.addWidget(self.log_view)

        # Apply style sheets
        self.apply_styles()

        # Log fetcher setup
        self.log_fetcher = LogFetcher()
        self.log_thread = QThread()
        self.log_fetcher.moveToThread(self.log_thread)
        self.log_fetcher.new_log.connect(self.append_log)
        self.log_thread.started.connect(self.log_fetcher.run)
        self.log_thread.start()

        # Worker and thread for data sending
        self.data_sender_thread = QThread()
        self.data_sender_worker = None

    def apply_styles(self):
        # Set the overall background color
        self.setStyleSheet("""
            QWidget {
                background-color: black;
                color: white;
            }
            QPushButton {
                background-color: gray;
                color: white;
                border: 1px solid white;
            }
            QTextEdit {
                background-color: black;
                color: white;
                border: 1px solid white;
            }
            QLineEdit {
                background-color: black;
                color: white;
                border: 1px solid gray;
            }
        """)

    def handle_send_data(self):
        population_size = self.population_size_input.text()
        number_of_islands = self.number_of_islands_input.text()
        number_of_generations = self.number_of_generations_input.text()
        date = self.date_input.text()

        if self.data_sender_worker:
            self.data_sender_worker.deleteLater()
        self.data_sender_worker = DataSenderWorker(population_size, number_of_islands, number_of_generations, date)
        self.data_sender_worker.moveToThread(self.data_sender_thread)

        self.data_sender_worker.finished.connect(self.on_data_sent)
        self.data_sender_worker.error.connect(self.on_data_error)
        self.data_sender_thread.started.connect(self.data_sender_worker.run)
        self.data_sender_thread.start()

    def on_data_sent(self, log_message):
        self.log_view.append(log_message)
        self.data_sender_thread.quit()
        self.data_sender_thread.wait()

    def on_data_error(self, log_message):
        self.log_view.append(log_message)
        self.data_sender_thread.quit()
        self.data_sender_thread.wait()

    def append_log(self, log_message):
        self.log_view.append(log_message)

    def closeEvent(self, event):
        self.log_fetcher.stop()
        self.log_thread.quit()
        self.log_thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()