import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO

class CameraYolo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Recognize an object with camera - YOLOv8 + PyQt5")
        self.setGeometry(100, 100, 900, 700)

        self.model = YOLO("best.pt")


        self.class_colors = {
            'car': (0, 252, 199),
            'chair': (255, 34, 134),
            'desk': (86, 0, 254),
            'door': (206, 255, 0),
            'notebook': (0, 128, 255),
            'pen': (235, 183, 0),
            'window': (0, 255, 255)
        }

        self.image_label = QLabel()
        self.image_label.setFixedSize(800, 600)

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.image_label.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        results = self.model(frame)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = self.model.names[class_id]
            confidence = float(box.conf[0])

            color = self.class_colors.get(label, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(), self.image_label.height()))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraYolo()
    window.show()
    sys.exit(app.exec_())

