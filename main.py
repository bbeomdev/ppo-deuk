import sys
import cv2
import torch
import pygame
from ultralytics import YOLO
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QSpacerItem, QSizePolicy
from PySide6.QtGui import QImage, QPixmap, QMovie, QPainter, QColor, QPen, QBrush, QFont
from PySide6.QtCore import Qt, QEvent, QThread, Signal, QTimer
import os

os.chdir('hand')

class WorkerThread(QThread):
    frame_processed = Signal(object)

    def __init__(self, model, cap):
        super().__init__()
        self.model = model
        self.cap = cap
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                results = self.model(frame, conf=0.6)
                self.frame_processed.emit((frame, results))

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.scale_factor = 1.0
        self.sounds_playing = False

        # 배경색 설정
        self.setStyleSheet("background-color: #E8F4F8;")  # 연파랑색

        self.gif_files = [
            '_internal/hand_img/00.jpg',
            '_internal/hand_img/01.jpg',
            '_internal/hand_img/02.jpg',
            '_internal/hand_img/03.jpg',
            '_internal/hand_img/04.jpg',
            '_internal/hand_img/05.jpg'
        ]

        self.actions = [
            "손바닥을 마주 대고 씻어주세요.",
            "손등과 손바닥을 마주 대고 씻어주세요.",
            "손깍지를 끼고 씻어주세요.",
            "손가락을 마주 잡고 씻어주세요.",
            "엄지손가락을 돌려주며 씻어주세요.",
            "손톱 손바닥에 문지르며 씻어주세요."
        ]
        self.required_sequence = [0, 1, 2, 3, 4, 5]
        self.required_detections = [15] * len(self.required_sequence)
        self.detection_counts = [0] * len(self.required_sequence)
        self.current_step = 0
        self.initial_detection_done = False

        pygame.mixer.init()
        self.sounds = [pygame.mixer.Sound(f'_internal/hand_sound/{i}.mp3') for i in range(8)]

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        self.progress_bar, self.progress_label, self.signal_labels, self.camera_label, self.gif_label = self.setup_ui(main_widget, self.scale_factor)

        self.cap = cv2.VideoCapture(0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO('_internal/best.pt')
        self.model.classes = [0, 1, 2, 3, 4, 5]

        self.worker = WorkerThread(self.model, self.cap)
        self.worker.frame_processed.connect(self.process_frame)
        self.worker.start()

        self.update_signals(0)
        self.load_gif(self.gif_files[0])

    def setup_ui(self, main_widget, scale_factor=1.0):
        main_layout = QVBoxLayout(main_widget)

        main_layout.addSpacerItem(QSpacerItem(20, 50, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed))

        self.title_label = QLabel("안녕하세요")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 60px; font-weight: bold;")

        top_layout = QVBoxLayout()
        top_layout.addWidget(self.title_label)
        top_layout.addSpacing(0)

        progress_signal_layout = QHBoxLayout()

        left_spacer = QSpacerItem(80, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        right_spacer = QSpacerItem(0, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        progress_signal_layout.addItem(left_spacer)

        progress_bar = QProgressBar()
        progress_bar.setTextVisible(False)
        progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 5px;
                background-color: #ffffff;  # 배경색 흰색
                text-align: center;
            }
            QProgressBar::chunk {
                width: 20px;
                margin: 0.5px;
            }
        """)
        progress_bar.setFixedWidth(880)
        progress_bar.setFixedHeight(80)
        progress_signal_layout.addWidget(progress_bar, 1)

        progress_label = QLabel("0%")
        progress_label.setStyleSheet("font-size: 24px;")
        progress_signal_layout.addWidget(progress_label)

        progress_signal_layout.addItem(right_spacer)

        signal_layout = QHBoxLayout()
        signal_layout.setSpacing(5)

        signal_labels = []
        for i in range(6):
            label = QLabel()
            label.setFixedSize(100, 100)
            signal_labels.append(label)
            signal_layout.addWidget(label)

        right_spacer = QSpacerItem(115, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        signal_layout.addItem(right_spacer)

        progress_signal_layout.addLayout(signal_layout)

        top_layout.addLayout(progress_signal_layout)
        top_layout.addSpacing(0)

        camera_label = QLabel()
        camera_label.setFixedSize(950, 700)
        gif_label = QLabel()
        gif_label.setFixedSize(700, 700)
        gif_label.setScaledContents(True)

        video_gif_layout = QHBoxLayout()
        video_gif_layout.addWidget(camera_label)
        video_gif_layout.addWidget(gif_label)

        main_layout.addLayout(top_layout)
        main_layout.addLayout(video_gif_layout)

        main_layout.addSpacerItem(QSpacerItem(20, 30, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed))

        return progress_bar, progress_label, signal_labels, camera_label, gif_label

    def keyPressEvent(self, event: QEvent):
        if event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
        super().keyPressEvent(event)

    def process_frame(self, data):
        frame, results = data

        if not self.initial_detection_done:
            # 초기 감지 여부를 확인
            detected = any(
                cls in self.required_sequence
                for result in results
                for box in result.boxes
                if (cls := int(box.cls.item()))
            )

            if detected:
                self.initial_detection_done = True
                self.title_label.setText("안녕하십니까! 반갑습니다! 손을 씻어주세요.")
                self.play_sound(6)  # "안녕하십니까! 반갑습니다! 손을 씻어주세요." 사운드
                QTimer.singleShot(4000, self.start_detection_process)
        else:
            if not self.sounds_playing:
                # 현재 단계의 요구된 동작이 감지되었는지 확인
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cls = int(box.cls.item())
                        label = result.names[cls] if cls < len(result.names) else "Unknown"
                        confidence = box.conf.item()

                        # 현재 단계의 동작 감지
                        if self.current_step < len(self.required_sequence):
                            if cls == self.required_sequence[self.current_step]:
                                if self.current_step == 0 and self.title_label.text() == "안녕하십니까! 반갑습니다! 손을 씻어주세요.":
                                    # 첫 번째 단계에서는 카운트를 업데이트 하지 않음
                                    continue
                                else:
                                    # 카운트 증가
                                    self.detection_counts[self.current_step] += 1
                                    self.update_progress(self.detection_counts[self.current_step])
                                    if self.detection_counts[self.current_step] >= self.required_detections[self.current_step]:
                                        self.current_step += 1
                                        if self.current_step < len(self.required_sequence):
                                            self.start_detection_process()
                                        else:
                                            self.show_final_message()
                                        break

        # 화면 업데이트
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.camera_label.setPixmap(scaled_pixmap)

    def start_detection_process(self):
        self.title_label.setText(self.actions[self.current_step])
        self.play_sound(self.current_step)
        self.sounds_playing = True
        self.update_progress(self.detection_counts[self.current_step])
        self.progress_bar.setMaximum(self.required_detections[self.current_step])
        self.update_signals(self.detection_counts[self.current_step])
        self.update_gif()

        QTimer.singleShot(self.sounds[self.current_step].get_length() * 1000, self.sound_finished)

    def sound_finished(self):
        self.sounds_playing = False

    def play_sound(self, index):
        if 0 <= index < len(self.sounds):
            self.sounds[index].play()

    def update_gif(self):
        if self.current_step < len(self.gif_files):
            gif_path = self.gif_files[self.current_step]
        else:
            gif_path = self.gif_files[0]

        self.load_gif(gif_path)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        if self.required_detections[self.current_step] > 0:
            percentage = value * 100 // self.required_detections[self.current_step]
        else:
            percentage = 100
        self.progress_label.setText(f"{percentage}%")

        # 진행률에 따른 색상 변화
        if percentage <= 50:
            chunk_color = "#CC0505"  # 빨간색
        elif percentage <= 80:
            chunk_color = "#B8CC05"  # 노란색
        else:
            chunk_color = "#05B8CC"  # 파란색

        # 진행바의 스타일시트를 동적으로 업데이트
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;  /* 연한 회색 테두리 */
                border-radius: 5px;
                background-color: #ffffff;  /* 배경색 흰색 */
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {chunk_color};
                width: 20px;
                margin: 0.5px;
            }}
        """)
        
        self.update_signals(percentage)

    def update_signals(self, value):
        pastel_green = QColor('#05B8CC')  # 파스텔톤 초록색 (Light Green)
        pastel_blue = QColor('#B8CC05')   # 파스텔톤 파란색 (Light Blue)

        for i, label in enumerate(self.signal_labels):
            if i < self.current_step:
                self.draw_signal(label, pastel_green, i + 1)
            elif i == self.current_step:
                self.draw_signal(label, pastel_blue, i + 1)
            else:
                self.draw_signal(label, QColor("gray"), i + 1)

    def draw_signal(self, label, color, number):
        # 윤곽선을 제거하고 색상을 파스텔톤으로 변경, 글자색은 흰색으로 설정
        pixmap = QPixmap(100, 100)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        
        # 파스텔톤의 색상과 약간의 형광 효과를 위한 붓 설정
        brush_color = color.lighter(120)  # 색상을 조금 더 밝게
        painter.setBrush(QBrush(brush_color))
        
        # 원 그리기
        painter.setPen(Qt.NoPen)  # 윤곽선 제거
        painter.drawEllipse(10, 10, 80, 80)
        
        # 숫자 텍스트 설정
        painter.setPen(QPen(Qt.black))  # 글자색을 흰색으로 설정
        painter.setFont(QFont("Arial", 20, QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, str(number))
        
        painter.end()
        label.setPixmap(pixmap)

    def reset(self):
        self.current_step = 0
        self.detection_counts = [0] * len(self.required_sequence)
        self.initial_detection_done = False
        self.update_progress(0)
        self.progress_bar.setMaximum(self.required_detections[self.current_step])
        self.title_label.setText("안녕하세요")
        self.update_signals(0)

        self.update_gif()

    def show_final_message(self):
        self.title_label.setText("감사합니다. 입장 가능하십니다.")
        self.play_sound(7)
        self.update_signals(100)
        QTimer.singleShot(4000, self.reset)

    def load_gif(self, gif_path):
        self.movie = QMovie(gif_path)
        self.gif_label.setMovie(self.movie)
        self.movie.frameChanged.connect(self.resize_gif)
        self.movie.start()

    def resize_gif(self):
        pixmap = self.movie.currentPixmap()
        scaled_pixmap = pixmap.scaled(self.gif_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.gif_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.worker.stop()
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())
