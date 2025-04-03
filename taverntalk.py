# TavernTalk.py

import sys
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QLabel, QTextEdit, QComboBox, QHBoxLayout, QMessageBox
)
from PyQt5.QtCore import Qt
import re


class TavernTalk(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TavernTalk")
        self.resize(800, 600)

        self.layout = QVBoxLayout()

        self.transcribe_button = QPushButton("Transcribe Audio (Whisper.cpp)")
        self.transcribe_button.clicked.connect(self.transcribe_audio)
        self.layout.addWidget(self.transcribe_button)

        self.load_button = QPushButton("Load Transcript")
        self.load_button.clicked.connect(self.load_transcript)
        self.layout.addWidget(self.load_button)

        self.segment_display = QTextEdit()
        self.segment_display.setReadOnly(True)
        self.layout.addWidget(self.segment_display)

        self.speaker_input_layout = QHBoxLayout()
        self.speaker_dropdown = QComboBox()
        self.speaker_dropdown.setEditable(True)
        self.speaker_dropdown.setInsertPolicy(QComboBox.InsertAtTop)
        self.speaker_input_layout.addWidget(QLabel("Assign speaker to all lines:"))
        self.speaker_input_layout.addWidget(self.speaker_dropdown)
        self.layout.addLayout(self.speaker_input_layout)

        self.tag_button = QPushButton("Tag All Segments")
        self.tag_button.clicked.connect(self.tag_speakers)
        self.layout.addWidget(self.tag_button)

        self.export_button = QPushButton("Export Campaign Log")
        self.export_button.clicked.connect(self.export_log)
        self.layout.addWidget(self.export_button)

        self.setLayout(self.layout)

        self.segments = []

    def transcribe_audio(self):
        audio_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.mp3 *.wav *.m4a)")
        if not audio_path:
            return

        model_path = "ggml-base.en.bin"  # Adjust if needed
        whisper_exe = "./whispercpp/main"  # Adjust path to whisper.cpp executable

        if not Path(whisper_exe).exists():
            QMessageBox.critical(self, "Whisper.cpp Not Found", f"Expected executable at {whisper_exe}.")
            return

        try:
            subprocess.run([
                whisper_exe,
                "-m", model_path,
                "-f", audio_path,
                "-otxt"
            ], check=True)

            txt_output = Path(audio_path).with_suffix(".txt")
            if txt_output.exists():
                with open(txt_output, 'r', encoding='utf-8') as f:
                    self.segments = [line.strip() for line in f.readlines() if line.strip()]
                self.segment_display.setPlainText("\n".join(self.segments))
            else:
                QMessageBox.warning(self, "No Output", "Transcription finished, but no .txt file found.")
        except subprocess.CalledProcessError:
            QMessageBox.critical(self, "Error", "Whisper.cpp transcription failed.")

    def load_transcript(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Transcript", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.segments = [line.strip() for line in f.readlines() if line.strip()]
            self.segment_display.setPlainText("\n".join(self.segments))

    def tag_speakers(self):
        speaker = self.speaker_dropdown.currentText().strip()
        if not speaker:
            QMessageBox.warning(self, "No Speaker", "Please enter or select a speaker name.")
            return

        tagged = []
        for line in self.segments:
            match = re.match(r"\[(.*?)\] (.*)", line)
            if match:
                timestamp, text = match.groups()
                tagged.append(f"**{speaker}:** {text}")
        self.segment_display.setPlainText("\n\n".join(tagged))
        self.segments = tagged

    def export_log(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Campaign Log", "", "Markdown Files (*.md);;Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(self.segments))
            QMessageBox.information(self, "Exported", f"Campaign log saved to {file_path}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TavernTalk()
    window.show()
    sys.exit(app.exec_())

