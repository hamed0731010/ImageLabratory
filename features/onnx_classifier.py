from __future__ import annotations

import json
import shutil
from pathlib import Path

from PyQt6.QtWidgets import (
    QFileDialog, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ONNXClassifier(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.model_edit = QLineEdit()
        self.labels_edit = QLineEdit()
        self.folder_edit = QLineEdit()
        self.size_edit = QLineEdit("auto")
        self.mode = QComboBox()
        self.mode.addItems(["Copy images", "Move images"])
        self.status = QLabel("Choose an ONNX model, labels, and image folder.")
        self.status.setWordWrap(True)

        form = QFormLayout()
        form.addRow("ONNX model:", self.file_row(self.model_edit, "Choose model", self.choose_model))
        form.addRow("Labels:", self.file_row(self.labels_edit, "Choose labels", self.choose_labels))
        form.addRow("Images:", self.file_row(self.folder_edit, "Choose folder", self.choose_folder))
        form.addRow("Input size:", self.size_edit)
        form.addRow("Action:", self.mode)

        convert = QPushButton("Classify all images")
        convert.clicked.connect(self.classify_images)
        box = QGroupBox("ONNX image classifier")
        box.setLayout(form)
        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addWidget(convert)
        layout.addWidget(self.status)
        layout.addStretch()

    @staticmethod
    def file_row(field: QLineEdit, text: str, handler) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(field)
        button = QPushButton(text)
        button.clicked.connect(handler)
        row.addWidget(button)
        return widget

    def choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose ONNX model", "", "ONNX (*.onnx)")
        if path:
            self.model_edit.setText(path)

    def choose_labels(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose labels", "", "Labels (*.json *.txt *.csv);;All files (*)")
        if path:
            self.labels_edit.setText(path)

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose image folder")
        if path:
            self.folder_edit.setText(path)

    def read_labels(self, path: Path) -> list[str]:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(value) for value in data]
            if isinstance(data, dict):
                return [str(data[key]) for key in sorted(data, key=lambda key: int(key) if str(key).isdigit() else str(key))]
            raise ValueError("JSON labels must be a list or an index-to-label object.")
        return [line.split(",", 1)[-1].strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def input_size(self, session) -> tuple[int, int]:
        value = self.size_edit.text().strip().lower()
        if value != "auto":
            parts = [int(part.strip()) for part in value.replace("x", ",").split(",")]
            if len(parts) != 2 or min(parts) <= 0:
                raise ValueError("Input size must be auto or WIDTH,HEIGHT.")
            return parts[0], parts[1]
        shape = session.get_inputs()[0].shape
        height, width = shape[-2], shape[-1]
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError("Model input size is dynamic. Enter it manually, for example 224,224.")
        return width, height

    def classify_images(self) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            from PIL import Image
        except ImportError:
            self.error("Install dependencies first: pip install onnxruntime pillow numpy")
            return
        model = Path(self.model_edit.text().strip())
        labels_file = Path(self.labels_edit.text().strip())
        folder = Path(self.folder_edit.text().strip())
        if not model.is_file() or not labels_file.is_file() or not folder.is_dir():
            self.error("Choose a valid ONNX model, labels file, and image folder.")
            return
        try:
            labels = self.read_labels(labels_file)
            session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
            width, height = self.input_size(session)
            input_meta = session.get_inputs()[0]
            images = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: p.name.lower())
            if not images:
                self.error("No supported images found.")
                return
            for number, source in enumerate(images, 1):
                image = Image.open(source).convert("RGB").resize((width, height))
                array = np.asarray(image, dtype=np.float32) / 255.0
                array = np.transpose(array, (2, 0, 1))[None, ...]
                result = session.run(None, {input_meta.name: array})[0]
                index = int(np.argmax(np.asarray(result).reshape(-1)))
                class_name = labels[index] if index < len(labels) else f"class_{index}"
                safe_name = "".join(char if char not in '<>:"/\\|?*' else "_" for char in class_name).strip() or f"class_{index}"
                destination_dir = folder / safe_name
                destination_dir.mkdir(exist_ok=True)
                destination = destination_dir / source.name
                if destination.exists():
                    destination.unlink()
                if self.mode.currentIndex() == 0:
                    shutil.copy2(source, destination)
                else:
                    shutil.move(str(source), str(destination))
                self.status.setText(f"Processed {number}/{len(images)}: {source.name} → {safe_name}")
            self.status.setText(f"Finished: {len(images)} images classified.")
        except Exception as error:
            self.error(str(error))

    def error(self, message: str) -> None:
        self.status.setText(f"Error: {message}")
        QMessageBox.critical(self, "Classification failed", message)
