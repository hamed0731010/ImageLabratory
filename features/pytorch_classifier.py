from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class PyTorchClassifier(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.model_edit = QLineEdit()
        self.labels_edit = QLineEdit()
        self.folder_edit = QLineEdit()
        self.architecture_edit = QLineEdit()
        self.class_edit = QLineEdit()
        self.size_edit = QLineEdit("224,224")
        self.model_type = QComboBox()
        self.model_type.addItems(["Complete model", "TorchScript", "State dictionary"])
        self.mode = QComboBox()
        self.mode.addItems(["Copy images", "Move images"])
        self.status = QLabel("Choose a PyTorch model, labels, and image folder.")
        self.status.setWordWrap(True)
        form = QFormLayout()
        form.addRow("Model type:", self.model_type)
        form.addRow("PyTorch model:", self.file_row(self.model_edit, "Choose model", self.choose_model))
        form.addRow("Labels:", self.file_row(self.labels_edit, "Choose labels", self.choose_labels))
        form.addRow("Architecture .py:", self.file_row(self.architecture_edit, "Choose file", self.choose_architecture))
        form.addRow("Model class:", self.class_edit)
        form.addRow("Input size:", self.size_edit)
        form.addRow("Images:", self.file_row(self.folder_edit, "Choose folder", self.choose_folder))
        form.addRow("Action:", self.mode)
        button = QPushButton("Classify all images")
        button.clicked.connect(self.classify)
        box = QGroupBox("PyTorch image classifier")
        box.setLayout(form)
        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addWidget(button)
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
        path, _ = QFileDialog.getOpenFileName(self, "Choose PyTorch model", "", "PyTorch (*.pt *.pth)")
        if path: self.model_edit.setText(path)

    def choose_labels(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose labels", "", "Labels (*.json *.txt *.csv);;All files (*)")
        if path: self.labels_edit.setText(path)

    def choose_architecture(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose architecture", "", "Python (*.py)")
        if path: self.architecture_edit.setText(path)

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose image folder")
        if path: self.folder_edit.setText(path)

    def labels(self, path: Path) -> list[str]:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return [str(value) for value in data] if isinstance(data, list) else [str(data[key]) for key in sorted(data, key=str)]
        return [line.split(",", 1)[-1].strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def load_model(self, torch, path: Path):
        if self.model_type.currentText() == "TorchScript":
            return torch.jit.load(str(path), map_location="cpu")
        if self.model_type.currentText() == "Complete model":
            model = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(model, dict):
                raise ValueError("This file is a state dictionary. Select State dictionary.")
            return model
        architecture = Path(self.architecture_edit.text().strip())
        class_name = self.class_edit.text().strip()
        if not architecture.is_file() or not class_name:
            raise ValueError("State dictionary requires an architecture .py file and model class name.")
        spec = importlib.util.spec_from_file_location("user_classifier_architecture", architecture)
        if not spec or not spec.loader: raise ValueError("Could not load architecture file.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        model_class = getattr(module, class_name, None)
        if model_class is None: raise ValueError(f"Class '{class_name}' was not found.")
        model = model_class()
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
        model.load_state_dict(state)
        return model

    def classify(self) -> None:
        try:
            import numpy as np
            import torch
            from PIL import Image
        except ImportError:
            self.error("Install dependencies first: pip install torch pillow numpy")
            return
        try:
            model_path, labels_path, folder = Path(self.model_edit.text()), Path(self.labels_edit.text()), Path(self.folder_edit.text())
            if not model_path.is_file() or not labels_path.is_file() or not folder.is_dir(): raise ValueError("Choose a valid model, labels file, and image folder.")
            width, height = [int(x.strip()) for x in self.size_edit.text().replace("x", ",").split(",")]
            labels = self.labels(labels_path)
            model = self.load_model(torch, model_path).eval()
            images = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: p.name.lower())
            for number, source in enumerate(images, 1):
                image = Image.open(source).convert("RGB").resize((width, height))
                array = np.asarray(image, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).unsqueeze(0)
                with torch.no_grad(): result = model(tensor)
                if isinstance(result, (tuple, list)): result = result[0]
                index = int(torch.argmax(result).item())
                name = labels[index] if index < len(labels) else f"class_{index}"
                safe = "".join(c if c not in '<>:"/\\|?*' else "_" for c in name).strip() or f"class_{index}"
                target = folder / safe / source.name
                target.parent.mkdir(exist_ok=True)
                if target.exists(): target.unlink()
                shutil.copy2(source, target) if self.mode.currentIndex() == 0 else shutil.move(str(source), str(target))
                self.status.setText(f"Processed {number}/{len(images)}: {source.name} → {safe}")
            self.status.setText(f"Finished: {len(images)} images classified.")
        except Exception as error:
            self.error(str(error))

    def error(self, message: str) -> None:
        self.status.setText(f"Error: {message}")
        QMessageBox.critical(self, "Classification failed", message)
