from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QPlainTextEdit, QProgressBar, QVBoxLayout, QWidget
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".pdf"}


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
        self.progress_bar = QProgressBar()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Classification log will appear here...")
        self.settings = QSettings("ImageLabelStudio", "PyTorchClassifier")
        self.load_setup()
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
        save_setup = QPushButton("Save setup")
        save_setup.clicked.connect(self.save_setup)
        box = QGroupBox("PyTorch image classifier")
        box.setLayout(form)
        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addWidget(button)
        layout.addWidget(save_setup)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status)
        layout.addWidget(self.log, stretch=1)
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

    def save_setup(self) -> None:
        self.settings.setValue("model", self.model_edit.text())
        self.settings.setValue("labels", self.labels_edit.text())
        self.settings.setValue("folder", self.folder_edit.text())
        self.settings.setValue("architecture", self.architecture_edit.text())
        self.settings.setValue("class_name", self.class_edit.text())
        self.settings.setValue("size", self.size_edit.text())
        self.settings.setValue("model_type", self.model_type.currentIndex())
        self.settings.setValue("mode", self.mode.currentIndex())
        self.settings.sync()
        self.status.setText("Setup saved. It will be restored next time.")

    def load_setup(self) -> None:
        self.model_edit.setText(self.settings.value("model", ""))
        self.labels_edit.setText(self.settings.value("labels", ""))
        self.folder_edit.setText(self.settings.value("folder", ""))
        self.architecture_edit.setText(self.settings.value("architecture", ""))
        self.class_edit.setText(self.settings.value("class_name", ""))
        self.size_edit.setText(self.settings.value("size", "224,224"))
        self.model_type.setCurrentIndex(int(self.settings.value("model_type", 0)))
        self.mode.setCurrentIndex(int(self.settings.value("mode", 0)))

    def labels(self, path: Path) -> list[str]:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return [str(value) for value in data] if isinstance(data, list) else [str(data[key]) for key in sorted(data, key=str)]
        return [line.split(",", 1)[-1].strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def load_model(self, torch, path: Path):
        loaded = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            model_name = loaded.get("model_name") or loaded.get("experiment", {}).get("model_name", "efficientnet_b1")
            return self.build_checkpoint_model(torch, model_name, len(loaded.get("class_names", [])), loaded["model_state_dict"])
        if self.model_type.currentText() == "TorchScript":
            return torch.jit.load(str(path), map_location="cpu")
        if self.model_type.currentText() == "Complete model":
            model = loaded
            if isinstance(model, dict):
                checkpoint_state = model.get("model_state_dict") or model.get("state_dict")
                if checkpoint_state is None:
                    raise ValueError("This file is a state dictionary. Select State dictionary.")
                class_names = model.get("class_names", [])
                model_name = model.get("model_name") or model.get("experiment", {}).get("model_name", "efficientnet_b1")
                model = self.build_checkpoint_model(torch, model_name, len(class_names), checkpoint_state)
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

    @staticmethod
    def build_checkpoint_model(torch, model_name, class_count, state):
        from torch import nn
        from torchvision import models
        if not class_count:
            raise ValueError("The training checkpoint does not contain class_names.")
        if model_name in {"efficientnet_b1", "EfficientNet-B1", "CustomEfficient"}:
            encoder = models.efficientnet_b1(weights=None)
            encoder.classifier[1] = nn.Linear(encoder.classifier[1].in_features, class_count)

            class CustomEfficientCheckpoint(nn.Module):
                def __init__(self, backbone):
                    super().__init__()
                    self.encoder = backbone

                def forward(self, x):
                    return self.encoder(x)

            model = CustomEfficientCheckpoint(encoder)
        elif model_name in {"resnet18", "ResNet-18"}:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, class_count)
        elif model_name in {"mobilenet_v3_small", "MobileNet-V3"}:
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
        else:
            raise ValueError(f"Unsupported checkpoint model: {model_name}")
        # Checkpoints saved from wrapper/DataParallel models may contain
        # prefixes that are not part of the selected backbone's state dict.
        normalized_state = {}
        for key, value in state.items():
            key = key.removeprefix("module.")
            if model_name in {"efficientnet_b1", "EfficientNet-B1", "CustomEfficient"} and key.startswith("encoder."):
                key = key.removeprefix("encoder.")
            normalized_state[key] = value
        model.load_state_dict(normalized_state)
        return model

    def classify(self) -> None:
        try:
            import time
            started_at = time.perf_counter()
            import io
            import numpy as np
            import torch
            from PIL import Image
        except ImportError:
            self.error("Install dependencies first: pip install torch pillow numpy")
            return
        try:
            model_path, labels_path, folder = Path(self.model_edit.text()), Path(self.labels_edit.text()), Path(self.folder_edit.text())
            if not model_path.is_file() or not folder.is_dir(): raise ValueError("Choose a valid model and image folder.")
            width, height = [int(x.strip()) for x in self.size_edit.text().replace("x", ",").split(",")]
            checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
            checkpoint_labels = checkpoint.get("class_names", []) if isinstance(checkpoint, dict) else []
            labels = self.labels(labels_path) if labels_path.is_file() else [str(value) for value in checkpoint_labels]
            if not labels:
                raise ValueError("Choose a labels file, or use a training checkpoint containing class_names.")
            model = self.load_model(torch, model_path).eval()
            files = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: p.name.lower())
            samples = []
            for source in files:
                if source.suffix.lower() == ".pdf":
                    try:
                        import fitz
                    except ImportError:
                        raise ValueError("PDF support requires PyMuPDF. Install it with: pip install PyMuPDF")
                    with fitz.open(str(source)) as document:
                        samples.extend((source, page_number) for page_number in range(document.page_count))
                else:
                    samples.append((source, None))
            self.progress_bar.setRange(0, len(samples)); self.progress_bar.setValue(0); self.log.clear(); self.log.appendPlainText(f"Starting classification: {len(samples)} image pages")
            for number, (source, page_number) in enumerate(samples, 1):
                if page_number is None:
                    image = Image.open(source).convert("RGB")
                    output_name = source.name
                else:
                    import fitz
                    with fitz.open(str(source)) as document:
                        pixmap = document[page_number].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
                        image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                    output_name = f"{source.stem}_page_{page_number + 1:03d}.png"
                output_image = image.copy()
                image = image.resize((width, height))
                array = np.asarray(image, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).unsqueeze(0)
                with torch.no_grad(): result = model(tensor)
                if isinstance(result, (tuple, list)): result = result[0]
                index = int(torch.argmax(result).item())
                name = labels[index] if index < len(labels) else f"class_{index}"
                safe = "".join(c if c not in '<>:"/\\|?*' else "_" for c in name).strip() or f"class_{index}"
                target = folder / safe / output_name
                target.parent.mkdir(exist_ok=True)
                if target.exists(): target.unlink()
                if page_number is None:
                    shutil.copy2(source, target) if self.mode.currentIndex() == 0 else shutil.move(str(source), str(target))
                else:
                    output_image.save(target, format="PNG")
                elapsed = time.perf_counter() - started_at
                source_label = f"{source.name} page {page_number + 1}" if page_number is not None else source.name
                message = f"{number}/{len(samples)} ({number * 100 // len(samples)}%) — {source_label} → {safe} — {elapsed:.1f}s"
                self.status.setText(message); self.log.appendPlainText(message); self.progress_bar.setValue(number); QApplication.processEvents()
            elapsed = time.perf_counter() - started_at
            finished = f"Finished: {len(samples)} image pages classified in {elapsed:.1f}s ({elapsed / max(1, len(samples)):.2f}s/page)."
            self.status.setText(finished); self.log.appendPlainText(finished)
        except Exception as error:
            self.error(str(error))

    def error(self, message: str) -> None:
        self.status.setText(f"Error: {message}")
        QMessageBox.critical(self, "Classification failed", message)
