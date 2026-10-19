from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ClusterWorker(QThread):
    progress = pyqtSignal(str)
    completed = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, source, output, method, count, mode, image_size):
        super().__init__(); self.source, self.output, self.method, self.count, self.mode, self.image_size = Path(source), Path(output), method, count, mode, image_size

    def run(self):
        try:
            import numpy as np
            import torch
            from PIL import Image
            from torchvision import models, transforms
            from sklearn.cluster import AgglomerativeClustering, KMeans
            files = sorted((p for p in self.source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: str(p).lower())
            if len(files) < self.count: raise ValueError("Cluster count cannot be larger than the number of images.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            backbone.fc = torch.nn.Identity(); backbone.eval().to(device)
            transform = transforms.Compose([transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
            embeddings = []
            with torch.no_grad():
                for number, path in enumerate(files, 1):
                    image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device); embeddings.append(backbone(image).squeeze(0).cpu().numpy()); self.progress.emit(f"Embedding {number}/{len(files)}: {path.name}")
            matrix = np.asarray(embeddings)
            if self.method == "K-Means": labels = KMeans(n_clusters=self.count, n_init=10, random_state=42).fit_predict(matrix)
            else: labels = AgglomerativeClustering(n_clusters=self.count).fit_predict(matrix)
            self.output.mkdir(parents=True, exist_ok=True)
            report = []
            for path, label in zip(files, labels):
                destination_dir = self.output / f"cluster_{int(label):03d}"; destination_dir.mkdir(exist_ok=True); target = destination_dir / path.name
                if target.exists(): target = destination_dir / f"{path.stem}_{len(list(destination_dir.iterdir()))}{path.suffix}"
                if self.mode == "Copy": shutil.copy2(path, target)
                else: shutil.move(str(path), str(target))
                report.append({"source": str(path), "cluster": f"cluster_{int(label):03d}", "destination": str(target)})
            with (self.output / "clusters.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=["source", "cluster", "destination"]); writer.writeheader(); writer.writerows(report)
            (self.output / "summary.json").write_text(__import__("json").dumps({"method": self.method, "clusters": self.count, "images": len(files), "device": str(device)}, indent=2), encoding="utf-8")
            self.completed.emit(str(self.output))
        except Exception as error: self.failed.emit(str(error))


class ClusteringFeature(QWidget):
    def __init__(self):
        super().__init__(); self.source = QLineEdit(); self.output = QLineEdit(); self.method = QComboBox(); self.method.addItems(["K-Means", "Agglomerative"]); self.mode = QComboBox(); self.mode.addItems(["Copy", "Move"]); self.count = QSpinBox(); self.count.setRange(2, 10000); self.count.setValue(5); self.size = QSpinBox(); self.size.setRange(32, 1024); self.size.setValue(224); self.status = QLabel("Visual clusters use ResNet-18 embeddings."); self.status.setWordWrap(True)
        form = QFormLayout(); form.addRow("Dataset:", self.row(self.source, "Choose folder", self.choose_source)); form.addRow("Output:", self.row(self.output, "Choose folder", self.choose_output)); form.addRow("Method:", self.method); form.addRow("Cluster count:", self.count); form.addRow("Image size:", self.size); form.addRow("Action:", self.mode); box = QGroupBox("Image clustering"); box.setLayout(form); button = QPushButton("Start clustering"); button.clicked.connect(self.start); layout = QVBoxLayout(self); layout.addWidget(box); layout.addWidget(button); layout.addWidget(self.status); layout.addStretch()

    @staticmethod
    def row(field, text, handler):
        widget = QWidget(); row = QHBoxLayout(widget); row.setContentsMargins(0,0,0,0); row.addWidget(field); button = QPushButton(text); button.clicked.connect(handler); row.addWidget(button); return widget

    def choose_source(self):
        path = QFileDialog.getExistingDirectory(self, "Choose dataset"); self.source.setText(path) if path else None

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "Choose output"); self.output.setText(path) if path else None

    def start(self):
        if not Path(self.source.text()).is_dir(): QMessageBox.warning(self, "Dataset required", "Choose a valid dataset folder."); return
        output = Path(self.output.text()) if self.output.text() else Path(self.source.text()) / "clusters"
        self.worker = ClusterWorker(self.source.text(), output, self.method.currentText(), self.count.value(), self.mode.currentText(), self.size.value()); self.worker.progress.connect(self.status.setText); self.worker.completed.connect(lambda path: self.finish(f"Clustering complete: {path}")); self.worker.failed.connect(lambda error: self.finish(f"Clustering failed: {error}")); self.worker.start()

    def finish(self, message): self.status.setText(message)
