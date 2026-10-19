from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget


class TrainingWorker(QThread):
    progress = pyqtSignal(str)
    completed = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, dataset, output, model_name, epochs, batch_size, image_size):
        super().__init__(); self.dataset, self.output, self.model_name = Path(dataset), Path(output), model_name
        self.epochs, self.batch_size, self.image_size = epochs, batch_size, image_size

    def run(self) -> None:
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, random_split
            from torchvision import datasets, models, transforms
            if (self.dataset / "train_data").is_dir():
                train_root = self.dataset / "train_data"; val_root = self.dataset / "val_data"
            else:
                train_root = val_root = self.dataset
            transform = transforms.Compose([transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
            full_train = datasets.ImageFolder(str(train_root), transform=transform)
            if val_root == train_root:
                split = max(1, int(len(full_train) * .2)); train_data, val_data = random_split(full_train, [len(full_train)-split, split], generator=torch.Generator().manual_seed(42))
            else:
                train_data, val_data = full_train, datasets.ImageFolder(str(val_root), transform=transform)
            classes = full_train.classes
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            constructors = {"EfficientNet-B1": lambda: models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT), "ResNet-18": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT), "MobileNet-V3": lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)}
            model = constructors[self.model_name]()
            if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
            elif hasattr(model, "fc"):
                model.fc = nn.Linear(model.fc.in_features, len(classes))
            model.to(device)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            self.output.mkdir(parents=True, exist_ok=True)
            history = []; best = 0.0; started = time.time()
            for epoch in range(1, self.epochs + 1):
                model.train(); total = correct = loss_total = 0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device); optimizer.zero_grad(); logits = model(images); loss = criterion(logits, labels); loss.backward(); optimizer.step()
                    loss_total += loss.item() * labels.size(0); correct += (logits.argmax(1) == labels).sum().item(); total += labels.size(0)
                model.eval(); val_total = val_correct = 0; val_loss = 0.0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device); logits = model(images); val_loss += criterion(logits, labels).item() * labels.size(0); val_correct += (logits.argmax(1) == labels).sum().item(); val_total += labels.size(0)
                row = {"epoch": epoch, "train_loss": loss_total/max(1,total), "train_accuracy": correct/max(1,total), "val_loss": val_loss/max(1,val_total), "val_accuracy": val_correct/max(1,val_total)}; history.append(row)
                self.progress.emit(f"Epoch {epoch}/{self.epochs} — train {row['train_accuracy']:.2%}, val {row['val_accuracy']:.2%}")
                payload = {"model_state_dict": model.state_dict(), "class_names": classes, "class_to_idx": full_train.class_to_idx, "model_name": self.model_name, "image_size": self.image_size, "epoch": epoch}
                torch.save(payload, self.output / "last.pt")
                if row["val_accuracy"] >= best: best = row["val_accuracy"]; torch.save(payload, self.output / "best.pt")
            with (self.output / "history.json").open("w", encoding="utf-8") as file: json.dump(history, file, indent=2)
            (self.output / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
            model.eval(); dummy = torch.randn(1, 3, self.image_size, self.image_size, device=device)
            torch.onnx.export(model, dummy, str(self.output / "model.onnx"), input_names=["input"], output_names=["output"], opset_version=17)
            summary = {"model": self.model_name, "classes": classes, "epochs": self.epochs, "best_val_accuracy": best, "device": str(device), "duration_seconds": time.time()-started, "artifacts": ["best.pt", "last.pt", "model.onnx", "history.json", "classes.json"]}
            (self.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            self.completed.emit(str(self.output))
        except Exception as error: self.failed.emit(str(error))


class TrainingFeature(QWidget):
    def __init__(self) -> None:
        super().__init__(); self.worker = None; self.dataset = QLineEdit(); self.output = QLineEdit(); self.model = QComboBox(); self.model.addItems(["EfficientNet-B1", "ResNet-18", "MobileNet-V3"]); self.epochs = QSpinBox(); self.epochs.setRange(1, 1000); self.epochs.setValue(10); self.batch = QSpinBox(); self.batch.setRange(1, 512); self.batch.setValue(16); self.size = QSpinBox(); self.size.setRange(32, 1024); self.size.setValue(224)
        form = QFormLayout(); form.addRow("Dataset:", self.row(self.dataset, "Choose folder", self.choose_dataset)); form.addRow("Output:", self.row(self.output, "Choose folder", self.choose_output)); form.addRow("Model:", self.model); form.addRow("Epochs:", self.epochs); form.addRow("Batch size:", self.batch); form.addRow("Image size:", self.size)
        self.start = QPushButton("Start training"); self.start.clicked.connect(self.start_training); self.status = QLabel("Dataset should contain train_data/val_data folders or class folders directly."); self.status.setWordWrap(True); box = QGroupBox("Train image classifier"); box.setLayout(form); layout = QVBoxLayout(self); layout.addWidget(box); layout.addWidget(self.start); layout.addWidget(self.status); layout.addStretch()

    @staticmethod
    def row(field, text, handler):
        widget = QWidget(); row = QHBoxLayout(widget); row.setContentsMargins(0,0,0,0); row.addWidget(field); button = QPushButton(text); button.clicked.connect(handler); row.addWidget(button); return widget

    def choose_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Choose dataset folder"); self.dataset.setText(path) if path else None

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "Choose output folder"); self.output.setText(path) if path else None

    def start_training(self):
        if not Path(self.dataset.text()).is_dir(): QMessageBox.warning(self, "Dataset required", "Choose a valid dataset folder."); return
        output = Path(self.output.text()) if self.output.text() else Path(self.dataset.text()) / "training_artifacts" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.worker = TrainingWorker(self.dataset.text(), output, self.model.currentText(), self.epochs.value(), self.batch.value(), self.size.value()); self.worker.progress.connect(self.status.setText); self.worker.completed.connect(lambda path: self.finish(f"Training complete. Artifacts: {path}")); self.worker.failed.connect(lambda error: self.finish(f"Training failed: {error}")); self.start.setEnabled(False); self.worker.start()

    def finish(self, message): self.start.setEnabled(True); self.status.setText(message)
