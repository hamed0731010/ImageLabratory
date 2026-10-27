from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QCheckBox, QComboBox, QDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QPlainTextEdit, QProgressBar, QScrollArea, QSpinBox, QVBoxLayout, QWidget
from PyQt6.QtGui import QIcon, QPixmap


class TrainingWorker(QThread):
    progress = pyqtSignal(str)
    completed = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, dataset, output, model_name, epochs, batch_size, image_size, resume):
        super().__init__(); self.dataset, self.output, self.model_name = Path(dataset), Path(output), model_name
        self.epochs, self.batch_size, self.image_size, self.resume = epochs, batch_size, image_size, resume

    def run(self) -> None:
        try:
            self.progress.emit("Loading PyTorch and dataset...")
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, random_split
            from torchvision import datasets, models, transforms
            from torchvision.utils import save_image
            if (self.dataset / "train_data").is_dir():
                train_root = self.dataset / "train_data"; val_root = self.dataset / "val_data"
            else:
                train_root = val_root = self.dataset
            self.progress.emit(f"Reading training images from {train_root}...")
            transform = transforms.Compose([transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
            full_train = datasets.ImageFolder(str(train_root), transform=transform)
            if val_root == train_root:
                split = max(1, int(len(full_train) * .2)); train_data, val_data = random_split(full_train, [len(full_train)-split, split], generator=torch.Generator().manual_seed(42))
            else:
                train_data, val_data = full_train, datasets.ImageFolder(str(val_root), transform=transform)
            classes = full_train.classes
            self.progress.emit(f"Dataset ready: {len(train_data)} training images, {len(val_data)} validation images, {len(classes)} classes.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.progress.emit(f"Loading cached pretrained {self.model_name} weights (offline mode)...")
            model_factories = {
                "EfficientNet-B1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
                "ResNet-18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
                "MobileNet-V3": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
            }
            factory, weights = model_factories[self.model_name]
            checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
            model = factory(weights=None)
            if checkpoint.is_file():
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                self.progress.emit(f"Loaded cached weights: {checkpoint.name}")
            else:
                self.progress.emit(f"Cached weights not found; continuing without pretrained weights: {checkpoint}")
            self.progress.emit(f"Building {self.model_name} with {len(classes)} classes on {device}...")
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
            misclassified_root = self.output / "misclassified"
            mean = torch.tensor([.485, .456, .406], device=device).view(3, 1, 1)
            std = torch.tensor([.229, .224, .225], device=device).view(3, 1, 1)
            history = []; best = 0.0; start_epoch = 1
            checkpoint_path = self.output / "last.pt"
            if self.resume:
                if not checkpoint_path.is_file():
                    raise ValueError(f"Checkpoint not found: {checkpoint_path}")
                saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if saved.get("model_name") != self.model_name or saved.get("class_names") != classes:
                    raise ValueError("The checkpoint model or classes do not match the selected dataset/model.")
                model.load_state_dict(saved["model_state_dict"])
                if saved.get("optimizer_state_dict"):
                    optimizer.load_state_dict(saved["optimizer_state_dict"])
                start_epoch = int(saved.get("epoch", 0)) + 1
                history_path = self.output / "history.json"
                if history_path.is_file():
                    history = json.loads(history_path.read_text(encoding="utf-8"))
                best = max((row.get("val_accuracy", 0.0) for row in history), default=0.0)
                self.progress.emit(f"Resuming from checkpoint at epoch {start_epoch - 1}; training {self.epochs} additional epochs...")
            started = time.time()
            end_epoch = start_epoch + self.epochs
            for epoch in range(start_epoch, end_epoch):
                epoch_misclassified = misclassified_root / f"epoch_{epoch:04d}"
                train_misclassified = epoch_misclassified / "train"
                val_misclassified = epoch_misclassified / "val"
                train_misclassified.mkdir(parents=True, exist_ok=True)
                val_misclassified.mkdir(parents=True, exist_ok=True)
                model.train(); total = correct = loss_total = 0
                total_train_batches = len(train_loader)
                self.progress.emit(f"Epoch {epoch}/{self.epochs} — training started ({total_train_batches} batches)")
                for batch_number, (images, labels) in enumerate(train_loader, 1):
                    images, labels = images.to(device), labels.to(device); optimizer.zero_grad(); logits = model(images); loss = criterion(logits, labels); loss.backward(); optimizer.step()
                    predictions = logits.argmax(1)
                    for item, actual, predicted in zip(images, labels, predictions):
                        if actual != predicted:
                            clean = ((item.detach() * std + mean).clamp(0, 1)).cpu()
                            save_image(clean, train_misclassified / f"{total:06d}_actual-{classes[actual.item()]}_pred-{classes[predicted.item()]}.png")
                    loss_total += loss.item() * labels.size(0); correct += (logits.argmax(1) == labels).sum().item(); total += labels.size(0)
                    if batch_number == 1 or batch_number % 10 == 0 or batch_number == total_train_batches:
                        self.progress.emit(f"Epoch {epoch}/{self.epochs} — training batch {batch_number}/{total_train_batches} ({batch_number * 100 // total_train_batches}%)")
                model.eval(); val_total = val_correct = 0; val_loss = 0.0
                with torch.no_grad():
                    total_val_batches = len(val_loader)
                    self.progress.emit(f"Epoch {epoch}/{self.epochs} — validation started ({total_val_batches} batches)")
                    for batch_number, (images, labels) in enumerate(val_loader, 1):
                        images, labels = images.to(device), labels.to(device); logits = model(images); predictions = logits.argmax(1)
                        for item, actual, predicted in zip(images, labels, predictions):
                            if actual != predicted:
                                clean = ((item.detach() * std + mean).clamp(0, 1)).cpu()
                                save_image(clean, val_misclassified / f"{val_total:06d}_actual-{classes[actual.item()]}_pred-{classes[predicted.item()]}.png")
                        val_loss += criterion(logits, labels).item() * labels.size(0); val_correct += (predictions == labels).sum().item(); val_total += labels.size(0)
                        if batch_number == 1 or batch_number % 10 == 0 or batch_number == total_val_batches:
                            self.progress.emit(f"Epoch {epoch}/{self.epochs} — validation batch {batch_number}/{total_val_batches} ({batch_number * 100 // total_val_batches}%)")
                row = {"epoch": epoch, "train_loss": loss_total/max(1,total), "train_accuracy": correct/max(1,total), "val_loss": val_loss/max(1,val_total), "val_accuracy": val_correct/max(1,val_total)}; history.append(row)
                train_errors = len(list(train_misclassified.glob("*.png")))
                val_errors = len(list(val_misclassified.glob("*.png")))
                self.progress.emit(f"Epoch {epoch}/{end_epoch - 1} — train {row['train_accuracy']:.2%}, val {row['val_accuracy']:.2%}; misclassified train={train_errors}, val={val_errors}")
                payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "class_names": classes, "class_to_idx": full_train.class_to_idx, "model_name": self.model_name, "image_size": self.image_size, "epoch": epoch}
                torch.save(payload, self.output / "last.pt")
                if row["val_accuracy"] >= best: best = row["val_accuracy"]; torch.save(payload, self.output / "best.pt")
            with (self.output / "history.json").open("w", encoding="utf-8") as file: json.dump(history, file, indent=2)
            (self.output / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
            self.progress.emit("Exporting the final model to ONNX...")
            model.eval(); dummy = torch.randn(1, 3, self.image_size, self.image_size, device=device)
            torch.onnx.export(model, dummy, str(self.output / "model.onnx"), input_names=["input"], output_names=["output"], opset_version=17)
            summary = {"model": self.model_name, "classes": classes, "epochs": end_epoch - 1, "additional_epochs": self.epochs, "best_val_accuracy": best, "device": str(device), "duration_seconds": time.time()-started, "misclassified_images": "misclassified/epoch_XXXX/{train,val}", "artifacts": ["best.pt", "last.pt", "model.onnx", "history.json", "classes.json"]}
            (self.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            self.completed.emit(str(self.output))
        except Exception as error: self.failed.emit(str(error))


class TrainingFeature(QWidget):
    def __init__(self) -> None:
        super().__init__(); self.worker = None; self.dataset = QLineEdit(); self.output = QLineEdit(); self.model = QComboBox(); self.model.addItems(["EfficientNet-B1", "ResNet-18", "MobileNet-V3"]); self.epochs = QSpinBox(); self.epochs.setRange(1, 1000); self.epochs.setValue(10); self.batch = QSpinBox(); self.batch.setRange(1, 512); self.batch.setValue(16); self.size = QSpinBox(); self.size.setRange(32, 1024); self.size.setValue(224); self.resume = QCheckBox("Continue from last checkpoint (additional epochs)")
        form = QFormLayout(); form.addRow("Dataset:", self.row(self.dataset, "Choose folder", self.choose_dataset)); form.addRow("Output:", self.row(self.output, "Choose folder", self.choose_output)); form.addRow("Model:", self.model); form.addRow("Epochs:", self.epochs); form.addRow("Batch size:", self.batch); form.addRow("Image size:", self.size); form.addRow("Resume:", self.resume)
        self.start = QPushButton("Start training"); self.start.clicked.connect(self.start_training); self.view_errors = QPushButton("View misclassified images"); self.view_errors.clicked.connect(self.show_misclassified); self.status = QLabel("Dataset should contain train_data/val_data folders or class folders directly."); self.status.setWordWrap(True); self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0); self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setPlaceholderText("Training logs will appear here..."); box = QGroupBox("Train image classifier"); box.setLayout(form); layout = QVBoxLayout(self); layout.addWidget(box); layout.addWidget(self.start); layout.addWidget(self.view_errors); layout.addWidget(self.progress_bar); layout.addWidget(self.status); layout.addWidget(self.log, stretch=1); layout.addStretch()

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
        if self.resume.isChecked() and not (output / "last.pt").is_file(): QMessageBox.warning(self, "Checkpoint required", "The selected output folder does not contain last.pt."); return
        self.log.clear(); self.progress_bar.setValue(0); self.log_message("Starting training..."); self.worker = TrainingWorker(self.dataset.text(), output, self.model.currentText(), self.epochs.value(), self.batch.value(), self.size.value(), self.resume.isChecked()); self.worker.progress.connect(self.log_message); self.worker.completed.connect(lambda path: self.finish(f"Training complete. Artifacts: {path}")); self.worker.failed.connect(lambda error: self.finish(f"Training failed: {error}")); self.start.setEnabled(False); self.worker.start()

    def log_message(self, message):
        self.status.setText(message); self.log.appendPlainText(message)
        import re
        match = re.search(r"Epoch (\d+)/(\d+)", message)
        if match: self.progress_bar.setValue(round(int(match.group(1)) * 100 / int(match.group(2))))

    def finish(self, message):
        self.start.setEnabled(True); self.status.setText(message); self.log.appendPlainText(message)
        if message.startswith("Training complete"): self.progress_bar.setValue(100)

    def show_misclassified(self):
        output = Path(self.output.text())
        root = output / "misclassified"
        images = sorted(root.rglob("*.png")) if root.is_dir() else []
        if not images:
            QMessageBox.information(self, "No images", "No misclassified images were found yet.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Misclassified images by epoch")
        dialog.resize(900, 650)
        scroll = QScrollArea(dialog); scroll.setWidgetResizable(True)
        content = QWidget(); grid = QGridLayout(content)
        for index, path in enumerate(images):
            item = QLabel(); item.setAlignment(Qt.AlignmentFlag.AlignCenter); item.setToolTip(str(path)); item.setPixmap(QPixmap(str(path)).scaled(180, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            caption = QLabel(f"{path.parent.parent.name} / {path.parent.name}\n{path.name}"); caption.setWordWrap(True); cell = QVBoxLayout(); cell.addWidget(item); cell.addWidget(caption); wrapper = QWidget(); wrapper.setLayout(cell); grid.addWidget(wrapper, index // 4, index % 4)
        scroll.setWidget(content); layout = QVBoxLayout(dialog); layout.addWidget(scroll); dialog.exec()
