from __future__ import annotations

import json
import shutil
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QApplication, QFileDialog, QComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget


class ModelEvaluationFeature(QWidget):
    def __init__(self):
        super().__init__(); self.model_path = QLineEdit(); self.test_path = QLineEdit(); self.output = QLineEdit(); self.architecture = QComboBox(); self.architecture.addItems(["EfficientNet-B1", "ResNet-18", "MobileNet-V3"]); self.matrix = QLabel(); self.report = QPlainTextEdit(); self.report.setReadOnly(True)
        form = QFormLayout(); form.addRow("Model checkpoint:", self.row(self.model_path, "Choose file", self.choose_model)); form.addRow("Model architecture:", self.architecture); form.addRow("Test dataset:", self.row(self.test_path, "Choose folder", self.choose_test)); form.addRow("Output:", self.row(self.output, "Choose folder", self.choose_output)); box = QGroupBox("Test model and analyze errors"); box.setLayout(form)
        run = QPushButton("Evaluate model"); run.clicked.connect(self.evaluate); load = QPushButton("Load result"); load.clicked.connect(self.load_last_result); reset = QPushButton("Reset"); reset.clicked.connect(self.reset_form); self.view_errors = QPushButton("View test misclassifications"); self.view_errors.clicked.connect(self.show_errors); self.progress = QProgressBar(); self.progress.setValue(0); self.status = QLabel("Ready"); self.status.setWordWrap(True)
        results = QHBoxLayout(); results.addWidget(self.matrix, stretch=1); results.addWidget(self.report, stretch=1)
        actions = QHBoxLayout(); actions.addWidget(run); actions.addWidget(load); actions.addWidget(self.view_errors); actions.addWidget(reset)
        layout = QVBoxLayout(self); layout.addWidget(box); layout.addLayout(actions); layout.addWidget(self.progress); layout.addWidget(self.status); layout.addLayout(results, stretch=1)

    @staticmethod
    def row(field, text, handler):
        widget = QWidget(); row = QHBoxLayout(widget); row.setContentsMargins(0, 0, 0, 0); row.addWidget(field); button = QPushButton(text); button.clicked.connect(handler); row.addWidget(button); return widget

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose checkpoint", "", "PyTorch checkpoint (*.pt *.pth)"); self.model_path.setText(path) if path else None

    def choose_test(self):
        path = QFileDialog.getExistingDirectory(self, "Choose test dataset"); self.test_path.setText(path) if path else None

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "Choose output folder"); self.output.setText(path) if path else None

    def evaluate(self):
        if not Path(self.model_path.text()).is_file() or not Path(self.test_path.text()).is_dir():
            QMessageBox.warning(self, "Inputs required", "Choose a valid checkpoint and ImageFolder test dataset."); return
        try:
            self.report.clear(); self.matrix.clear(); self.progress.setRange(0, 0); self.status.setText("Loading model and test dataset..."); QApplication.processEvents()
            import numpy as np
            import torch
            from PIL import Image, ImageDraw, ImageOps
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve
            from torchvision import datasets, models, transforms
            checkpoint = torch.load(self.model_path.text(), map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict): raise ValueError("The selected file is not a supported checkpoint dictionary.")
            experiment = checkpoint.get("experiment", {}) if isinstance(checkpoint.get("experiment", {}), dict) else {}
            model_name = checkpoint.get("model_name") or experiment.get("model_name") or self.architecture.currentText(); image_size = checkpoint.get("image_size") or experiment.get("image_size", 224)
            checkpoint_classes = checkpoint.get("class_names")
            classes = list(checkpoint_classes) if checkpoint_classes else None
            factories = {"EfficientNet-B1": models.efficientnet_b1, "efficientnet_b1": models.efficientnet_b1, "CustomEfficient": models.efficientnet_b1, "ResNet-18": models.resnet18, "resnet18": models.resnet18, "MobileNet-V3": models.mobilenet_v3_small, "mobilenet_v3_small": models.mobilenet_v3_small}; model = factories[model_name](weights=None)
            from torch import nn
            if classes is None:
                classes = datasets.ImageFolder(self.test_path.text()).classes
            if not classes:
                raise ValueError("No classes were found in the checkpoint or test dataset.")
            if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential): model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
            else: model.fc = nn.Linear(model.fc.in_features, len(classes))
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
            if state_dict is None and all(hasattr(value, "shape") for value in checkpoint.values()): state_dict = checkpoint
            if state_dict is None: raise ValueError("Checkpoint does not contain model_state_dict or state_dict.")
            normalized_state = {}
            for key, value in state_dict.items():
                key = key.removeprefix("module.")
                if model_name in {"EfficientNet-B1", "efficientnet_b1", "CustomEfficient"} and key.startswith("encoder."):
                    key = key.removeprefix("encoder.")
                normalized_state[key] = value
            model.load_state_dict(normalized_state); model.eval()
            class ResizeWithPad:
                def __call__(self, image):
                    contained = ImageOps.contain(image, (image_size, image_size), method=Image.Resampling.BICUBIC)
                    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255)); canvas.paste(contained, ((image_size-contained.width)//2, (image_size-contained.height)//2)); return canvas
            transform = transforms.Compose([ResizeWithPad(), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
            data = datasets.ImageFolder(self.test_path.text(), transform=transform)
            if list(classes) != list(data.classes): raise ValueError(f"Checkpoint classes do not match test classes. Checkpoint: {classes}; test: {data.classes}")
            loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False)
            y_true=[]; y_pred=[]; probabilities=[]; bad=[]
            self.progress.setRange(0, len(loader)); self.progress.setValue(0); self.status.setText(f"Evaluating 0/{len(loader)} batches..."); QApplication.processEvents()
            with torch.no_grad():
                for batch_number, (images, labels) in enumerate(loader, 1):
                    probs = torch.softmax(model(images), dim=1); preds = probs.argmax(1); y_true.extend(labels.tolist()); y_pred.extend(preds.tolist()); probabilities.extend(probs.tolist())
                    for i, (actual, predicted) in enumerate(zip(labels.tolist(), preds.tolist())):
                        if actual != predicted: bad.append((data.samples[len(y_true)-len(labels)+i][0], actual, predicted))
                    self.progress.setValue(batch_number); self.status.setText(f"Evaluating batch {batch_number}/{len(loader)} — errors found: {len(bad)}"); QApplication.processEvents()
            out = Path(self.output.text()) if self.output.text() else Path(self.test_path.text()) / "evaluation"; out.mkdir(parents=True, exist_ok=True); bad_dir = out / "misclassified"; bad_dir.mkdir(exist_ok=True)
            def safe(value): return "".join(char if char not in '<>:"/\\|?*' else "_" for char in value).strip() or "unknown"
            for index, (source, actual, predicted) in enumerate(bad):
                pair_dir = bad_dir / f"actual-{safe(classes[actual])}" / f"predicted-{safe(classes[predicted])}"; pair_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, pair_dir / f"{index:05d}_{Path(source).name}")
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))); image = Image.new("RGB", (max(500, len(classes)*80), max(450, len(classes)*70)), "white"); draw = ImageDraw.Draw(image); cell = min(60, 360 // max(1, len(classes))); ox, oy = 180, 60
            for r in range(len(classes)):
                for c in range(len(classes)):
                    x, y = ox+c*cell, oy+r*cell; draw.rectangle((x,y,x+cell,y+cell), outline="black", fill=(255, max(0, 255-int(cm[r,c]*20)), max(0, 255-int(cm[r,c]*20)))); draw.text((x+cell//3,y+cell//3), str(cm[r,c]), fill="black")
            for i, name in enumerate(classes): draw.text((ox+i*cell, 35), name[:10], fill="black"); draw.text((20, oy+i*cell+cell//3), name[:18], fill="black")
            matrix_path = out / "confusion_matrix.png"; image.save(matrix_path); self.matrix.setPixmap(QPixmap(str(matrix_path)).scaled(650, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(classes))), zero_division=0); metrics = {"model_path": str(Path(self.model_path.text()).resolve()), "test_dataset_path": str(Path(self.test_path.text()).resolve()), "model_architecture": model_name, "output_path": str(out.resolve()), "accuracy": accuracy_score(y_true,y_pred), "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()), "weighted_f1": float((f1*support/max(1,sum(support))).sum()), "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i,name in enumerate(classes)}, "roc_auc": {}, "roc_threshold_youden": {}, "calibration_ece": {}}
            probs = np.asarray(probabilities); true = np.asarray(y_true)
            for i, name in enumerate(classes):
                binary = (true == i).astype(int); fpr, tpr, thresholds = roc_curve(binary, probs[:,i]); valid = np.isfinite(thresholds); metrics["roc_auc"][name] = float(auc(fpr,tpr)); metrics["roc_threshold_youden"][name] = float(thresholds[np.argmax((tpr-fpr)[valid])]) if valid.any() else 0.5; frac, mean = calibration_curve(binary, probs[:,i], n_bins=10, strategy="uniform"); metrics["calibration_ece"][name] = float(np.average(np.abs(frac-mean), weights=np.ones(len(frac)))) if len(frac) else 0.0
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8"); (out / "classification_report.txt").write_text(classification_report(y_true, y_pred, target_names=classes, zero_division=0), encoding="utf-8")
            self.report.setPlainText(json.dumps(metrics, indent=2) + f"\n\nMisclassified images: {len(bad)}\nSaved to: {out}"); self._error_dir = bad_dir; self.status.setText(f"Evaluation complete — {len(y_true)} images, {len(bad)} misclassified.")
        except Exception as error:
            self.progress.setRange(0, 1); self.progress.setValue(0); self.status.setText("Evaluation failed."); QMessageBox.critical(self, "Evaluation failed", str(error))

    def show_errors(self):
        if not hasattr(self, "_error_dir") or not self._error_dir.is_dir(): QMessageBox.information(self, "No results", "Run evaluation first."); return
        dialog = QWidget(); dialog.setWindowTitle("Test misclassified images — grouped by labels"); dialog.resize(1000, 700); tabs = QTabWidget(dialog)
        for actual_dir in sorted(self._error_dir.glob("actual-*")):
            for predicted_dir in sorted(actual_dir.glob("predicted-*")):
                paths = sorted(predicted_dir.iterdir()); page = QWidget(); scroll = QScrollArea(page); scroll.setWidgetResizable(True); content = QWidget(); grid = QGridLayout(content)
                for index, path in enumerate(paths):
                    image = QLabel(); image.setAlignment(Qt.AlignmentFlag.AlignCenter); image.setPixmap(QPixmap(str(path)).scaled(180, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)); image.setToolTip(path.name)
                    caption = QLabel(f'<a href="{QUrl.fromLocalFile(str(path)).toString()}">{path.name}</a>'); caption.setOpenExternalLinks(False); caption.linkActivated.connect(lambda url: QDesktopServices.openUrl(QUrl(url))); caption.setWordWrap(True); cell = QVBoxLayout(); cell.addWidget(image); cell.addWidget(caption); wrapper = QWidget(); wrapper.setLayout(cell); grid.addWidget(wrapper, index // 4, index % 4)
                scroll.setWidget(content); layout = QVBoxLayout(page); layout.addWidget(scroll); tabs.addTab(page, f"{actual_dir.name[7:]} → {predicted_dir.name[10:]} ({len(paths)})")
        outer = QVBoxLayout(dialog); outer.addWidget(tabs); dialog.show(); self._error_window = dialog

    def load_last_result(self):
        import json
        output = Path(self.output.text())
        metrics_path = output / "metrics.json"
        matrix_path = output / "confusion_matrix.png"
        if not metrics_path.is_file():
            QMessageBox.information(self, "No saved result", "No previous evaluation was found in the selected output folder.")
            return
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")); self.report.setPlainText(json.dumps(metrics, indent=2)); self.status.setText(f"Last evaluation loaded from: {output}")
            if matrix_path.is_file(): self.matrix.setPixmap(QPixmap(str(matrix_path)).scaled(650, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._error_dir = output / "misclassified"
        except Exception as error:
            QMessageBox.critical(self, "Could not load result", str(error))

    def reset_form(self):
        self.model_path.clear(); self.test_path.clear(); self.output.clear(); self.architecture.setCurrentIndex(0); self.matrix.clear(); self.report.clear(); self.progress.setRange(0, 1); self.progress.setValue(0); self.status.setText("Ready"); self._error_dir = None
