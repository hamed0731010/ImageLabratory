"""Batch image labeling through an OpenAI-compatible vision-language API."""

from __future__ import annotations

import base64
import json
import mimetypes
import shutil
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QPlainTextEdit, QProgressBar, QSpinBox,
    QVBoxLayout, QWidget,
)


class VLMWorker(QThread):
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.stop_requested = False

    def stop(self):
        self.stop_requested = True

    def run(self):
        try:
            dataset = Path(self.config["dataset"])
            images = sorted(p for p in dataset.rglob("*") if p.is_file() and p.suffix.lower() in
                            {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
            if not images:
                raise ValueError("No supported images were found in the dataset.")
            output = Path(self.config["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if output.exists():
                for line in output.read_text(encoding="utf-8").splitlines():
                    try:
                        item = json.loads(line)
                        existing[item.get("image", "")] = item
                    except json.JSONDecodeError:
                        continue
            schema = json.loads(self.config["schema"])
            results = dict(existing)
            for number, path in enumerate(images, 1):
                key = str(path.resolve())
                if key in existing:
                    self.progress.emit(number * 100 // len(images), f"Skipped existing: {path.name}")
                    continue
                if self.stop_requested:
                    break
                result = self.request(path, schema)
                label = self.extract_label(result)
                destination_dir = Path(self.config["label_output"]) / self.safe_label(label)
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / path.name
                if destination.exists() and destination.resolve() != path.resolve():
                    destination = destination_dir / f"{path.stem}_{number}{path.suffix}"
                if self.config["transfer"] == "Move":
                    shutil.move(str(path), str(destination))
                else:
                    shutil.copy2(str(path), str(destination))
                results[key] = {"image": key, "relative_image": str(path.relative_to(dataset)),
                                "label": result, "label_folder": str(destination_dir),
                                "transfer": self.config["transfer"], "timestamp": datetime.now().isoformat(timespec="seconds")}
                output.write_text("\n".join(json.dumps(value, ensure_ascii=False) for value in results.values()) + "\n", encoding="utf-8")
                self.progress.emit(number * 100 // len(images), f"Labeled {number}/{len(images)}: {path.name}")
            summary = output.with_name(output.stem + "_summary.json")
            summary.write_text(json.dumps({"dataset": str(dataset.resolve()), "images": len(images),
                                           "completed": len(results), "output": str(output.resolve())}, indent=2), encoding="utf-8")
            self.completed.emit(str(output))
        except Exception as error:
            self.failed.emit(str(error))

    @staticmethod
    def extract_label(result):
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            for key in ("label", "class", "category", "name"):
                if result.get(key) is not None:
                    return str(result[key])
        raise ValueError("The API response must contain a 'label', 'class', 'category', or 'name' field.")

    @staticmethod
    def safe_label(label):
        cleaned = "".join(character if character.isalnum() or character in " _-." else "_" for character in label).strip(" .")
        return cleaned[:120] or "unlabeled"

    def request(self, path: Path, schema: dict):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        body = {"model": self.config["model"], "temperature": 0,
                "max_tokens": self.config["tokens"],
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": self.config["prompt"]},
                    {"type": "text", "text": "Return only JSON matching this schema:\n" + json.dumps(schema)},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ]}]}
        request = urllib.request.Request(self.config["endpoint"], data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": "Bearer " + self.config["api_key"],
                                                  "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"API error {error.code}: {error.read().decode(errors='replace')[:500]}") from error
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        content = content.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(content)


class VLMLabeling(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.endpoint = QLineEdit("https://api.openai.com/v1/chat/completions")
        self.api_key = QLineEdit(); self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.model = QLineEdit("gpt-4o-mini")
        self.tokens = QSpinBox(); self.tokens.setRange(1, 100000); self.tokens.setValue(1000)
        self.dataset = QLineEdit(); self.output = QLineEdit(); self.label_output = QLineEdit()
        from PyQt6.QtWidgets import QComboBox
        self.transfer = QComboBox(); self.transfer.addItems(["Copy", "Move"])
        self.prompt = QPlainTextEdit("Inspect this image and produce the requested structured label.")
        self.schema = QPlainTextEdit('{\n  "type": "object",\n  "properties": {\n    "label": {"type": "string"},\n    "confidence": {"type": "number"}\n  },\n  "required": ["label"]\n}')
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.progress = QProgressBar(); self.status = QLabel("Ready")
        form = QFormLayout()
        form.addRow("API endpoint:", self.endpoint); form.addRow("API token:", self.api_key)
        form.addRow("Model:", self.model); form.addRow("Max output tokens:", self.tokens)
        form.addRow("Dataset:", self.row(self.dataset, "Choose folder", self.choose_dataset))
        form.addRow("Label output:", self.row(self.label_output, "Choose folder", self.choose_label_output))
        form.addRow("Transfer images:", self.transfer)
        form.addRow("Output JSONL:", self.row(self.output, "Choose file", self.choose_output))
        box = QGroupBox("VLM / JEV-compatible labeling")
        box.setLayout(form)
        run = QPushButton("Start labeling"); run.clicked.connect(self.start)
        stop = QPushButton("Stop after current image"); stop.clicked.connect(self.stop)
        layout = QVBoxLayout(self); layout.addWidget(box)
        layout.addWidget(QLabel("Prompt:")); layout.addWidget(self.prompt)
        layout.addWidget(QLabel("JSON schema:")); layout.addWidget(self.schema)
        buttons = QHBoxLayout(); buttons.addWidget(run); buttons.addWidget(stop); layout.addLayout(buttons)
        layout.addWidget(self.progress); layout.addWidget(self.status); layout.addWidget(self.log)

    @staticmethod
    def row(edit, label, callback):
        widget = QWidget(); layout = QHBoxLayout(widget); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit); button = QPushButton(label); button.clicked.connect(callback); layout.addWidget(button)
        return widget

    def choose_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Choose image dataset")
        if path: self.dataset.setText(path)

    def choose_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save labeling results", "vlm_labels.jsonl", "JSON Lines (*.jsonl)")
        if path: self.output.setText(path)

    def choose_label_output(self):
        path = QFileDialog.getExistingDirectory(self, "Choose label output folder")
        if path: self.label_output.setText(path)

    def start(self):
        try:
            if not self.api_key.text().strip() or not self.dataset.text().strip(): raise ValueError("API token and dataset are required.")
            json.loads(self.schema.toPlainText())
            output = self.output.text().strip() or str(Path(self.dataset.text()) / "vlm_labels.jsonl")
            label_output = self.label_output.text().strip() or str(Path(self.dataset.text()) / "labeled")
            config = {"endpoint": self.endpoint.text().strip(), "api_key": self.api_key.text().strip(), "model": self.model.text().strip(),
                      "tokens": self.tokens.value(), "dataset": self.dataset.text().strip(), "output": output,
                      "label_output": label_output, "transfer": self.transfer.currentText(),
                      "prompt": self.prompt.toPlainText().strip(), "schema": self.schema.toPlainText()}
            self.worker = VLMWorker(config); self.worker.progress.connect(self.update); self.worker.completed.connect(self.done); self.worker.failed.connect(self.error)
            self.progress.setValue(0); self.log.clear(); self.worker.start()
        except Exception as error: QMessageBox.warning(self, "Invalid input", str(error))

    def stop(self):
        if self.worker and self.worker.isRunning(): self.worker.stop(); self.status.setText("Stopping after current image...")

    def update(self, value, message): self.progress.setValue(value); self.status.setText(message); self.log.appendPlainText(message)
    def done(self, path): self.status.setText("Complete"); self.log.appendPlainText(f"Results: {path}")
    def error(self, message): self.status.setText("Failed"); QMessageBox.critical(self, "VLM labeling failed", message)
