from __future__ import annotations

import csv
import json
import os
import re
import shutil
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif"}


class OCRKeywordClassifier(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.folder = QLineEdit()
        self.keywords = QLineEdit()
        self.tesseract = QLineEdit(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        self.language = QLineEdit("eng")
        self.max_dimension = QLineEdit("1800")
        self.mode = QComboBox()
        self.mode.addItems(["Copy matches", "Move matches"])
        self.unmatched = QComboBox()
        self.unmatched.addItems(["Keep unmatched in place", "Move unmatched to _unmatched", "Copy unmatched to _unmatched"])
        self.status = QLabel("Provide a keyword JSON file and an image folder.")
        self.status.setWordWrap(True)
        form = QFormLayout()
        form.addRow("Images:", self.file_row(self.folder, "Choose folder", self.choose_folder))
        form.addRow("Class keywords:", self.file_row(self.keywords, "Choose JSON", self.choose_keywords))
        form.addRow("Tesseract:", self.file_row(self.tesseract, "Choose exe", self.choose_tesseract))
        form.addRow("Language:", self.language)
        form.addRow("Max OCR size:", self.max_dimension)
        form.addRow("Matched images:", self.mode)
        form.addRow("No keyword match:", self.unmatched)
        button = QPushButton("OCR classify all images")
        button.clicked.connect(self.classify)
        box = QGroupBox("OCR keyword classifier")
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
        browse = QPushButton(text)
        browse.clicked.connect(handler)
        row.addWidget(browse)
        return widget

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose image folder")
        if path: self.folder.setText(path)

    def choose_keywords(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose keyword JSON", "", "JSON (*.json)")
        if path: self.keywords.setText(path)

    def choose_tesseract(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose Tesseract", "", "Executable (*.exe);;All files (*)")
        if path: self.tesseract.setText(path)

    def read_keyword_map(self, path: Path) -> dict[str, list[str]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data:
            raise ValueError("Keyword JSON must be an object: class name mapped to a list of words.")
        result = {}
        for name, words in data.items():
            if isinstance(words, str): words = [words]
            if not isinstance(words, list) or not words:
                raise ValueError(f"Class '{name}' must have a non-empty keyword list.")
            result[str(name)] = [str(word).strip().lower() for word in words if str(word).strip()]
        return result

    def classify(self) -> None:
        try:
            import pytesseract
            from PIL import Image, ImageEnhance, ImageOps
        except ImportError:
            self.error("Install OCR dependencies: pip install pytesseract pillow")
            return
        try:
            folder = Path(self.folder.text().strip())
            keyword_file = Path(self.keywords.text().strip())
            tess_cmd = Path(self.tesseract.text().strip())
            if not folder.is_dir() or not keyword_file.is_file(): raise ValueError("Choose a valid image folder and keyword JSON file.")
            if not tess_cmd.is_file(): raise ValueError(f"Tesseract was not found: {tess_cmd}")
            language = self.language.text().strip() or "eng"
            max_dimension = int(self.max_dimension.text().strip())
            if max_dimension < 400: raise ValueError("Max OCR size must be at least 400.")
            keyword_map = self.read_keyword_map(keyword_file)
            pytesseract.pytesseract.tesseract_cmd = str(tess_cmd)
            files = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: p.name.lower())
            rows = []
            for number, source in enumerate(files, 1):
                try:
                    with Image.open(source) as original:
                        image = ImageOps.exif_transpose(original).convert("L")
                        scale = min(1.0, max_dimension / max(image.size))
                        if scale < 1:
                            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
                        image = ImageEnhance.Contrast(image).enhance(1.5)
                        text = pytesseract.image_to_string(image, lang=language, config="--oem 1 --psm 6", timeout=45)
                    normalized = text.casefold()
                    scores = {name: sum(len(re.findall(re.escape(word), normalized)) for word in words) for name, words in keyword_map.items()}
                    best = max(scores, key=scores.get) if scores and max(scores.values()) > 0 else "_unmatched"
                    row = {"file": str(source), "class": best, "score": scores.get(best, 0), "scores": json.dumps(scores), "ocr_text": " ".join(text.split())[:1000], "status": ""}
                    if best == "_unmatched":
                        action = self.unmatched.currentIndex()
                        row["status"] = "unmatched"
                        if action == 1: self.transfer(source, folder / best)
                        elif action == 2: self.copy(source, folder / best)
                    else:
                        row["status"] = "matched"
                        if self.mode.currentIndex() == 0: self.copy(source, folder / best)
                        else: self.transfer(source, folder / best)
                    rows.append(row)
                    self.status.setText(f"Processed {number}/{len(files)}: {source.name} → {best}")
                except Exception as error:
                    rows.append({"file": str(source), "class": "", "score": 0, "scores": "{}", "ocr_text": "", "status": f"error: {error}"})
            report = folder / "ocr_keyword_report.csv"
            with report.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=["file", "class", "score", "scores", "ocr_text", "status"])
                writer.writeheader(); writer.writerows(rows)
            self.status.setText(f"Finished: {len(files)} images. Report: {report}")
        except Exception as error:
            self.error(str(error))

    @staticmethod
    def copy(source: Path, folder: Path) -> None:
        folder.mkdir(exist_ok=True)
        target = folder / source.name
        shutil.copy2(source, target)

    @staticmethod
    def transfer(source: Path, folder: Path) -> None:
        folder.mkdir(exist_ok=True)
        shutil.move(str(source), str(folder / source.name))

    def error(self, message: str) -> None:
        self.status.setText(f"Error: {message}")
        QMessageBox.critical(self, "OCR classification failed", message)
