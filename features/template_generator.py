from __future__ import annotations

import random
import shutil
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_TEXT = "Sample\nExample\nPositive\nNegative\nDetected\nNot detected"
DEFAULT_NAMES = "Sara Ahmadi, Mohammad Karimi, Maryam Hosseini, Ali Rezaei, Mina Ebrahimi"
DEFAULT_RESULTS = "Positive, Negative, Detected, Not detected, Borderline"
DEFAULT_CATEGORIES = "Type A, Type B, Unknown"
DEFAULT_PROPERTIES = "Property A, Property B, Property C"


class TemplateWorker(QThread):
    progress = pyqtSignal(str)
    completed = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, templates, output, texts, names, results, categories, properties, per_template, font_dir, min_size, max_size):
        super().__init__(); self.templates, self.output, self.texts = Path(templates), Path(output), texts; self.names, self.results, self.categories, self.properties = names, results, categories, properties; self.per_template, self.font_dir, self.min_size, self.max_size = per_template, Path(font_dir) if font_dir else None, min_size, max_size

    def run(self):
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageEnhance
            templates = sorted((p for p in self.templates.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: str(p).lower())
            if not templates: raise ValueError("No template images found.")
            fonts = []
            if self.font_dir and self.font_dir.is_dir(): fonts = [p for p in self.font_dir.rglob("*") if p.suffix.lower() in {".ttf", ".otf"}]
            if not fonts:
                windows_fonts = Path("C:/Windows/Fonts")
                if windows_fonts.is_dir(): fonts = [p for p in windows_fonts.iterdir() if p.suffix.lower() in {".ttf", ".otf"}]
            if not fonts: raise ValueError("No .ttf or .otf fonts found.")
            self.output.mkdir(parents=True, exist_ok=True); total = len(templates) * self.per_template; done = 0
            for template in templates:
                with Image.open(template) as original:
                    base = original.convert("RGBA")
                for variant in range(self.per_template):
                    image = base.copy(); name = random.choice(self.names); result = random.choice(self.results); category = random.choice(self.categories); property_text = random.choice(self.properties); custom = random.choice(self.texts) if self.texts else "Synthetic sample"; text = f"Name: {name}\nResult: {result}\nCategory: {category}\nProperty: {property_text}\n{custom}"; font_path = random.choice(fonts); size = random.randint(self.min_size, self.max_size); font = ImageFont.truetype(str(font_path), size)
                    layer = Image.new("RGBA", image.size, (0, 0, 0, 0)); draw = ImageDraw.Draw(layer); bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=max(2, size // 5)); text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    if text_width >= image.width or text_height >= image.height: raise ValueError(f"Font size is too large for template {template.name}.")
                    x = random.randint(0, max(0, image.width - text_width)); y = random.randint(0, max(0, image.height - text_height)); color = random.choice([(255,255,255,220), (0,0,0,220), (255,0,0,210), (0,80,180,210)])
                    draw.multiline_text((x, y), text, font=font, fill=color, spacing=max(2, size // 5), stroke_width=random.choice([0, 1, 2]), stroke_fill=(255,255,255,160) if color[0] < 100 else (0,0,0,160))
                    if random.random() < .35: layer = layer.rotate(random.uniform(-8, 8), expand=False, resample=Image.Resampling.BICUBIC)
                    image = Image.alpha_composite(image, layer)
                    if random.random() < .25: image = ImageEnhance.Contrast(image).enhance(random.uniform(.9, 1.1))
                    output = self.output / f"{template.stem}_{variant+1:04d}.png"; image.convert("RGB").save(output, quality=95)
                    done += 1
                    if done == 1 or done % 10 == 0 or done == total: self.progress.emit(f"Generated {done}/{total}: {output.name}")
            self.completed.emit(str(self.output))
        except Exception as error: self.failed.emit(str(error))


class TemplateGenerator(QWidget):
    def __init__(self):
        super().__init__(); self.templates = QLineEdit(); self.output = QLineEdit(); self.fonts = QLineEdit(); self.texts = QLineEdit(DEFAULT_TEXT); self.names = QLineEdit(DEFAULT_NAMES); self.results = QLineEdit(DEFAULT_RESULTS); self.categories = QLineEdit(DEFAULT_CATEGORIES); self.properties = QLineEdit(DEFAULT_PROPERTIES); self.per_template = QSpinBox(); self.per_template.setRange(1, 10000); self.per_template.setValue(10); self.min_size = QSpinBox(); self.min_size.setRange(8, 1000); self.min_size.setValue(18); self.max_size = QSpinBox(); self.max_size.setRange(8, 1000); self.max_size.setValue(48); self.status = QLabel("Generate synthetic text variations from any image template folder."); self.status.setWordWrap(True)
        form = QFormLayout(); form.addRow("Templates:", self.row(self.templates, "Choose folder", self.choose_templates)); form.addRow("Output:", self.row(self.output, "Choose folder", self.choose_output)); form.addRow("Fonts folder:", self.row(self.fonts, "Optional", self.choose_fonts)); form.addRow("Custom text:", self.texts); form.addRow("Names/IDs:", self.names); form.addRow("Results/statuses:", self.results); form.addRow("Categories:", self.categories); form.addRow("Properties:", self.properties); form.addRow("Images per template:", self.per_template); form.addRow("Min font size:", self.min_size); form.addRow("Max font size:", self.max_size); box = QGroupBox("Synthetic template generator"); box.setLayout(form); button = QPushButton("Generate images"); button.clicked.connect(self.start); self.log = QLabel(); self.log.setWordWrap(True); layout = QVBoxLayout(self); layout.addWidget(box); layout.addWidget(button); layout.addWidget(self.status); layout.addWidget(self.log); layout.addStretch()

    @staticmethod
    def row(field, text, handler):
        widget = QWidget(); row = QHBoxLayout(widget); row.setContentsMargins(0,0,0,0); row.addWidget(field); button = QPushButton(text); button.clicked.connect(handler); row.addWidget(button); return widget

    def choose_templates(self):
        path = QFileDialog.getExistingDirectory(self, "Choose template folder"); self.templates.setText(path) if path else None

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "Choose output folder"); self.output.setText(path) if path else None

    def choose_fonts(self):
        path = QFileDialog.getExistingDirectory(self, "Choose fonts folder"); self.fonts.setText(path) if path else None

    def start(self):
        if not Path(self.templates.text()).is_dir(): QMessageBox.warning(self, "Templates required", "Choose a valid template folder."); return
        split = lambda value: [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]
        texts, names, results, categories, properties = split(self.texts.text()), split(self.names.text()), split(self.results.text()), split(self.categories.text()), split(self.properties.text())
        if not all((texts, names, results, categories, properties)): QMessageBox.warning(self, "Data required", "Enter at least one value in every data field."); return
        output = Path(self.output.text()) if self.output.text() else Path(self.templates.text()) / "synthetic_templates"
        self.worker = TemplateWorker(self.templates.text(), output, texts, names, results, categories, properties, self.per_template.value(), self.fonts.text(), self.min_size.value(), self.max_size.value()); self.worker.progress.connect(self.status.setText); self.worker.completed.connect(lambda path: self.status.setText(f"Finished. Images saved to: {path}")); self.worker.failed.connect(lambda error: self.status.setText(f"Generation failed: {error}")); self.worker.start()
