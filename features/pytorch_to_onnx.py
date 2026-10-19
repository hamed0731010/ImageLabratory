from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PyTorchToONNX(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.model_path = QLineEdit()
        self.architecture_path = QLineEdit()
        self.class_name = QLineEdit()
        self.input_shape = QLineEdit("1,3,224,224")
        self.output_path = QLineEdit()
        self.model_type = QComboBox()
        self.model_type.addItems(["Complete model", "TorchScript", "State dictionary"])
        self.opset = QLineEdit("17")
        self.dynamic_batch = QComboBox()
        self.dynamic_batch.addItems(["No", "Yes"])
        self.architecture_path.setPlaceholderText("Required for state dictionary")
        self.class_name.setPlaceholderText("For example: MyModel")

        form = QFormLayout()
        form.addRow("Model type:", self.model_type)
        form.addRow("PyTorch file:", self.file_row(self.model_path, "Choose model", self.choose_model))
        form.addRow("Architecture .py:", self.file_row(self.architecture_path, "Choose Python file", self.choose_architecture))
        form.addRow("Model class:", self.class_name)
        form.addRow("Input shape:", self.input_shape)
        form.addRow("ONNX output:", self.file_row(self.output_path, "Choose output", self.choose_output, save=True))
        form.addRow("ONNX opset:", self.opset)
        form.addRow("Dynamic batch:", self.dynamic_batch)

        self.convert_button = QPushButton("Convert to ONNX")
        self.convert_button.clicked.connect(self.convert)
        self.status = QLabel("Select a PyTorch model to begin.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #9aa4b2;")

        box = QGroupBox("PyTorch → ONNX")
        box.setLayout(form)
        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addWidget(self.convert_button)
        layout.addWidget(self.status)
        layout.addStretch()

    @staticmethod
    def file_row(field: QLineEdit, text: str, handler, save: bool = False) -> QWidget:
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
        if path:
            self.model_path.setText(path)
            if not self.output_path.text():
                self.output_path.setText(str(Path(path).with_suffix(".onnx")))

    def choose_architecture(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose model architecture", "", "Python (*.py)")
        if path:
            self.architecture_path.setText(path)

    def choose_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save ONNX model", "model.onnx", "ONNX (*.onnx)")
        if path:
            self.output_path.setText(path)

    def convert(self) -> None:
        try:
            import torch
        except ImportError:
            self.error("PyTorch is not installed. Install it in the active virtual environment.")
            return
        model_path = Path(self.model_path.text().strip())
        output_path = Path(self.output_path.text().strip())
        if not model_path.is_file() or not output_path:
            self.error("Choose a valid PyTorch file and ONNX output path.")
            return
        try:
            shape = tuple(int(value.strip()) for value in self.input_shape.text().split(","))
            if not shape or any(value <= 0 for value in shape):
                raise ValueError
            opset = int(self.opset.text().strip())
        except ValueError:
            self.error("Input shape must contain positive integers, and opset must be a number.")
            return
        try:
            if self.model_type.currentText() == "TorchScript":
                model = torch.jit.load(str(model_path), map_location="cpu")
            elif self.model_type.currentText() == "State dictionary":
                model = self.load_state_dict_model(torch, model_path)
            else:
                model = torch.load(str(model_path), map_location="cpu", weights_only=False)
                if isinstance(model, dict):
                    raise ValueError("This file is a state dictionary. Select State dictionary and provide its architecture.")
            model.eval()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dummy = torch.randn(*shape)
            dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if self.dynamic_batch.currentText() == "Yes" else None
            torch.onnx.export(model, dummy, str(output_path), opset_version=opset, input_names=["input"], output_names=["output"], dynamic_axes=dynamic_axes)
            self.status.setText(f"Export complete: {output_path}")
        except Exception as error:  # conversion errors vary by model/operator
            self.error(str(error))

    def load_state_dict_model(self, torch, model_path: Path):
        architecture = Path(self.architecture_path.text().strip())
        class_name = self.class_name.text().strip()
        if not architecture.is_file() or not class_name:
            raise ValueError("State dictionary conversion requires an architecture .py file and model class name.")
        spec = importlib.util.spec_from_file_location("user_model_architecture", architecture)
        if not spec or not spec.loader:
            raise ValueError("Could not load the architecture Python file.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        model_class = getattr(module, class_name, None)
        if model_class is None:
            raise ValueError(f"Class '{class_name}' was not found in {architecture.name}.")
        model = model_class()
        state = torch.load(str(model_path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        return model

    def error(self, message: str) -> None:
        self.status.setText(f"Error: {message}")
        QMessageBox.critical(self, "Conversion failed", message)
