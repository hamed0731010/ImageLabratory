from __future__ import annotations

import shutil
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QTimer, QRect, Qt
from PyQt6.QtGui import QAction, QIcon, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QRubberBand,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from features.pytorch_to_onnx import PyTorchToONNX
from features.onnx_classifier import ONNXClassifier
from features.pytorch_classifier import PyTorchClassifier
from features.ocr_keyword_classifier import OCRKeywordClassifier
from features.training import TrainingFeature
from features.model_evaluation import ModelEvaluationFeature
from features.clustering import ClusteringFeature
from features.template_generator import TemplateGenerator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
DATABASE_PATH = Path(__file__).with_name("image_label_studio.db")


class ImageView(QGraphicsView):
    """Displays the original image, fitting it to the available area without changing the file."""

    def __init__(self) -> None:
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item: QGraphicsPixmapItem | None = None
        self.user_zoomed = False
        self.crop_mode = False
        self.crop_band = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport())
        self.crop_start = None
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def show_image(self, path: Path) -> tuple[int, int] | None:
        image = QImage(str(path))
        if image.isNull():
            return None
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(QPixmap.fromImage(image))
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.user_zoomed = False
        self.fit_image()
        return image.width(), image.height()

    def fit_image(self) -> None:
        if self.pixmap_item:
            self.user_zoomed = False
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_in(self) -> None:
        if self.pixmap_item:
            self.user_zoomed = True
            self.scale(1.25, 1.25)

    def zoom_out(self) -> None:
        if self.pixmap_item:
            self.user_zoomed = True
            self.scale(0.8, 0.8)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if not self.pixmap_item:
            return
        self.user_zoomed = True
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def set_crop_mode(self, enabled: bool) -> None:
        self.crop_mode = enabled
        self.setDragMode(QGraphicsView.DragMode.NoDrag if enabled else QGraphicsView.DragMode.ScrollHandDrag)
        if not enabled:
            self.crop_start = None
            self.crop_band.hide()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.crop_mode and event.button() == Qt.MouseButton.LeftButton:
            self.crop_start = event.position().toPoint()
            self.crop_band.setGeometry(QRect(self.crop_start, self.crop_start))
            self.crop_band.show()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.crop_mode and self.crop_start is not None:
            self.crop_band.setGeometry(QRect(self.crop_start, event.position().toPoint()).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self.crop_mode and event.button() == Qt.MouseButton.LeftButton:
            self.crop_band.setGeometry(QRect(self.crop_start, event.position().toPoint()).normalized())
            self.crop_start = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def selected_image_rect(self) -> QRect | None:
        if not self.pixmap_item or not self.crop_band.isVisible() or self.crop_band.width() < 5 or self.crop_band.height() < 5:
            return None
        viewport_rect = self.crop_band.geometry()
        top_left = self.mapToScene(viewport_rect.topLeft()).toPoint()
        bottom_right = self.mapToScene(viewport_rect.bottomRight()).toPoint()
        image_rect = self.pixmap_item.boundingRect().toRect()
        return QRect(top_left, bottom_right).normalized().intersected(image_rect)


class LabelStudio(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Laboratory")
        self.setWindowIcon(QIcon(str(Path(__file__).with_name("image_laboratory_logo.png"))))
        self.resize(1200, 800)
        self.image_dir: Path | None = None
        self.images: list[Path] = []
        self.index = -1
        self.classes: list[str] = []
        self.current_size: tuple[int, int] | None = None
        self.rotation = 0
        self.mirrored = False
        self.undo_record = None
        self.next_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self.next_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.next_shortcut.activated.connect(self.next_image)
        self.previous_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.previous_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.previous_shortcut.activated.connect(self.previous_image)
        self.zoom_in_shortcut = QShortcut(QKeySequence("Ctrl++"), self)
        self.zoom_in_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.zoom_in_shortcut.activated.connect(self.image_view_zoom_in)
        self.zoom_out_shortcut = QShortcut(QKeySequence("Ctrl+-"), self)
        self.zoom_out_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.zoom_out_shortcut.activated.connect(self.image_view_zoom_out)
        self.cancel_crop_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.cancel_crop_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.cancel_crop_shortcut.activated.connect(lambda: self.image_view.set_crop_mode(False))
        self.rotate_shortcut = QShortcut(QKeySequence("Ctrl+Right"), self)
        self.rotate_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.rotate_shortcut.activated.connect(self.rotate_current)
        self.mirror_shortcut = QShortcut(QKeySequence("Ctrl+Up"), self)
        self.mirror_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.mirror_shortcut.activated.connect(self.mirror_current)

        self.directory_edit = QLineEdit()
        self.directory_edit.setReadOnly(True)
        browse = QPushButton("Choose image folder…")
        browse.clicked.connect(self.choose_directory)

        self.class_edit = QLineEdit()
        self.class_edit.setPlaceholderText("cat, dog, car")
        self.class_edit.returnPressed.connect(self.set_classes)
        class_button = QPushButton("Set classes")
        class_button.clicked.connect(self.set_classes)
        save_setup_button = QPushButton("Save setup")
        save_setup_button.clicked.connect(self.save_setup)
        load_setup_button = QPushButton("Load from database")
        load_setup_button.clicked.connect(self.load_setup)

        self.transfer_mode = QComboBox()
        self.transfer_mode.addItems(["Copy labeled images", "Move labeled images"])

        rotate_button = QPushButton("Rotate 90°")
        rotate_button.clicked.connect(self.rotate_current)
        mirror_button = QPushButton("Mirror")
        mirror_button.clicked.connect(self.mirror_current)
        rotate_all_button = QPushButton("Rotate all source images 90°")
        rotate_all_button.clicked.connect(self.rotate_all_images)
        zoom_in_button = QPushButton("Zoom in")
        zoom_in_button.clicked.connect(self.image_view_zoom_in)
        zoom_out_button = QPushButton("Zoom out")
        zoom_out_button.clicked.connect(self.image_view_zoom_out)
        fit_button = QPushButton("Fit")
        fit_button.clicked.connect(self.image_view_fit)
        save_edits_button = QPushButton("Save rotation/mirror")
        save_edits_button.clicked.connect(self.save_current_edits)
        start_crop_button = QPushButton("Start crop selection")
        start_crop_button.clicked.connect(self.start_crop_selection)
        save_crop_button = QPushButton("Crop selected & save")
        save_crop_button.clicked.connect(self.crop_selected_and_save)
        save_crop_new_button = QPushButton("Crop selected as new")
        save_crop_new_button.clicked.connect(self.crop_selected_as_new)
        cancel_crop_button = QPushButton("Cancel crop (Esc)")
        cancel_crop_button.clicked.connect(lambda: self.image_view.set_crop_mode(False))

        setup = QGroupBox("1. Setup")
        form = QFormLayout(setup)
        folder_row = QWidget()
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.addWidget(self.directory_edit)
        folder_layout.addWidget(browse)
        class_row = QWidget()
        class_layout = QHBoxLayout(class_row)
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_layout.addWidget(self.class_edit)
        class_layout.addWidget(class_button)
        form.addRow("Images:", folder_row)
        form.addRow("Classes:", class_row)
        form.addRow("Transfer:", self.transfer_mode)
        setup_buttons = QWidget()
        setup_buttons_layout = QHBoxLayout(setup_buttons)
        setup_buttons_layout.setContentsMargins(0, 0, 0, 0)
        setup_buttons_layout.addWidget(save_setup_button)
        setup_buttons_layout.addWidget(load_setup_button)
        form.addRow("Saved setups:", setup_buttons)
        image_tools = QWidget()
        image_tools_layout = QHBoxLayout(image_tools)
        image_tools_layout.setContentsMargins(0, 0, 0, 0)
        image_tools_layout.addWidget(rotate_button)
        image_tools_layout.addWidget(mirror_button)
        image_tools_layout.addWidget(rotate_all_button)
        image_tools_layout.addWidget(zoom_in_button)
        image_tools_layout.addWidget(zoom_out_button)
        image_tools_layout.addWidget(fit_button)
        image_tools_layout.addWidget(save_edits_button)
        image_tools_layout.addWidget(start_crop_button)
        image_tools_layout.addWidget(save_crop_button)
        image_tools_layout.addWidget(save_crop_new_button)
        image_tools_layout.addWidget(cancel_crop_button)
        form.addRow("Image tools:", image_tools)

        self.image_view = ImageView()
        self.placeholder = QLabel("Choose an image folder and enter your classes to begin.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #9aa4b2; font-size: 18px;")
        self.image_view.scene.addWidget(self.placeholder)

        self.position_label = QLabel("No images loaded")
        self.size_label = QLabel("")
        self.current_label = QLabel("Select a class below")
        self.current_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.class_buttons = QWidget()
        self.class_buttons_layout = QHBoxLayout(self.class_buttons)
        self.class_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.class_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.previous_button = QPushButton("← Previous")
        self.skip_button = QPushButton("Skip")
        self.next_button = QPushButton("Next →")
        self.undo_button = QPushButton("Undo last label")
        self.previous_button.clicked.connect(self.previous_image)
        self.skip_button.clicked.connect(self.skip_image)
        self.next_button.clicked.connect(self.next_image)
        self.undo_button.clicked.connect(self.undo_last_label)
        self.previous_button.setEnabled(False)
        self.skip_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.undo_button.setEnabled(False)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(self.current_label, alignment=Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.class_buttons)
        nav = QHBoxLayout()
        nav.addWidget(self.previous_button)
        nav.addStretch()
        nav.addWidget(self.skip_button)
        nav.addWidget(self.next_button)
        nav.addWidget(self.undo_button)
        controls_layout.addLayout(nav)

        manual_page = QWidget()
        layout = QVBoxLayout(manual_page)
        self.sidebar_toggle_button = QPushButton("Hide sidebar")
        self.sidebar_toggle_button.setFixedWidth(120)
        self.sidebar_toggle_button.clicked.connect(lambda: self.toggle_sidebar(not self.sidebar.isVisible()))
        layout.addWidget(self.sidebar_toggle_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(setup)
        layout.addWidget(self.image_view, stretch=1)
        info = QHBoxLayout()
        info.addWidget(self.position_label)
        info.addStretch()
        info.addWidget(self.size_label)
        layout.addLayout(info)
        layout.addWidget(controls)

        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(190)
        self.sidebar.setStyleSheet("QWidget { background: #20252b; } QPushButton { padding: 12px 8px; text-align: left; color: #e8edf2; border: 0; } QPushButton:checked { background: #3b82f6; }")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(8, 12, 8, 8)
        sidebar_title = QLabel("FEATURES")
        sidebar_title.setStyleSheet("color: #9aa4b2; font-size: 11px; font-weight: bold; padding: 4px 8px;")
        sidebar_layout.addWidget(sidebar_title)
        manual_button = QPushButton("Manual labeling")
        manual_button.setCheckable(True)
        manual_button.setChecked(True)
        sidebar_layout.addWidget(manual_button)
        onnx_button = QPushButton("PyTorch → ONNX   [DISABLED]")
        onnx_button.setCheckable(True)
        onnx_button.setToolTip("Temporarily disabled")
        classifier_button = QPushButton("ONNX classifier   [DISABLED]")
        classifier_button.setCheckable(True)
        classifier_button.setToolTip("Temporarily disabled")
        pytorch_classifier_button = QPushButton("PyTorch classifier")
        pytorch_classifier_button.setCheckable(True)
        sidebar_layout.addWidget(pytorch_classifier_button)
        ocr_button = QPushButton("OCR keyword classifier")
        ocr_button.setCheckable(True)
        sidebar_layout.addWidget(ocr_button)
        training_button = QPushButton("Image classification (train)")
        training_button.setCheckable(True)
        sidebar_layout.addWidget(training_button)
        evaluation_button = QPushButton("Model test & metrics")
        evaluation_button.setCheckable(True)
        sidebar_layout.addWidget(evaluation_button)
        clustering_button = QPushButton("Image clustering   [DISABLED]")
        clustering_button.setCheckable(True)
        clustering_button.setToolTip("Temporarily disabled")
        sidebar_layout.addWidget(clustering_button)
        template_button = QPushButton("Template generator   [DISABLED]")
        template_button.setCheckable(True)
        template_button.setToolTip("Temporarily disabled")
        sidebar_layout.addWidget(template_button)
        # ONNX workflows are kept available in the code but temporarily disabled.
        onnx_button.setEnabled(False)
        classifier_button.setEnabled(False)
        clustering_button.setEnabled(False)
        template_button.setEnabled(False)
        onnx_button.setStyleSheet("color: #7f8793;")
        classifier_button.setStyleSheet("color: #7f8793;")
        clustering_button.setStyleSheet("color: #7f8793;")
        template_button.setStyleSheet("color: #7f8793;")
        sidebar_layout.addWidget(onnx_button)
        sidebar_layout.addWidget(classifier_button)
        sidebar_layout.addStretch()

        onnx_page = PyTorchToONNX()
        classifier_page = ONNXClassifier()
        pytorch_classifier_page = PyTorchClassifier()
        ocr_page = OCRKeywordClassifier()
        training_page = TrainingFeature()
        evaluation_page = ModelEvaluationFeature()
        clustering_page = ClusteringFeature()
        template_page = TemplateGenerator()

        content_stack = QStackedWidget()
        content_stack.addWidget(manual_page)
        content_stack.addWidget(onnx_page)
        content_stack.addWidget(classifier_page)
        content_stack.addWidget(pytorch_classifier_page)
        content_stack.addWidget(ocr_page)
        content_stack.addWidget(training_page)
        content_stack.addWidget(evaluation_page)
        content_stack.addWidget(clustering_page)
        content_stack.addWidget(template_page)
        manual_button.clicked.connect(lambda: content_stack.setCurrentIndex(0))
        manual_button.clicked.connect(lambda: onnx_button.setChecked(False))
        manual_button.clicked.connect(lambda: classifier_button.setChecked(False))
        manual_button.clicked.connect(lambda: pytorch_classifier_button.setChecked(False))
        manual_button.clicked.connect(lambda: ocr_button.setChecked(False))
        onnx_button.clicked.connect(lambda: content_stack.setCurrentIndex(1))
        onnx_button.clicked.connect(lambda: manual_button.setChecked(False))
        onnx_button.clicked.connect(lambda: classifier_button.setChecked(False))
        classifier_button.clicked.connect(lambda: content_stack.setCurrentIndex(2))
        classifier_button.clicked.connect(lambda: manual_button.setChecked(False))
        classifier_button.clicked.connect(lambda: onnx_button.setChecked(False))
        classifier_button.clicked.connect(lambda: pytorch_classifier_button.setChecked(False))
        pytorch_classifier_button.clicked.connect(lambda: content_stack.setCurrentIndex(3))
        pytorch_classifier_button.clicked.connect(lambda: manual_button.setChecked(False))
        pytorch_classifier_button.clicked.connect(lambda: onnx_button.setChecked(False))
        pytorch_classifier_button.clicked.connect(lambda: classifier_button.setChecked(False))
        pytorch_classifier_button.clicked.connect(lambda: ocr_button.setChecked(False))
        ocr_button.clicked.connect(lambda: content_stack.setCurrentIndex(4))
        ocr_button.clicked.connect(lambda: manual_button.setChecked(False))
        ocr_button.clicked.connect(lambda: onnx_button.setChecked(False))
        ocr_button.clicked.connect(lambda: classifier_button.setChecked(False))
        ocr_button.clicked.connect(lambda: pytorch_classifier_button.setChecked(False))
        ocr_button.clicked.connect(lambda: training_button.setChecked(False))
        training_button.clicked.connect(lambda: content_stack.setCurrentIndex(5))
        training_button.clicked.connect(lambda: manual_button.setChecked(False))
        training_button.clicked.connect(lambda: onnx_button.setChecked(False))
        training_button.clicked.connect(lambda: classifier_button.setChecked(False))
        training_button.clicked.connect(lambda: pytorch_classifier_button.setChecked(False))
        training_button.clicked.connect(lambda: ocr_button.setChecked(False))
        training_button.clicked.connect(lambda: clustering_button.setChecked(False))
        evaluation_button.clicked.connect(lambda: content_stack.setCurrentIndex(6))
        evaluation_button.clicked.connect(lambda: manual_button.setChecked(False))
        evaluation_button.clicked.connect(lambda: training_button.setChecked(False))
        evaluation_button.clicked.connect(lambda: clustering_button.setChecked(False))
        evaluation_button.clicked.connect(lambda: template_button.setChecked(False))
        clustering_button.clicked.connect(lambda: content_stack.setCurrentIndex(7))
        clustering_button.clicked.connect(lambda: manual_button.setChecked(False))
        clustering_button.clicked.connect(lambda: onnx_button.setChecked(False))
        clustering_button.clicked.connect(lambda: classifier_button.setChecked(False))
        clustering_button.clicked.connect(lambda: pytorch_classifier_button.setChecked(False))
        clustering_button.clicked.connect(lambda: ocr_button.setChecked(False))
        clustering_button.clicked.connect(lambda: training_button.setChecked(False))
        clustering_button.clicked.connect(lambda: template_button.setChecked(False))
        template_button.clicked.connect(lambda: content_stack.setCurrentIndex(8))
        template_button.clicked.connect(lambda: manual_button.setChecked(False))
        template_button.clicked.connect(lambda: onnx_button.setChecked(False))
        template_button.clicked.connect(lambda: classifier_button.setChecked(False))
        template_button.clicked.connect(lambda: pytorch_classifier_button.setChecked(False))
        template_button.clicked.connect(lambda: ocr_button.setChecked(False))
        template_button.clicked.connect(lambda: training_button.setChecked(False))
        template_button.clicked.connect(lambda: clustering_button.setChecked(False))

        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addWidget(self.sidebar)
        central_layout.addWidget(content_stack, stretch=1)
        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        for button in self.findChildren(QPushButton):
            button.clicked.connect(self.show_processing_feedback)
        self.status.showMessage("Ready")
        self.create_menu()
        self.init_database()
        self.restore_last_setup()

    def show_processing_feedback(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.status.showMessage("Processing...")
        QTimer.singleShot(1200, self.clear_processing_feedback)

    def clear_processing_feedback(self) -> None:
        if QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        self.status.showMessage("Ready")

    def create_menu(self) -> None:
        menu = self.menuBar().addMenu("View")
        sidebar_action = QAction("Show sidebar", self)
        sidebar_action.setCheckable(True)
        sidebar_action.setChecked(True)
        sidebar_action.triggered.connect(self.toggle_sidebar)
        menu.addAction(sidebar_action)
        fit = QAction("Fit image to window", self)
        fit.setShortcut("F")
        fit.triggered.connect(self.image_view.fit_image)
        menu.addAction(fit)
        reset = QAction("Reset / choose another folder", self)
        reset.triggered.connect(self.reset)
        menu.addAction(reset)
        edit_menu = self.menuBar().addMenu("Edit")
        undo = QAction("Undo last label", self)
        undo.setShortcut("Ctrl+Z")
        undo.triggered.connect(self.undo_last_label)
        edit_menu.addAction(undo)

    def toggle_sidebar(self, visible: bool) -> None:
        self.sidebar.setVisible(visible)
        self.sidebar_toggle_button.setText("Hide sidebar" if visible else "Show sidebar")

    def choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose image folder")
        if directory:
            self.image_dir = Path(directory)
            self.directory_edit.setText(str(self.image_dir))
            self.load_images()

    def init_database(self) -> None:
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS setups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    image_dir TEXT NOT NULL,
                    classes TEXT NOT NULL,
                    current_image TEXT,
                    updated_at TEXT NOT NULL
                )"""
            )
            columns = [row[1] for row in connection.execute("PRAGMA table_info(setups)")]
            if "current_image" not in columns:
                connection.execute("ALTER TABLE setups ADD COLUMN current_image TEXT")

    def save_setup(self) -> None:
        if not self.image_dir or not self.classes:
            QMessageBox.information(self, "Nothing to save", "Choose an image folder and set your classes first.")
            return
        default_name = self.image_dir.name or "My labeling setup"
        name, accepted = QInputDialog.getText(self, "Save setup", "Setup name:", text=default_name)
        name = name.strip()
        if not accepted or not name:
            return
        try:
            with sqlite3.connect(DATABASE_PATH) as connection:
                connection.execute(
                    "INSERT INTO setups(name, image_dir, classes, updated_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET image_dir=excluded.image_dir, classes=excluded.classes, "
                    "updated_at=excluded.updated_at",
                    (name, str(self.image_dir), json.dumps(self.classes), datetime.now().isoformat(timespec="seconds")),
                )
                connection.execute(
                    "UPDATE setups SET current_image = ? WHERE name = ?",
                    (self.images[self.index].name if 0 <= self.index < len(self.images) else None, name),
                )
            self.status.showMessage(f"Saved setup: {name}")
        except sqlite3.Error as error:
            QMessageBox.critical(self, "Database error", str(error))

    def load_setup(self) -> None:
        with sqlite3.connect(DATABASE_PATH) as connection:
            rows = connection.execute("SELECT name FROM setups ORDER BY updated_at DESC").fetchall()
        if not rows:
            QMessageBox.information(self, "No saved setups", "Save a setup first.")
            return
        names = [row[0] for row in rows]
        name, accepted = QInputDialog.getItem(self, "Load setup", "Choose a saved setup:", names, 0, False)
        if accepted and name:
            self.apply_setup(name)

    def apply_setup(self, name: str) -> None:
        with sqlite3.connect(DATABASE_PATH) as connection:
            row = connection.execute("SELECT image_dir, classes, current_image FROM setups WHERE name = ?", (name,)).fetchone()
        if not row:
            return
        image_dir, classes_json, current_image = row
        try:
            classes = json.loads(classes_json)
        except json.JSONDecodeError:
            QMessageBox.warning(self, "Invalid setup", "The saved class list is not valid.")
            return
        self.image_dir = Path(image_dir)
        self.directory_edit.setText(str(self.image_dir))
        self.classes = classes
        self.class_edit.setText(", ".join(self.classes))
        self.rebuild_class_buttons()
        if self.image_dir.is_dir():
            self.load_images()
            if current_image:
                matching = next((i for i, image in enumerate(self.images) if image.name == current_image), None)
                if matching is not None:
                    self.index = matching
                    self.show_current()
        else:
            self.clear_image("Saved folder is missing. Choose the folder again.")
            QMessageBox.warning(self, "Folder not found", str(self.image_dir))

    def restore_last_setup(self) -> None:
        with sqlite3.connect(DATABASE_PATH) as connection:
            row = connection.execute("SELECT name FROM setups ORDER BY updated_at DESC LIMIT 1").fetchone()
        if row:
            self.apply_setup(row[0])

    def set_classes(self) -> None:
        values = [part.strip() for part in self.class_edit.text().split(",")]
        self.classes = list(dict.fromkeys(value for value in values if value))
        if not self.classes:
            QMessageBox.warning(self, "Classes required", "Enter one or more class names separated by commas.")
            return
        self.rebuild_class_buttons()
        if self.image_dir:
            self.load_images()

    def rebuild_class_buttons(self) -> None:
        while self.class_buttons_layout.count():
            item = self.class_buttons_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for number, class_name in enumerate(self.classes, 1):
            button = QPushButton(f"{number}: {class_name}")
            button.setToolTip(f"Press {number} to label as {class_name}")
            button.setMinimumWidth(110)
            button.clicked.connect(lambda checked=False, name=class_name: self.label_current(name))
            self.class_buttons_layout.addWidget(button)

    def load_images(self) -> None:
        if not self.image_dir:
            return
        self.images = sorted(
            (p for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda p: p.name.lower(),
        )
        self.index = 0 if self.images else -1
        if not self.images:
            self.clear_image("No supported images found in this folder.")
        else:
            self.show_current()

    def show_current(self) -> None:
        if self.index < 0 or self.index >= len(self.images):
            self.clear_image("Finished — no more images.")
            return
        path = self.images[self.index]
        self.rotation = 0
        self.mirrored = False
        self.current_size = self.image_view.show_image(path)
        self.position_label.setText(f"Image {self.index + 1} of {len(self.images)}: {path.name}")
        self.size_label.setText(f"{self.current_size[0]} × {self.current_size[1]} px" if self.current_size else "Unreadable image")
        self.previous_button.setEnabled(self.index > 0)
        self.skip_button.setEnabled(True)
        self.next_button.setEnabled(True)
        self.current_label.setText("Choose a class to copy and advance")
        self.status.showMessage("Tip: scroll over the image to zoom; drag to pan.")
        self.save_progress()

    def save_progress(self) -> None:
        if not self.image_dir or self.index < 0 or self.index >= len(self.images):
            return
        try:
            with sqlite3.connect(DATABASE_PATH) as connection:
                connection.execute(
                    "UPDATE setups SET current_image = ?, updated_at = ? WHERE image_dir = ?",
                    (self.images[self.index].name, datetime.now().isoformat(timespec="seconds"), str(self.image_dir)),
                )
        except sqlite3.Error:
            pass

    def clear_image(self, message: str) -> None:
        self.image_view.scene.clear()
        self.placeholder = QLabel(message)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #9aa4b2; font-size: 18px;")
        self.image_view.scene.addWidget(self.placeholder)
        self.position_label.setText(message)
        self.size_label.clear()
        self.previous_button.setEnabled(False)
        self.skip_button.setEnabled(False)
        self.next_button.setEnabled(False)

    def label_current(self, class_name: str) -> None:
        if self.index < 0 or not self.image_dir:
            return
        source = self.images[self.index]
        destination_dir = self.image_dir / class_name
        destination_dir.mkdir(exist_ok=True)
        destination = destination_dir / source.name
        if destination.exists():
            answer = QMessageBox.question(self, "File already exists", f"Replace {destination.name} in {class_name}?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            was_move = self.transfer_mode.currentIndex() == 1
            if self.rotation or self.mirrored:
                transformed = self.transformed_image(source)
                if transformed.isNull() or not transformed.save(str(destination)):
                    raise OSError("The transformed image could not be saved.")
                if was_move:
                    source.unlink()
            elif not was_move:
                shutil.copy2(source, destination)
            else:
                shutil.move(str(source), str(destination))
        except OSError as error:
            QMessageBox.critical(self, "Copy failed", str(error))
            return
        self.status.showMessage(f"Copied to {class_name}\\{source.name}")
        self.undo_record = {"source": source, "destination": destination, "was_move": was_move, "index": self.index}
        self.undo_button.setEnabled(True)
        self.next_image()

    def undo_last_label(self) -> None:
        if not self.undo_record:
            return
        record = self.undo_record
        source = record["source"]
        destination = record["destination"]
        try:
            if record["was_move"]:
                if destination.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(destination), str(source))
            elif destination.exists():
                destination.unlink()
            self.index = record["index"]
            self.undo_record = None
            self.undo_button.setEnabled(False)
            self.show_current()
            self.status.showMessage(f"Undid label: {source.name}")
        except OSError as error:
            QMessageBox.critical(self, "Undo failed", str(error))

    def transformed_image(self, source: Path) -> QImage:
        image = QImage(str(source))
        if self.rotation:
            image = image.transformed(self.rotation_transform())
        if self.mirrored:
            image = image.mirrored(True, False)
        return image

    def rotation_transform(self):
        from PyQt6.QtGui import QTransform
        transform = QTransform()
        transform.rotate(self.rotation)
        return transform

    def refresh_current_transform(self) -> None:
        if self.index < 0 or self.index >= len(self.images):
            return
        image = self.transformed_image(self.images[self.index])
        self.image_view.scene.clear()
        self.image_view.pixmap_item = self.image_view.scene.addPixmap(QPixmap.fromImage(image))
        self.image_view.scene.setSceneRect(self.image_view.pixmap_item.boundingRect())
        self.image_view.user_zoomed = False
        self.image_view.fit_image()
        self.current_size = image.width(), image.height()
        self.size_label.setText(f"{image.width()} × {image.height()} px")

    def rotate_current(self) -> None:
        if self.index >= 0 and self.index < len(self.images):
            self.rotation = (self.rotation + 90) % 360
            self.refresh_current_transform()

    def image_view_zoom_in(self) -> None:
        self.image_view.zoom_in()

    def image_view_zoom_out(self) -> None:
        self.image_view.zoom_out()

    def image_view_fit(self) -> None:
        self.image_view.fit_image()

    def mirror_current(self) -> None:
        if self.index >= 0 and self.index < len(self.images):
            self.mirrored = not self.mirrored
            self.refresh_current_transform()

    def save_current_edits(self) -> None:
        if self.index < 0 or self.index >= len(self.images):
            return
        source = self.images[self.index]
        if not self.rotation and not self.mirrored:
            QMessageBox.information(self, "No edits", "Rotate or mirror the image first.")
            return
        answer = QMessageBox.question(self, "Replace original image?", f"Save rotation/mirror changes to {source.name} and replace the original?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.replace_source_image(self.transformed_image(source), source)
            self.rotation = 0
            self.mirrored = False
            self.show_current()
            self.status.showMessage(f"Saved edits to {source.name}")
        except OSError as error:
            QMessageBox.critical(self, "Save failed", str(error))

    def crop_current(self) -> None:
        if self.index < 0 or self.index >= len(self.images):
            return
        source = self.images[self.index]
        image = self.transformed_image(source)
        if image.isNull():
            return
        x, accepted = QInputDialog.getInt(self, "Crop image", "Left (x):", 0, 0, image.width() - 1)
        if not accepted:
            return
        y, accepted = QInputDialog.getInt(self, "Crop image", "Top (y):", 0, 0, image.height() - 1)
        if not accepted:
            return
        width, accepted = QInputDialog.getInt(self, "Crop image", "Width:", image.width() - x, 1, image.width() - x)
        if not accepted:
            return
        height, accepted = QInputDialog.getInt(self, "Crop image", "Height:", image.height() - y, 1, image.height() - y)
        if not accepted:
            return
        cropped = image.copy(QRect(x, y, width, height))
        answer = QMessageBox.question(self, "Replace original image?", f"Save this crop and replace {source.name}?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.replace_source_image(cropped, source)
            self.rotation = 0
            self.mirrored = False
            self.show_current()
            self.status.showMessage(f"Saved crop to {source.name}")
        except OSError as error:
            QMessageBox.critical(self, "Crop save failed", str(error))

    def start_crop_selection(self) -> None:
        if self.index < 0 or self.index >= len(self.images):
            return
        self.image_view.set_crop_mode(True)
        self.status.showMessage("Drag over the image to select a crop, then click 'Crop selected & save'.")

    def crop_selected_and_save(self) -> None:
        if self.index < 0 or self.index >= len(self.images):
            return
        rect = self.image_view.selected_image_rect()
        if rect is None or rect.width() < 2 or rect.height() < 2:
            QMessageBox.information(self, "Select crop", "Click 'Start crop selection' and drag over the image first.")
            return
        source = self.images[self.index]
        image = self.transformed_image(source)
        cropped = image.copy(rect)
        answer = QMessageBox.question(self, "Replace original image?", f"Save this selected crop and replace {source.name}?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.replace_source_image(cropped, source)
            self.image_view.set_crop_mode(False)
            self.rotation = 0
            self.mirrored = False
            self.show_current()
            self.status.showMessage(f"Saved selected crop to {source.name}")
        except OSError as error:
            QMessageBox.critical(self, "Crop save failed", str(error))

    def crop_selected_as_new(self) -> None:
        if self.index < 0 or self.index >= len(self.images):
            return
        rect = self.image_view.selected_image_rect()
        if rect is None or rect.width() < 2 or rect.height() < 2:
            QMessageBox.information(self, "Select crop", "Click 'Start crop selection' and drag over the image first.")
            return
        source = self.images[self.index]
        cropped = self.transformed_image(source).copy(rect)
        output = source.with_name(f"{source.stem}_crop_001{source.suffix}")
        number = 1
        while output.exists():
            number += 1
            output = source.with_name(f"{source.stem}_crop_{number:03d}{source.suffix}")
        try:
            self.replace_source_image(cropped, output)
            self.image_view.set_crop_mode(False)
            self.status.showMessage(f"Saved new cropped image: {output.name}")
            QMessageBox.information(self, "Done", f"New cropped image saved successfully:\n{output}")
        except OSError as error:
            QMessageBox.critical(self, "Crop save failed", str(error))

    @staticmethod
    def replace_source_image(image: QImage, source: Path) -> None:
        temporary = source.with_name(f".{source.stem}.codex_tmp{source.suffix}")
        image_format = source.suffix.lstrip(".").upper()
        if image_format == "JPG":
            image_format = "JPEG"
        if not image.save(str(temporary), image_format, 95):
            raise OSError("Could not save the edited image.")
        temporary.replace(source)

    def rotate_all_images(self) -> None:
        if not self.images:
            return
        answer = QMessageBox.question(
            self,
            "Rotate all source images?",
            "This will permanently rotate every image in the selected folder by 90 degrees. Continue?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        failed = []
        for path in self.images:
            image = QImage(str(path))
            rotated = image.transformed(self.rotation_transform_for(90))
            if rotated.isNull() or not rotated.save(str(path)):
                failed.append(path.name)
        if failed:
            QMessageBox.warning(self, "Some images failed", "Could not rotate: " + ", ".join(failed))
        self.show_current()

    @staticmethod
    def rotation_transform_for(degrees: int):
        from PyQt6.QtGui import QTransform
        transform = QTransform()
        transform.rotate(degrees)
        return transform

    def next_image(self) -> None:
        if self.index < len(self.images) - 1:
            self.index += 1
            self.show_current()
        else:
            self.index = len(self.images)
            self.clear_image("Finished — all images processed.")

    def previous_image(self) -> None:
        if self.images and self.index > 0:
            self.index -= 1
            self.show_current()

    def skip_image(self) -> None:
        self.next_image()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Left:
            self.previous_image()
            event.accept()
            return
        if key == Qt.Key.Key_Right:
            self.next_image()
            event.accept()
            return
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            class_index = key - Qt.Key.Key_1
            if class_index < len(self.classes):
                self.label_current(self.classes[class_index])
                event.accept()
                return
        super().keyPressEvent(event)

    def reset(self) -> None:
        self.image_dir = None
        self.images = []
        self.index = -1
        self.directory_edit.clear()
        self.clear_image("Choose an image folder and enter your classes to begin.")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Image Laboratory")
    app.setWindowIcon(QIcon(str(Path(__file__).with_name("image_laboratory_logo.png"))))
    window = LabelStudio()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
