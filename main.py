"""Entry point for the Basic Photo Editor application."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Any

import cv2
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
)


class ImageLabel(QLabel):
    """Simple QLabel that keeps aspect ratio when scaling pixmaps."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText(f"{title}\n(No image loaded)")
        self.setMinimumSize(320, 240)
        self.setStyleSheet("border: 1px solid #ccc; background: #f9f9f9;")
        self._title = title

    def set_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        if pixmap is None:
            self.setText(f"{self._title}\n(No image loaded)")
            self.setPixmap(QPixmap())
            return
        scaled = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)
        self.setText("")

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        if self.pixmap() is not None and not self.pixmap().isNull():
            self.set_pixmap(self.pixmap())
        super().resizeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Basic Photo Editor")
        self.resize(1200, 700)

        # Image states
        self.original_image: Optional[np.ndarray] = None  # set on load only
        self.base_image: Optional[np.ndarray] = None      # committed image
        self.preview_image: Optional[np.ndarray] = None   # shown as Current

        # legacy names (for existing logic)
        self.working_image: Optional[np.ndarray] = None
        self.current_image: Optional[np.ndarray] = None

        # unified history entries with snapshots
        self.history: list[dict[str, Any]] = []
        self.history_list: Optional[QListWidget] = None

        # non-destructive pipeline state
        self.filter_state: dict[str, Any] = {
            "grayscale": False,
            "blur": False,
            "sharpen": False,
            "edge": False,
            "sepia": False,
            "blur_kernel": 5,
        }
        self.adjustment_state: dict[str, int] = {
            "brightness": 0,
            "contrast": 0,
            "saturation": 0,
        }

        # noise cache to avoid stacking repeated applications
        self._noise_seed_image: Optional[np.ndarray] = None
        self._last_noise_kind: Optional[str] = None

        # debounce timer for live adjustments
        self.adjust_timer = QTimer(self)
        self.adjust_timer.setSingleShot(True)
        self.adjust_timer.timeout.connect(self._apply_live_adjustments)

        self._init_ui()

    def _init_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout()
        central.setLayout(root_layout)
        self.setCentralWidget(central)

        # Top action bar
        actions_layout = QHBoxLayout()
        load_btn = QPushButton("Load Image")
        reset_btn = QPushButton("Reset")
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setEnabled(False)
        save_btn = QPushButton("Save Current")
        actions_layout.addWidget(load_btn)
        actions_layout.addWidget(reset_btn)
        actions_layout.addWidget(self.undo_btn)
        actions_layout.addWidget(save_btn)
        actions_layout.addStretch()
        root_layout.addLayout(actions_layout)

        load_btn.clicked.connect(self.load_image)
        reset_btn.clicked.connect(self.reset_image)
        self.undo_btn.clicked.connect(self.undo)
        save_btn.clicked.connect(self.save_image)

        # Main content area
        main_layout = QHBoxLayout()
        root_layout.addLayout(main_layout, stretch=1)

        # Left tools panel
        tools_panel = self._build_tools_panel()
        main_layout.addWidget(tools_panel)

        # Center image views
        center_container = QWidget()
        center_layout = QGridLayout()
        center_container.setLayout(center_layout)
        self.original_view = ImageLabel("Original")
        self.current_view = ImageLabel("Current")
        center_layout.addWidget(self.original_view, 0, 0)
        center_layout.addWidget(self.current_view, 0, 1)
        main_layout.addWidget(center_container, stretch=1)

        # Right analysis panel with histogram
        self.right_panel = self._build_analysis_panel()
        main_layout.addWidget(self.right_panel)

    def _build_analysis_panel(self) -> QGroupBox:
        box = QGroupBox("Analysis")
        layout = QVBoxLayout()

        # Histogram canvas
        self.hist_fig = Figure(figsize=(3, 2), tight_layout=True)
        self.hist_canvas = FigureCanvas(self.hist_fig)
        self.hist_ax = self.hist_fig.add_subplot(111)
        self.hist_ax.set_title("Histogram")
        self.hist_ax.set_xlabel("Intensity")
        self.hist_ax.set_ylabel("Frequency")
        layout.addWidget(self.hist_canvas)

        equalize_btn = QPushButton("Equalize Histogram")
        equalize_btn.clicked.connect(self.equalize_histogram)
        layout.addWidget(equalize_btn)

        # FFT canvas
        self.fft_fig = Figure(figsize=(3, 2), tight_layout=True)
        self.fft_canvas = FigureCanvas(self.fft_fig)
        self.fft_ax = self.fft_fig.add_subplot(111)
        self.fft_ax.set_title("FFT Magnitude")
        layout.addWidget(self.fft_canvas)

        notch_row1 = QHBoxLayout()
        self.notch_radius_slider = self._make_slider(1, 30, 6)
        self.notch_radius_slider.setSingleStep(1)
        notch_row1.addWidget(QLabel("Notch r"))
        notch_row1.addWidget(self.notch_radius_slider)
        layout.addLayout(notch_row1)

        auto_notch_row = QHBoxLayout()
        self.auto_notch_max_slider = self._make_slider(1, 8, 4)
        self.auto_notch_max_slider.setSingleStep(1)
        auto_notch_row.addWidget(QLabel("Auto peaks"))
        auto_notch_row.addWidget(self.auto_notch_max_slider)
        layout.addLayout(auto_notch_row)

        notch_row2 = QHBoxLayout()
        start_notch_btn = QPushButton("Select 2 Notches")
        clear_notch_btn = QPushButton("Clear Notches")
        start_notch_btn.clicked.connect(self.start_notch_selection)
        clear_notch_btn.clicked.connect(self.clear_notches)
        notch_row2.addWidget(start_notch_btn)
        notch_row2.addWidget(clear_notch_btn)
        layout.addLayout(notch_row2)

        auto_notch_btn = QPushButton("Auto Notch Remove")
        auto_notch_btn.clicked.connect(self.auto_notch_filter)
        layout.addWidget(auto_notch_btn)

        band_row = QHBoxLayout()
        self.band_width_slider = self._make_slider(1, 40, 6)
        self.band_width_slider.setSingleStep(1)
        band_row.addWidget(QLabel("Band width"))
        band_row.addWidget(self.band_width_slider)
        layout.addLayout(band_row)

        auto_band_btn = QPushButton("Auto Band-Reject")
        auto_band_btn.clicked.connect(self.auto_band_reject_filter)
        layout.addWidget(auto_band_btn)

        layout.addStretch()
        box.setLayout(layout)
        box.setMinimumWidth(280)
        return box

    def load_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff)",
        )
        if not file_path:
            return

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.original_image = image
        self.base_image = np.copy(image)
        self.preview_image = np.copy(image)
        self.working_image = np.copy(image)
        self.current_image = np.copy(image)
        self._clear_noise_cache()
        self.filter_state = {"grayscale": False, "blur": False, "sharpen": False, "edge": False, "sepia": False}
        self.filter_state["blur_kernel"] = 5
        self.adjustment_state = {"brightness": 0, "contrast": 0, "saturation": 0}
        self._init_history("Loaded image")
        self._reset_notches()
        self._set_transform_defaults(image)
        self._reset_adjustments()
        self._sync_filter_buttons()
        self._recompute_current_image()
        self._update_undo_button()

    def reset_image(self) -> None:
        if self.original_image is None:
            return
        self.base_image = np.copy(self.original_image)
        self.preview_image = np.copy(self.original_image)
        self.working_image = np.copy(self.original_image)
        self.current_image = np.copy(self.original_image)
        self._clear_noise_cache()
        self.filter_state = {"grayscale": False, "blur": False, "sharpen": False, "edge": False, "sepia": False}
        self.filter_state["blur_kernel"] = 5
        self.adjustment_state = {"brightness": 0, "contrast": 0, "saturation": 0}
        self._init_history("Reset")
        self._reset_notches()
        self._set_transform_defaults(self.original_image)
        self._reset_adjustments()
        self._sync_filter_buttons()
        self._recompute_current_image()
        self._update_undo_button()

    def save_image(self) -> None:
        if self.current_image is None:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            str(Path.home() / "edited.png"),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff)",
        )
        if not file_path:
            return

        target = self.preview_image if self.preview_image is not None else self.current_image
        bgr_image = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, bgr_image)

    # --------------------
    # UI building helpers
    # --------------------
    def _build_tools_panel(self) -> QGroupBox:
        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(10)

        # Adjustments
        adjustments_box = QGroupBox("Adjustments")
        adjustments_layout = QVBoxLayout()
        self.brightness_label = QLabel("Brightness: 0")
        self.brightness_slider = self._make_slider(-100, 100, 0)
        self.contrast_label = QLabel("Contrast: 0")
        self.contrast_slider = self._make_slider(-100, 100, 0)
        self.saturation_label = QLabel("Saturation: 0")
        self.saturation_slider = self._make_slider(-100, 100, 0)
        adjustments_layout.addWidget(self.brightness_label)
        adjustments_layout.addWidget(self.brightness_slider)
        adjustments_layout.addWidget(self.contrast_label)
        adjustments_layout.addWidget(self.contrast_slider)
        adjustments_layout.addWidget(self.saturation_label)
        adjustments_layout.addWidget(self.saturation_slider)
        commit_btn = QPushButton("Commit Adjustments")
        commit_btn.clicked.connect(self.commit_adjustments)
        adjustments_layout.addWidget(commit_btn)
        adjustments_box.setLayout(adjustments_layout)
        content_layout.addWidget(adjustments_box)

        # Preset filters
        filters_box = QGroupBox("Preset Filters")
        filters_vlayout = QVBoxLayout()
        filters_grid = QGridLayout()
        filters = [
            ("Grayscale", "grayscale"),
            ("Blur", "blur"),
            ("Sharpen", "sharpen"),
            ("Edge", "edge"),
            ("Sepia", "sepia"),
        ]
        self.filter_buttons: dict[str, QPushButton] = {}
        for idx, (text, key) in enumerate(filters):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, k=key: self.toggle_filter(k, checked))
            self.filter_buttons[key] = btn
            row, col = divmod(idx, 2)
            filters_grid.addWidget(btn, row, col)
        filters_vlayout.addLayout(filters_grid)
        blur_kernel_row = QHBoxLayout()
        blur_kernel_row.addWidget(QLabel("Blur kernel"))
        self.blur_kernel_combo = QComboBox()
        self.blur_kernel_combo.addItems(["3", "5", "7", "9"])
        self.blur_kernel_combo.setCurrentText(str(self.filter_state.get("blur_kernel", 5)))
        self.blur_kernel_combo.currentTextChanged.connect(self.on_blur_kernel_changed)
        blur_kernel_row.addWidget(self.blur_kernel_combo)
        blur_kernel_row.addStretch()
        filters_vlayout.addLayout(blur_kernel_row)
        filters_box.setLayout(filters_vlayout)
        content_layout.addWidget(filters_box)

        # Edge filters
        edges_box = QGroupBox("Edges")
        edges_layout = QVBoxLayout()

        sobel_row = QHBoxLayout()
        sobel_row.addWidget(QLabel("Sobel dir"))
        self.sobel_direction = QComboBox()
        self.sobel_direction.addItems(["X", "Y", "Both"])
        sobel_btn = QPushButton("Apply Sobel")
        sobel_btn.clicked.connect(self.apply_sobel)
        sobel_row.addWidget(self.sobel_direction)
        sobel_row.addWidget(sobel_btn)
        edges_layout.addLayout(sobel_row)

        lap_row = QHBoxLayout()
        lap_row.addWidget(QLabel("Laplacian k"))
        self.laplacian_kernel = QComboBox()
        self.laplacian_kernel.addItems(["1", "3", "5", "7"])
        lap_btn = QPushButton("Apply Laplacian")
        lap_btn.clicked.connect(self.apply_laplacian)
        lap_row.addWidget(self.laplacian_kernel)
        lap_row.addWidget(lap_btn)
        edges_layout.addLayout(lap_row)

        edges_box.setLayout(edges_layout)
        content_layout.addWidget(edges_box)

        # Noise
        noise_box = QGroupBox("Noise")
        noise_layout = QVBoxLayout()

        sp_row = QHBoxLayout()
        sp_row.addWidget(QLabel("Salt & pepper"))
        self.sp_density_slider = self._make_slider(0, 100, 5)
        sp_apply = QPushButton("Apply")
        sp_apply.clicked.connect(self.apply_salt_pepper_noise)
        sp_row.addWidget(self.sp_density_slider)
        sp_row.addWidget(sp_apply)
        noise_layout.addLayout(sp_row)

        periodic_row1 = QHBoxLayout()
        periodic_row1.addWidget(QLabel("Amp"))
        self.per_amp_slider = self._make_slider(0, 100, 20)
        periodic_row1.addWidget(self.per_amp_slider)
        noise_layout.addLayout(periodic_row1)

        periodic_row2 = QHBoxLayout()
        periodic_row2.addWidget(QLabel("fx"))
        self.per_fx_slider = self._make_slider(0, 50, 5)
        periodic_row2.addWidget(self.per_fx_slider)
        periodic_row2.addWidget(QLabel("fy"))
        self.per_fy_slider = self._make_slider(0, 50, 5)
        periodic_row2.addWidget(self.per_fy_slider)
        noise_layout.addLayout(periodic_row2)

        periodic_row3 = QHBoxLayout()
        periodic_row3.addWidget(QLabel("Phase"))
        self.per_phase_slider = self._make_slider(0, 360, 0)
        periodic_row3.addWidget(self.per_phase_slider)
        per_apply = QPushButton("Apply Periodic")
        per_apply.clicked.connect(self.apply_periodic_noise)
        periodic_row3.addWidget(per_apply)
        noise_layout.addLayout(periodic_row3)

        median_row = QHBoxLayout()
        median_row.addWidget(QLabel("Median k"))
        self.median_kernel = QComboBox()
        self.median_kernel.addItems(["1", "3", "5", "7", "9"])
        median_apply = QPushButton("Apply Median")
        median_apply.clicked.connect(self.apply_median_filter)
        median_row.addWidget(self.median_kernel)
        median_row.addWidget(median_apply)
        noise_layout.addLayout(median_row)

        noise_box.setLayout(noise_layout)
        content_layout.addWidget(noise_box)

        # Transformations
        transform_box = QGroupBox("Transform")
        transform_layout = QVBoxLayout()

        crop_row1 = QHBoxLayout()
        crop_row1.addWidget(QLabel("x"))
        self.crop_x = QSpinBox()
        self.crop_x.setRange(0, 10000)
        crop_row1.addWidget(self.crop_x)
        crop_row1.addWidget(QLabel("y"))
        self.crop_y = QSpinBox()
        self.crop_y.setRange(0, 10000)
        crop_row1.addWidget(self.crop_y)
        transform_layout.addLayout(crop_row1)

        crop_row2 = QHBoxLayout()
        crop_row2.addWidget(QLabel("w"))
        self.crop_w = QSpinBox()
        self.crop_w.setRange(1, 10000)
        crop_row2.addWidget(self.crop_w)
        crop_row2.addWidget(QLabel("h"))
        self.crop_h = QSpinBox()
        self.crop_h.setRange(1, 10000)
        crop_row2.addWidget(self.crop_h)
        crop_apply = QPushButton("Crop")
        crop_apply.clicked.connect(self.apply_crop)
        crop_row2.addWidget(crop_apply)
        transform_layout.addLayout(crop_row2)

        rotate_row = QHBoxLayout()
        rot_left = QPushButton("-90°")
        rot_right = QPushButton("+90°")
        rot_left.clicked.connect(lambda: self.apply_rotate(-90))
        rot_right.clicked.connect(lambda: self.apply_rotate(90))
        rotate_row.addWidget(rot_left)
        rotate_row.addWidget(rot_right)
        transform_layout.addLayout(rotate_row)

        rot_row2 = QHBoxLayout()
        rot_row2.addWidget(QLabel("Angle"))
        self.rotate_angle = QDoubleSpinBox()
        self.rotate_angle.setRange(-360.0, 360.0)
        self.rotate_angle.setSingleStep(1.0)
        rot_row2.addWidget(self.rotate_angle)
        rot_apply = QPushButton("Rotate")
        rot_apply.clicked.connect(lambda: self.apply_rotate(self.rotate_angle.value()))
        rot_row2.addWidget(rot_apply)
        transform_layout.addLayout(rot_row2)

        resize_row1 = QHBoxLayout()
        resize_row1.addWidget(QLabel("W"))
        self.resize_w = QSpinBox()
        self.resize_w.setRange(1, 20000)
        resize_row1.addWidget(self.resize_w)
        resize_row1.addWidget(QLabel("H"))
        self.resize_h = QSpinBox()
        self.resize_h.setRange(1, 20000)
        resize_row1.addWidget(self.resize_h)
        transform_layout.addLayout(resize_row1)

        resize_row2 = QHBoxLayout()
        self.resize_keep_aspect = QCheckBox("Keep aspect")
        self.resize_keep_aspect.setChecked(True)
        resize_row2.addWidget(self.resize_keep_aspect)
        resize_apply = QPushButton("Resize")
        resize_apply.clicked.connect(self.apply_resize)
        resize_row2.addWidget(resize_apply)
        transform_layout.addLayout(resize_row2)

        transform_box.setLayout(transform_layout)
        content_layout.addWidget(transform_box)

        # History
        history_box = QGroupBox("History")
        history_layout = QVBoxLayout()
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.on_history_item_clicked)
        history_layout.addWidget(self.history_list)
        history_box.setLayout(history_layout)
        content_layout.addWidget(history_box)

        content_layout.addStretch()
        content.setLayout(content_layout)
        scroll.setWidget(content)
        container_layout.addWidget(scroll)
        container.setLayout(container_layout)
        container.setMinimumWidth(260)

        # Connect sliders after layout build to ensure attributes exist
        self.brightness_slider.valueChanged.connect(self.on_adjustment_changed)
        self.contrast_slider.valueChanged.connect(self.on_adjustment_changed)
        self.saturation_slider.valueChanged.connect(self.on_adjustment_changed)
        self.resize_w.valueChanged.connect(lambda _: setattr(self, "_resize_last_changed", "w"))
        self.resize_h.valueChanged.connect(lambda _: setattr(self, "_resize_last_changed", "h"))

        return container

    def _make_slider(self, minimum: int, maximum: int, value: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        slider.setTickInterval(20)
        slider.setTickPosition(QSlider.TicksBelow)
        return slider

    def _set_spin_value(self, spin: QSpinBox, value: int) -> None:
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)

    def _set_transform_defaults(self, image: Optional[np.ndarray]) -> None:
        if image is None:
            return
        h, w = image.shape[:2]
        self._resize_last_changed = "w"

        for spin, val, minimum in [
            (self.crop_x, 0, 0),
            (self.crop_y, 0, 0),
            (self.crop_w, w, 1),
            (self.crop_h, h, 1),
            (self.resize_w, w, 1),
            (self.resize_h, h, 1),
        ]:
            spin.setMinimum(minimum)
            self._set_spin_value(spin, val)

        self.rotate_angle.blockSignals(True)
        self.rotate_angle.setValue(0.0)
        self.rotate_angle.blockSignals(False)

    # --------------------
    # Adjustment helpers
    # --------------------
    def _schedule_adjustments(self) -> None:
        if self.adjust_timer.isActive():
            self.adjust_timer.stop()
        self.adjust_timer.start(150)

    def _apply_live_adjustments(self) -> None:
        self.recompute_preview()

    def recompute_preview(self) -> None:
        if self.base_image is None:
            return
        img = np.copy(self.base_image)

        fs = self.filter_state if hasattr(self, "filter_state") else {}

        # 1) grayscale / sepia (sepia takes precedence if both toggled)
        if fs.get("sepia"):
            transform = np.array(
                [
                    [0.393, 0.769, 0.189],
                    [0.349, 0.686, 0.168],
                    [0.272, 0.534, 0.131],
                ],
                dtype=np.float32,
            )
            flat = img.reshape((-1, 3)).astype(np.float32)
            sepia_flat = flat @ transform.T
            sepia_flat = np.clip(sepia_flat, 0, 255)
            img = sepia_flat.reshape(img.shape).astype(np.uint8)
        elif fs.get("grayscale"):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 2) blur
        if fs.get("blur"):
            k = int(fs.get("blur_kernel", 5) or 5)
            if k % 2 == 0:
                k += 1
            if k < 3:
                k = 3
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 3) sharpen
        if fs.get("sharpen"):
            kernel = np.array(
                [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
            )
            img = cv2.filter2D(img, -1, kernel)

        # 4) edge
        if fs.get("edge"):
            kernel = np.array(
                [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32
            )
            img = cv2.filter2D(img, -1, kernel)

        img = self._apply_adjustments(img)
        self.preview_image = np.copy(img)
        self.current_image = np.copy(img)
        self._update_views()
        self._update_histogram()
        self._update_fft()

    def finalize_adjustments_if_any(self) -> None:
        if self.base_image is None or self.preview_image is None:
            return
        if (
            self.adjustment_state.get("brightness", 0) == 0
            and self.adjustment_state.get("contrast", 0) == 0
            and self.adjustment_state.get("saturation", 0) == 0
        ):
            return

        self._clear_noise_cache()
        label = (
            f"Adjustments: B={self.adjustment_state['brightness']}, "
            f"C={self.adjustment_state['contrast']}, S={self.adjustment_state['saturation']}"
        )
        self.commit_image(self.preview_image, label=label)
        self._reset_adjustments()
        self.recompute_preview()

    def commit_adjustments(self) -> None:
        self.finalize_adjustments_if_any()

    def _reset_adjustments(self) -> None:
        for label, slider, text in [
            (self.brightness_label, self.brightness_slider, "Brightness"),
            (self.contrast_label, self.contrast_slider, "Contrast"),
            (self.saturation_label, self.saturation_slider, "Saturation"),
        ]:
            slider.blockSignals(True)
            slider.setValue(0)
            label.setText(f"{text}: 0")
            slider.blockSignals(False)
        self.adjustment_state = {"brightness": 0, "contrast": 0, "saturation": 0}

    def _sync_filter_buttons(self) -> None:
        for key, btn in self.filter_buttons.items():
            btn.blockSignals(True)
            btn.setChecked(self.filter_state.get(key, False))
            btn.blockSignals(False)
        if hasattr(self, "blur_kernel_combo"):
            self.blur_kernel_combo.blockSignals(True)
            k = self.filter_state.get("blur_kernel", 5)
            self.blur_kernel_combo.setCurrentText(str(k))
            self.blur_kernel_combo.blockSignals(False)

    def _init_history(self, label: str) -> None:
        self.history = []
        if self.history_list:
            self.history_list.blockSignals(True)
            self.history_list.clear()
            self.history_list.blockSignals(False)
        self.push_history(label)

    def push_history(self, label: str) -> None:
        if self.base_image is None:
            return
        # truncate future entries if user had jumped back
        current_row = self.history_list.currentRow() if self.history_list else -1
        if current_row == -1:
            current_row = len(self.history) - 1
        if current_row < len(self.history) - 1:
            self.history = self.history[: current_row + 1]
            if self.history_list:
                self.history_list.blockSignals(True)
                while self.history_list.count() > current_row + 1:
                    self.history_list.takeItem(self.history_list.count() - 1)
                self.history_list.blockSignals(False)

        entry = {
            "label": label,
            "image": np.copy(self.base_image),
            "filter_state": dict(self.filter_state),
            "adjustment_state": dict(self.adjustment_state),
        }
        self.history.append(entry)
        if self.history_list:
            self.history_list.blockSignals(True)
            self.history_list.addItem(label)
            self.history_list.setCurrentRow(self.history_list.count() - 1)
            self.history_list.blockSignals(False)
        self._update_undo_button()

    def _restore_history_entry(self, entry: dict[str, Any]) -> None:
        restored = entry.get("image")
        if restored is None:
            return
        self.base_image = np.copy(restored)
        self.preview_image = np.copy(restored)
        self.working_image = np.copy(restored)
        self.current_image = np.copy(restored)
        self.filter_state = dict(entry.get("filter_state", {}))
        self.adjustment_state = dict(entry.get("adjustment_state", {}))
        # sync sliders to restored adjustments without triggering signals
        for slider, key, label_widget, text in [
            (self.brightness_slider, "brightness", self.brightness_label, "Brightness"),
            (self.contrast_slider, "contrast", self.contrast_label, "Contrast"),
            (self.saturation_slider, "saturation", self.saturation_label, "Saturation"),
        ]:
            slider.blockSignals(True)
            slider.setValue(self.adjustment_state.get(key, 0))
            label_widget.setText(f"{text}: {self.adjustment_state.get(key, 0)}")
            slider.blockSignals(False)
        self._sync_filter_buttons()
        self._clear_noise_cache()
        self.recompute_preview()
        self._update_undo_button()

    def on_history_item_clicked(self, item) -> None:
        if not self.history_list:
            return
        row = self.history_list.row(item)
        if row < 0 or row >= len(self.history):
            return
        # truncate future history beyond selection
        self.history = self.history[: row + 1]
        self.history_list.blockSignals(True)
        while self.history_list.count() > row + 1:
            self.history_list.takeItem(self.history_list.count() - 1)
        self.history_list.setCurrentRow(row)
        self.history_list.blockSignals(False)
        self._restore_history_entry(self.history[-1])

    # --------------------
    # Adjustments and filters
    # --------------------
    def on_adjustment_changed(self) -> None:
        if self.base_image is None:
            return
        self.adjustment_state["brightness"] = self.brightness_slider.value()
        self.adjustment_state["contrast"] = self.contrast_slider.value()
        self.adjustment_state["saturation"] = self.saturation_slider.value()
        self.brightness_label.setText(
            f"Brightness: {self.adjustment_state['brightness']}"
        )
        self.contrast_label.setText(
            f"Contrast: {self.adjustment_state['contrast']}"
        )
        self.saturation_label.setText(
            f"Saturation: {self.adjustment_state['saturation']}"
        )
        self._schedule_adjustments()

    def _recompute_current_image(self) -> None:
        self.recompute_preview()

    def _update_undo_button(self) -> None:
        if hasattr(self, "undo_btn"):
            self.undo_btn.setEnabled(len(self.history) > 1)

    def commit_image(self, new_base: np.ndarray, label: Optional[str] = None) -> None:
        self.base_image = np.copy(new_base)
        self.preview_image = np.copy(self.base_image)
        self.working_image = np.copy(self.base_image)
        self.current_image = np.copy(self.base_image)
        self.recompute_preview()
        self.push_history(label or "Commit")

    def undo(self) -> None:
        if len(self.history) <= 1:
            return
        target_idx = len(self.history) - 2
        # truncate to target
        self.history = self.history[: target_idx + 1]
        if self.history_list:
            self.history_list.blockSignals(True)
            while self.history_list.count() > target_idx + 1:
                self.history_list.takeItem(self.history_list.count() - 1)
            self.history_list.setCurrentRow(target_idx)
            self.history_list.blockSignals(False)
        self._restore_history_entry(self.history[-1])
        self._update_undo_button()

    def _clear_noise_cache(self) -> None:
        self._noise_seed_image = None
        self._last_noise_kind = None

    def _apply_adjustments(self, image: np.ndarray) -> np.ndarray:
        result = image.astype(np.float32)

        alpha = 1.0 + (self.adjustment_state.get("contrast", 0) / 100.0)
        beta = float(self.adjustment_state.get("brightness", 0))
        result = result * alpha + beta
        result = np.clip(result, 0, 255)

        sat_scale = 1.0 + (self.adjustment_state.get("saturation", 0) / 100.0)
        if sat_scale != 1.0:
            if result.ndim == 2 or (result.ndim == 3 and result.shape[2] == 1):
                result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(
                np.float32
            )
            hsv[..., 1] *= sat_scale
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return result.astype(np.uint8)

    def on_blur_kernel_changed(self, value: str) -> None:
        try:
            k = int(value)
        except (TypeError, ValueError):
            k = 5
        if k % 2 == 0:
            k += 1
        if k < 3:
            k = 3
        if k > 99:
            k = 99 if k % 2 == 1 else 99 - 1
        prev = self.filter_state.get("blur_kernel", 5)
        self.filter_state["blur_kernel"] = k
        # normalize combo display in case value was adjusted
        if hasattr(self, "blur_kernel_combo") and self.blur_kernel_combo.currentText() != str(k):
            self.blur_kernel_combo.blockSignals(True)
            self.blur_kernel_combo.setCurrentText(str(k))
            self.blur_kernel_combo.blockSignals(False)
        if self.base_image is None:
            return
        self.recompute_preview()
        if prev != k:
            self.push_history(f"blur kernel {k}")

    def toggle_filter(self, key: str, checked: bool) -> None:
        if self.base_image is None:
            # ensure UI reflects disabled state when no image loaded
            btn = self.filter_buttons.get(key)
            if btn:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
            return
        prev = self.filter_state.get(key, False)
        self.filter_state[key] = bool(checked)
        # keep button state in sync without re-triggering signals
        btn = self.filter_buttons.get(key)
        if btn:
            btn.blockSignals(True)
            btn.setChecked(bool(checked))
            btn.blockSignals(False)
        self.recompute_preview()
        if prev != self.filter_state[key]:
            self.push_history(f"filter {key} {'on' if checked else 'off'}")

    def apply_grayscale(self) -> None:
        self.toggle_filter("grayscale", True)

    def apply_blur(self) -> None:
        self.toggle_filter("blur", True)

    def apply_sharpen(self) -> None:
        self.toggle_filter("sharpen", True)

    def apply_edge_enhance(self) -> None:
        self.toggle_filter("edge", True)

    def apply_sepia(self) -> None:
        self.toggle_filter("sepia", True)

    def _apply_filter(self, func) -> None:
        # unused with filter_state pipeline
        return

    def apply_sobel(self) -> None:
        if self.preview_image is None:
            return
        img = self.preview_image
        gray = self._to_grayscale(img).astype(np.float32)
        direction = self.sobel_direction.currentText().lower()

        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        if direction == "x":
            mag = np.abs(sobel_x)
        elif direction == "y":
            mag = np.abs(sobel_y)
        else:
            mag = np.hypot(sobel_x, sobel_y)

        norm = self._normalize_to_uint8(mag)
        sobel_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        self.commit_image(sobel_rgb, label="sobel")
        self._reset_adjustments()

    def apply_laplacian(self) -> None:
        if self.preview_image is None:
            return
        img = self.preview_image
        gray = self._to_grayscale(img).astype(np.float32)
        ksize = int(self.laplacian_kernel.currentText())
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
        mag = np.abs(lap)
        norm = self._normalize_to_uint8(mag)
        lap_rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        self.commit_image(lap_rgb, label="laplacian")
        self._reset_adjustments()

    def apply_salt_pepper_noise(self) -> None:
        if self.preview_image is None:
            return
        density = self.sp_density_slider.value() / 100.0
        if density <= 0:
            return

        img = self.preview_image.astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        noisy = img.copy()
        h, w, _ = noisy.shape
        rand = np.random.rand(h, w)
        pepper_mask = rand < (density / 2.0)
        salt_mask = (rand >= (density / 2.0)) & (rand < density)
        noisy[pepper_mask] = 0
        noisy[salt_mask] = 255

        self.commit_image(noisy, label="noise_salt_pepper")
        self._last_noise_kind = "salt_pepper"
        self._reset_adjustments()

    def apply_periodic_noise(self) -> None:
        if self.preview_image is None:
            return
        amp = self.per_amp_slider.value()
        fx = self.per_fx_slider.value()
        fy = self.per_fy_slider.value()
        phase_deg = self.per_phase_slider.value()

        img = self.preview_image.astype(np.float32)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)

        h, w, _ = img.shape
        if (fx == 0 and fy == 0) or amp == 0:
            self.commit_image(img.astype(np.uint8), label="noise_periodic")
            self._reset_adjustments()
            return

        y = np.arange(h).reshape(-1, 1)
        x = np.arange(w).reshape(1, -1)
        phase_rad = np.deg2rad(phase_deg)
        sinusoid = np.sin(2 * np.pi * (fx * x / w + fy * y / h) + phase_rad)
        noise = amp * sinusoid
        noisy = img + noise[..., None]
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        self.commit_image(noisy, label="noise_periodic")
        self._last_noise_kind = "periodic"
        self._reset_adjustments()

    def apply_median_filter(self) -> None:
        if self.preview_image is None:
            return
        k = int(self.median_kernel.currentText())
        if k <= 1:
            # k=1 is no-op but we still just refresh
            self.commit_image(self.preview_image, label="median")
            self._reset_adjustments()
            return

        img = self.preview_image
        if img.ndim == 2:
            filtered = cv2.medianBlur(img, k)
            filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
        else:
            # medianBlur supports multi-channel directly
            filtered_rgb = cv2.medianBlur(img, k)
        self.commit_image(filtered_rgb, label="median")
        self._reset_adjustments()

    # --------------------
    # Transformations: crop, rotate, resize
    # --------------------
    def apply_crop(self) -> None:
        if self.preview_image is None:
            return
        h, w = self.preview_image.shape[:2]
        x = max(0, self.crop_x.value())
        y = max(0, self.crop_y.value())
        cw = max(1, self.crop_w.value())
        ch = max(1, self.crop_h.value())

        if x >= w or y >= h:
            return
        x2 = min(w, x + cw)
        y2 = min(h, y + ch)
        if x2 <= x or y2 <= y or cw <= 0 or ch <= 0:
            return

        cropped = self.preview_image[y:y2, x:x2]
        self.commit_image(cropped, label="crop")
        self._set_transform_defaults(cropped)
        self._reset_adjustments()

    def apply_rotate(self, angle: float) -> None:
        if self.preview_image is None:
            return
        img = self.preview_image
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(m[0, 0])
        sin = abs(m[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        m[0, 2] += (new_w / 2) - center[0]
        m[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(img, m, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.commit_image(rotated, label="rotate")
        self._set_transform_defaults(rotated)
        self._reset_adjustments()

    def apply_resize(self) -> None:
        if self.preview_image is None:
            return
        img = self.preview_image
        h0, w0 = img.shape[:2]
        w_target = max(1, self.resize_w.value())
        h_target = max(1, self.resize_h.value())

        if self.resize_keep_aspect.isChecked():
            last = getattr(self, "_resize_last_changed", "w")
            if last == "h":
                w_target = max(1, int(round(w0 * h_target / h0)))
                self._set_spin_value(self.resize_w, w_target)
            else:
                h_target = max(1, int(round(h0 * w_target / w0)))
                self._set_spin_value(self.resize_h, h_target)

            resized = cv2.resize(img, (w_target, h_target), interpolation=cv2.INTER_CUBIC)
        self.commit_image(resized, label="resize")
        self._set_transform_defaults(resized)
        self._reset_adjustments()

    def equalize_histogram(self) -> None:
        if self.preview_image is None:
            return

        if self.preview_image.ndim == 2 or self.preview_image.shape[2] == 1:
            gray = self._to_grayscale(self.preview_image)
            eq = cv2.equalizeHist(gray)
            eq_rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
        else:
            # Equalize luminance channel
            ycrcb = cv2.cvtColor(self.preview_image, cv2.COLOR_RGB2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y_eq = cv2.equalizeHist(y)
            eq_ycrcb = cv2.merge((y_eq, cr, cb))
            eq_rgb = cv2.cvtColor(eq_ycrcb, cv2.COLOR_YCrCb2RGB)

        self.commit_image(eq_rgb, label="equalize")
        self._reset_adjustments()

    def _update_views(self) -> None:
        self.original_view.set_pixmap(self._to_pixmap(self.original_image))
        self.current_view.set_pixmap(self._to_pixmap(self.current_image))

    def _update_histogram(self) -> None:
        if self.current_image is None:
            self.hist_ax.clear()
            self.hist_ax.set_title("Histogram")
            self.hist_ax.set_xlabel("Intensity")
            self.hist_ax.set_ylabel("Frequency")
            self.hist_canvas.draw()
            return

        gray = self._to_grayscale(self.current_image)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

        self.hist_ax.clear()
        self.hist_ax.plot(hist, color="black")
        self.hist_ax.set_xlim(0, 255)
        self.hist_ax.set_title("Histogram (Current)")
        self.hist_ax.set_xlabel("Intensity")
        self.hist_ax.set_ylabel("Frequency")
        self.hist_canvas.draw()

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.shape[2] == 1:
            return image[..., 0]
        # Luminance approximation
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _to_pixmap(image: Optional[np.ndarray]) -> Optional[QPixmap]:
        if image is None:
            return None
        if image.ndim == 2:
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(
                image.data, width, height, bytes_per_line, QImage.Format_Grayscale8
            )
            return QPixmap.fromImage(q_image)

        height, width, channels = image.shape
        if channels == 1:
            bytes_per_line = width
            q_image = QImage(
                image.data, width, height, bytes_per_line, QImage.Format_Grayscale8
            )
        else:
            bytes_per_line = channels * width
            q_image = QImage(
                image.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
        return QPixmap.fromImage(q_image)

    def _update_fft(self) -> None:
        if self.current_image is None:
            self.fft_ax.clear()
            self.fft_ax.set_title("FFT Magnitude")
            self.fft_canvas.draw()
            return

        gray = self._to_grayscale(self.current_image).astype(np.float32)
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude = 20 * np.log1p(np.abs(dft_shift))

        self.fft_ax.clear()
        self.fft_ax.imshow(magnitude, cmap="magma")
        self.fft_ax.set_title("FFT (log |F|)")
        self.fft_ax.axis("off")
        self.fft_canvas.draw()
        self._dft_cached = dft_shift

    def start_notch_selection(self) -> None:
        if self.current_image is None:
            return
        self.notch_points: list[tuple[int, int]] = []
        if hasattr(self, "_fft_click_cid") and self._fft_click_cid is not None:
            self.fft_canvas.mpl_disconnect(self._fft_click_cid)
        self._fft_click_cid = self.fft_canvas.mpl_connect(
            "button_press_event", self._on_fft_click
        )

    def clear_notches(self) -> None:
        self.notch_points = []
        if hasattr(self, "_fft_click_cid") and self._fft_click_cid is not None:
            self.fft_canvas.mpl_disconnect(self._fft_click_cid)
            self._fft_click_cid = None

    def _on_fft_click(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        self.notch_points.append((y, x))
        if len(self.notch_points) == 2:
            if hasattr(self, "_fft_click_cid") and self._fft_click_cid is not None:
                self.fft_canvas.mpl_disconnect(self._fft_click_cid)
                self._fft_click_cid = None
            self.apply_notch_filter()

    def apply_notch_filter(self) -> None:
        if self.preview_image is None:
            return
        if not hasattr(self, "_dft_cached"):
            return
        if len(getattr(self, "notch_points", [])) != 2:
            return

        gray = self._to_grayscale(self.preview_image).astype(np.float32)
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)

        h, w = gray.shape
        mask = np.ones((h, w), dtype=np.float32)
        r = max(1, self.notch_radius_slider.value())

        yy, xx = np.ogrid[:h, :w]

        for (y, x) in self.notch_points:
            sym_y = (h - y) % h
            sym_x = (w - x) % w
            mask[(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = 0
            mask[(yy - sym_y) ** 2 + (xx - sym_x) ** 2 <= r * r] = 0

        dft_filtered = dft_shift * mask
        ishift = np.fft.ifftshift(dft_filtered)
        img_back = np.fft.ifft2(ishift)
        img_back = np.abs(img_back)
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)

        cleaned_rgb = cv2.cvtColor(img_back, cv2.COLOR_GRAY2RGB)
        self.commit_image(cleaned_rgb, label="notch_filter")
        self._reset_adjustments()
        self.clear_notches()

    # --------------------
    # Automatic periodic noise removal
    # --------------------
    def auto_notch_filter(self) -> None:
        """Detect dominant spikes in FFT magnitude and notch them symmetrically."""
        if self.preview_image is None:
            return

        gray = self._to_grayscale(self.preview_image).astype(np.float32)
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        mag = np.log1p(np.abs(dft_shift))

        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        mag_blur = cv2.GaussianBlur(mag_norm.astype(np.float32), (5, 5), 0)

        # Local maxima detection: peak if it equals its neighborhood max and exceeds a threshold.
        local_max = mag_blur == cv2.dilate(mag_blur, np.ones((3, 3), np.float32))
        thresh = np.median(mag_blur) + 2.5 * np.std(mag_blur)
        mask_peaks = (mag_blur >= thresh) & local_max

        coords = np.argwhere(mask_peaks)
        h, w = mag_blur.shape
        cy, cx = h // 2, w // 2
        min_dc_radius = 8

        # Remove peaks near DC.
        filtered = []
        for y, x in coords:
            if (y - cy) ** 2 + (x - cx) ** 2 <= min_dc_radius * min_dc_radius:
                continue
            filtered.append((y, x, mag_blur[y, x]))

        if not filtered:
            return

        # Sort by strength and keep top-N.
        filtered.sort(key=lambda t: t[2], reverse=True)
        max_peaks = max(1, self.auto_notch_max_slider.value())
        peaks = filtered[:max_peaks]

        mask = np.ones((h, w), dtype=np.float32)
        r = max(1, self.notch_radius_slider.value())
        yy, xx = np.ogrid[:h, :w]

        for y, x, _ in peaks:
            sym_y = (h - y) % h
            sym_x = (w - x) % w
            mask[(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = 0
            mask[(yy - sym_y) ** 2 + (xx - sym_x) ** 2 <= r * r] = 0

        dft_filtered = dft_shift * mask
        ishift = np.fft.ifftshift(dft_filtered)
        img_back = np.fft.ifft2(ishift)
        img_back = np.abs(img_back)
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)

        cleaned_rgb = cv2.cvtColor(img_back, cv2.COLOR_GRAY2RGB)
        self.commit_image(cleaned_rgb, label="auto_notch")
        self._reset_adjustments()

    def auto_band_reject_filter(self) -> None:
        """Detect dominant ring in FFT and apply an annular attenuation (band-reject)."""
        if self.preview_image is None:
            return

        gray = self._to_grayscale(self.preview_image).astype(np.float32)
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        mag = np.abs(dft_shift)

        h, w = mag.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_int = r.astype(np.int32)
        max_r = r_int.max() + 1

        # Radial profile of magnitude to find dominant ring.
        sums = np.bincount(r_int.ravel(), weights=mag.ravel(), minlength=max_r)
        counts = np.bincount(r_int.ravel(), minlength=max_r)
        profile = sums / np.maximum(counts, 1)

        skip_dc = 5
        if profile.size <= skip_dc + 1:
            return
        peak_r = int(np.argmax(profile[skip_dc:]) + skip_dc)

        width = max(1, self.band_width_slider.value())
        mask = np.ones((h, w), dtype=np.float32)
        ring = np.abs(r - peak_r) <= width
        mask[ring] = 0.1  # attenuate band

        dft_filtered = dft_shift * mask
        ishift = np.fft.ifftshift(dft_filtered)
        img_back = np.fft.ifft2(ishift)
        img_back = np.abs(img_back)
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)

        cleaned_rgb = cv2.cvtColor(img_back, cv2.COLOR_GRAY2RGB)
        self.commit_image(cleaned_rgb, label="auto_band")
        self._reset_adjustments()

    def _reset_notches(self) -> None:
        self.notch_points = []
        if hasattr(self, "_fft_click_cid") and self._fft_click_cid is not None:
            self.fft_canvas.mpl_disconnect(self._fft_click_cid)
            self._fft_click_cid = None

    @staticmethod
    def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
        arr = np.abs(arr)
        min_val = float(arr.min())
        max_val = float(arr.max())
        if max_val - min_val < 1e-6:
            return np.zeros_like(arr, dtype=np.uint8)
        norm = (255.0 * (arr - min_val) / (max_val - min_val)).astype(np.uint8)
        return norm


def main() -> None:
    """Start the Qt application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
