import sys
import os
from math import floor
from pathlib import Path
from collections import namedtuple
from typing import Callable

import cv2
import numpy as np
from PySide6.QtWidgets import QGridLayout, QSpinBox, QLayout, QSizePolicy
from PySide6.QtCore import Qt

from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (QApplication, QWidget, QPushButton, QComboBox, QFileDialog,
                             QHBoxLayout, QVBoxLayout, QLabel, QFrame, QSlider, QCheckBox)

SPRITE_VERT_GAP = 7
BACKGROUND_GRAY_COLOR = 224
SHADOW_COLOR = (40,40,40,127)

CANVAS_SIZES = {
    "Tiny Legacy (35x35)": 35,
    "Tiny (40x40)": 40,
    "Oversize Tiny (60x60)": 60,
    "Small/Medium Legacy (70x70)": 70,
    "Small/Medium (80x80)": 80,
    "Oversize Small/Medium (100x100)": 100,
    "Legacy Large (140x140)": 140,
    "Large (160x160)": 160,
    "Oversize Large (200x200)": 200
}

def convert_cv_qt(cv_img: np.ndarray) -> QPixmap:
    height, width, channel = cv_img.shape
    bytes_per_line = 4 * width
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888).rgbSwapped()
    return QPixmap.fromImage(q_img)


def is_valid_piskel_size(width: int, height: int) -> bool:
    base_width = width // 4
    base_height = height // 4
    return (
        4 * base_width == width
        and 4 * base_height == height
        and width == height
        and base_width in CANVAS_SIZES.values()
    )

# Overlays image a onto image b
def alpha_overlay(a: np.ndarray, b: np.ndarray):
    alpha_a = a[:, :, 3].astype(np.float32) / 255.0
    alpha_b = b[:, :, 3].astype(np.float32) / 255.0
    alpha_o = alpha_a + (1.0 - alpha_a) * alpha_b

    alpha_img_a = np.repeat(np.expand_dims(alpha_a, axis=2), 3, axis=2)
    alpha_img_b = np.repeat(np.expand_dims(alpha_b, axis=2), 3, axis=2)
    alpha_img_o = np.repeat(np.expand_dims(alpha_o, axis=2), 3, axis=2)

    # Compute alpha overlay
    overlay = (a[:, :, :3] * alpha_img_a) + (b[:, :, :3] * alpha_img_b * (1.0 - alpha_img_a))
    bad_alphas = alpha_img_o == 0
    alpha_img_o[bad_alphas] = 1.0
    overlay = np.floor(overlay / alpha_img_o).astype(np.uint8)
    overlay[bad_alphas] = 255
    overlay = np.dstack((overlay, np.floor(alpha_o * 255).astype(np.uint8)))

    return overlay


def get_visible_sprite(sprite):
    visible_rows, visible_cols = np.where(sprite[:, :, 3] != 0)
    min_row = np.min(visible_rows)
    max_row = np.max(visible_rows)
    min_col = np.min(visible_cols)
    max_col = np.max(visible_cols)
    return sprite[min_row:max_row + 1, min_col:max_col + 1]


class SpriteFormatter(QWidget):
    def __init__(self):
        super().__init__()

        self.setFixedSize(800, 340)
        self.setWindowTitle("Foundry Pixel Art Formatter")

        # create the label that holds the image
        self.image_size_px = 80
        self.default_viewport_size_px = 4 * self.image_size_px
        self.canvas_size_dropdown = None
        self.sprite_type_dropdown = None
        self.shadow_slider = None
        self.shadow_y_slider = None
        self.pixel_art_shadow = None
        self.enable_shadow_checkbox = None
        self.auto_shadow_checkbox = None
        self.textbox = None

        self.prev_info_text = ""
        self.error_displayed = False
        self.background = self.create_background_grid()
        self.background_pixmap = convert_cv_qt(self.background)
        self.open_sprite = None
        self.open_sprite_list = None
        self.output_sprite = None
        self.filename = None

        self.options_bar = self.create_options_bar()
        self.image_label = self.create_image_display()

        # create the overall hbox layout separating the image and settings.
        hbox = QHBoxLayout()
        hbox.addLayout(self.options_bar)
        hbox.addWidget(self.image_label)

        # set the hbox layout as the widgets layout
        self.setLayout(hbox)

    def create_background_grid(self) -> np.ndarray:
        background = np.full((self.image_size_px, self.image_size_px, 4), 255, dtype=np.uint8)
        background[::2, ::2, :3] = BACKGROUND_GRAY_COLOR
        background[1::2, 1::2, :3] = BACKGROUND_GRAY_COLOR
        scaled_bg = cv2.resize(src=background, dsize=(self.default_viewport_size_px, self.default_viewport_size_px), interpolation=cv2.INTER_NEAREST_EXACT)
        return scaled_bg

    def update_image(self):
        if self.open_sprite is not None and (
                self.sprite_type_dropdown.getValue() or (
                not self.sprite_type_dropdown.getValue() and is_valid_piskel_size(self.open_sprite.shape[1], self.open_sprite.shape[0])
            )
        ):
            try:
                self.output_sprite = self.process_sprite(self.open_sprite)
                # Scale to canvas size for viewing
                output_scaled = cv2.resize(
                    src=self.output_sprite,
                    dsize=(self.default_viewport_size_px, self.default_viewport_size_px),
                    interpolation=cv2.INTER_NEAREST_EXACT
                )
                viewport_sprite = alpha_overlay(output_scaled, self.background)

                viewport_pixmap = convert_cv_qt(viewport_sprite)
                self.image_label.setPixmap(viewport_pixmap)
            except cv2.error:
                if not self.error_displayed:
                    self.prev_info_text = self.textbox.text()
                self.textbox.setText("<font color='red'>An error occurred while processing the image.</font>")
                self.error_displayed = True
                self.image_label.setPixmap(self.background_pixmap)
                self.set_sprite_options_visibility(False)
                self.open_sprite = None
                self.output_sprite = None
        elif self.open_sprite is None:
            self.image_label.setPixmap(self.background_pixmap)

    def process_sprite(self, sprite: np.ndarray):
        shadow_canvas = np.zeros((self.image_size_px, self.image_size_px, 4), dtype=np.uint8)
        sprite_canvas = np.zeros((self.image_size_px, self.image_size_px, 4), dtype=np.uint8)

        if self.enable_shadow_checkbox.isChecked():
            if self.pixel_art_shadow.isChecked():
                shadow_x = self.image_size_px // 2
                shadow_y = self.image_size_px - self.shadow_y_slider.getValue()
            else:
                # If we are drawing the non-pixelated shadow, we scale the values up to compensate
                # for the larger canvas size
                shadow_x = 2 * self.image_size_px
                shadow_y = 4 * self.image_size_px - self.shadow_y_slider.getValue()
                shadow_canvas = cv2.resize(
                    src=shadow_canvas,
                    dsize=None,
                    fx=4.0, fy=4.0,
                    interpolation=cv2.INTER_NEAREST_EXACT
                )
            shadow_height = self.shadow_slider.getValue() // 5

            # Draw drop shadow on blank canvas
            cv2.ellipse(
                img=shadow_canvas,
                center=(shadow_x, shadow_y),
                axes=(self.shadow_slider.getValue(), shadow_height),
                color=SHADOW_COLOR,
                thickness=-1,
                angle=0,
                startAngle=0,
                endAngle=360,
                lineType=cv2.LINE_AA if not self.pixel_art_shadow.isChecked() else cv2.LINE_8
            )

        if (
            not self.sprite_type_dropdown.getValue()
            and (
                not self.enable_shadow_checkbox.isChecked()
                or self.pixel_art_shadow.isChecked()
            )
        ):
            # Either we're in Piskel mode, and so are assuming the sprite is pre-scaled and so need to make the shadow match (if it doesn't already)
            # or we
            # This is also true if we have no shadow/the shadow is disabled.
            shadow_canvas = cv2.resize(
                src=shadow_canvas,
                dsize=None,
                fx=4.0, fy=4.0,
                interpolation=cv2.INTER_NEAREST_EXACT
            )

        # OMM sprite mode
        if self.sprite_type_dropdown.getValue():
            # Trim transparent parts of the sprite if present. This probably isn't strictly necessary.
            visible_sprite = get_visible_sprite(sprite)

            content_limit_rows = min(visible_sprite.shape[0], self.image_size_px)
            content_limit_cols = min(visible_sprite.shape[1], self.image_size_px)

            row_offset = max(0, self.image_size_px - content_limit_rows - SPRITE_VERT_GAP)
            col_offset = max(0, (self.image_size_px - content_limit_cols) // 2)

            # Cuts the open sprite off at the borders of the canvas if it is too large
            safe_sprite = visible_sprite[:content_limit_rows, :content_limit_cols]

            sprite_canvas_section = sprite_canvas[
                row_offset:row_offset + content_limit_rows,
                col_offset:col_offset + content_limit_cols
            ]

            sprite_canvas_section[...] = safe_sprite[...]
        # Piskel sprite mode
        else:
            # Sprites are pre-scaled and aligned in Piskel mode.
            sprite_canvas = sprite

        if self.enable_shadow_checkbox.isChecked() and not self.pixel_art_shadow.isChecked() and self.sprite_type_dropdown.getValue():
            # Shadow is scaled and sprite isn't, so we need to scale the sprite to match.
            sprite_canvas = cv2.resize(
                src=sprite_canvas,
                dsize=None,
                fx=4.0, fy=4.0,
                interpolation=cv2.INTER_NEAREST_EXACT
            )

        # Place the sprite on the shadow.
        output_sprite = alpha_overlay(sprite_canvas, shadow_canvas)
        if self.pixel_art_shadow.isChecked() and self.sprite_type_dropdown.getValue():
            # Neither the sprite nor shadow was pre-scaled, so we need to do that now.
            output_sprite = cv2.resize(
                src=output_sprite,
                dsize=None,
                fx=4.0, fy=4.0,
                interpolation=cv2.INTER_NEAREST_EXACT
            )

        return output_sprite

    def update_size_dependent_widgets(self):
        if self.pixel_art_shadow.isChecked():
            self.shadow_slider.setRange(0, self.image_size_px)
            self.shadow_y_slider.setRange(0, self.image_size_px)
        else:
            self.shadow_slider.setRange(0, 4 * self.image_size_px)
            self.shadow_y_slider.setRange(0, 4 * self.image_size_px)

        self.background = self.create_background_grid()
        self.background_pixmap = convert_cv_qt(self.background)

    def update_canvas_size(self):
        self.image_size_px = self.canvas_size_dropdown.getValue()
        self.update_size_dependent_widgets()
        self.update_image()

    def sprite_mode_change(self):
        # OMM mode
        if self.sprite_type_dropdown.getValue():
            self.textbox.setText(self.prev_info_text)
            self.error_displayed = False
            self.canvas_size_dropdown.setVisible(True)
            if self.open_sprite is not None:
                self.set_sprite_options_visibility(True)
            self.update_canvas_size()
        # Piskel mode
        else:
            self.canvas_size_dropdown.setVisible(False)
            if self.open_sprite is not None:
                if not is_valid_piskel_size(self.open_sprite.shape[0], self.open_sprite.shape[1]):
                    if not self.error_displayed:
                        self.prev_info_text = self.textbox.text()
                    self.textbox.setText("<font color='red'>Incorrectly sized sprite for piskel mode.</font>")
                    self.error_displayed = True
                    self.set_sprite_options_visibility(False)
                else:
                    self.image_size_px = self.open_sprite.shape[0] // 4
                    self.update_size_dependent_widgets()
                    self.update_image()
        if self.open_sprite is not None:
            self.set_shadow_dims(self.open_sprite)

    def create_dropdown(self, label: str, options: dict[str, int], default_option_idx: int, on_change: Callable, row: int, layout: QGridLayout):
        dropdown_label = QLabel(label)

        dropdown_entry = QComboBox(self)
        for option, val in options.items():
            dropdown_entry.addItem(option, val)
        dropdown_entry.setCurrentIndex(default_option_idx)
        dropdown_entry.currentIndexChanged.connect(on_change)
        policy = QSizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.MinimumExpanding)
        dropdown_entry.setSizePolicy(policy)
        layout.addWidget(dropdown_label, row, 0)
        layout.addWidget(dropdown_entry, row, 1)

        def set_visible(visible: bool):
            dropdown_label.setVisible(visible)
            dropdown_entry.setVisible(visible)

        dropdown = namedtuple('Dropdown', ["getValue", "set_visible"])
        dropdown.getValue = dropdown_entry.currentData
        dropdown.setVisible = set_visible

        return dropdown

    def toggle_shadow_pixel_art_checkbox(self):
        if self.pixel_art_shadow.isChecked():
            self.shadow_slider.setValue(self.shadow_slider.getValue() // 4)
            self.shadow_slider.setRange(0, self.image_size_px)
            self.shadow_y_slider.setValue(self.shadow_y_slider.getValue() // 4)
            self.shadow_y_slider.setRange(0, self.image_size_px)
        else:
            self.shadow_slider.setRange(0, 4 * self.image_size_px)
            self.shadow_slider.setValue(4 * self.shadow_slider.getValue())
            self.shadow_y_slider.setRange(0, 4 * self.image_size_px)
            self.shadow_y_slider.setValue(4 * self.shadow_y_slider.getValue())
        self.update_image()

    def toggle_shadow_visibility_checkbox(self):
        self.set_shadow_options_visibility(self.enable_shadow_checkbox.isChecked())
        self.auto_shadow_checkbox.setVisible(self.open_sprite_list is not None and self.enable_shadow_checkbox.isChecked())
        self.update_image()

    def toggle_auto_shadow_checkbox(self):
        self.set_shadow_options_visibility(self.enable_shadow_checkbox.isChecked())
        self.update_image()

    def create_int_slider(self, slider_label: str, initial_value: int, slider_min: int, slider_max: int, layout: QGridLayout, row: int):
        label = QLabel(self)
        label.setText(slider_label)

        manual_entry = QSpinBox(self)
        manual_entry.setMaximumWidth(80)
        manual_entry.setAlignment(Qt.AlignmentFlag.AlignLeft)
        manual_entry.setRange(0, 9999)

        slider_entry = QSlider(Qt.Orientation.Horizontal, self)
        slider_entry.setTickInterval(1)
        slider_entry.setRange(slider_min, slider_max)
        slider_entry.setValue(initial_value)

        def truncate_entry():
            text_val = int(manual_entry.text())
            if text_val < slider_entry.minimum():
                manual_entry.setValue(slider_entry.minimum())
                slider_entry.setValue(slider_entry.minimum())
            elif text_val > slider_entry.maximum():
                manual_entry.setValue(slider_entry.maximum())
                slider_entry.setValue(slider_entry.maximum())

        manual_entry.editingFinished.connect(truncate_entry)

        def match_bar_to_text():
            if manual_entry.text() != "" and manual_entry.text() != "-":
                text_value = int(manual_entry.text())
                if text_value < slider_entry.minimum():
                    slider_entry.setValue(slider_entry.minimum())
                elif text_value > slider_entry.maximum():
                    slider_entry.setValue(slider_entry.maximum())
                else:
                    slider_entry.setValue(text_value)
                self.update_image()

        def match_text_to_bar():
            manual_entry.setValue(slider_entry.value())
            self.update_image()

        manual_entry.textChanged.connect(match_bar_to_text)
        slider_entry.sliderMoved.connect(match_text_to_bar)
        slider_entry.sliderReleased.connect(match_text_to_bar)

        min_label = QLabel(self)
        min_label.setText(str(slider_min))

        max_label = QLabel(self)
        max_label.setText(str(slider_max))

        layout.addWidget(label, row, 0)
        layout.addWidget(manual_entry, row, 1)
        layout.addWidget(min_label, row, 2)
        layout.addWidget(slider_entry, row, 3)
        layout.addWidget(max_label, row, 4)

        def get_value() -> int:
            return slider_entry.value()

        def set_value(val: int):
            clipped_val = max(slider_entry.minimum(), min(slider_entry.maximum(), val))
            slider_entry.setValue(clipped_val)
            manual_entry.setValue(clipped_val)

        def set_range(min_val: int, max_val: int):
            slider_entry.setRange(min_val, max_val)
            min_label.setText(str(min_val))
            max_label.setText(str(max_val))
            set_value(slider_entry.value())

        def set_visible(visible: bool):
            label.setVisible(visible)
            manual_entry.setVisible(visible)
            min_label.setVisible(visible)
            slider_entry.setVisible(visible)
            max_label.setVisible(visible)

        slider = namedtuple('Slider', ["get_value", "set_value", "set_range", "set_visible"])
        slider.getValue = get_value
        slider.setValue = set_value
        slider.setRange = set_range
        slider.setVisible = set_visible

        return slider

    def create_options_bar(self) -> QLayout:
        open_sprite_btn = QPushButton("Open")
        open_sprite_btn.clicked.connect(self.open_sprite_file)

        export_sprite_btn = QPushButton("Export")
        export_sprite_btn.clicked.connect(self.save_sprite_file)

        file_layout = QHBoxLayout()
        file_layout.addWidget(open_sprite_btn)
        file_layout.addWidget(export_sprite_btn)

        dropdown_layout = QGridLayout()
        sprite_type_dropdown = self.create_dropdown("Sprite Type:", {
            "OMM Sprite": True,
            "Piskel Sprite": False
        }, 0, self.sprite_mode_change, 0, dropdown_layout)
        canvas_size_dropdown = self.create_dropdown("Canvas Size:", CANVAS_SIZES, 4, self.update_canvas_size, 1, dropdown_layout)
        dropdown_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        slider_options = QGridLayout()
        shadow_slider = self.create_int_slider("Shadow Size (px):", 0, 0, self.image_size_px, slider_options, 0)
        shadow_y_slider = self.create_int_slider("Shadow Height (px):", 0, 0, self.image_size_px, slider_options, 1)

        pixel_art_shadow_checkbox = QCheckBox("Pixelate Shadow")
        pixel_art_shadow_checkbox.setChecked(True)
        pixel_art_shadow_checkbox.stateChanged.connect(self.toggle_shadow_pixel_art_checkbox)

        enable_shadow_checkbox = QCheckBox("Enable Shadow")
        enable_shadow_checkbox.setChecked(True)
        enable_shadow_checkbox.stateChanged.connect(self.toggle_shadow_visibility_checkbox)

        checkbox_options = QHBoxLayout()
        checkbox_options.addWidget(enable_shadow_checkbox)
        checkbox_options.addWidget(pixel_art_shadow_checkbox)
        checkbox_options.setAlignment(Qt.AlignmentFlag.AlignLeft)

        automatic_shadow_checkbox = QCheckBox("Automatic Shadow")
        automatic_shadow_checkbox.setChecked(True)
        automatic_shadow_checkbox.setVisible(False)
        automatic_shadow_checkbox.stateChanged.connect(self.toggle_auto_shadow_checkbox)

        textbox = QLabel()

        # Create a vertical box layout on the left for changing options.
        options_bar = QVBoxLayout()
        options_bar.addLayout(file_layout)
        options_bar.addLayout(dropdown_layout)
        options_bar.addLayout(checkbox_options)
        options_bar.addWidget(automatic_shadow_checkbox)
        options_bar.addLayout(slider_options)
        options_bar.addStretch()
        options_bar.addWidget(textbox)

        self.canvas_size_dropdown = canvas_size_dropdown
        self.sprite_type_dropdown = sprite_type_dropdown
        self.shadow_slider = shadow_slider
        self.shadow_y_slider = shadow_y_slider
        self.pixel_art_shadow = pixel_art_shadow_checkbox
        self.enable_shadow_checkbox = enable_shadow_checkbox
        self.auto_shadow_checkbox = automatic_shadow_checkbox
        self.textbox = textbox

        self.set_sprite_options_visibility(False)

        return options_bar

    def set_shadow_options_visibility(self, visible: bool):
        visibility_allowed_filter = self.open_sprite_list is None or (self.open_sprite_list is not None and not self.auto_shadow_checkbox.isChecked())

        self.shadow_slider.setVisible(visible and visibility_allowed_filter)
        self.shadow_y_slider.setVisible(visible and visibility_allowed_filter)
        self.pixel_art_shadow.setVisible(visible and visibility_allowed_filter)

    def set_sprite_options_visibility(self, visible: bool):
        if self.enable_shadow_checkbox.isChecked():
            self.set_shadow_options_visibility(visible)
        self.enable_shadow_checkbox.setVisible(visible)

    def create_image_display(self):
        image_label = QLabel(self)

        image_label.setFrameStyle(QFrame.Shape.Box)
        image_label.setLineWidth(2)
        image_label.setFixedSize(self.default_viewport_size_px, self.default_viewport_size_px)

        image_label.setPixmap(self.background_pixmap)

        return image_label

    def sprite_width(self, sprite: np.ndarray) -> int:
        visible_sprite = get_visible_sprite(sprite)
        _, y = np.where(visible_sprite[:, :, 3] != 0)
        min_col = int(np.min(y))
        max_col = int(np.max(y))

        est_shadow_width = max_col - min_col
        est_shadow_width //= 4 if not self.sprite_type_dropdown.getValue() else 1

        return est_shadow_width

    def set_shadow_dims(self, sprite):
        width = floor(self.sprite_width(sprite) * 0.75)
        height = floor(1.5 * SPRITE_VERT_GAP)
        self.shadow_slider.setValue(width)
        self.shadow_y_slider.setValue(height)


    def open_sprite_file(self):
        files = QFileDialog.getOpenFileNames(
            parent=self,
            caption='Select one or more sprites.',
            dir=os.getcwd(),
            filter='Image File (*.png *.webp)',
            selectedFilter='Image File (*.png *.webp)'
        )

        if len(files[0]) != 0:
            img_path = files[0][0]
            if len(files[0]) == 1:
                self.open_sprite_list = None
                self.auto_shadow_checkbox.setVisible(False)
            else:
                self.open_sprite_list = files[0]
                self.auto_shadow_checkbox.setVisible(True)


            self.filename = Path(img_path).stem
            try:
                sprite = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            except cv2.error:
                if not self.error_displayed:
                    self.prev_info_text = self.textbox.text()
                self.textbox.setText(f"<font color='red'>Could not open sprite at {img_path}.</font>")
                self.error_displayed = True
                return

            if (
                self.sprite_type_dropdown.getValue()
                or (is_valid_piskel_size(sprite.shape[0], sprite.shape[1]) and not self.sprite_type_dropdown.getValue())
            ):
                self.image_size_px = self.canvas_size_dropdown.getValue() if self.sprite_type_dropdown.getValue() else sprite.shape[0] // 4
                self.background = self.create_background_grid()

                if self.open_sprite_list is None:
                    self.prev_info_text = f"Currently Open: {os.path.basename(img_path)}"
                else:
                    self.prev_info_text = f"Currently Open ({len(self.open_sprite_list)} Total Files): {os.path.basename(img_path)}"

                self.error_displayed = False
                self.open_sprite = sprite
                self.set_shadow_dims(sprite)
                self.textbox.setText(self.prev_info_text)
                self.set_sprite_options_visibility(True)
                self.update_image()
            else:
                if not self.error_displayed:
                    self.prev_info_text = self.textbox.text()
                self.textbox.setText("<font color='red'>Incorrectly sized sprite for piskel mode.</font>")
                self.error_displayed = True

    def save_sprite_file(self):
        if self.output_sprite is not None:

            if self.open_sprite_list is None:
                response = QFileDialog.getSaveFileName(
                    parent=self,
                    caption="Select where to export this sprite.",
                    dir=f"{self.filename}.webp",
                    filter="Image File (*.webp)",
                    selectedFilter="Image File (*.webp)"
                )
                export_path = response[0]
            else:
                response = QFileDialog.getExistingDirectory(
                    parent=self,
                    caption="Select where to export these sprites"
                )
                export_path = response

            if export_path != "":
                try:
                    if self.open_sprite_list is None:
                        cv2.imwrite(export_path, self.output_sprite, [int(cv2.IMWRITE_WEBP_QUALITY), 100])
                    else:
                        for sprite_path in self.open_sprite_list:
                            sprite = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
                            if self.auto_shadow_checkbox.isChecked():
                                self.set_shadow_dims(sprite)
                            output_sprite = self.process_sprite(sprite)
                            filename = Path(sprite_path).stem
                            cv2.imwrite(f'{export_path}/{filename}.webp' , output_sprite, [int(cv2.IMWRITE_WEBP_QUALITY), 100])
                        self.set_shadow_dims(self.open_sprite)
                except cv2.error:
                    if not self.error_displayed:
                        self.prev_info_text = self.textbox.text()
                    self.textbox.setText(f"<font color='red'>Could not save sprite at provided path.</font>")
                    self.error_displayed = True

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet('''
        QWidget {
            font-size: 20px;
        }
    ''')

    myApp = SpriteFormatter()
    myApp.show()

    try:
        sys.exit(app.exec())
    except SystemExit:
        print('Closing Window...')