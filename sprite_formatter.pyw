import sys
import os
from math import floor
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt

from PySide6.QtGui import QPixmap, QImage, QColorConstants, QIntValidator
from PySide6.QtWidgets import (QApplication, QWidget, QPushButton, QComboBox, QFileDialog,
                             QHBoxLayout, QVBoxLayout, QLabel, QFrame, QSlider, QLineEdit, QCheckBox)


px_v_gap = 7

def convert_cv_qt(cv_img):
    height, width, channel = cv_img.shape
    bytes_per_line = 4 * width
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888).rgbSwapped()
    return QPixmap.fromImage(q_img)

def createEmptyPixmap(size):
    bg = QPixmap(size, size)
    bg.fill(QColorConstants.Transparent)
    return bg


class SpriteFormatter(QWidget):
    def __init__(self):
        super().__init__()

        self.window_width, self.window_height = 800, 100
        self.setMinimumSize(self.window_width, self.window_height)
        self.setWindowTitle("Foundry Pixel Art Formatter")

        # create the label that holds the image
        self.image_size_px = 80
        self.default_viewport_size_px = 4 * self.image_size_px
        self.canvas_size_dropdown = None
        self.shadow_min_label = None
        self.shadow_max_label = None
        self.shadow_current_val = None
        self.shadow_current_label = None
        self.shadow_current_validator = None
        self.shadow_width_slider = None
        self.pixel_art_shadow_checkbox = None
        self.resize_checkbox = None
        self.textbox = None

        self.open_sprite = None
        self.output_sprite = None
        self.filename = None
        self.shadow_width = None

        self.options_bar = self.createOptionsBar()
        self.image_label = self.createImageDisplay()

        # create the overall hbox layout separating the image and settings.
        hbox = QHBoxLayout()
        hbox.addLayout(self.options_bar)
        hbox.addWidget(self.image_label)

        # set the hbox layout as the widgets layout
        self.setLayout(hbox)

    def updateImage(self):
        if self.open_sprite is not None:
            try:
                canvas = np.zeros((self.image_size_px, self.image_size_px, 4), dtype=np.uint8)

                if self.pixel_art_shadow_checkbox.isChecked():

                    # Draw drop shadow on canvas
                    shadow_x = self.image_size_px // 2
                    shadow_y = max(0, self.image_size_px - floor(1.5 * px_v_gap))
                    shadow_height = self.shadow_width // 5
                    cv2.ellipse(img=canvas, center=(shadow_x, shadow_y), axes=(self.shadow_width, shadow_height), color=(0,0,0,127), thickness=-1, angle=0, startAngle=0, endAngle=360)

                    # Draw sprite on canvas
                    width_offset = max(0, (self.image_size_px - self.open_sprite.shape[1]) // 2)
                    height_offset = max(0, self.image_size_px - self.open_sprite.shape[0] - px_v_gap)

                    content_limit_width = min(self.open_sprite.shape[1], self.image_size_px)
                    content_limit_height = min(self.open_sprite.shape[0], self.image_size_px)

                    # This combines the drop shadow and the sprite drawing
                    visible_pixels = self.open_sprite[:content_limit_height, :content_limit_width][:, :, 3] != 0
                    canvas[
                        height_offset:height_offset + self.open_sprite.shape[0],
                        width_offset:width_offset + self.open_sprite.shape[1]
                    ][visible_pixels] = self.open_sprite[:content_limit_height, :content_limit_width][visible_pixels]

                    # Scale to correct output size
                    scaled_sprite = cv2.resize(src=canvas, dsize=None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST_EXACT)
                    self.output_sprite = scaled_sprite
                else:
                    # Draw sprite on canvas
                    width_offset = max(0, (self.image_size_px - self.open_sprite.shape[1]) // 2)
                    height_offset = max(0, self.image_size_px - self.open_sprite.shape[0] - px_v_gap)

                    content_limit_width = min(self.open_sprite.shape[1], self.image_size_px)
                    content_limit_height = min(self.open_sprite.shape[0], self.image_size_px)

                    canvas[
                        height_offset:height_offset + self.open_sprite.shape[0],
                        width_offset:width_offset + self.open_sprite.shape[1]
                    ] = self.open_sprite[:content_limit_height, :content_limit_width]

                    # Scale to correct output size
                    scaled_sprite = cv2.resize(src=canvas, dsize=None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST_EXACT)

                    shadow_canvas = np.zeros_like(scaled_sprite)

                    # Draw drop shadow on scaled canvas
                    shadow_x = 2 * self.image_size_px
                    shadow_y = max(0, 4 * self.image_size_px - floor(1.5 * 4 * px_v_gap))
                    shadow_height = self.shadow_width // 5
                    cv2.ellipse(img=shadow_canvas, center=(shadow_x, shadow_y), axes=(self.shadow_width, shadow_height),
                                color=(0, 0, 0, 127), thickness=-1, angle=0, startAngle=0, endAngle=360)

                    shadow_canvas[scaled_sprite[:, :, 3] != 0] = scaled_sprite[scaled_sprite[:, :, 3] != 0]

                    scaled_sprite = shadow_canvas
                    self.output_sprite = shadow_canvas

                # Scale to canvas size for viewing
                if self.resize_checkbox.isChecked():
                    scaled_canvas = cv2.resize(src=self.output_sprite, dsize=(self.default_viewport_size_px, self.default_viewport_size_px), interpolation=cv2.INTER_NEAREST_EXACT)
                else:
                    scaled_canvas = scaled_sprite

                canvas_pixmap = convert_cv_qt(scaled_canvas)
                self.image_label.setPixmap(canvas_pixmap)
            except cv2.error:
                self.textbox.setText("<font color='red'>An error occurred while drawing the image. :(</font>")
                bg = createEmptyPixmap(self.default_viewport_size_px)
                self.image_label.setPixmap(bg)
                self.setSpriteOptionVisibility(False)
                self.open_sprite = None
                self.output_sprite = None
                return
        else:
            bg = createEmptyPixmap(self.default_viewport_size_px)
            self.image_label.setPixmap(bg)

    def updateCanvasSize(self):
        self.image_size_px = self.canvas_size_dropdown.currentData()
        self.shadow_width_slider.setMaximum(self.image_size_px)
        self.shadow_current_validator.setTop(self.image_size_px)
        self.shadow_max_label.setText(str(self.image_size_px))
        if self.shadow_width_slider.value() > self.image_size_px:
            self.shadow_width_slider.setValue(self.image_size_px)
            self.shadow_current_val.setText(str(self.image_size_px))
        if not self.resize_checkbox.isChecked():
            self.image_label.setFixedSize(4 * self.image_size_px, 4 * self.image_size_px)
        self.updateImage()

    def createCanvasSizeSelect(self):
        canvas_size_label = QLabel("Canvas Size:")
        canvas_size_dropdown = QComboBox(self)
        canvas_size_dropdown.addItem("Tiny Legacy (35x35)", 35)
        canvas_size_dropdown.addItem("Tiny (40x40)", 40)
        canvas_size_dropdown.addItem("Oversize Tiny (60x60)", 60)
        canvas_size_dropdown.addItem("Small/Medium Legacy (70x70)", 70)
        canvas_size_dropdown.addItem("Small/Medium (80x80)", 80)
        canvas_size_dropdown.addItem("Oversize Small/Medium (100x100)", 100)
        canvas_size_dropdown.addItem("Legacy Large (140x140)", 140)
        canvas_size_dropdown.addItem("Large (160x160)", 160)
        canvas_size_dropdown.addItem("Oversize Large (200x200)", 200)
        canvas_size_dropdown.setCurrentIndex(4)
        canvas_size_dropdown.currentIndexChanged.connect(self.updateCanvasSize)

        canvas_size_select = QHBoxLayout()
        canvas_size_select.addWidget(canvas_size_label)
        canvas_size_select.addWidget(canvas_size_dropdown)
        canvas_size_select.addStretch()

        return canvas_size_select, canvas_size_dropdown

    def changeShadowWidth(self):
        self.shadow_width = self.shadow_width_slider.value()
        self.shadow_current_val.setText(str(self.shadow_width))
        self.updateImage()

    def changeShadowWidthTextEntry(self):
        if self.shadow_current_val.text() != "":
            self.shadow_width = int(self.shadow_current_val.text())
            self.shadow_width_slider.setValue(self.shadow_width)
            self.updateImage()

    def toggleResizeCheckbox(self):
        if self.resize_checkbox.isChecked():
            self.image_label.setFixedSize(self.default_viewport_size_px, self.default_viewport_size_px)
        else:
            self.image_label.setFixedSize(4 * self.image_size_px, 4 * self.image_size_px)
        self.updateImage()

    def toggleShadowPixelArtCheckbox(self):
        if not self.pixel_art_shadow_checkbox.isChecked():
            self.shadow_width_slider.setMaximum(4 * self.image_size_px)
            self.shadow_current_validator.setTop(4 * self.image_size_px)
            self.shadow_max_label.setText(str(4 * self.image_size_px))
            self.shadow_width_slider.setValue(4 * self.shadow_width)
        else:
            self.shadow_width_slider.setMaximum(self.image_size_px)
            self.shadow_current_validator.setTop(self.image_size_px)
            self.shadow_max_label.setText(str(self.image_size_px))
            self.shadow_width_slider.setValue(self.shadow_width // 4)
        self.updateImage()

    def createOptionsBar(self):
        open_sprite_btn = QPushButton("Open Sprite")
        open_sprite_btn.clicked.connect(self.openSpriteFile)

        export_sprite_btn = QPushButton("Export Sprite")
        export_sprite_btn.clicked.connect(self.saveSpriteFile)

        canvas_size_select, canvas_size_dropdown = self.createCanvasSizeSelect()

        shadow_width_slider = QSlider(Qt.Orientation.Horizontal, self)
        shadow_width_slider.setTickInterval(1)
        shadow_width_slider.setRange(0, self.image_size_px)
        shadow_width_slider.valueChanged.connect(self.changeShadowWidth)

        shadow_min_label = QLabel(shadow_width_slider)
        shadow_min_label.setText("0")

        shadow_max_label = QLabel(shadow_width_slider)
        shadow_max_label.setText(str(self.image_size_px))

        shadow_current_validator = QIntValidator(0, self.image_size_px, self)
        shadow_current_val = QLineEdit(self)
        shadow_current_val.setValidator(shadow_current_validator)
        shadow_current_val.setMaximumWidth(60)
        shadow_current_val.setAlignment(Qt.AlignmentFlag.AlignLeft)
        shadow_current_val.textChanged.connect(self.changeShadowWidthTextEntry)

        shadow_current_label = QLabel(shadow_width_slider)
        shadow_current_label.setText("Shadow Size (px):")

        shadow_current = QHBoxLayout()
        shadow_current.addWidget(shadow_current_label)
        shadow_current.addWidget(shadow_current_val)

        labeled_shadow_slider = QHBoxLayout()
        labeled_shadow_slider.addWidget(shadow_min_label)
        labeled_shadow_slider.addWidget(shadow_width_slider)
        labeled_shadow_slider.addWidget(shadow_max_label)

        shadow_slider_layout = QHBoxLayout()
        shadow_slider_layout.addLayout(shadow_current)
        shadow_slider_layout.addLayout(labeled_shadow_slider)

        pixel_art_shadow_checkbox = QCheckBox("Pixel Art Style Shadow")
        pixel_art_shadow_checkbox.setChecked(True)
        pixel_art_shadow_checkbox.stateChanged.connect(self.toggleShadowPixelArtCheckbox)

        resize_checkbox = QCheckBox("Viewport Autoscaling", self)
        resize_checkbox.setChecked(True)
        resize_checkbox.stateChanged.connect(self.toggleResizeCheckbox)

        textbox = QLabel()

        # Create a vertical box layout on the left for changing options.
        options_bar = QVBoxLayout()
        options_bar.addWidget(open_sprite_btn)
        options_bar.addWidget(export_sprite_btn)
        options_bar.addLayout(canvas_size_select)
        options_bar.addLayout(shadow_slider_layout)
        options_bar.addWidget(pixel_art_shadow_checkbox)
        options_bar.addWidget(resize_checkbox)
        options_bar.addStretch()
        options_bar.addWidget(textbox)

        self.canvas_size_dropdown = canvas_size_dropdown
        self.shadow_min_label = shadow_min_label
        self.shadow_max_label = shadow_max_label
        self.shadow_current_val = shadow_current_val
        self.shadow_current_label = shadow_current_label
        self.shadow_current_validator = shadow_current_validator
        self.shadow_width_slider = shadow_width_slider
        self.pixel_art_shadow_checkbox = pixel_art_shadow_checkbox
        self.resize_checkbox = resize_checkbox
        self.textbox = textbox

        self.setSpriteOptionVisibility(False)

        return options_bar

    def setSpriteOptionVisibility(self, visible):
        self.shadow_min_label.setVisible(visible)
        self.shadow_max_label.setVisible(visible)
        self.shadow_current_val.setVisible(visible)
        self.shadow_width_slider.setVisible(visible)
        self.pixel_art_shadow_checkbox.setVisible(visible)
        self.shadow_current_label.setVisible(visible)

    def createImageDisplay(self):
        image_label = QLabel(self)
        image_label.setPixmap(createEmptyPixmap(self.default_viewport_size_px))
        image_label.setFrameStyle(QFrame.Shape.Box)
        image_label.setFixedSize(self.default_viewport_size_px, self.default_viewport_size_px)

        return image_label

    def openSpriteFile(self):
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a file',
            dir=os.getcwd(),
            filter='Image File (*.png *.webp)',
            selectedFilter='Image File (*.png *.webp)'
        )
        img_path = response[0]

        if img_path != '':
            self.filename = Path(img_path).stem
            try:
                self.textbox.setText(f"Currently Open: {os.path.basename(img_path)}")
                self.open_sprite = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                self.shadow_width = floor(self.open_sprite.shape[1] * 0.75)
                self.shadow_width_slider.setValue(self.shadow_width)
                self.shadow_current_val.setText(str(self.shadow_width))
                self.setSpriteOptionVisibility(True)
            except cv2.error:
                self.textbox.setText(f"<font color='red'>Could not open sprite at {img_path}.</font>")
                return
            self.updateImage()

    def saveSpriteFile(self):
        if self.output_sprite is not None:
            response = QFileDialog.getSaveFileName(
                parent=self,
                caption="Select where to export this sprite.",
                dir=f"{self.filename}_sprite.webp",
                filter="Image File (*.webp)",
                selectedFilter="Image File (*.webp)"
            )

            export_path = response[0]
            if export_path != "":
                try:
                    cv2.imwrite(export_path, self.output_sprite, [int(cv2.IMWRITE_WEBP_QUALITY), 100])
                except cv2.error:
                    self.textbox.setText(f"<font color='red'>Could not save sprite at provided path.</font>")

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