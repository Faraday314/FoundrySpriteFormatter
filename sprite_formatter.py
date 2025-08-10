import sys
import os
from math import floor
from pathlib import Path
from collections import namedtuple

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
        self.setFixedWidth(800)
        self.setFixedHeight(340)

        # create the label that holds the image
        self.image_size_px = 80
        self.default_viewport_size_px = 4 * self.image_size_px
        self.canvas_size_dropdown = None
        self.shadow_slider = None
        self.pixel_art_shadow = None
        self.textbox = None

        self.open_sprite = None
        self.output_sprite = None
        self.filename = None

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

                width_offset = max(0, (self.image_size_px - self.open_sprite.shape[1]) // 2)
                height_offset = max(0, self.image_size_px - self.open_sprite.shape[0] - px_v_gap)

                content_limit_width = min(self.open_sprite.shape[1], self.image_size_px)
                content_limit_height = min(self.open_sprite.shape[0], self.image_size_px)

                if self.pixel_art_shadow.isChecked():

                    # Draw drop shadow on canvas
                    shadow_x = self.image_size_px // 2
                    shadow_y = max(0, self.image_size_px - floor(1.5 * px_v_gap))
                    shadow_height = self.shadow_slider.getValue() // 5
                    cv2.ellipse(
                        img=canvas,
                        center=(shadow_x, shadow_y),
                        axes=(self.shadow_slider.getValue(), shadow_height),
                        color=(0,0,0,127),
                        thickness=-1,
                        angle=0,
                        startAngle=0,
                        endAngle=360
                    )

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
                    shadow_height = self.shadow_slider.getValue() // 5
                    cv2.ellipse(img=shadow_canvas, center=(shadow_x, shadow_y), axes=(self.shadow_slider.getValue(), shadow_height),
                                color=(0, 0, 0, 127), thickness=-1, angle=0, startAngle=0, endAngle=360)

                    shadow_canvas[scaled_sprite[:, :, 3] != 0] = scaled_sprite[scaled_sprite[:, :, 3] != 0]

                    self.output_sprite = shadow_canvas

                # Scale to canvas size for viewing
                scaled_canvas = cv2.resize(src=self.output_sprite, dsize=(self.default_viewport_size_px, self.default_viewport_size_px), interpolation=cv2.INTER_NEAREST_EXACT)
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
        self.shadow_slider.setRange(0, self.image_size_px if self.pixel_art_shadow.isChecked() else 4 * self.image_size_px)
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

    def toggleShadowPixelArtCheckbox(self):
        if self.pixel_art_shadow.isChecked():
            self.shadow_slider.setValue(self.shadow_slider.getValue() // 4)
            self.shadow_slider.setRange(0, self.image_size_px)
        else:
            self.shadow_slider.setRange(0, 4 * self.image_size_px)
            self.shadow_slider.setValue(4 * self.shadow_slider.getValue())
        self.updateImage()

    def createIntSlider(self, initial_value, slider_min, slider_max, slider_label):
        label = QLabel(self)
        label.setText(slider_label)

        validator = QIntValidator(self)
        text_entry = QLineEdit(self)
        text_entry.setValidator(validator)
        text_entry.setMaximumWidth(60)
        text_entry.setMaxLength(4)
        text_entry.setAlignment(Qt.AlignmentFlag.AlignLeft)

        slider_entry = QSlider(Qt.Orientation.Horizontal, self)
        slider_entry.setTickInterval(1)
        slider_entry.setRange(slider_min, slider_max)
        slider_entry.setValue(initial_value)

        def truncateEntry():
            text_val = int(text_entry.text())
            if text_val < slider_entry.minimum():
                text_entry.setText(str(slider_entry.minimum()))
                slider_entry.setValue(slider_entry.minimum())
            elif text_val > slider_entry.maximum():
                text_entry.setText(str(slider_entry.maximum()))
                slider_entry.setValue(slider_entry.maximum())

        text_entry.editingFinished.connect(truncateEntry)

        def match_bar_to_text():
            if text_entry.text() != "" and text_entry.text() != "-":
                text_value = int(text_entry.text())
                if text_value < slider_entry.minimum():
                    slider_entry.setValue(slider_entry.minimum())
                elif text_value > slider_entry.maximum():
                    slider_entry.setValue(slider_entry.maximum())
                else:
                    slider_entry.setValue(text_value)
                self.updateImage()

        def match_text_to_bar():
            text_entry.setText(str(slider_entry.value()))
            self.updateImage()

        text_entry.textEdited.connect(match_bar_to_text)
        slider_entry.sliderMoved.connect(match_text_to_bar)
        slider_entry.sliderReleased.connect(match_text_to_bar)

        min_label = QLabel(self)
        min_label.setText(str(slider_min))

        max_label = QLabel(self)
        max_label.setText(str(slider_max))

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(label)
        slider_layout.addWidget(text_entry)
        slider_layout.addWidget(min_label)
        slider_layout.addWidget(slider_entry)
        slider_layout.addWidget(max_label)

        def getValue():
            return slider_entry.value()

        def setValue(val: int):
            clipped_val = max(slider_entry.minimum(), min(slider_entry.maximum(), val))
            slider_entry.setValue(clipped_val)
            text_entry.setText(str(clipped_val))

        def setRange(min_val: int, max_val: int):
            slider_entry.setRange(min_val, max_val)
            min_label.setText(str(min_val))
            max_label.setText(str(max_val))
            setValue(slider_entry.value())

        def setVisible(visible: bool):
            label.setVisible(visible)
            text_entry.setVisible(visible)
            min_label.setVisible(visible)
            slider_entry.setVisible(visible)
            max_label.setVisible(visible)

        slider = namedtuple('Slider', ["layout", "getValue", "setValue", "setRange", "setVisible"])
        slider.layout = slider_layout
        slider.getValue = getValue
        slider.setValue = setValue
        slider.setRange = setRange
        slider.setVisible = setVisible

        return slider

    def createOptionsBar(self):
        open_sprite_btn = QPushButton("Open Sprite")
        open_sprite_btn.clicked.connect(self.openSpriteFile)

        export_sprite_btn = QPushButton("Export Sprite")
        export_sprite_btn.clicked.connect(self.saveSpriteFile)

        file_layout = QHBoxLayout()
        file_layout.addWidget(open_sprite_btn)
        file_layout.addWidget(export_sprite_btn)

        canvas_size_select, canvas_size_dropdown = self.createCanvasSizeSelect()

        shadow_slider = self.createIntSlider(0, 0, self.image_size_px, "Shadow Size (px):")
        shadow_height_slider = self.createIntSlider(0, 0, self.image_size_px, "Shadow Height (px)")

        pixel_art_shadow_checkbox = QCheckBox("Pixel Art Style Shadow")
        pixel_art_shadow_checkbox.setChecked(True)
        pixel_art_shadow_checkbox.stateChanged.connect(self.toggleShadowPixelArtCheckbox)

        textbox = QLabel()

        # Create a vertical box layout on the left for changing options.
        options_bar = QVBoxLayout()
        options_bar.addLayout(file_layout)
        options_bar.addLayout(canvas_size_select)
        options_bar.addLayout(shadow_slider.layout)
        options_bar.addLayout(shadow_height_slider.layout)
        options_bar.addWidget(pixel_art_shadow_checkbox)
        options_bar.addStretch()
        options_bar.addWidget(textbox)

        self.canvas_size_dropdown = canvas_size_dropdown
        self.shadow_slider = shadow_slider
        self.pixel_art_shadow = pixel_art_shadow_checkbox
        self.textbox = textbox

        self.setSpriteOptionVisibility(False)

        return options_bar

    def setSpriteOptionVisibility(self, visible):
        self.shadow_slider.setVisible(visible)
        self.pixel_art_shadow.setVisible(visible)

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
                self.shadow_slider.setValue(floor(self.open_sprite.shape[1] * 0.75))
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