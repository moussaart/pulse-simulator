import cv2
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFileDialog, QDoubleSpinBox, 
                             QGroupBox, QFormLayout, QCheckBox, QSlider, 
                             QWidget, QScrollArea, QMessageBox)
from src.gui.widgets import ModernGroupBox, ActionButton, create_themed_spinbox, LabeledSlider
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from src.gui.theme import MODERN_STYLESHEET, COLORS

class ImagePreviewWidget(QLabel):
    # Signals for feedback
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet(f"background-color: {COLORS['widget_bg']}; border: 1px solid {COLORS['border']};")
        self.setMouseTracking(True) # Enable mouse tracking for preview
        
        self.scaled_pixmap = None
        self.original_image = None
        
        self.detected_lines = [] # List of ((x1, y1), (x2, y2))
        self.manual_segments = [] # List of ((x1, y1), (x2, y2))
        
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Editing State
        self.mode = "NONE" # NONE, LINE, POLYGON, DELETE
        self.temp_points = [] # For current drawing
        self.current_mouse_pos = None
        
    def set_image(self, image):
        self.original_image = image
        self.update_preview()
        
    def set_lines(self, lines):
        self.detected_lines = lines
        self.update_preview()
        
    def set_mode(self, mode):
        self.mode = mode
        self.temp_points = []
        self.update_preview()
        
    def clear_manual(self):
        self.manual_segments = []
        self.temp_points = []
        self.update_preview()

    def update_preview(self):
        if self.original_image is None:
            self.setText("No Image Selected")
            return
            
        # Create clear preview from original
        preview_img = self.original_image.copy()
        
        # Convert to RGB for Qt
        if len(preview_img.shape) == 2:
            preview_img = cv2.cvtColor(preview_img, cv2.COLOR_GRAY2RGB)
        else:
            preview_img = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
            
        h, w, ch = preview_img.shape
        bytes_per_line = ch * w
        q_img = QImage(preview_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Draw on pixmap
        painter = QPainter(pixmap)
        
        # 1. Draw Detected Lines
        pen_detected = QPen(QColor(255, 0, 0), 2)  # Red lines
        painter.setPen(pen_detected)
        for p1, p2 in self.detected_lines:
            painter.drawLine(p1[0], p1[1], p2[0], p2[1])
            
        # 2. Draw Manual Segments
        pen_manual = QPen(QColor(0, 0, 255), 2) # Blue lines
        painter.setPen(pen_manual)
        for p1, p2 in self.manual_segments:
            painter.drawLine(p1[0], p1[1], p2[0], p2[1])
            
        # 3. Draw Pending (Temp) Geometry
        pen_temp = QPen(QColor(0, 255, 0), 2, Qt.DashLine) # Green Dashed
        painter.setPen(pen_temp)
        
        if self.mode == "LINE" and len(self.temp_points) == 1 and self.current_mouse_pos:
            # Draw line from start to current mouse
            p1 = self.temp_points[0]
            # Need to inverse map mouse pos to image coords? 
            # Actually we shouldn't draw on the unscaled pixmap with mouse coords.
            # We need to wait until we scale to draw strictly UI overlays, OR map mouse to image.
            # Let's map mouse to image coords.
            img_pos = self._map_to_image(self.current_mouse_pos)
            if img_pos:
                painter.drawLine(p1[0], p1[1], img_pos[0], img_pos[1])
                
        elif self.mode == "POLYGON":
            if len(self.temp_points) > 0:
                # Draw existing segments
                for i in range(len(self.temp_points) - 1):
                    painter.drawLine(self.temp_points[i][0], self.temp_points[i][1], 
                                     self.temp_points[i+1][0], self.temp_points[i+1][1])
                
                # Draw closing line to mouse
                if self.current_mouse_pos:
                    img_pos = self._map_to_image(self.current_mouse_pos)
                    if img_pos:
                        painter.drawLine(self.temp_points[-1][0], self.temp_points[-1][1], 
                                         img_pos[0], img_pos[1])

        painter.end()
        
        # Scale for display
        w_widget = self.width()
        h_widget = self.height()
        self.scaled_pixmap = pixmap.scaled(w_widget, h_widget, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Calculate scaling factors/offsets for mouse mapping
        if self.scaled_pixmap.width() > 0 and self.scaled_pixmap.height() > 0:
            self.scale_x = self.scaled_pixmap.width() / w
            self.scale_y = self.scaled_pixmap.height() / h
            # Center the image in the label
            self.offset_x = (w_widget - self.scaled_pixmap.width()) / 2
            self.offset_y = (h_widget - self.scaled_pixmap.height()) / 2
        
        self.setPixmap(self.scaled_pixmap)

    def _map_to_image(self, pos):
        if not self.original_image is None and self.scale_x > 0:
            x = int((pos.x() - self.offset_x) / self.scale_x)
            y = int((pos.y() - self.offset_y) / self.scale_y)
            # Clamp
            h, w = self.original_image.shape[:2]
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            return (x, y)
        return None

    def mousePressEvent(self, event):
        if self.original_image is None: return
        
        img_pos = self._map_to_image(event.pos())
        if not img_pos: return
        
        if event.button() == Qt.LeftButton:
            if self.mode == "LINE":
                self.temp_points.append(img_pos)
                if len(self.temp_points) == 2:
                    self.manual_segments.append((self.temp_points[0], self.temp_points[1]))
                    self.temp_points = []
                    
            elif self.mode == "POLYGON":
                self.temp_points.append(img_pos)
                # Double click handling is tricky in press, usually done in DblClick event
                # We'll just rely on double click event or checking proximity to start
                if len(self.temp_points) > 2:
                    # Check if closed (clicked near start)
                    start = self.temp_points[0]
                    dist = np.sqrt((img_pos[0]-start[0])**2 + (img_pos[1]-start[1])**2)
                    if dist < 10: # Close polygon
                        # Add last segment back to start
                        self.temp_points.pop() # Remove the click that was just added
                        for i in range(len(self.temp_points) - 1):
                            self.manual_segments.append((self.temp_points[i], self.temp_points[i+1]))
                        self.manual_segments.append((self.temp_points[-1], self.temp_points[0]))
                        self.temp_points = []
            
            elif self.mode == "DELETE":
                self.delete_segment_at(img_pos)
        
        elif event.button() == Qt.RightButton:
            # Cancel current operation
            self.temp_points = []
            
        self.update_preview()

    def mouseMoveEvent(self, event):
        self.current_mouse_pos = event.pos()
        if self.mode in ["LINE", "POLYGON"]:
            self.update_preview()

    def mouseDoubleClickEvent(self, event):
        if self.mode == "POLYGON" and len(self.temp_points) > 1:
            # Finish polygon (open chain) or close it? 
            # Let's say double click finishes the chain as open line strip
            # Or usually polygons are closed. Let's assume user wants to close or finish.
            # For "Floor Plan", walls are lines. So Polygon tool is really "Polyline" tool.
            # Let's just convert temp points to segments
             if len(self.temp_points) >= 2:
                 for i in range(len(self.temp_points) - 1):
                     self.manual_segments.append((self.temp_points[i], self.temp_points[i+1]))
             self.temp_points = []
             self.update_preview()

    def delete_segment_at(self, pos):
        threshold = 10 # pixels
        
        # Check manual segments
        to_remove = []
        for i, (p1, p2) in enumerate(self.manual_segments):
            if self._dist_point_to_segment(pos, p1, p2) < threshold:
                to_remove.append(i)
        
        for i in sorted(to_remove, reverse=True):
            self.manual_segments.pop(i)
            
        # Check detected segments
        to_remove_det = []
        for i, (p1, p2) in enumerate(self.detected_lines):
            if self._dist_point_to_segment(pos, p1, p2) < threshold:
                to_remove_det.append(i)
                
        for i in sorted(to_remove_det, reverse=True):
            self.detected_lines.pop(i)

    def _dist_point_to_segment(self, p, s1, s2):
        x, y = p
        x1, y1 = s1
        x2, y2 = s2
        
        A = x - x1
        B = y - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        param = -1
        if len_sq != 0:
            param = dot / len_sq
            
        if param < 0:
            xx = x1
            yy = y1
        elif param > 1:
            xx = x2
            yy = y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D
            
        dx = x - xx
        dy = y - yy
        return np.sqrt(dx * dx + dy * dy)

    def resizeEvent(self, event):
        self.update_preview()
        super().resizeEvent(event)

class ImageImportWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Floor Plan Image")
        self.resize(1000, 700)
        
        # Apply Theme
        self.setStyleSheet(MODERN_STYLESHEET + f"""
            QDialog {{ background-color: {COLORS['background']}; }}
            QScrollArea {{ background-color: transparent; border: none; }}
            QWidget#ValidBackground {{ background-color: {COLORS['background']}; }}
        """)
        
        self.image_path = None
        self.cv_image = None
        self.gray_image = None
        self.skeleton = None
        
        # UI Setup
        self.setup_ui()
        
        # Processing timer (debounce)
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.process_image)
        
    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # --- Left Panel (Preview) ---
        left_panel = QVBoxLayout()
        
        self.preview_label = ImagePreviewWidget()
        left_panel.addWidget(self.preview_label, stretch=1)
        
        # Info label
        self.info_label = QLabel("No image selected")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold; margin-top: 10px;")
        left_panel.addWidget(self.info_label)
        
        main_layout.addLayout(left_panel, stretch=60)
        
        # --- Right Panel (Controls) ---
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_widget = QWidget()
        right_widget.setObjectName("ValidBackground")
        right_widget.setStyleSheet(f"background-color: {COLORS['background']};")
        right_layout = QVBoxLayout(right_widget)
        
        # 1. File Selection
        file_group = ModernGroupBox("Image File")
        file_layout = QVBoxLayout()
        
        self.select_btn = ActionButton("Select Image...", variant="primary")
        self.select_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.select_btn)
        
        self.path_label = QLabel("None")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #aaa; font-style: italic;")
        file_layout.addWidget(self.path_label)
        
        file_group.setLayout(file_layout)
        right_layout.addWidget(file_group)

        # 2. Manual Edit (Moved to Top)
        manual_group = ModernGroupBox("Manual Edit")
        manual_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_line = ActionButton("Line", variant="secondary")
        self.btn_line.clicked.connect(lambda: self.set_edit_mode("LINE"))
        self.btn_line.setToolTip("Click start, click end to draw a line")
        
        self.btn_poly = ActionButton("Polygon", variant="secondary")
        self.btn_poly.clicked.connect(lambda: self.set_edit_mode("POLYGON"))
        self.btn_poly.setToolTip("Click points to draw, double-click to finish")
        
        self.btn_del = ActionButton("Delete", variant="secondary")
        self.btn_del.clicked.connect(lambda: self.set_edit_mode("DELETE"))
        self.btn_del.setToolTip("Click on a line to delete it")
        
        btn_layout.addWidget(self.btn_line)
        btn_layout.addWidget(self.btn_poly)
        btn_layout.addWidget(self.btn_del)
        manual_layout.addLayout(btn_layout)
        
        self.btn_clear = ActionButton("Clear Manual", variant="danger")
        self.btn_clear.clicked.connect(self.clear_manual)
        manual_layout.addWidget(self.btn_clear)
        
        self.edit_status = QLabel("Mode: None")
        self.edit_status.setAlignment(Qt.AlignCenter)
        manual_layout.addWidget(self.edit_status)
        
        manual_group.setLayout(manual_layout)
        right_layout.addWidget(manual_group)
        
        # 3. Scale & Origin
        geo_group = ModernGroupBox("Geometry")
        geo_layout = QFormLayout()
        
        self.width_spin = create_themed_spinbox(min_val=1, max_val=1000, default=20.0, suffix=" m")
        geo_layout.addRow("Width:", self.width_spin)
        
        self.height_spin = create_themed_spinbox(min_val=1, max_val=1000, default=15.0, suffix=" m")
        geo_layout.addRow("Height:", self.height_spin)
        
        self.off_x_spin = create_themed_spinbox(min_val=-500, max_val=500, default=0.0, suffix=" m")
        geo_layout.addRow("Offset X:", self.off_x_spin)
        
        self.off_y_spin = create_themed_spinbox(min_val=-500, max_val=500, default=0.0, suffix=" m")
        geo_layout.addRow("Offset Y:", self.off_y_spin)
        
        geo_group.setLayout(geo_layout)
        right_layout.addWidget(geo_group)
        
        # 3. Detection Settings
        detect_group = ModernGroupBox("Line Detection")
        detect_layout = QFormLayout()
        
        # Threshold
        self.thresh_slider = LabeledSlider(min_val=1, max_val=200, default=40)
        self.thresh_slider.value_changed.connect(self.update_detection_params)
        detect_layout.addRow("Threshold:", self.thresh_slider)
        
        # Min Line Length
        self.min_len_slider = LabeledSlider(min_val=1, max_val=50, default=3, suffix="%")
        self.min_len_slider.value_changed.connect(self.update_detection_params)
        detect_layout.addRow("Min Length:", self.min_len_slider)
        
        # Max Line Gap
        self.max_gap_slider = LabeledSlider(min_val=1, max_val=50, default=5, suffix="%")
        self.max_gap_slider.value_changed.connect(self.update_detection_params)
        detect_layout.addRow("Max Gap:", self.max_gap_slider)
        
        detect_group.setLayout(detect_layout)
        right_layout.addWidget(detect_group)
        
        # 4. NLOS Parameters
        nlos_group = ModernGroupBox("NLOS Zone Parameters")
        nlos_layout = QFormLayout()
        
        self.wall_thick_spin = create_themed_spinbox(min_val=0.01, max_val=2.0, default=0.2, step=0.05, suffix=" m")
        nlos_layout.addRow("Thickness:", self.wall_thick_spin)
        
        self.error_bias_spin = create_themed_spinbox(default=0.05, step=0.01)
        nlos_layout.addRow("Error Bias:", self.error_bias_spin)
        
        self.noise_factor_spin = create_themed_spinbox(default=1.5)
        nlos_layout.addRow("Noise Factor:", self.noise_factor_spin)

        self.path_loss_spin = create_themed_spinbox(default=2.5)
        nlos_layout.addRow("Path Loss Exp:", self.path_loss_spin)
        
        self.shadow_std_spin = create_themed_spinbox(default=4.0)
        nlos_layout.addRow("Shadow Std:", self.shadow_std_spin)
        
        nlos_group.setLayout(nlos_layout)
        right_layout.addWidget(nlos_group)
        
        
        right_layout.addStretch()
        
        # 5. Actions
        action_layout = QHBoxLayout()
        self.cancel_btn = ActionButton("Cancel", variant="secondary")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.import_btn = ActionButton("Import", variant="primary")
        self.import_btn.clicked.connect(self.accept)
        self.import_btn.setEnabled(False)
        
        action_layout.addWidget(self.cancel_btn)
        action_layout.addWidget(self.import_btn)
        right_layout.addLayout(action_layout)
        
        right_scroll.setWidget(right_widget)
        main_layout.addWidget(right_scroll, stretch=40)
        
    def select_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Floor Plan", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if fname:
            self.image_path = fname
            self.path_label.setText(fname.split('/')[-1]) # Show filename
            self.load_image(fname)
            
    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            return
            
        self.cv_image = img
        self.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Pre-calculate skeleton
        _, binary = cv2.threshold(self.gray_image, 127, 255, cv2.THRESH_BINARY_INV)
        self.skeleton = self._skeletonize(binary)
        
        self.process_image()
        self.import_btn.setEnabled(True)
        
    def _skeletonize(self, binary_img):
        # Same logic as in manager
        skeleton = np.zeros(binary_img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        img = binary_img.copy()
        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skeleton
        
    def update_detection_params(self):
        # Use timer to avoid too frequent updates
        self.update_timer.start(300)
        
    def process_image(self):
        if self.skeleton is None:
            return
            
        h, w = self.skeleton.shape
        
        # Get parameters
        min_len_pct = self.min_len_slider.value() / 100.0
        max_gap_pct = self.max_gap_slider.value() / 100.0
        threshold = self.thresh_slider.value()
        
        min_line_length = int(min(w, h) * min_len_pct)
        max_line_gap = int(min(w, h) * max_gap_pct)
        
        lines = cv2.HoughLinesP(
            self.skeleton, 
            rho=1, 
            theta=np.pi/180, 
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        detected_segments = []
        if lines is not None:
            # Convert for preview (pixel coords)
            sim_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                detected_segments.append(((x1, y1), (x2, y2)))
                sim_lines.append(((x1, y1), (x2, y2)))
                
            # Run merge logic (using pixel coords for preview simplicity)
            # Note: For real import, we merge in sim coords, but logic is same
            merged = self._merge_collinear_segments(sim_lines, 5.0) # 5 pixels merge dist
            detected_segments = merged
            
        self.preview_label.set_image(self.cv_image)
        self.preview_label.set_lines(detected_segments)
        self.info_label.setText(f"Detected {len(detected_segments)} wall segments")
        
        # Store for retrieval
        self.final_segments_px = detected_segments

    def _merge_collinear_segments(self, lines, max_distance):
        # Simplified merge for preview visualization
        # Real merge happens in manager with real units
        if not lines: return []
        segments = list(lines)
        merged = []
        used = set()
        
        # Helper check function
        def can_merge(s1, s2):
            (x1, y1), (x2, y2) = s1
            (x3, y3), (x4, y4) = s2
            angle1 = np.arctan2(y2-y1, x2-x1) % np.pi
            angle2 = np.arctan2(y4-y3, x4-x3) % np.pi
            if abs(angle1 - angle2) > np.pi/18: return False
            
            endpoints = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            min_d = float('inf')
            for i in range(2):
                for j in range(2,4):
                    d = np.sqrt((endpoints[i][0]-endpoints[j][0])**2 + (endpoints[i][1]-endpoints[j][1])**2)
                    min_d = min(min_d, d)
            return min_d < max_distance

        for i, seg1 in enumerate(segments):
            if i in used: continue
            start, end = seg1
            used.add(i)
            
            merged_any = True
            while merged_any:
                merged_any = False
                for j, seg2 in enumerate(segments):
                    if j in used: continue
                    if can_merge((start, end), seg2):
                        # Merge logic (picking farthest points)
                        pts = [start, end, seg2[0], seg2[1]]
                        max_d = 0
                        best_pair = (start, end)
                        for p1 in pts:
                            for p2 in pts:
                                d = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                                if d > max_d:
                                    max_d = d
                                    best_pair = (p1, p2)
                        start, end = best_pair
                        used.add(j)
                        merged_any = True
            merged.append((start, end))
        return merged

    def set_edit_mode(self, mode):
        # Toggle check
        if self.preview_label.mode == mode:
            mode = "NONE"
            
        self.preview_label.set_mode(mode)
        self.edit_status.setText(f"Mode: {mode}")
        
        # Update cursor
        if mode in ["LINE", "POLYGON"]:
            self.preview_label.setCursor(Qt.CrossCursor)
        elif mode == "DELETE":
            self.preview_label.setCursor(Qt.ForbiddenCursor)
        else:
            self.preview_label.setCursor(Qt.ArrowCursor)
        
        # Update button styles
        def update_btn(btn, active):
            if active:
                btn.setStyleSheet(f"background-color: {COLORS['primary']}; color: white;")
            else:
                btn.setStyleSheet("") # Default
                
        update_btn(self.btn_line, mode == "LINE")
        update_btn(self.btn_poly, mode == "POLYGON")
        update_btn(self.btn_del, mode == "DELETE")

    def clear_manual(self):
        self.preview_label.clear_manual()
        
    def get_data(self):
        # Combine detected and manual segments
        all_segments = self.preview_label.detected_lines + self.preview_label.manual_segments
        
        return {
            'segments_px': all_segments,
            'image_size': (self.cv_image.shape[1], self.cv_image.shape[0]), # w, h
            'target_size': (self.width_spin.value(), self.height_spin.value()),
            'offset': (self.off_x_spin.value(), self.off_y_spin.value()),
            'wall_width': self.wall_thick_spin.value(),
            'nlos_params': {
                'error_bias': self.error_bias_spin.value(),
                'noise_factor': self.noise_factor_spin.value(),
                'path_loss_exp': self.path_loss_spin.value(),
                'shadow_std': self.shadow_std_spin.value()
            }
        }
