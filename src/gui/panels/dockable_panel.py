"""
Dockable Panel System
Provides flexible panel management with docked and floating modes
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QDialog, QSizePolicy, QScrollArea, QStyle
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont

from src.gui.theme import COLORS


class PanelTitleBar(QFrame):
    """Custom title bar for dockable panels"""
    
    close_clicked = pyqtSignal()
    float_clicked = pyqtSignal()
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("PanelTitleBar")
        self.setup_ui(title)
    
    def setup_ui(self, title: str):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 6, 6)
        layout.setSpacing(6)
        
        # Title label with icon
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.title_label.setStyleSheet(f"color: {COLORS['text_bright']};")
        layout.addWidget(self.title_label)
        
        layout.addStretch()
        
        # Float/dock toggle button
        self.float_btn = QPushButton()
        self.float_btn.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.float_btn.setFixedSize(28, 28)
        self.float_btn.setIconSize(QSize(16, 16))
        self.float_btn.setToolTip("Float window")
        self.float_btn.clicked.connect(self.float_clicked.emit)
        self.float_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['secondary_hover']};
            }}
        """)
        layout.addWidget(self.float_btn)
        
        # Close button
        self.close_btn = QPushButton()
        self.close_btn.setIcon(self.style().standardIcon(QStyle.SP_TitleBarCloseButton))
        self.close_btn.setFixedSize(28, 28)
        self.close_btn.setIconSize(QSize(16, 16))
        self.close_btn.setToolTip("Close panel")
        self.close_btn.clicked.connect(self.close_clicked.emit)
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: #c42b1c;
                border: 1px solid #c42b1c;
            }}
        """)
        layout.addWidget(self.close_btn)
        
        self.setStyleSheet(f"""
            #PanelTitleBar {{
                background-color: {COLORS['panel_bg']};
                border-bottom: 2px solid {COLORS['accent']};
                border-radius: 4px 4px 0 0;
            }}
        """)
        self.setFixedHeight(40)


class DockablePanel(QFrame):
    """
    A panel that can be docked or floated.
    Contains a title bar and content area.
    """
    
    closed = pyqtSignal(str)  # Emits panel_id when closed
    mode_changed = pyqtSignal(str, str)  # Emits (panel_id, new_mode)
    
    def __init__(self, panel_id: str, title: str, content_widget: QWidget, parent=None):
        super().__init__(parent)
        self.panel_id = panel_id
        self.title = title
        self.content_widget = content_widget
        self.is_floating = False
        self.floating_window = None
        
        self.setObjectName("DockablePanel")
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Title bar
        self.title_bar = PanelTitleBar(self.title)
        self.title_bar.close_clicked.connect(self.close_panel)
        self.title_bar.float_clicked.connect(self.toggle_float)
        layout.addWidget(self.title_bar)
        
        # Content area with scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background: {COLORS['background']};
                width: 10px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['secondary']};
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {COLORS['secondary_hover']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        scroll.setWidget(self.content_widget)
        layout.addWidget(scroll)
        
        self.setStyleSheet(f"""
            #DockablePanel {{
                background-color: {COLORS['panel_bg']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
            }}
        """)
        
        self.setMinimumWidth(280)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    
    def close_panel(self):
        """Close/hide the panel"""
        if self.floating_window:
            self.floating_window.close()
            self.floating_window = None
        self.hide()
        self.closed.emit(self.panel_id)
    
    def toggle_float(self):
        """Toggle between docked and floating mode"""
        if self.is_floating:
            self.dock()
        else:
            self.make_float()
    
    def make_float(self):
        """Convert to floating window"""
        if self.is_floating:
            return
        
        self.is_floating = True
        # Use Normal/Restore icon to indicate "Dock back"
        self.title_bar.float_btn.setIcon(self.style().standardIcon(QStyle.SP_TitleBarNormalButton))
        self.title_bar.float_btn.setToolTip("Dock panel")
        
        # Make sure panel is visible before adding to floating window
        self.show()
        
        # Create floating window
        self.floating_window = FloatingPanelWindow(self.title, self)
        self.floating_window.closed.connect(self._on_floating_closed)
        self.floating_window.show()
        
        self.mode_changed.emit(self.panel_id, "floating")
    
    def dock(self):
        """Return to docked mode"""
        if not self.is_floating:
            return
        
        self.is_floating = False
        # Use Maximize/Pop-out icon to indicate "Float"
        self.title_bar.float_btn.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.title_bar.float_btn.setToolTip("Float panel")
        
        # Emit signal FIRST to allow PanelManager to reparent the panel before window closes
        self.mode_changed.emit(self.panel_id, "docked")
        
        if self.floating_window:
            self.floating_window.close()
            self.floating_window = None
    
    def _on_floating_closed(self):
        """Handle floating window close"""
        self.floating_window = None
        self.is_floating = False
        self.title_bar.float_btn.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.close_panel()


class FloatingPanelWindow(QDialog):
    """Floating window container for panels"""
    
    closed = pyqtSignal()
    
    def __init__(self, title: str, panel: DockablePanel, parent=None):
        super().__init__(parent)
        self.panel = panel
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Take the content from the panel temporarily
        layout.addWidget(self.panel)
        
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['background']};
                color: {COLORS['text']};
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }}
            QLabel {{
                color: {COLORS['text']};
            }}
            QGroupBox {{
                color: {COLORS['text']};
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
            }}
            QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {{
                background-color: {COLORS['input_bg']};
                color: {COLORS['text_bright']};
                border: 1px solid {COLORS['border']};
                border-radius: 2px;
                padding: 4px;
            }}
            QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                background-color: {COLORS['secondary']};
                border: none;
                width: 16px;
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: {COLORS['secondary_hover']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
                background-color: {COLORS['secondary']};
            }}
            QComboBox::down-arrow {{
                image: none;
                border: none;
                width: 0;
            }}
        """)
        
        self.resize(320, 400)
    
    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class PanelManager:
    """
    Manages all dockable panels in the application.
    Handles registration, opening, closing, and mode switching.
    """
    
    def __init__(self, dock_container: QVBoxLayout, main_window):
        self.dock_container = dock_container
        self.main_window = main_window
        self.panels: dict[str, DockablePanel] = {}
        self.panel_factories: dict[str, callable] = {}
        self.open_panels: set[str] = set()
    
    def register_panel(self, panel_id: str, title: str, factory: callable):
        """
        Register a panel with its factory function.
        Factory should return the content widget for the panel.
        """
        self.panel_factories[panel_id] = (title, factory)
    
    def get_panel_list(self) -> list[tuple[str, str]]:
        """Get list of (panel_id, title) tuples"""
        return [(pid, info[0]) for pid, info in self.panel_factories.items()]
    
    def init_panel_hidden(self, panel_id: str):
        """
        Initialize a panel (create widgets) but keep it hidden.
        This allows widget values to be accessed without showing the panel.
        """
        if panel_id in self.panels:
            return  # Already initialized
        
        if panel_id not in self.panel_factories:
            return
        
        title, factory = self.panel_factories[panel_id]
        
        # Create panel but don't add to layout or show
        content = factory()
        panel = DockablePanel(panel_id, title, content)
        panel.closed.connect(self._on_panel_closed)
        panel.mode_changed.connect(self._on_mode_changed)
        panel.hide()  # Keep hidden
        self.panels[panel_id] = panel
    
    def open_panel(self, panel_id: str, mode: str = "docked"):
        """
        Open a panel in specified mode.
        mode: "docked" or "floating"
        """
        if panel_id in self.open_panels:
            # Panel already open, just bring to focus if floating
            panel = self.panels.get(panel_id)
            if panel and panel.floating_window:
                panel.floating_window.raise_()
                panel.floating_window.activateWindow()
            return
        
        if panel_id not in self.panel_factories:
            return
        
        title, factory = self.panel_factories[panel_id]
        
        # Check if existing panel is still valid (C++ object not deleted)
        panel = None
        if panel_id in self.panels:
            try:
                # Try to access a property to check if object is valid
                _ = self.panels[panel_id].isVisible()
                panel = self.panels[panel_id]
            except RuntimeError:
                # C++ object was deleted, remove stale reference
                del self.panels[panel_id]
        
        if panel is None:
            # Create new panel
            content = factory()
            panel = DockablePanel(panel_id, title, content)
            panel.closed.connect(self._on_panel_closed)
            panel.mode_changed.connect(self._on_mode_changed)
            self.panels[panel_id] = panel
        
        self.open_panels.add(panel_id)
        
        if mode == "floating":
            panel.make_float()
        else:
            # Add to dock container
            self.dock_container.insertWidget(self.dock_container.count() - 1, panel)
            panel.show()
            # Auto-expand dock area when opening docked panel
            self._update_dock_visibility()
    
    def close_panel(self, panel_id: str):
        """Close a panel"""
        if panel_id in self.panels:
            self.panels[panel_id].close_panel()
    
    def toggle_panel(self, panel_id: str, mode: str = "docked"):
        """Toggle a panel open/closed"""
        if panel_id in self.open_panels:
            self.close_panel(panel_id)
        else:
            self.open_panel(panel_id, mode)
    
    def is_panel_open(self, panel_id: str) -> bool:
        """Check if a panel is currently open"""
        return panel_id in self.open_panels
    
    def _count_docked_panels(self) -> int:
        """Count how many panels are currently docked (open but not floating)"""
        count = 0
        for panel_id in self.open_panels:
            panel = self.panels.get(panel_id)
            if panel and not panel.is_floating:
                count += 1
        return count
    
    def _update_dock_visibility(self):
        """Show or hide the dock area based on whether there are docked panels"""
        if not hasattr(self.main_window, 'main_splitter'):
            return
        
        docked_count = self._count_docked_panels()
        # current_sizes = self.main_window.main_splitter.sizes()
        
        if docked_count > 0:
            # Expand dock area when panels are docked - maintain reasonable ratio
            # Use specific sizes to avoid layout issues
            self.main_window.main_splitter.setSizes([320, 960])
        else:
            # Collapse dock area when no panels are docked
            self.main_window.main_splitter.setSizes([0, 1280])
    
    def _on_panel_closed(self, panel_id: str):
        """Handle panel closed event - just hide, don't delete"""
        self.open_panels.discard(panel_id)
        # Remove from dock container layout if docked (but don't delete widget)
        if panel_id in self.panels:
            panel = self.panels[panel_id]
            try:
                if not panel.is_floating:
                    self.dock_container.removeWidget(panel)
                # DON'T delete the panel - just hide it so widgets stay valid
                panel.hide()
            except RuntimeError:
                pass  # Panel was already deleted somehow
        
        # Auto-hide dock area if no docked panels remain
        self._update_dock_visibility()
    
    def _on_mode_changed(self, panel_id: str, new_mode: str):
        """Handle panel mode change"""
        panel = self.panels.get(panel_id)
        if not panel:
            return
        
        if new_mode == "floating":
            # Remove from dock container
            self.dock_container.removeWidget(panel)
        else:
            # Add back to dock container
            self.dock_container.insertWidget(self.dock_container.count() - 1, panel)
            panel.show()
        
        # Update dock visibility after mode change
        self._update_dock_visibility()

