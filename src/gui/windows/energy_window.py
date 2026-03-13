"""
Energy Consumption Display Window

Standalone window showing detailed UWB tag energy metrics:
- Summary cards with key metrics
- Power breakdown chart (matplotlib)
- Protocol and configuration details
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QFrame, QWidget, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from src.gui.theme import COLORS
from src.core.uwb.energy_model import EnergyCalculator, EnergyResult

# Try to import matplotlib for charts
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class MetricCard(QFrame):
    """A styled card showing a single metric with title, value, and unit."""

    def __init__(self, title: str, value: str = "—", unit: str = "", color: str = COLORS['accent'], parent=None):
        super().__init__(parent)
        self._color = color
        self.setObjectName("MetricCard")
        self.setStyleSheet(f"""
            #MetricCard {{
                background-color: {COLORS['panel_bg']};
                border: 1px solid {COLORS['border']};
                border-left: 3px solid {color};
                border-radius: 4px;
                padding: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        layout.addWidget(self.title_label)

        value_row = QHBoxLayout()
        value_row.setSpacing(4)

        self.value_label = QLabel(value)
        self.value_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.value_label.setStyleSheet(f"color: {color};")
        value_row.addWidget(self.value_label)

        self.unit_label = QLabel(unit)
        self.unit_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        self.unit_label.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        value_row.addWidget(self.unit_label)
        value_row.addStretch()

        layout.addLayout(value_row)

    def set_value(self, value: str, unit: str = ""):
        """Update the displayed value and unit."""
        self.value_label.setText(value)
        if unit:
            self.unit_label.setText(unit)


class PowerBreakdownChart(QWidget):
    """Matplotlib-based horizontal bar chart showing power breakdown."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)

        if not HAS_MATPLOTLIB:
            layout = QVBoxLayout(self)
            label = QLabel("📊 Install matplotlib for charts")
            label.setStyleSheet(f"color: {COLORS['text_dim']};")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            return

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(4, 2.5), facecolor=COLORS['background'])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

    def update_chart(self, result: EnergyResult):
        """Redraw the power breakdown chart."""
        if not HAS_MATPLOTLIB:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Data
        categories = ['UWB TX+RX', 'Idle', 'IMU']
        values = [
            result.uwb_active_power_mW,
            result.tag_idle_power_mW,
            result.imu_power_mW
        ]
        colors_chart = ['#4fc3f7', '#ff8a65', '#81c784']

        # Horizontal bar
        bars = ax.barh(categories, values, color=colors_chart, height=0.5, edgecolor='none')

        # Value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{val:.2f} mW', va='center', ha='left',
                        color=COLORS['text'], fontsize=9)

        # Styling
        ax.set_xlabel('Power (mW)', color=COLORS['text_dim'], fontsize=9)
        ax.set_title('Power Breakdown', color=COLORS['text_bright'],
                      fontsize=11, fontweight='bold', pad=10)
        ax.set_facecolor(COLORS['background'])
        ax.tick_params(colors=COLORS['text'], labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(COLORS['border'])
        ax.spines['left'].set_color(COLORS['border'])
        ax.xaxis.label.set_color(COLORS['text_dim'])

        # Add margin to right for labels
        max_val = max(values) if values else 1
        ax.set_xlim(0, max_val * 1.4)

        self.figure.tight_layout()
        self.canvas.draw()


class EnergyWindow(QDialog):
    """
    Standalone window displaying detailed UWB tag energy consumption metrics.
    Shows summary cards, power breakdown chart, and protocol details.
    """

    def __init__(self, energy_calculator: EnergyCalculator, parent=None):
        super().__init__(parent)
        self.calculator = energy_calculator
        self.setWindowTitle("⚡ UWB Energy Consumption")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.setMinimumSize(520, 600)
        self.resize(560, 680)

        self._setup_ui()
        self._apply_theme()
        self.refresh()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # ── Header ────────────────────────────────────────────────────────
        header = QLabel("⚡ UWB Tag Energy Consumption")
        header.setFont(QFont("Segoe UI", 14, QFont.Bold))
        header.setStyleSheet(f"color: {COLORS['text_bright']};")
        main_layout.addWidget(header)

        # ── Summary Cards (3 up top, 2 below) ────────────────────────────
        cards_grid = QGridLayout()
        cards_grid.setSpacing(8)

        self.card_total_power = MetricCard("Current Power", color="#26c6da")
        self.card_total_energy = MetricCard("Total Consumed", color="#4fc3f7")
        self.card_battery_life = MetricCard("Battery Life", color="#81c784")
        self.card_energy_ranging = MetricCard("Energy / Ranging", color="#ffb74d")
        self.card_duty_cycle = MetricCard("Duty Cycle", color="#ce93d8")

        cards_grid.addWidget(self.card_total_power, 0, 0)
        cards_grid.addWidget(self.card_total_energy, 0, 1)
        cards_grid.addWidget(self.card_battery_life, 0, 2)
        cards_grid.addWidget(self.card_energy_ranging, 1, 0)
        cards_grid.addWidget(self.card_duty_cycle, 1, 1)

        main_layout.addLayout(cards_grid)

        # ── Power Breakdown Chart ─────────────────────────────────────────
        self.chart = PowerBreakdownChart()
        main_layout.addWidget(self.chart)

        # ── Protocol Details ──────────────────────────────────────────────
        details_frame = QFrame()
        details_frame.setObjectName("DetailsFrame")
        details_frame.setStyleSheet(f"""
            #DetailsFrame {{
                background-color: {COLORS['panel_bg']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
            }}
        """)

        details_layout = QGridLayout(details_frame)
        details_layout.setSpacing(6)
        details_layout.setContentsMargins(12, 10, 12, 10)

        # Title
        details_title = QLabel("Protocol Details")
        details_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        details_title.setStyleSheet(f"color: {COLORS['accent']};")
        details_layout.addWidget(details_title, 0, 0, 1, 2)

        self._detail_labels = {}
        detail_items = [
            ("Ranging Mode", "mode"),
            ("UWB Frequency (Dynamic)", "freq"),
            ("Anchors (Dynamic)", "anchors"),
            ("IMU Power (Dynamic)", "imu_enabled"),
            ("Messages / Ranging", "msgs"),
            ("TX Energy / Message", "e_tx"),
            ("RX Energy / Message", "e_rx"),
            ("Total Current Draw", "current"),
            ("Total Time Active", "sim_time"),
        ]

        for i, (label_text, key) in enumerate(detail_items, start=1):
            lbl = QLabel(f"{label_text}:")
            lbl.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
            val = QLabel("—")
            val.setStyleSheet(f"color: {COLORS['text_bright']}; font-size: 11px;")
            val.setAlignment(Qt.AlignRight)
            details_layout.addWidget(lbl, i, 0)
            details_layout.addWidget(val, i, 1)
            self._detail_labels[key] = val

        main_layout.addWidget(details_frame)
        main_layout.addStretch()

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['background']};
                color: {COLORS['text']};
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
        """)

    def refresh(self):
        """Recalculate and update all displayed values."""
        result = self.calculator.calculate()

        # Summary cards
        self.card_total_power.set_value(f"{result.total_power_mW:.2f}", "mW")
        self.card_total_energy.set_value(f"{result.total_energy_consumed_J:.4f}", "J")
        
        if result.battery_life_days < 1:
            self.card_battery_life.set_value(f"{result.battery_life_hours:.1f}", "hours")
        elif result.battery_life_days > 365:
            years = result.battery_life_days / 365
            self.card_battery_life.set_value(f"{years:.1f}", "years")
        else:
            self.card_battery_life.set_value(f"{result.battery_life_days:.1f}", "days")

        self.card_energy_ranging.set_value(f"{result.energy_per_ranging_uJ:.2f}", "µJ")
        self.card_duty_cycle.set_value(f"{result.duty_cycle_percent:.3f}", "%")

        # Chart
        self.chart.update_chart(result)

        # Details
        self._detail_labels["mode"].setText(result.ranging_mode)
        self._detail_labels["freq"].setText(f"{result.uwb_frequency_hz:.2f} Hz")
        self._detail_labels["anchors"].setText(str(result.num_anchors))
        
        # Determine if IMU is enabled by checking if uwb is disabled or imu is explicitly on
        imu_status = "Enabled" if (self.calculator.config.imu_enabled) else "Disabled"
        if self.calculator.config.uwb_disabled:
             imu_status = "Enabled (IMU Only)"
        self._detail_labels["imu_enabled"].setText(imu_status)
        
        self._detail_labels["msgs"].setText(
            f"{result.messages_per_ranging} ({result.tx_messages_per_ranging} TX + {result.rx_messages_per_ranging} RX)"
        )
        self._detail_labels["e_tx"].setText(f"{result.energy_per_tx_message_uJ:.4f} µJ")
        self._detail_labels["e_rx"].setText(f"{result.energy_per_rx_message_uJ:.4f} µJ")
        self._detail_labels["current"].setText(f"{result.total_current_mA:.2f} mA")
        self._detail_labels["sim_time"].setText(f"{self.calculator.total_simulation_time_s:.2f} s")
