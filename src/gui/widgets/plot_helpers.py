"""
Plot helper functions - Factory for creating themed pyqtgraph PlotWidgets.
Replaces 6+ identical plot setup blocks scattered across imu_window.py,
Distance_plot_window.py, cir_window.py, and Plots.py.
"""
import pyqtgraph as pg
from src.gui.theme import COLORS


def create_themed_plot(title="", x_label="", y_label="", x_units="", y_units="",
                       show_grid=True, enable_legend=False):
    """
    Create a pre-configured pg.PlotWidget with the application's dark theme.

    Args:
        title: Plot title text.
        x_label: X-axis label.
        y_label: Y-axis label.
        x_units: X-axis units string.
        y_units: Y-axis units string.
        show_grid: Whether to show grid lines.
        enable_legend: Whether to add a legend.

    Returns:
        A fully styled pg.PlotWidget.
    """
    plot = pg.PlotWidget()
    plot.setBackground(COLORS['background'])

    if title:
        plot.setTitle(title, color=COLORS['text'], size='11pt', bold=True)

    label_style = {'color': COLORS['text'], 'font-size': '10pt'}

    if x_label:
        kwargs = {'units': x_units} if x_units else {}
        plot.setLabel('bottom', x_label, **label_style, **kwargs)

    if y_label:
        kwargs = {'units': y_units} if y_units else {}
        plot.setLabel('left', y_label, **label_style, **kwargs)

    if show_grid:
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.getAxis('left').setGrid(100)
        plot.getAxis('bottom').setGrid(100)

    # Style axes
    axis_pen = pg.mkPen(color=COLORS['text'], width=1)
    plot.getAxis('bottom').setPen(axis_pen)
    plot.getAxis('left').setPen(axis_pen)

    if enable_legend:
        plot.addLegend(
            offset=(-10, 10),
            labelTextColor=COLORS['text'],
            brush=pg.mkBrush(COLORS['background']),
            pen=pg.mkPen(color='#404040')
        )

    return plot
