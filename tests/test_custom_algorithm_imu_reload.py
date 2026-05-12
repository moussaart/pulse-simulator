import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.localization.algorithm_loader import AlgorithmLoader
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmOutput
from src.core.uwb.imu import IMUData
from src.core.uwb.uwb_devices import Position, Tag
from src.gui.managers.file_manager import FileManager
from src.gui.managers.simulation_manager import SimulationManager


class _CaptureImuAlgorithm(BaseLocalizationAlgorithm):
    uses_imu = True
    seen_inputs = []

    @property
    def name(self):
        return "Capture IMU"

    def initialize(self):
        pass

    def update(self, input_data):
        type(self).seen_inputs.append(input_data)
        return AlgorithmOutput(
            position=(0.0, 0.0),
            state=np.zeros(4),
            covariance=np.eye(4),
            initialized=True,
        )


class _NoImuAlgorithm(BaseLocalizationAlgorithm):
    uses_imu = False
    seen_inputs = []

    @property
    def name(self):
        return "No IMU"

    def initialize(self):
        pass

    def update(self, input_data):
        type(self).seen_inputs.append(input_data)
        return AlgorithmOutput(
            position=(0.0, 0.0),
            state=np.zeros(4),
            covariance=np.eye(4),
            initialized=True,
        )


def _make_parent(algorithm, tag):
    return SimpleNamespace(
        algorithm=algorithm,
        algorithm_imu_overrides={},
        tag=tag,
        anchors=[],
        dt=0.02,
        kf_state=np.zeros(4),
        kf_P=np.eye(4),
        kf_initialized=False,
        aekf_Q=None,
        aekf_R=None,
    )


def _tag_with_imu(ax, ay, az, gx=0.0, gy=0.0, gz=0.0, enabled=True):
    tag = Tag(Position(0.0, 0.0, 0.0))
    tag.imu_data = IMUData()
    tag.imu_data_on = enabled
    tag.imu_data.add_measurement(0.0, ax, ay, az, gx, gy, gz)
    return tag


def test_legacy_custom_algorithm_imu_requirement_survives_loader_restart(tmp_path):
    algorithm_file = tmp_path / "legacy_imu_algorithm.py"
    algorithm_file.write_text(
        "import numpy as np\n"
        "from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmOutput\n"
        "class LegacyImuAlgorithm(BaseLocalizationAlgorithm):\n"
        "    @property\n"
        "    def name(self):\n"
        "        return 'Legacy IMU Algorithm'\n"
        "    def initialize(self):\n"
        "        pass\n"
        "    def update(self, input_data):\n"
        "        if input_data.imu_data_on and input_data.accel is not None:\n"
        "            accel = input_data.accel\n"
        "        return AlgorithmOutput((0.0, 0.0), np.zeros(4), np.eye(4), True)\n",
        encoding="utf-8",
    )

    first_loader = AlgorithmLoader(str(tmp_path))
    first_algorithms = first_loader.discover_algorithms()
    assert first_algorithms["Legacy IMU Algorithm"].uses_imu is True
    assert "imu" in first_algorithms["Legacy IMU Algorithm"].required_sensors

    # Simulate application restart: a fresh loader re-reads the file from disk.
    second_loader = AlgorithmLoader(str(tmp_path))
    second_algorithms = second_loader.discover_algorithms()
    assert second_algorithms["Legacy IMU Algorithm"].uses_imu is True
    assert "imu" in second_algorithms["Legacy IMU Algorithm"].required_sensors


def test_reloaded_custom_algorithm_receives_live_changing_imu_data():
    _CaptureImuAlgorithm.seen_inputs = []
    tag = _tag_with_imu(1.0, 2.0, 9.81)
    parent = _make_parent("Capture IMU", tag)
    manager = SimulationManager(parent)
    manager._algorithm_methods = {"Capture IMU": _CaptureImuAlgorithm}

    manager.estimate_position([1.0, 2.0, 3.0], [True, True, True])
    tag.imu_data.add_measurement(0.02, 3.0, 4.0, 9.81, 0.0, 0.0, 0.5)
    manager.estimate_position([1.0, 2.0, 3.0], [True, True, True])

    assert len(_CaptureImuAlgorithm.seen_inputs) == 2
    first_input, second_input = _CaptureImuAlgorithm.seen_inputs
    assert first_input.imu_data_on is True
    assert second_input.imu_data_on is True
    assert np.allclose(first_input.accel, [1.0, 2.0, 9.81])
    assert np.allclose(second_input.accel, [3.0, 4.0, 9.81])
    assert not np.allclose(first_input.accel, second_input.accel)
    assert np.allclose(second_input.gyro, [0.0, 0.0, 0.5])


def test_non_imu_custom_algorithm_does_not_receive_imu_data_even_when_tag_has_imu():
    _NoImuAlgorithm.seen_inputs = []
    tag = _tag_with_imu(1.0, 2.0, 9.81)
    parent = _make_parent("No IMU", tag)
    manager = SimulationManager(parent)
    manager._algorithm_methods = {"No IMU": _NoImuAlgorithm}

    manager.estimate_position([1.0, 2.0, 3.0], [True, True, True])

    input_data = _NoImuAlgorithm.seen_inputs[-1]
    assert input_data.imu_data_on is False
    assert input_data.accel is None
    assert input_data.gyro is None


def test_project_config_saves_and_restores_custom_algorithm_imu_flags(tmp_path, monkeypatch):
    config_path = tmp_path / "project.json"

    class Combo:
        def __init__(self, text):
            self.text = text

        def currentText(self):
            return self.text

        def setCurrentText(self, text):
            self.text = text

    class Slider:
        def __init__(self, value):
            self._value = value

        def value(self):
            return self._value

        def setValue(self, value):
            self._value = value

    parent = SimpleNamespace(
        algo_combo=Combo("Capture IMU"),
        pattern_combo=Combo("Circular"),
        speed_slider=Slider(10),
        timestep_slider=Slider(5),
        tag=SimpleNamespace(imu_data_on=True),
        anchors=[],
        channel_conditions=SimpleNamespace(nlos_zones=[], moving_nlos_zones=[]),
        nlos_manager=SimpleNamespace(zone_colors=[], update_nlos_zones=lambda: None),
        distance_plots_window=None,
        algorithm_imu_overrides={},
        renormalize_anchors=lambda: None,
        show_error_message=lambda title, message: (_ for _ in ()).throw(AssertionError(message)),
    )

    manager = FileManager(parent)
    monkeypatch.setattr(FileManager, "algorithm_uses_imu", staticmethod(lambda name: name == "Capture IMU"))
    monkeypatch.setattr(
        "src.gui.managers.file_manager.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(config_path), "JSON Files (*.json)"),
    )
    manager.save_map_config()

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["algorithm_config"] == {"name": "Capture IMU", "uses_imu": True}
    assert saved["tag_config"] == {"imu_data_on": True}

    parent.tag.imu_data_on = False
    parent.algorithm_imu_overrides = {}
    monkeypatch.setattr(
        "src.gui.managers.file_manager.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(config_path), "JSON Files (*.json)"),
    )
    manager.load_map_config()

    assert parent.tag.imu_data_on is True
    assert parent.algorithm_imu_overrides["Capture IMU"] is True
