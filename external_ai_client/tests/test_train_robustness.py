import json

import pytest

from agent_switch_mo.train import (
    BoundedStats,
    MetricsWriter,
    atomic_torch_save,
    checkpoint_payload_is_resumable,
    make_checkpoint_payload,
)


def test_bounded_stats_stay_flat():
    stats = BoundedStats(limit=10)
    for i in range(10000):
        stats.add(float(i))
    assert len(stats.values) == 10
    assert stats.values[-1] == 9999.0
    assert stats.mean() == pytest.approx(sum(range(9990, 10000)) / 10)


def test_atomic_torch_save_leaves_no_partial_file(tmp_path):
    import torch

    target = tmp_path / "policy.pth"
    atomic_torch_save({"model": torch.zeros(2, 2)}, str(target))
    assert target.exists()
    assert not list(tmp_path.glob("policy.pth.*.tmp"))
    payload = torch.load(str(target), map_location="cpu", weights_only=False)
    assert payload["model"].shape == (2, 2)


def test_checkpoint_payload_is_resumable():
    keys = {
        "step", "model", "optimizer", "target_model", "replay",
        "rng_states", "episode", "agent_cumul_rewards", "algorithm",
    }
    assert checkpoint_payload_is_resumable({k: None for k in keys}) is True
    broken = {k: None for k in keys}
    del broken["optimizer"]
    assert checkpoint_payload_is_resumable(broken) is False


def test_checkpoint_payload_contains_all_training_state_for_resume():
    payload = make_checkpoint_payload(
        step=12,
        model={"w": 1},
        optimizer={"state": 2},
        target_model={"w": 3},
        replay={"buffer": [], "position": 0},
        rng_states={"python": "state"},
        episode=4,
        agent_cumul_rewards=[2.0, 3.0],
        algorithm="Duty-Cycled UWB-IMU NA-AEKF",
    )
    assert checkpoint_payload_is_resumable(payload)
    assert payload["step"] == 12
    assert payload["episode"] == 4


def test_metrics_writer_appends_valid_jsonl_and_closes(tmp_path):
    path = tmp_path / "training.jsonl"
    writer = MetricsWriter(str(path))
    writer.write({"step": 1, "reward": 0.5})
    writer.write({"step": 2, "reward": 0.75})
    writer.close()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert all("wall_time" in row for row in rows)
