"""A worker step is pushed to GUI viewers without touching the RL socket."""

import json
import socket

from src.api.ai_training_worker import TrainingWorker


def _subscribe(port):
    sock = socket.create_connection(("127.0.0.1", port), timeout=3.0)
    sock.sendall(b'{"command":"subscribe","kind":"frame"}\n')
    reader = sock.makefile("r", encoding="utf-8")
    ready = json.loads(reader.readline())
    assert ready["type"] == "ready"
    return sock, reader


def test_each_worker_step_is_published_to_a_passive_viewer(tmp_path):
    worker = TrainingWorker(
        run_dir=str(tmp_path), serve_http=True, http_port=0,
        serve_frames=True, frames_port=0,
    )
    try:
        worker.start_services()
        sock, reader = _subscribe(worker.frames.port)
        try:
            returned = worker.tick()
            published = json.loads(reader.readline())
            assert published == returned
            assert published["type"] == "state"
            assert published["step_id"] == 0
            assert len(published["agents"]) == 1
        finally:
            reader.close()
            sock.close()
    finally:
        worker.close()


def test_frame_stream_is_independent_of_the_rl_action_socket(tmp_path):
    worker = TrainingWorker(
        run_dir=str(tmp_path), serve_http=True, http_port=0,
        serve_frames=True, frames_port=0,
    )
    try:
        worker.start_services()
        sock, reader = _subscribe(worker.frames.port)
        try:
            assert worker.engine is not None
            assert worker.tick()["step_id"] == 0
            assert json.loads(reader.readline())["step_id"] == 0
            # No AIGymServer client was connected, but the viewer still received
            # the frame and the engine stepped independently.
        finally:
            reader.close()
            sock.close()
    finally:
        worker.close()
