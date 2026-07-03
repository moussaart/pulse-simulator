"""
Performance Benchmark Suite for PULSE Simulator

Measures throughput and timing for the core simulation components:
- LOS checks (GPU batch vs sequential)
- Distance computation (vectorized vs per-anchor)
- Measurement pipeline
- Anchor visualization data prep

Run with:
    python -m pytest tests/test_performance.py -v -s
"""
import numpy as np
import time
import sys
import os
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.uwb.uwb_devices import Position, Anchor as UWBAnchor, Tag as UWBTag
from src.core.uwb.channel_model import UWBChannelModel
from src.core.parallel.gpu_backend import gpu_manager
from src.core.parallel.geometry_kernels import batch_los_check_gpu, _batch_los_check_numpy
from src.core.uwb.Nlos_zones import NLOSZone


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _create_anchors(n, spread=20.0):
    """Create N anchors spread in a circle of given radius."""
    anchors = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = spread * np.cos(angle)
        y = spread * np.sin(angle)
        anchor = UWBAnchor(Position(x, y))
        anchor.id = f"A{i}"
        anchors.append(anchor)
    return anchors


def _create_nlos_zones(n_zones=5, spread=20.0):
    """Create N rectangular NLOS zones scattered in the area."""
    zones = []
    rng = np.random.RandomState(42)
    for _ in range(n_zones):
        cx, cy = rng.uniform(-spread, spread, 2)
        hw, hh = rng.uniform(0.5, 3.0, 2)
        zones.append(NLOSZone(cx - hw, cy - hh, cx + hw, cy + hh))
    return zones


# ─── Benchmarks ──────────────────────────────────────────────────────────────

ANCHOR_COUNTS = [4, 10, 25, 50, 100]
N_ITERATIONS = 50


class TestLOSPerformance:
    """Benchmark LOS check performance: GPU vs CPU vs sequential."""
    
    @pytest.mark.parametrize("n_anchors", ANCHOR_COUNTS)
    def test_batch_los_cpu_vectorized(self, n_anchors):
        """Vectorized NumPy LOS check (CPU fallback)."""
        anchors = _create_anchors(n_anchors)
        zones = _create_nlos_zones(10)
        tag_pos = (1.0, 2.0)
        
        anchor_pos = np.array([[a.position.x, a.position.y] for a in anchors])
        anchor_segs = np.empty((n_anchors, 4), dtype=np.float64)
        anchor_segs[:, 0] = anchor_pos[:, 0]
        anchor_segs[:, 1] = anchor_pos[:, 1]
        anchor_segs[:, 2] = tag_pos[0]
        anchor_segs[:, 3] = tag_pos[1]
        
        # Extract edges manually for direct benchmark
        zone_edges = []
        for z in zones:
            x1, y1, x2, y2 = z.x1, z.y1, z.x2, z.y2
            zone_edges.extend([[x1,y1,x2,y1],[x2,y1,x2,y2],[x2,y2,x1,y2],[x1,y2,x1,y1]])
        zone_edges = np.array(zone_edges, dtype=np.float64)
        
        # Warmup
        _batch_los_check_numpy(anchor_segs, zone_edges)
        
        t0 = time.perf_counter()
        for _ in range(N_ITERATIONS):
            result = _batch_los_check_numpy(anchor_segs, zone_edges)
        elapsed = time.perf_counter() - t0
        
        per_call_us = (elapsed / N_ITERATIONS) * 1e6
        print(f"\n  CPU vectorized LOS: {n_anchors} anchors × 10 zones = {per_call_us:.1f} µs/call")
        assert result.shape == (n_anchors,)
    
    @pytest.mark.parametrize("n_anchors", ANCHOR_COUNTS)
    def test_batch_los_gpu(self, n_anchors):
        """GPU-accelerated LOS check (or CPU fallback)."""
        anchors = _create_anchors(n_anchors)
        zones = _create_nlos_zones(10)
        tag_pos = (1.0, 2.0)
        anchor_pos = np.array([[a.position.x, a.position.y] for a in anchors])
        
        # Warmup
        batch_los_check_gpu(anchor_pos, tag_pos, zones, [])
        
        t0 = time.perf_counter()
        for _ in range(N_ITERATIONS):
            result = batch_los_check_gpu(anchor_pos, tag_pos, zones, [])
        elapsed = time.perf_counter() - t0
        
        per_call_us = (elapsed / N_ITERATIONS) * 1e6
        gpu_label = "GPU" if gpu_manager.available else "CPU-fallback"
        print(f"\n  {gpu_label} batch LOS: {n_anchors} anchors × 10 zones = {per_call_us:.1f} µs/call")
        assert result.shape == (n_anchors,)
    
    @pytest.mark.parametrize("n_anchors", ANCHOR_COUNTS)
    def test_sequential_los(self, n_anchors):
        """Sequential per-anchor LOS check (baseline)."""
        anchors = _create_anchors(n_anchors)
        zones = _create_nlos_zones(10)
        tag = UWBTag(Position(1.0, 2.0))
        tag.id = "T1"
        
        channel = UWBChannelModel()
        for z in zones:
            channel.nlos_zones.append(z)
        
        # Warmup
        for a in anchors:
            channel.check_los_to_anchor(a.position, tag.position)
        
        t0 = time.perf_counter()
        for _ in range(N_ITERATIONS):
            results = [channel.check_los_to_anchor(a.position, tag.position) for a in anchors]
        elapsed = time.perf_counter() - t0
        
        per_call_us = (elapsed / N_ITERATIONS) * 1e6
        print(f"\n  Sequential LOS:     {n_anchors} anchors × 10 zones = {per_call_us:.1f} µs/call")
        assert len(results) == n_anchors


class TestDistancePerformance:
    """Benchmark distance computation: vectorized vs per-anchor."""
    
    @pytest.mark.parametrize("n_anchors", ANCHOR_COUNTS)
    def test_vectorized_distance(self, n_anchors):
        """Vectorized distance computation using NumPy."""
        anchor_pos = np.random.randn(n_anchors, 2) * 20
        tag_xy = np.array([1.0, 2.0])
        
        t0 = time.perf_counter()
        for _ in range(N_ITERATIONS * 10):
            dists = np.sqrt(np.sum((anchor_pos - tag_xy) ** 2, axis=1))
        elapsed = time.perf_counter() - t0
        
        per_call_us = (elapsed / (N_ITERATIONS * 10)) * 1e6
        print(f"\n  Vectorized distance: {n_anchors} anchors = {per_call_us:.1f} µs/call")
        assert dists.shape == (n_anchors,)
    
    @pytest.mark.parametrize("n_anchors", ANCHOR_COUNTS)
    def test_per_anchor_distance(self, n_anchors):
        """Per-anchor distance computation using Position.distance_to()."""
        anchors = _create_anchors(n_anchors)
        tag_pos = Position(1.0, 2.0)
        
        t0 = time.perf_counter()
        for _ in range(N_ITERATIONS * 10):
            dists = [a.position.distance_to(tag_pos) for a in anchors]
        elapsed = time.perf_counter() - t0
        
        per_call_us = (elapsed / (N_ITERATIONS * 10)) * 1e6
        print(f"\n  Per-anchor distance: {n_anchors} anchors = {per_call_us:.1f} µs/call")
        assert len(dists) == n_anchors


class TestMeasurementPerformance:
    """Benchmark the channel model measurement pipeline."""
    
    @pytest.mark.parametrize("n_anchors", ANCHOR_COUNTS)
    def test_channel_measurement(self, n_anchors):
        """Full channel model measurement (includes CIR generation)."""
        anchors = _create_anchors(n_anchors)
        tag = UWBTag(Position(1.0, 2.0))
        tag.id = "T1"
        channel = UWBChannelModel()
        
        # Warmup
        for a in anchors[:min(2, n_anchors)]:
            channel.update_los_condition(a.position, tag.position)
            channel.measure_distance_detailed(a.position.distance_to(tag.position), tag.position)
        
        t0 = time.perf_counter()
        for _ in range(max(1, N_ITERATIONS // n_anchors)):
            for a in anchors:
                channel.update_los_condition(a.position, tag.position)
                channel.measure_distance_detailed(a.position.distance_to(tag.position), tag.position)
        elapsed = time.perf_counter() - t0
        
        iters = max(1, N_ITERATIONS // n_anchors)
        per_frame_ms = (elapsed / iters) * 1e3
        per_anchor_us = (elapsed / (iters * n_anchors)) * 1e6
        print(f"\n  Channel measurement: {n_anchors} anchors = {per_frame_ms:.1f} ms/frame, {per_anchor_us:.1f} µs/anchor")


class TestAnchorPositionCache:
    """Benchmark anchor position array caching."""
    
    @pytest.mark.parametrize("n_anchors", ANCHOR_COUNTS)
    def test_rebuild_vs_update_inplace(self, n_anchors):
        """Compare rebuilding anchor array from scratch vs in-place update."""
        anchors = _create_anchors(n_anchors)
        cached_array = np.array([[a.position.x, a.position.y] for a in anchors], dtype=np.float64)
        
        # Rebuild from scratch
        t0 = time.perf_counter()
        for _ in range(N_ITERATIONS * 100):
            arr = np.array([[a.position.x, a.position.y] for a in anchors], dtype=np.float64)
        elapsed_rebuild = time.perf_counter() - t0
        
        # In-place update
        t0 = time.perf_counter()
        for _ in range(N_ITERATIONS * 100):
            for i, a in enumerate(anchors):
                cached_array[i, 0] = a.position.x
                cached_array[i, 1] = a.position.y
        elapsed_inplace = time.perf_counter() - t0
        
        rebuild_us = (elapsed_rebuild / (N_ITERATIONS * 100)) * 1e6
        inplace_us = (elapsed_inplace / (N_ITERATIONS * 100)) * 1e6
        speedup = elapsed_rebuild / max(elapsed_inplace, 1e-9)
        
        print(f"\n  Anchor pos cache {n_anchors}: rebuild={rebuild_us:.1f}µs, inplace={inplace_us:.1f}µs, speedup={speedup:.1f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
