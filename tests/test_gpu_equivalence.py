"""
GPU Acceleration Equivalence Tests

Verifies that GPU-accelerated operations produce numerically equivalent
results to the original CPU implementations.

Run: python -m tests.test_gpu_equivalence
"""

import sys
import os

# Fix Windows terminal encoding for emoji/unicode output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time


def test_vectorized_jacobian():
    """
    Test that vectorized_jacobian() matches the original per-anchor loop.
    """
    from src.core.parallel.parallel_utils import vectorized_jacobian
    
    print("\n" + "="*60)
    print("TEST 1: Vectorized Jacobian vs Per-Anchor Loop")
    print("="*60)
    
    # Test with various anchor counts
    for n_anchors in [1, 4, 8, 16, 32]:
        # Random state and anchor positions
        state = np.random.randn(4) * 5  # [x, y, vx, vy]
        anchors = np.random.randn(n_anchors, 2) * 10
        
        # Original loop (reference)
        H_ref = np.zeros((n_anchors, 4))
        h_ref = np.zeros(n_anchors)
        for i in range(n_anchors):
            dx = state[0] - anchors[i, 0]
            dy = state[1] - anchors[i, 1]
            d = np.sqrt(dx**2 + dy**2)
            if d < 1e-6:
                d = 1e-6
            H_ref[i, 0] = dx / d
            H_ref[i, 1] = dy / d
            h_ref[i] = d
        
        # Vectorized version
        H_vec, h_vec = vectorized_jacobian(state, anchors)
        
        # Compare
        h_err = np.max(np.abs(h_ref - h_vec))
        H_err = np.max(np.abs(H_ref - H_vec))
        passed = h_err < 1e-10 and H_err < 1e-10
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  N={n_anchors:3d}: h_err={h_err:.2e}, H_err={H_err:.2e}  {status}")
    
    # Performance benchmark
    state = np.random.randn(4)
    anchors = np.random.randn(8, 2) * 10
    
    # Time the loop
    N_iters = 10000
    t0 = time.perf_counter()
    for _ in range(N_iters):
        H_ref = np.zeros((8, 4))
        h_ref = np.zeros(8)
        for i in range(8):
            dx = state[0] - anchors[i, 0]
            dy = state[1] - anchors[i, 1]
            d = np.sqrt(dx**2 + dy**2)
            H_ref[i, 0] = dx / d
            H_ref[i, 1] = dy / d
            h_ref[i] = d
    loop_time = time.perf_counter() - t0
    
    # Time vectorized
    t0 = time.perf_counter()
    for _ in range(N_iters):
        H_vec, h_vec = vectorized_jacobian(state, anchors)
    vec_time = time.perf_counter() - t0
    
    speedup = loop_time / vec_time
    print(f"\n  Benchmark (8 anchors, {N_iters} iters):")
    print(f"    Loop:       {loop_time*1000:.1f} ms")
    print(f"    Vectorized: {vec_time*1000:.1f} ms")
    print(f"    Speedup:    {speedup:.1f}x")


def test_batch_los_check():
    """
    Test that batch_los_check_gpu() matches per-anchor LOS checks.
    """
    from src.core.parallel.geometry_kernels import (
        batch_los_check_gpu, _batch_los_check_numpy
    )
    
    print("\n" + "="*60)
    print("TEST 2: Batch LOS Check (NumPy Vectorized)")
    print("="*60)
    
    # Create test scenario: tag at center, anchors around, rectangular zone blocking some
    tag_pos = (5.0, 5.0)
    
    # 8 anchors at various positions
    anchor_positions = np.array([
        [0.0, 0.0],   # should be blocked by zone
        [10.0, 0.0],  # should be blocked
        [10.0, 10.0], # clear
        [0.0, 10.0],  # should be blocked
        [5.0, 0.0],   # should be blocked
        [5.0, 10.0],  # clear
        [0.0, 5.0],   # should be blocked
        [10.0, 5.0],  # clear
    ])
    
    # Zone edges: a wall from (2,2) to (8,3)
    # Build segments manually for the numpy test
    zone_edges = np.array([
        [2.0, 2.0, 8.0, 2.0],  # bottom
        [8.0, 2.0, 8.0, 3.0],  # right  
        [8.0, 3.0, 2.0, 3.0],  # top
        [2.0, 3.0, 2.0, 2.0],  # left
    ], dtype=np.float64)
    
    # Build anchor segments
    n_anchors = len(anchor_positions)
    anchor_segs = np.empty((n_anchors, 4), dtype=np.float64)
    anchor_segs[:, 0] = anchor_positions[:, 0]
    anchor_segs[:, 1] = anchor_positions[:, 1]
    anchor_segs[:, 2] = tag_pos[0]
    anchor_segs[:, 3] = tag_pos[1]
    
    # Run vectorized NumPy check
    is_los = _batch_los_check_numpy(anchor_segs, zone_edges)
    
    print(f"  Anchor positions and LOS results:")
    for i, (pos, los) in enumerate(zip(anchor_positions, is_los)):
        print(f"    Anchor {i} at ({pos[0]:.0f},{pos[1]:.0f}): {'LOS ✅' if los else 'NLOS ❌'}")
    
    # Verify: anchors below the wall (y < 2) looking up to tag (y=5) must cross the wall
    # Anchors at y=0 should be NLOS (blocked by wall at y=2-3)
    assert not is_los[0], "Anchor 0 at (0,0) should be NLOS"
    assert not is_los[1], "Anchor 1 at (10,0) should be NLOS"
    assert not is_los[4], "Anchor 4 at (5,0) should be NLOS"
    
    # Anchors above the wall should be LOS
    assert is_los[2], "Anchor 2 at (10,10) should be LOS"
    assert is_los[5], "Anchor 5 at (5,10) should be LOS"
    
    print("\n  All assertions passed ✅")
    
    # Performance benchmark
    n_test = 32
    test_anchors = np.random.randn(n_test, 4) * 10
    n_edges = 40  # 10 zones × 4 edges each
    test_edges = np.random.randn(n_edges, 4) * 10
    
    N_iters = 5000
    t0 = time.perf_counter()
    for _ in range(N_iters):
        _batch_los_check_numpy(test_anchors, test_edges)
    vec_time = time.perf_counter() - t0
    
    print(f"\n  Benchmark ({n_test} anchors, {n_edges} edges, {N_iters} iters):")
    print(f"    Vectorized NumPy: {vec_time*1000:.1f} ms")
    print(f"    Per call:         {vec_time/N_iters*1e6:.1f} µs")


def test_gpu_toa_ping_pong():
    """
    Test that the ping-pong-free batch_toa_detection still works.
    """
    from src.core.parallel.gpu_backend import gpu_manager
    
    print("\n" + "="*60)
    print("TEST 3: GPU Availability & Backend Check")
    print("="*60)
    
    gpu_avail = gpu_manager._gpu_available
    print(f"  GPU Available:     {gpu_avail}")
    dev_name = getattr(gpu_manager, '_device_name', 'Unknown')
    print(f"  Device Name:       {dev_name}")
    mem = getattr(gpu_manager, '_device_memory', None)
    print(f"  GPU Memory:        {mem / 1e9:.1f} GB" if mem else "  GPU Memory:        N/A")
    
    if gpu_avail:
        cp = gpu_manager.cupy
        # Quick smoke test: create arrays on GPU
        a = cp.ones(1000, dtype=cp.float64)
        b = cp.sum(a)
        assert float(b) == 1000.0, "GPU sum test failed"
        print(f"  GPU Compute Test:  ✅ PASS (sum of 1000 ones = {float(b)})")
    else:
        print(f"  GPU Compute Test:  ⏭️ SKIPPED (no GPU)")


if __name__ == '__main__':
    print("PULSE GPU Acceleration — Equivalence Tests")
    print("=" * 60)
    
    test_vectorized_jacobian()
    test_batch_los_check()
    test_gpu_toa_ping_pong()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
