"""
GPU-Accelerated Geometry Kernels for UWB Simulation

Batched LOS/NLOS zone intersection tests using CuPy.
Replaces per-anchor Python loops in channel_model.py with a single
vectorized GPU operation.

All functions fall back to NumPy when CuPy is unavailable.
"""

import numpy as np
from typing import List, Tuple, Optional
from src.core.parallel.gpu_backend import (
    get_array_module, to_cpu, to_gpu, to_gpu_batch, gpu_manager
)
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CuPy RawKernel for batched segment-segment intersection (LOS check)
# ─────────────────────────────────────────────────────────────────────────────

_los_kernel = None


def _get_los_kernel():
    """
    Lazily create a CuPy RawKernel that tests segment-segment intersection
    for all anchor-tag pairs against all zone edges simultaneously.

    Each thread handles one (anchor, edge) pair.
    If any edge is crossed, the anchor is marked as NLOS via atomicOr.
    """
    global _los_kernel
    if _los_kernel is not None:
        return _los_kernel

    cp = gpu_manager.cupy
    if cp is None:
        return None

    try:
        _los_kernel = cp.RawKernel(r'''
extern "C" __global__
void check_los_batch(
    const double* __restrict__ anchor_segs,   // [N_anchors, 4]: x1,y1,x2,y2
    const double* __restrict__ zone_edges,    // [N_edges, 4]:   x1,y1,x2,y2
    int* __restrict__ is_blocked,             // [N_anchors]:    0=LOS, 1=NLOS
    const int n_anchors,
    const int n_edges
) {
    // 2D grid: blockIdx.x = anchor index, threadIdx.x = edge index
    int aid = blockIdx.x;
    int eid = threadIdx.x;
    if (aid >= n_anchors || eid >= n_edges) return;

    // Segment A-B: anchor-tag link
    double ax = anchor_segs[aid * 4 + 0];
    double ay = anchor_segs[aid * 4 + 1];
    double bx = anchor_segs[aid * 4 + 2];
    double by = anchor_segs[aid * 4 + 3];

    // Segment C-D: zone edge
    double cx = zone_edges[eid * 4 + 0];
    double cy = zone_edges[eid * 4 + 1];
    double dx = zone_edges[eid * 4 + 2];
    double dy = zone_edges[eid * 4 + 3];

    // CCW orientation test: ccw(P,Q,R) = (R.y-P.y)*(Q.x-P.x) > (Q.y-P.y)*(R.x-P.x)
    int d1 = ((dy - ay) * (bx - ax)) > ((by - ay) * (dx - ax));
    int d2 = ((cy - ay) * (bx - ax)) > ((by - ay) * (cx - ax));
    int d3 = ((by - cy) * (dx - cx)) > ((dy - cy) * (bx - cx));
    int d4 = ((ay - cy) * (dx - cx)) > ((dy - cy) * (ax - cx));

    // Segments intersect iff d1 != d2 AND d3 != d4
    if (d1 != d2 && d3 != d4) {
        atomicOr(&is_blocked[aid], 1);
    }
}
''', 'check_los_batch')
        logger.debug("Created batched LOS check CUDA kernel")
    except Exception as e:
        logger.warning(f"Failed to create LOS kernel: {e}")
        _los_kernel = None

    return _los_kernel


def _extract_zone_edges(nlos_zones, moving_nlos_zones) -> np.ndarray:
    """
    Flatten all NLOS zone boundaries into an (N_edges, 4) array.
    Each row is (x1, y1, x2, y2) for one edge segment.

    Handles NLOSZone (rect), PolygonNLOSZone, and MovingNLOSZone.
    """
    from src.core.uwb.Nlos_zones import NLOSZone, PolygonNLOSZone, MovingNLOSZone

    edges = []
    for zone in nlos_zones:
        if isinstance(zone, NLOSZone):
            # Rectangle → 4 edges
            x1, y1, x2, y2 = zone.x1, zone.y1, zone.x2, zone.y2
            edges.append([x1, y1, x2, y1])  # bottom
            edges.append([x2, y1, x2, y2])  # right
            edges.append([x2, y2, x1, y2])  # top
            edges.append([x1, y2, x1, y1])  # left
        elif isinstance(zone, PolygonNLOSZone):
            pts = zone.points
            for i in range(len(pts)):
                p1, p2 = pts[i], pts[(i + 1) % len(pts)]
                edges.append([p1[0], p1[1], p2[0], p2[1]])

    for zone in moving_nlos_zones:
        if isinstance(zone, MovingNLOSZone):
            pts = zone.get_points() if hasattr(zone, 'get_points') else zone.points
            for i in range(len(pts)):
                p1, p2 = pts[i], pts[(i + 1) % len(pts)]
                edges.append([p1[0], p1[1], p2[0], p2[1]])

    if not edges:
        return np.empty((0, 4), dtype=np.float64)
    return np.array(edges, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def batch_los_check_gpu(
    anchor_positions: np.ndarray,
    tag_position: Tuple[float, float],
    nlos_zones: list,
    moving_nlos_zones: list
) -> np.ndarray:
    """
    Check LOS condition for ALL anchors at once using GPU.

    Args:
        anchor_positions: (N_anchors, 2) array of anchor [x, y] positions
        tag_position: (x, y) tuple of the tag's current position
        nlos_zones: List of static NLOSZone / PolygonNLOSZone objects
        moving_nlos_zones: List of MovingNLOSZone objects

    Returns:
        Boolean NumPy array (N_anchors,): True = LOS, False = NLOS
    """
    n_anchors = len(anchor_positions)
    if n_anchors == 0:
        return np.array([], dtype=bool)

    # Build per-anchor segments: anchor_pos → tag_pos
    tag_x, tag_y = float(tag_position[0]), float(tag_position[1])
    anchor_segs = np.empty((n_anchors, 4), dtype=np.float64)
    anchor_segs[:, 0] = anchor_positions[:, 0]  # anchor x
    anchor_segs[:, 1] = anchor_positions[:, 1]  # anchor y
    anchor_segs[:, 2] = tag_x                   # tag x
    anchor_segs[:, 3] = tag_y                   # tag y

    # Extract zone edges
    zone_edges = _extract_zone_edges(nlos_zones, moving_nlos_zones)
    n_edges = len(zone_edges)

    # If no zones → everything is LOS
    if n_edges == 0:
        return np.ones(n_anchors, dtype=bool)

    # Try GPU kernel
    kernel = _get_los_kernel()
    if kernel is not None and gpu_manager.should_use_gpu(n_anchors * n_edges):
        cp = gpu_manager.cupy
        try:
            segs_gpu = cp.asarray(anchor_segs)
            edges_gpu = cp.asarray(zone_edges)
            is_blocked = cp.zeros(n_anchors, dtype=cp.int32)

            # Grid: one block per anchor, threads = edges (pad to warp of 32)
            threads_per_block = min(1024, max(32, ((n_edges + 31) // 32) * 32))
            grid = (n_anchors,)
            block = (threads_per_block,)

            kernel(grid, block, (segs_gpu, edges_gpu, is_blocked,
                                  np.int32(n_anchors), np.int32(n_edges)))

            # Return: True = LOS (not blocked)
            return ~to_cpu(is_blocked).astype(bool)
        except Exception as e:
            logger.warning(f"GPU LOS check failed, falling back to CPU: {e}")

    # CPU fallback — vectorized NumPy (still faster than Python loops)
    return _batch_los_check_numpy(anchor_segs, zone_edges)


def _batch_los_check_numpy(
    anchor_segs: np.ndarray,
    zone_edges: np.ndarray
) -> np.ndarray:
    """
    Vectorized NumPy fallback for batched LOS checks.
    Uses broadcasting to test all anchor × edge pairs simultaneously.
    """
    n_anchors = anchor_segs.shape[0]
    n_edges = zone_edges.shape[0]

    if n_edges == 0:
        return np.ones(n_anchors, dtype=bool)

    # Broadcast: (N_anchors, 1, 4) vs (1, N_edges, 4)
    A = anchor_segs[:, np.newaxis, :]  # (N_anchors, 1, 4)
    E = zone_edges[np.newaxis, :, :]   # (1, N_edges, 4)

    ax, ay = A[:, :, 0], A[:, :, 1]   # anchor x, y
    bx, by = A[:, :, 2], A[:, :, 3]   # tag x, y
    cx, cy = E[:, :, 0], E[:, :, 1]   # edge start x, y
    dx, dy = E[:, :, 2], E[:, :, 3]   # edge end x, y

    # CCW orientation tests
    d1 = ((dy - ay) * (bx - ax)) > ((by - ay) * (dx - ax))
    d2 = ((cy - ay) * (bx - ax)) > ((by - ay) * (cx - ax))
    d3 = ((by - cy) * (dx - cx)) > ((dy - cy) * (bx - cx))
    d4 = ((ay - cy) * (dx - cx)) > ((dy - cy) * (ax - cx))

    # Intersection: d1 != d2 AND d3 != d4
    intersects = (d1 != d2) & (d3 != d4)  # (N_anchors, N_edges)

    # Any intersection → NLOS
    is_blocked = np.any(intersects, axis=1)  # (N_anchors,)
    return ~is_blocked  # True = LOS
