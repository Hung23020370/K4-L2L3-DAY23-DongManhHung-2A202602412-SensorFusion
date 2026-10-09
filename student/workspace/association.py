"""Measurement-to-track association via Mahalanobis gating and greedy matching.

Part F supplies the association stage shown in docs/HUONG_DAN_KY_THUAT.md §2.
Load ``kalman`` with ``load_workspace_module`` for innovation helpers and tracking parameters
for the chi-square gate.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
from scipy.stats import chi2

from fusion_lab.workspace_loader import load_workspace_module
from fusion_lab.workspace_support import get_tracking_params

# Load kalman from the selected workspace, never with a plain `import kalman`.
kalman = load_workspace_module("kalman")


def mahalanobis_distance(track: Any, meas: Any) -> float:
    """Return squared Mahalanobis distance between a track and a measurement.

    Args:
        track: Track with ``x``, ``P``.
        meas: Measurement with ``sensor``.

    Returns:
        Scalar squared Mahalanobis distance.
    """
    H = meas.sensor.get_H(track.x)
    gamma = kalman.innovation(track.x, meas)
    S = kalman.innovation_covariance(track.P, meas, H)
    # d^2 = gamma' S^{-1} gamma; solve() avoids forming the explicit inverse.
    return float((gamma.T @ np.linalg.solve(S, gamma))[0, 0])


def chi2_gate(mhd_sq: float, sensor: Any) -> bool:
    """Return True if squared Mahalanobis distance lies inside the chi-square gate.

    Args:
        mhd_sq: Squared Mahalanobis distance.
        sensor: Sensor with ``dim_meas``.

    Returns:
        True if inside gate.
    """
    params = get_tracking_params()
    limit = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)
    return bool(mhd_sq < limit)


def association_cost_matrix(
    track_list: Sequence[Any], meas_list: Sequence[Any]
) -> np.matrix:
    """Build gated costs, checking each sensor's visibility before projection.

    Args:
        track_list: Active tracks.
        meas_list: Measurements for this sensor pass.

    Returns:
        Cost matrix; ``np.inf`` for invisible tracks or rejected chi-square gates.
        Invisible pairs must never call the Mahalanobis/projection helpers.
    """
    costs = np.full((len(track_list), len(meas_list)), np.inf)
    for i, track in enumerate(track_list):
        for j, meas in enumerate(meas_list):
            # Visibility first: an invisible pair must never reach the projection.
            if not meas.sensor.in_fov(track.x):
                continue
            distance = mahalanobis_distance(track, meas)
            if chi2_gate(distance, meas.sensor):
                costs[i, j] = distance
    return np.asmatrix(costs)


def pick_next_pair(
    association_matrix: np.matrix,
    unassigned_tracks: Sequence[Any],
    unassigned_meas: Sequence[Any],
) -> tuple[Any, Any, np.matrix, list[Any], list[Any]]:
    """Pick the minimum-cost track/measurement pair and shrink the association problem.

    Args:
        association_matrix: Current cost matrix.
        unassigned_tracks: Track objects still free.
        unassigned_meas: Measurement objects still free.

    Returns:
        Tuple (track, meas, new_matrix, remaining_tracks, remaining_meas).
        If no finite pair exists, return np.nan for track and meas and retain both lists.
    """
    costs = np.asarray(association_matrix, dtype=float)
    # Treat nan like inf: only finite costs are valid pairs.
    candidates = np.where(np.isfinite(costs), costs, np.inf)
    if candidates.size == 0 or not np.isfinite(candidates.min()):
        return np.nan, np.nan, association_matrix, list(unassigned_tracks), list(unassigned_meas)
    i, j = np.unravel_index(np.argmin(candidates), candidates.shape)
    track, meas = unassigned_tracks[i], unassigned_meas[j]
    remaining_matrix = np.asmatrix(np.delete(np.delete(costs, i, axis=0), j, axis=1))
    remaining_tracks = [t for k, t in enumerate(unassigned_tracks) if k != i]
    remaining_meas = [m for k, m in enumerate(unassigned_meas) if k != j]
    return track, meas, remaining_matrix, remaining_tracks, remaining_meas


def associate_and_update(
    manager: Any,
    meas_list: Sequence[Any],
    filter_obj: Any,
    sensor: Any,
) -> None:
    """Greedy association loop with EKF updates and track management.

    Args:
        manager: Track manager (``track_list``, ``manage_tracks``, ...).
        meas_list: Lidar or camera measurements for this frame pass.
        filter_obj: Filter with ``predict`` / ``update``.
        sensor: Explicit lidar/camera pass sensor, including empty measurement frames.

    Returns:
        None; updates tracks in place and always finishes the lifecycle pass.
        Visibility is handled in the cost matrix, before pair removal. Camera
        updates refine state only; lidar hits alone increase existence scores.
    """
    unassigned_tracks = list(manager.track_list)
    unassigned_meas = list(meas_list)
    if unassigned_tracks and unassigned_meas:
        # Visibility and the chi-square gate are already folded into the costs.
        matrix = association_cost_matrix(unassigned_tracks, unassigned_meas)
        while matrix.size > 0:
            track, meas, matrix, unassigned_tracks, unassigned_meas = pick_next_pair(
                matrix, unassigned_tracks, unassigned_meas
            )
            if isinstance(track, float) and np.isnan(track):
                break  # No finite pair left.
            filter_obj.update(track, meas)
            manager.handle_updated_track(track, sensor)
    # Always close the pass, even with no measurements (lidar misses, deletions, births).
    manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)