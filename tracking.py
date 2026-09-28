"""
Target Tracking & Data Association Module.

This module is responsible for maintaining target continuity across multiple radar frames.
It implements a highly optimized, vectorized Batch Kalman Filter to track hundreds of
targets simultaneously without Python loop overhead.

Coupled with a Global Nearest Neighbor (GNN) association algorithm, this module
ensures that momentary occlusions or CFAR misses do not result in dropped tracks
or the creation of "ghost" targets, providing a stable tactical picture to the C4I dashboard.
"""

import numpy as np


class BatchKalmanFilter:
    """
    Vectorized Kalman Filter processing multiple targets simultaneously using 3D Tensors.

    State Vector Formulation (x):
        [x_position, y_position, x_velocity, y_velocity]^T

    Mathematical Matrices:
        F: State Transition Matrix (Kinematic model)
        H: Measurement Mapping Matrix (Sensor to State mapping)
        R: Measurement Noise Covariance (Sensor uncertainty)
        Q: Process Noise Covariance (Uncertainty in target behavior / maneuvers)
        P: State Covariance Matrix (Current uncertainty of the track)
    """

    def __init__(self, dt):
        self.dt = dt

        # Initialize empty 3D arrays to hold states and covariances for N targets
        self.state_vectors = np.empty((0, 4, 1))
        self.covariance_matrices = np.empty((0, 4, 4))

        # F - State Transition Matrix (Constant Velocity Model)
        self.state_transition_matrix = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # H - Measurement Matrix (Radar measures position [x,y], but not directly cartesian velocity)
        self.measurement_mapping_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])

        # R - Measurement Noise Covariance (Configured for radar spatial resolution limits)
        self.measurement_noise_matrix = np.eye(2) * 5.0

        # Q - Process Noise Covariance (Allows the filter to track maneuvering targets)
        self.process_noise_matrix = np.eye(4) * 0.1
        self.identity_matrix = np.eye(4)

    def add_target(self, initial_x, initial_y):
        """
        Initializes a new track in the filter.
        New tracks are born with extremely high covariance (uncertainty) until further measurements arrive.
        """
        new_state = np.array([[[initial_x], [initial_y], [0.0], [0.0]]])
        new_covariance = np.expand_dims(np.eye(4) * 1000.0, axis=0)  # P_0 = 1000

        if self.state_vectors.size == 0:
            self.state_vectors = new_state
            self.covariance_matrices = new_covariance
        else:
            self.state_vectors = np.concatenate((self.state_vectors, new_state), axis=0)
            self.covariance_matrices = np.concatenate((self.covariance_matrices, new_covariance), axis=0)

    def delete_targets(self, indices_to_remove):
        """
        Prunes dead or lost tracks from the vectorized matrices.
        """
        if len(indices_to_remove) > 0:
            self.state_vectors = np.delete(self.state_vectors, indices_to_remove, axis=0)
            self.covariance_matrices = np.delete(self.covariance_matrices, indices_to_remove, axis=0)

    def predict(self):
        """
        Prediction Step (Time Update).
        Projects the current state and covariance forward in time based on the kinematic model.
        Equations:
            x_{k|k-1} = F * x_{k-1|k-1}
            P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        """
        if self.state_vectors.shape[0] == 0:
            return

        self.state_vectors = self.state_transition_matrix @ self.state_vectors
        self.covariance_matrices = (
                                               self.state_transition_matrix @ self.covariance_matrices @ self.state_transition_matrix.T) + self.process_noise_matrix

    def update(self, matched_indices, measurements):
        """
        Update Step (Measurement Update).
        Corrects the predicted state using actual radar detections, weighted by the Kalman Gain.
        """
        if len(matched_indices) == 0:
            return

        # Format incoming measurements to match tensor dimensions
        measurements_tensor = np.array(measurements).reshape(-1, 2, 1)

        # Extract only the matrices corresponding to targets that were successfully detected this frame
        matched_state_vectors = self.state_vectors[matched_indices]
        matched_covariance_matrices = self.covariance_matrices[matched_indices]

        # 1. Innovation (Residual): y = z - H * x
        innovation = measurements_tensor - (self.measurement_mapping_matrix @ matched_state_vectors)

        # 2. Innovation Covariance: S = H * P * H^T + R
        innovation_covariance = (
                                            self.measurement_mapping_matrix @ matched_covariance_matrices @ self.measurement_mapping_matrix.T) + self.measurement_noise_matrix

        # 3. Optimal Kalman Gain: K = P * H^T * S^-1
        kalman_gain = matched_covariance_matrices @ self.measurement_mapping_matrix.T @ np.linalg.inv(
            innovation_covariance)

        # 4. State Update: x = x + K * y
        matched_state_vectors = matched_state_vectors + (kalman_gain @ innovation)

        # 5. Covariance Update: P = (I - K * H) * P
        matched_covariance_matrices = (self.identity_matrix - (
                    kalman_gain @ self.measurement_mapping_matrix)) @ matched_covariance_matrices

        # Write back the corrected states to the global tracking vectors
        self.state_vectors[matched_indices] = matched_state_vectors
        self.covariance_matrices[matched_indices] = matched_covariance_matrices


class Track:
    """
    Metadata container for tracking lifecycle management.
    Handles track age (maturity) and Coasting (time since last valid measurement).
    """

    def __init__(self, track_id, measured_velocity=0.0):
        self.track_id = track_id
        self.age = 1  # Frames since target birth. Used to prevent plotting unconfirmed 'ghosts'.
        self.time_since_update = 0  # Coasting timer. Increments when radar misses the target.
        self.latest_measured_velocity = measured_velocity


def polar_to_cartesian(range_val, angle_deg):
    """
    Converts Radar Polar coordinates (Range, Azimuth) to Cartesian coordinates (X, Y).
    Assumes standard PPI display format where 0 degrees is True North (Y-axis).
    """
    angle_rad = np.deg2rad(angle_deg)
    x_range = range_val * np.sin(angle_rad)
    y_range = range_val * np.cos(angle_rad)
    return x_range, y_range


def associate_detections(tracks_meta, batch_kf, detections, distance_threshold=5.0):
    """
    Global Nearest Neighbor (GNN) Association Algorithm.
    Correlates raw CFAR detections to existing Kalman tracks using an Euclidean distance gate.

    Args:
        tracks_meta: List of Track metadata objects.
        batch_kf: The active BatchKalmanFilter containing predicted states.
        detections: List of current frame (X,Y) radar detections.
        distance_threshold: Association Gate radius. Detections outside this radius
                            cannot be assigned to the track, allowing for maneuvering tolerance.

    Returns:
        matched_indices: Tuples of (Track_Index, Detection_Index).
        unmatched_tracks: Tracks that were not seen this frame (will enter Coasting mode).
        unmatched_detections: New detections not belonging to any track (will spawn new tracks).
    """
    matched_indices = []
    unmatched_tracks = []
    unmatched_detections = []
    assigned_tracks = set()

    for det_idx, det in enumerate(detections):
        best_track_idx = -1
        min_distance = float('inf')

        det_x = det['x']
        det_y = det['y']

        # Scan all active tracks to find the closest predicted position
        for track_idx, track in enumerate(tracks_meta):
            if track_idx in assigned_tracks:
                continue

            track_x = batch_kf.state_vectors[track_idx, 0, 0]
            track_y = batch_kf.state_vectors[track_idx, 1, 0]
            euclidean_distance = np.hypot(det_x - track_x, det_y - track_y)

            # Association Gate Check
            if euclidean_distance < min_distance and euclidean_distance <= distance_threshold:
                min_distance = euclidean_distance
                best_track_idx = track_idx

        # Assign detection to the nearest valid track
        if best_track_idx != -1:
            matched_indices.append((best_track_idx, det_idx))
            assigned_tracks.add(best_track_idx)
        else:
            unmatched_detections.append(det_idx)

    # Flag tracks that received no measurements this frame
    for track_idx in range(len(tracks_meta)):
        if track_idx not in assigned_tracks:
            unmatched_tracks.append(track_idx)

    return matched_indices, unmatched_tracks, unmatched_detections