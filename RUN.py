"""
Primary System Orchestrator & Real-Time Pipeline Manager.

This module initializes the physical target environment and establishes the
asynchronous Multiprocessing architecture. By isolating the computationally
heavy physics, DSP, and tracking algorithms on a dedicated background CPU/GPU
process, it ensures the Frontend UI thread remains unblocked and highly responsive.

Communication between the Backend and Frontend is managed via a thread-safe
IPC (Inter-Process Communication) Queue.
"""

import time
import queue
import numpy as np
import tkinter as tk
import multiprocessing as mp

from physics import RadarTarget, RadarTransmitter, RadarReceiver, RadarMixer, run_hardware_simulation
from dsp import process_range_doppler, calculate_angle_of_arrival, run_2d_ca_cfar
from tracking import BatchKalmanFilter, Track, polar_to_cartesian, associate_detections
from gui import RadarDashboard


def get_targets():
    """
    Generates the simulated physical environment (Test Bench).
    Returns a list of RadarTarget objects with varying ranges, velocities,
    azimuth angles, and Radar Cross Section (RCS) values.
    """
    t1 = RadarTarget(initial_distance=60.0, radial_velocity=-15.0, azimuth_angle=10.0, rcs=10.0)
    t2 = RadarTarget(initial_distance=30.0, radial_velocity=20.0, azimuth_angle=-20.0, rcs=15.0)
    t3 = RadarTarget(initial_distance=15.0, radial_velocity=0.0, azimuth_angle=5.0, rcs=5.0)
    t4 = RadarTarget(initial_distance=85.0, radial_velocity=0.0, azimuth_angle=30.0, rcs=50.0)
    t5 = RadarTarget(initial_distance=98.0, radial_velocity=-5.0, azimuth_angle=-5.0, rcs=100.0)
    t6 = RadarTarget(initial_distance=5.0, radial_velocity=1.5, azimuth_angle=0.0, rcs=1.0)
    t7 = RadarTarget(initial_distance=40.0, radial_velocity=10.0, azimuth_angle=35.0, rcs=500.0)
    t8 = RadarTarget(initial_distance=40.0, radial_velocity=10.0, azimuth_angle=40.0, rcs=0.1)
    t9 = RadarTarget(initial_distance=50.0, radial_velocity=-10.0, azimuth_angle=-10.0, rcs=10.0)
    t10 = RadarTarget(initial_distance=52.0, radial_velocity=-10.0, azimuth_angle=-10.0, rcs=10.0)
    t11 = RadarTarget(initial_distance=70.0, radial_velocity=25.0, azimuth_angle=20.0, rcs=15.0)
    t12 = RadarTarget(initial_distance=70.0, radial_velocity=28.0, azimuth_angle=20.0, rcs=15.0)
    t13 = RadarTarget(initial_distance=60.0, radial_velocity=20.0, azimuth_angle=135.0, rcs=50.0)
    t14 = RadarTarget(initial_distance=65.0, radial_velocity=20.0, azimuth_angle=135.0, rcs=50.0)
    t15 = RadarTarget(initial_distance=20.0, radial_velocity=85.0, azimuth_angle=150.0, rcs=5.0)
    t16 = RadarTarget(initial_distance=55.0, radial_velocity=0.0, azimuth_angle=180.0, rcs=10.0)
    t17 = RadarTarget(initial_distance=45.0, radial_velocity=-8.0, azimuth_angle=-120.0, rcs=0.5)
    t18 = RadarTarget(initial_distance=47.0, radial_velocity=-8.0, azimuth_angle=-122.0, rcs=0.5)
    t19 = RadarTarget(initial_distance=49.0, radial_velocity=-8.0, azimuth_angle=-124.0, rcs=0.5)
    t20 = RadarTarget(initial_distance=22.0, radial_velocity=-2.0, azimuth_angle=75.0, rcs=2.0)
    return [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20]


def radar_engine_process(physical_targets_list, out_queue):
    """
    Dedicated OS Process handling all mathematical and computational overhead.
    Executes the 6-stage Radar Pipeline:
    Physics -> DSP -> CFAR -> Angle Extraction -> Tracking -> IPC Transfer.
    """
    # System Hardware Parameters (77GHz Automotive/Tactical Band)
    carrier_frequency_value = 77e9
    bandwidth_value = 4e9
    duration_value = 20e-6
    sample_rate_value = 200e6
    chirps_amount = 128

    # Timing Configuration (20 Hz Physical Update Rate)
    UPDATE_RATE_MS = 50
    macro_dt = UPDATE_RATE_MS / 1000.0
    time_array = np.arange(0, duration_value, 1 / sample_rate_value)

    # Initialize Hardware Simulators
    transmitter = RadarTransmitter(carrier_frequency_value, bandwidth_value, duration_value)
    receiver = RadarReceiver(transmitter)
    mixer = RadarMixer()

    # Initialize Tracking Subsystem
    batch_kf = BatchKalmanFilter(macro_dt)
    active_tracks = []
    next_track_id = 1

    while True:
        start_time = time.time()

        # ==========================================
        # Stage 1: Electromagnetic Physics Simulation
        # ==========================================
        radar_cube = run_hardware_simulation(
            physical_targets_list, transmitter, receiver, mixer, time_array, chirps_amount, duration_value, macro_dt)

        # ==========================================
        # Stage 2: Fast Fourier Transforms (DSP)
        # ==========================================
        range_doppler_cube, range_axis_cropped, velocity_axis, max_range_idx = process_range_doppler(
            radar_cube, sample_rate_value, carrier_frequency_value, bandwidth_value, duration_value)

        # ==========================================
        # Stage 3: CA-CFAR Target Detection
        # ==========================================
        rd_db_thresholded, peak_doppler_indices, peak_range_indices, peak_power = run_2d_ca_cfar(
            range_doppler_cube, max_range_idx)

        # ==========================================
        # Stage 4: Spatial Processing (Angle of Arrival)
        # ==========================================
        angles_list = calculate_angle_of_arrival(range_doppler_cube, peak_doppler_indices, peak_range_indices)

        # Format raw physical detections for the tracking engine
        current_detections = []
        for i in range(len(peak_range_indices)):
            r = range_axis_cropped[peak_range_indices[i]]
            ang = angles_list[i]
            v = velocity_axis[peak_doppler_indices[i]]
            x, y = polar_to_cartesian(r, ang)
            current_detections.append({'x': x, 'y': y, 'v': v})

        # ==========================================
        # Stage 5: Target Tracking & Data Association
        # ==========================================
        # Associate current frame detections to persistent Kalman filter tracks
        matched, unmatched_trks, unmatched_dets = associate_detections(
            active_tracks, batch_kf, current_detections, distance_threshold=20.0
        )

        # Update existing tracks with new measurements
        measurements_for_update = []
        matched_indices = []
        for trk_idx, det_idx in matched:
            det = current_detections[det_idx]
            measurements_for_update.append([det['x'], det['y']])
            matched_indices.append(trk_idx)
            active_tracks[trk_idx].latest_measured_velocity = det['v']
            active_tracks[trk_idx].time_since_update = 0  # Reset coasting timer

        batch_kf.update(matched_indices, measurements_for_update)

        # Initialize tracks for completely new detections
        for det_idx in unmatched_dets:
            det = current_detections[det_idx]
            batch_kf.add_target(det['x'], det['y'])
            active_tracks.append(Track(next_track_id, det['v']))
            next_track_id += 1

        # Increment missing timer for tracks not found in this frame
        for trk_idx in unmatched_trks:
            active_tracks[trk_idx].time_since_update += 1

        # Track Coasting Execution: Drop tracks that have been missing for over 5 seconds (100 frames)
        dead_indices = [i for i, trk in enumerate(active_tracks) if trk.time_since_update > 100]
        batch_kf.delete_targets(dead_indices)
        for i in sorted(dead_indices, reverse=True):
            del active_tracks[i]

        # Predict next states using kinematic models
        batch_kf.predict()
        for trk in active_tracks:
            trk.age += 1

        # ==========================================
        # Stage 6: IPC Payload Packaging
        # ==========================================
        gui_targets_data = []
        for i, trk in enumerate(active_tracks):
            # Only send confirmed tracks (Age >= 2) to prevent ghost rendering
            if trk.age >= 2:
                x = batch_kf.state_vectors[i, 0, 0]
                y = batch_kf.state_vectors[i, 1, 0]
                r = float(np.hypot(x, y))
                ang = float(np.rad2deg(np.arctan2(x, y)))
                v_radial = float(trk.latest_measured_velocity)
                gui_targets_data.append({
                    'id': trk.track_id,
                    'r': r,
                    'v': v_radial,
                    'ang': ang,
                    'tsu': trk.time_since_update
                })

        # Cast large arrays to float32 to drastically reduce IPC queue memory bandwidth
        payload = {
            'rd_matrix': rd_db_thresholded.astype(np.float32),
            'peak_power': float(peak_power),
            'targets': gui_targets_data,
            'range_axis': range_axis_cropped.astype(np.float32),
            'velocity_axis': velocity_axis.astype(np.float32)
        }

        # Backpressure Management: Discard stale queue data if the GUI lags behind real-time
        while out_queue.qsize() > 1:
            try:
                out_queue.get_nowait()
            except queue.Empty:
                break

        # Push the finalized frame to the Frontend UI
        out_queue.put(payload)

        # Regulate processing speed to maintain an authentic simulation timeline
        elapsed = time.time() - start_time
        sleep_time = macro_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def start_gui(in_queue):
    """
    Initializes the isolated Main OS thread for GUI rendering.
    Continuously polls the IPC queue for asynchronous physical data updates.
    """
    root = tk.Tk()
    dummy_matrix = np.zeros((10, 10))
    app = RadarDashboard(root, dummy_matrix, [0, 1], [0, 1], 0, [])

    def poll_queue():
        try:
            payload = None
            # Drain the queue to fetch only the most instantaneous radar snapshot
            while not in_queue.empty():
                payload = in_queue.get_nowait()

            if payload is not None:
                app.range_axis = payload['range_axis']
                app.velocity_axis = payload['velocity_axis']
                app.update_dashboard_data(payload['rd_matrix'], payload['peak_power'], payload['targets'])

        except queue.Empty:
            pass

        # Extremely aggressive polling rate ensures zero perception of latency
        root.after(10, poll_queue)

    poll_queue()
    root.mainloop()


if __name__ == "__main__":
    # Freeze Support allows Windows OS to properly fork Multiprocessing processes without recursive crashes
    mp.freeze_support()

    # 1. Initialize simulated world environment
    targets_list = get_targets()

    # 2. Establish Thread-Safe IPC channel
    data_queue = mp.Queue()

    # 3. Spin up Backend Physics/DSP thread
    engine_process = mp.Process(target=radar_engine_process, args=(targets_list, data_queue))
    engine_process.daemon = True
    engine_process.start()

    # 4. Spin up Frontend UI (Blocks the main thread)
    start_gui(data_queue)