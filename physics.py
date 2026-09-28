"""
Physics and Hardware Simulation Engine.

This module simulates the physical transmission, propagation, and reception
of FMCW (Frequency Modulated Continuous Wave) radar signals.
It utilizes a highly optimized 3D Tensor architecture executed directly on
NVIDIA GPUs via CuPy, capable of calculating tens of millions of complex
electromagnetic samples in milliseconds without Python loop overhead.
"""

import numpy as np
from constants import SPEED_OF_LIGHT

# ==============================================================================
# GPU ACCELERATION INIT FOR PHYSICS ENGINE
# ==============================================================================
try:
    import cupy as cp

    GPU_AVAILABLE = True
    # GPU successfully detected in the system! Physics engine will run on VRAM.
except ImportError:
    GPU_AVAILABLE = False


class RadarTarget:
    """Represents a physical target in the simulated environment."""

    def __init__(self, initial_distance, radial_velocity, azimuth_angle=0.0, rcs=1.0):
        self.current_distance = initial_distance
        self.radial_velocity = radial_velocity
        self.azimuth_angle = azimuth_angle
        self.rcs = rcs  # Radar Cross Section (reflectivity magnitude)

    def update_position(self, dt):
        """Linearly updates the target's distance based on its radial velocity."""
        self.current_distance += self.radial_velocity * dt


class RadarTransmitter:
    """Simulates the FMCW waveform generator (RF Synthesizer)."""

    def __init__(self, carrier_frequency, bandwidth, duration):
        self.carrier_frequency = carrier_frequency
        self.bandwidth = bandwidth
        self.duration = duration


class RadarReceiver:
    """Simulates the receiving antenna array hardware."""

    def __init__(self, transmitter):
        self.transmitter = transmitter
        # 2D Cross-Array configuration for true 360-degree spatial reception
        self.num_antennas_per_axis = 4
        self.num_antennas = 8  # 4 antennas for the X-axis, 4 antennas for the Y-axis
        self.wavelength = SPEED_OF_LIGHT / self.transmitter.carrier_frequency
        self.antenna_spacing = self.wavelength / 2


class RadarMixer:
    """
    Simulates the physical RF mixer that multiplies transmitted (TX)
    and received (RX) signals to extract the beat frequencies (IF signal).
    """

    def __init__(self):
        pass


def run_hardware_simulation(targets_list, transmitter, receiver, mixer, time_array, chirps_amount, duration_value,
                            macro_dt):
    """
    ULTRA-FAST 3D Tensor Physics Engine running entirely on NVIDIA GPU (CuPy).

    Data Dimensions generated:
    - Fast-Time: Samples per chirp
    - Slow-Time: Number of chirps per frame (Doppler phase accumulation)
    - Spatial: Number of receiving antennas (Cross-Array MIMO)
    """
    num_targets = len(targets_list)

    # Return empty noise block if no targets are present to maintain pipeline stability
    if num_targets == 0:
        return np.zeros((receiver.num_antennas, chirps_amount, len(time_array)), dtype=complex)

    # 1. Extract physical states from Python objects into contiguous arrays for tensor math
    distances = np.array([tgt.current_distance for tgt in targets_list])
    velocities = np.array([tgt.radial_velocity for tgt in targets_list])
    rcs_vals = np.array([tgt.rcs for tgt in targets_list])
    angles_rad = np.deg2rad(np.array([tgt.azimuth_angle for tgt in targets_list]))

    # ==========================================
    # GPU MEMORY UPLOAD & TENSOR ROUTING
    # ==========================================
    if GPU_AVAILABLE:
        xp = cp  # Route math operations to CuPy (VRAM)
        distances = cp.asarray(distances)
        velocities = cp.asarray(velocities)
        rcs_vals = cp.asarray(rcs_vals)
        angles_rad = cp.asarray(angles_rad)
        time_array = cp.asarray(time_array)
    else:
        xp = np  # Fallback to standard NumPy (CPU RAM)

    # 2. Build 2D Slow-Time Distance Matrix (Targets x Chirps)
    # Simulates microscopic target movement during the frame to induce Doppler shifts
    m_idx = xp.arange(chirps_amount)
    dist_matrix = distances[:, None] + velocities[:, None] * (m_idx[None, :] * duration_value)

    # 3. Time Delay (Tau) Tensor (Targets x Chirps x Samples)
    tau = (2 * dist_matrix) / SPEED_OF_LIGHT
    delayed_time = time_array[None, None, :] - tau[:, :, None]

    # 4. Generate TX (Transmitted) and RX (Received) FMCW Signals directly in Memory
    tx_phase = 2 * xp.pi * (
            transmitter.carrier_frequency * time_array + (transmitter.bandwidth / (2 * transmitter.duration)) * (
            time_array ** 2))
    tx_signal = xp.exp(1j * tx_phase)

    rx_phase = 2 * xp.pi * (
            transmitter.carrier_frequency * delayed_time + (transmitter.bandwidth / (2 * transmitter.duration)) * (
            delayed_time ** 2))
    rx_signal = xp.exp(1j * rx_phase) * rcs_vals[:, None, None]

    # 5. Spatial Phase Shift (2D Cross-Array configuration for 360-degree vision)
    k_indices = xp.arange(receiver.num_antennas_per_axis)

    # Calculate phase difference for the X-axis antennas (extracts the Sine component)
    phase_x = (2 * xp.pi / receiver.wavelength) * receiver.antenna_spacing * xp.sin(angles_rad[:, None]) * k_indices[
        None, :]
    spatial_exp_x = xp.exp(-1j * phase_x)

    # Calculate phase difference for the Y-axis antennas (extracts the Cosine component)
    phase_y = (2 * xp.pi / receiver.wavelength) * receiver.antenna_spacing * xp.cos(angles_rad[:, None]) * k_indices[
        None, :]
    spatial_exp_y = xp.exp(-1j * phase_y)

    # Concatenate both channels into a single 8-antenna MIMO array (Dimensions: Targets x 8)
    spatial_exp = xp.concatenate((spatial_exp_x, spatial_exp_y), axis=1)

    # 6. Hardware Mixer Operation (TX * conjugate(RX))
    mixed_signal = tx_signal[None, None, :] * xp.conj(rx_signal)

    # 7. Electromagnetic Superposition
    # Einstein Summation (einsum) collapses the Target dimension to yield the raw antenna data
    radar_cube = xp.einsum('nk, nmt -> kmt', spatial_exp, mixed_signal)

    # 8. Inject Hardware Thermal Noise (AWGN - Additive White Gaussian Noise)
    noise_power = 0.5
    noise = xp.random.normal(0, noise_power, radar_cube.shape) + 1j * xp.random.normal(0, noise_power, radar_cube.shape)
    radar_cube += noise

    # Download processed tensor back to Host RAM (CPU) for downstream DSP and OS processing
    if GPU_AVAILABLE:
        radar_cube = cp.asnumpy(radar_cube)
        velocities = cp.asnumpy(velocities)

    # 9. Macro-Kinematics Update
    # Advance physical positions over the non-transmitting (dead) time for the next OS frame
    frame_transmit_time = chirps_amount * duration_value
    dead_time = macro_dt - frame_transmit_time
    total_time = frame_transmit_time + (dead_time if dead_time > 0 else 0)

    for k, tgt in enumerate(targets_list):
        tgt.current_distance += velocities[k] * total_time

    return radar_cube