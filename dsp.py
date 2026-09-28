"""
Digital Signal Processing (DSP) Pipeline Module.

Responsible for translating raw baseband I/Q data (Radar Cube) into physical
measurements (Range, Velocity, Angle).
Features a Hybrid computing architecture:
1. GPU (CUDA) for massively parallel 2D-FFT transformations.
2. CPU Numba (JIT compilation) for pixel-level, C++ speed CA-CFAR detection.
"""

import numpy as np
from scipy.ndimage import label, maximum_position
from numba import njit
from constants import SPEED_OF_LIGHT

# ==============================================================================
# GPU ACCELERATION INIT FOR FFT PROCESSING
# ==============================================================================
try:
    import cupy as cp

    GPU_AVAILABLE = True
    print("[SYSTEM] NVIDIA GPU Detected. CuPy Acceleration ENABLED for DSP.")
except ImportError:
    GPU_AVAILABLE = False
    print("[SYSTEM] CuPy not found. Falling back to CPU (NumPy) for DSP.")


def process_range_doppler(radar_cube, sample_rate, carrier_freq, bandwidth, duration):
    """
    Executes a 2D Fast Fourier Transform to extract Range (Distance) and Doppler (Velocity).
    Applies 2D Hanning windowing to reduce spectral leakage and sidelobe levels.

    Args:
        radar_cube: Complex tensor of shape (antennas, chirps, samples).
        sample_rate: ADC sampling rate in Hz.
        carrier_freq: Radar start frequency in Hz (e.g., 77 GHz).
        bandwidth: FMCW sweep bandwidth in Hz.
        duration: Chirp duration (Sweep time) in seconds.

    Returns:
        range_doppler_cube: Processed 3D FFT tensor.
        range_axis_cropped: Physical distance axis in meters.
        velocity_axis: Physical radial velocity axis in m/s.
        max_range_idx: Index limit for cropping redundant far-field ranges.
    """
    n_doppler_pad = 128
    n_range_pad = 2048

    if GPU_AVAILABLE:
        # 1. TRANSFER TO GPU (VRAM)
        cube_gpu = cp.asarray(radar_cube)

        # 2. GPU WINDOWING
        window_range = cp.hanning(cube_gpu.shape[2])
        window_doppler = cp.hanning(cube_gpu.shape[1])
        window_2d = window_doppler[:, cp.newaxis] * window_range[cp.newaxis, :]
        cube_gpu = cube_gpu * window_2d

        # 3. MASSIVELY PARALLEL GPU 2D-FFT
        rd_cube_gpu = cp.fft.fft2(cube_gpu, s=(n_doppler_pad, n_range_pad), axes=(1, 2))
        rd_cube_gpu = cp.fft.fftshift(rd_cube_gpu, axes=1)

        # 4. TRANSFER BACK TO CPU (RAM) for Numba CA-CFAR processing
        range_doppler_cube = cp.asnumpy(rd_cube_gpu)

    else:
        # TRADITIONAL CPU PATH (NumPy Fallback)
        window_range = np.hanning(radar_cube.shape[2])
        window_doppler = np.hanning(radar_cube.shape[1])
        window_2d = window_doppler[:, np.newaxis] * window_range[np.newaxis, :]
        radar_cube = radar_cube * window_2d

        range_doppler_cube = np.fft.fft2(radar_cube, s=(n_doppler_pad, n_range_pad), axes=(1, 2))
        range_doppler_cube = np.fft.fftshift(range_doppler_cube, axes=1)

    # ==========================================
    # Physical Axis Calculations (CPU operations)
    # ==========================================
    half_samples = n_range_pad // 2
    range_freqs = np.fft.fftfreq(n_range_pad, d=(1 / sample_rate))[:half_samples]
    range_axis = (range_freqs * SPEED_OF_LIGHT * duration) / (2 * bandwidth)

    doppler_freqs = np.fft.fftshift(np.fft.fftfreq(n_doppler_pad, d=duration))
    velocity_axis = (doppler_freqs * SPEED_OF_LIGHT) / (2 * carrier_freq)

    # Crop redundant distant ranges to minimize downstream CFAR computational load
    max_range_idx = np.searchsorted(range_axis, 100)
    range_axis_cropped = range_axis[:max_range_idx]

    return range_doppler_cube, range_axis_cropped, velocity_axis, max_range_idx


def calculate_angle_of_arrival(range_doppler_cube, peak_doppler_indices, peak_range_indices):
    """
    Extracts true 360-degree Spatial Angle (Azimuth) using a 2D Cross-Array configuration.
    Extracts the spatial vectors for X (sine) and Y (cosine) axes, runs a 1D-FFT on each,
    and reconstructs the angle using arctan2 to eliminate front/back ambiguity.
    """
    angles_list = []
    angle_fft_size = 256
    spatial_freqs = np.fft.fftshift(np.fft.fftfreq(angle_fft_size))

    for i in range(len(peak_range_indices)):
        doppler_i = peak_doppler_indices[i]
        range_i = peak_range_indices[i]

        # Extract the spatial vector across all 8 antennas for the detected target cell
        spatial_vector = range_doppler_cube[:, doppler_i, range_i]

        # Extract Azimuth spatial frequency from the X-axis (horizontal) antennas
        spatial_x = spatial_vector[0:4]
        spatial_fft_x = np.fft.fftshift(np.fft.fft(spatial_x, n=angle_fft_size))
        peak_angle_idx_x = np.argmax(np.abs(spatial_fft_x))
        sine_val = np.clip(spatial_freqs[peak_angle_idx_x] * 2, -1.0, 1.0)

        # Extract Elevation/Depth spatial frequency from the Y-axis (vertical) antennas
        spatial_y = spatial_vector[4:8]
        spatial_fft_y = np.fft.fftshift(np.fft.fft(spatial_y, n=angle_fft_size))
        peak_angle_idx_y = np.argmax(np.abs(spatial_fft_y))
        cosine_val = np.clip(spatial_freqs[peak_angle_idx_y] * 2, -1.0, 1.0)

        # Reconstruct the absolute 360-degree physical angle using the four-quadrant inverse tangent
        calculated_angle = np.rad2deg(np.arctan2(sine_val, cosine_val))
        angles_list.append(calculated_angle)

    return angles_list


# ==============================================================================
# NUMBA COMPILED CA-CFAR ENGINE (Runs at bare-metal C++ speeds)
# ==============================================================================

@njit
def numba_cfar_core(linear_map, rd_db, d_span, r_span, guard_d, guard_r, offset_db):
    """
    Core implementation of the 2D Cell-Averaging Constant False Alarm Rate (CA-CFAR) window.
    Bypasses Python's GIL using the @njit decorator for extreme nested loop performance.

    Args:
        linear_map: Power map in linear scale for accurate noise averaging.
        rd_db: Power map in logarithmic (dB) scale for threshold comparison.
        d_span, r_span: Outer window dimensions (Training + Guard cells).
        guard_d, guard_r: Inner guard window dimensions to prevent self-masking.
        offset_db: SNR threshold margin above the local noise floor.
    """
    rows, cols = linear_map.shape
    target_mask = np.zeros((rows, cols), dtype=np.bool_)

    num_training_cells = ((2 * d_span + 1) * (2 * r_span + 1)) - ((2 * guard_d + 1) * (2 * guard_r + 1))

    # Slide the CUT (Cell Under Test) across the Range-Doppler map
    for d in range(d_span, rows - d_span):
        for r in range(r_span, cols - r_span):

            # Aggregate local background noise power
            noise_sum = 0.0
            for i in range(d - d_span, d + d_span + 1):
                for j in range(r - r_span, r + r_span + 1):
                    # Exclude the inner guard cells to prevent the target from raising its own threshold
                    if abs(i - d) > guard_d or abs(j - r) > guard_r:
                        noise_sum += linear_map[i, j]

            noise_avg = noise_sum / num_training_cells
            noise_db = 20 * np.log10(noise_avg + 1e-10)

            # Target is declared if its power exceeds the adaptive local noise floor + offset
            if rd_db[d, r] > noise_db + offset_db:
                target_mask[d, r] = True

    return target_mask


@njit
def numba_fast_dilation(mask):
    """
    Machine-code compiled Morphological Dilation.
    Connects fragmented detection pixels (sidelobes) belonging to the same physical target cluster.
    """
    rows, cols = mask.shape
    dilated = np.zeros((rows, cols), dtype=np.bool_)

    for d in range(1, rows - 1):
        for r in range(1, cols - 1):
            if mask[d, r]:
                # Stamp a 3x3 boolean 'True' area around the detection
                for i in range(d - 1, d + 2):
                    for j in range(r - 1, r + 2):
                        dilated[i, j] = True

    return dilated


def run_2d_ca_cfar(range_doppler_cube, max_range_idx):
    """
    Orchestrates the dynamic thresholding detection process.
    Prepares arrays, executes Numba cores, and clusters detections using SciPy.
    """
    # Non-coherent integration: Average magnitude across all antennas to improve SNR
    averaged_magnitude = np.mean(np.abs(range_doppler_cube), axis=0)
    rd_db = 20 * np.log10(averaged_magnitude + 1e-10)

    rd_db_cropped = rd_db[:, :max_range_idx]
    linear_map_cropped = averaged_magnitude[:, :max_range_idx]
    peak_power = np.max(rd_db_cropped)

    # CFAR Window Configuration
    training_range = 8
    training_doppler = 4
    guard_range = 2
    guard_doppler = 1
    offset_db = 15.0

    # 1. Execute highly optimized Numba CFAR core
    target_mask = numba_cfar_core(
        linear_map_cropped, rd_db_cropped,
        training_doppler, training_range,
        guard_doppler, guard_range, offset_db
    )

    # 2. Iterative Morphological Dilation (Clustering)
    for _ in range(3):
        target_mask = numba_fast_dilation(target_mask)

    # 3. Label connected components to identify discrete physical targets
    labeled_array, num_features = label(target_mask)

    peak_doppler_indices = []
    peak_range_indices = []

    if num_features > 0:
        # Extract the coordinate of the maximum intensity pixel within each target cluster
        cluster_peaks = maximum_position(rd_db_cropped, labels=labeled_array, index=np.arange(1, num_features + 1))
        peak_doppler_indices, peak_range_indices = zip(*cluster_peaks)
        peak_doppler_indices = list(peak_doppler_indices)
        peak_range_indices = list(peak_range_indices)

    # Blank out the rest of the map to display only verified detections in the GUI
    rd_db_thresholded = np.full(rd_db_cropped.shape, -100.0)
    rd_db_thresholded[target_mask] = rd_db_cropped[target_mask]

    return rd_db_thresholded, peak_doppler_indices, peak_range_indices, peak_power