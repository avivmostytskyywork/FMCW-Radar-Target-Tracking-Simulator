# FMCW Radar & Target Tracking Simulator

A professional-grade, Object-Oriented Python engineering tool for simulating FMCW phased-array radar physics, Digital Signal Processing (DSP) pipelines, and real-time multi-target tracking. 

This project bridges theoretical electromagnetics and algorithmic software development, demonstrating how raw RF echoes are processed into actionable kinematic data for autonomous systems and defense architectures.

## Key Features

* **FMCW Physics Engine:** Simulates linear frequency-modulated continuous wave (FMCW) transmission and the superposition of electromagnetic waves across a multi-antenna phased-array receiver.
* **DSP Pipeline:** Implements a full digital signal processing chain, utilizing 2D-FFT for precise Range-Doppler extraction and 1D Spatial FFT for Angle of Arrival (AoA) estimation.
* **Adaptive Target Detection:** Features a vectorized 2D CA-CFAR (Cell Averaging Constant False Alarm Rate) algorithm integrated with morphological clustering (binary dilation and connected components) to detect multiple dynamic targets in high-clutter environments[cite: 6].
* **Tracking & Data Association:** Incorporates a custom Batch Kalman Filter leveraging NumPy tensor operations for parallel state estimation, paired with a Greedy Nearest Neighbor (GNN) engine for real-time kinematic data association[cite: 6].
* **Interactive Engineering Dashboard:** An event-driven GUI built with Tkinter and Matplotlib, featuring a live PPI spatial sweep (radar screen), Range-Doppler heatmaps, precision target markers, and a real-time kinematic tracking table[cite: 6].

## Repository Structure

The project follows a modular software architecture, strictly separating the physics backend from the graphical frontend:


FMCW-Radar-Simulator/
├── radar_engine.py     # The physical backend (Radar, Transmitter, Receiver, Mixer, Targets)
├── tracking_engine.py  # DSP, CA-CFAR, Batch Kalman Filter, and Data Association
├── main.py             # The frontend (Tkinter GUI and Event Handlers)
├── requirements.txt    # Project dependencies
├── .gitignore          # Git exclusion rules (e.g., __pycache__/)
└── assets/             # Directory for screenshots and documentation visuals


## Installation & Usage

1) Clone the repository:
    git clone https://github.com/avivmostytskyywork/FMCW-Radar-Simulator.git
    cd FMCW-Radar-Simulator

2) Install dependencies:
    It is recommended to use a virtual environment: pip install -r requirements.txt

3) Run the Simulator:
    Launch the graphical interface by executing the main script: python main.py

4) Workflow:
    Initialize the real-time simulation loop incorporating physical target kinematics and RF generation.
    Process incoming signals through the DSP pipeline, 2D CA-CFAR detection, and spatial angle extraction.
    Monitor live tracking data via the Tkinter GUI dashboard, featuring both Range-Doppler heatmaps and polar PPI spatial views.

## Engineering Analysis Dashboard
Displays the live Range-Doppler heatmap with CFAR thresholding, precision target markers, and the real-time kinematic tracking table.

## PPI Spatial View
Visualizes the radar's Field of View (FOV) with a dynamic rotational sweep and active target tracking in polar coordinates.

## Engineering & Academic Context
This project was developed as a portfolio asset for hardware, logic design, and algorithm engineering.

It translates core academic concepts—such as wave physics, algorithm efficiency, and object-oriented Python—into a practical EDA and radar processing application.

By managing memory allocations efficiently with NumPy and structuring state management cleanly, it demonstrates the intersection of software engineering and physical hardware design.

## Tech Stack
Language: Python
Computation & Math: NumPy (Vectorized array/tensor operations, `np.einsum`), SciPy (Morphological image processing, Clustering)
Data Visualization: Matplotlib (Dynamic plots
Graphical User Interface: Tkinter
