"""
Radar Constants Module.

Defines the global physical, mathematical, and environmental constants 
used across the FMCW radar simulation architecture. Isolating these values 
allows for centralized configuration, ensuring consistency across physics 
engines, DSP pipelines, and tracking algorithms.
"""

# Speed of electromagnetic waves in a vacuum (meters per second).
# Critical constant for calculating time delays (tau), spatial wavelengths, 
# and radial velocity Doppler shifts in the RF hardware simulation.
SPEED_OF_LIGHT: float = 3e8