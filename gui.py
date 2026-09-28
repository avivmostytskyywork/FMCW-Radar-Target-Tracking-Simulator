"""
Tactical C4I (Command, Control, Communications, Computers, and Intelligence) Dashboard Module.

This module renders the frontend graphical user interface (GUI) for the radar system.
It utilizes a "Clock Decoupling" architecture: the GUI runs its own high-speed animation
loop (e.g., 50 FPS) independently of the backend physics/DSP engine's update rate.
This ensures buttery-smooth tactical displays (PPI sweeping, phosphor decay, fading targets)
even if the mathematical backend experiences latency spikes.
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class RadarDashboard:
    """
    Main Application Class for the Radar Dashboard.
    Integrates Tkinter for layout/tables and Matplotlib for high-performance plotting.
    """

    def __init__(self, root, rd_matrix, range_axis, velocity_axis, peak_power, targets_list):
        self.root = root
        self.root.title("FMCW Radar Tactical C4I Dashboard")
        self.root.geometry("1200x700")

        # ==========================================
        # 1. Data Initialization & Memory Structures
        # ==========================================
        self.rd_matrix = rd_matrix
        self.range_axis = range_axis
        self.velocity_axis = velocity_axis
        self.peak_power = peak_power

        self.backend_targets = targets_list

        # Local cache for simulating visual persistence (Afterglow/Fade out)
        self.target_memory = {}
        # Persistent database tracking the full lifecycle of all discovered targets
        self.global_db = {}

        # ==========================================
        # 2. Animation Engine & Clock Decoupling Config
        # ==========================================
        self.dark_mode = True
        self.scan_angle = 0.0

        # UI renders at 50 FPS regardless of the backend physics tick rate
        self.fps = 50
        self.ms_per_frame = int(1000 / self.fps)

        # Tactical sweep speed: 720 degrees/sec = 2 full rotations per second
        self.degrees_per_sec = 720.0
        self.degrees_per_frame = self.degrees_per_sec / self.fps

        # Color palette for differentiating multiple tracks
        self.target_colors = ['#4da6ff', '#ff9933', '#4dff4d', '#ff4d4d', '#b366ff',
                              '#d2b48c', '#ffb3e6', '#cccccc', '#ffff66', '#33ffff']

        # ==========================================
        # 3. Tkinter UI Layout Construction
        # ==========================================
        self.style = ttk.Style()
        self.style.theme_use('default')

        self.sidebar = ttk.Frame(self.root, width=200, padding=15)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        self.main_area = ttk.Frame(self.root, padding=10)
        self.main_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.plot_frame = ttk.Frame(self.main_area)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.list_frame = ttk.Frame(self.main_area, width=450, padding=5)
        self.list_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # ==========================================
        # 4. Matplotlib Canvas Integration
        # ==========================================
        self.fig = plt.Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax_rd = self.fig.add_subplot(111)
        self.ax_ppi = self.fig.add_subplot(111, polar=True)
        self.ax_rd.set_visible(False)
        self.ax_ppi.set_visible(False)
        self.cb = None

        # ==========================================
        # 5. Tabbed Data Environments (Live vs History)
        # ==========================================
        self.notebook = ttk.Notebook(self.list_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_live = ttk.Frame(self.notebook)
        self.tab_hist = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_live, text="Live Sector")
        self.notebook.add(self.tab_hist, text="Global History")

        # --- Setup Live Tracking Treeview ---
        self.table_title = ttk.Label(self.tab_live, text="Live Tracking: 0 Targets", font=("Segoe UI", 12, "bold"))
        self.table_title.pack(pady=(5, 5))

        columns_live = ("id", "range", "velocity", "angle")
        self.tree = ttk.Treeview(self.tab_live, columns=columns_live, show="headings", height=20)
        self.tree.heading("id", text="ID")
        self.tree.heading("range", text="Range (m)")
        self.tree.heading("velocity", text="Vel (m/s)")
        self.tree.heading("angle", text="Angle (°)")
        self.tree.column("id", anchor=tk.CENTER, width=60)
        self.tree.column("range", anchor=tk.CENTER, width=90)
        self.tree.column("velocity", anchor=tk.CENTER, width=90)
        self.tree.column("angle", anchor=tk.CENTER, width=90)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # --- Setup Historical Database Treeview ---
        self.hist_title = ttk.Label(self.tab_hist, text="Target Database Log", font=("Segoe UI", 12, "bold"))
        self.hist_title.pack(pady=(5, 5))

        columns_hist = ("id", "status", "range", "velocity")
        self.hist_tree = ttk.Treeview(self.tab_hist, columns=columns_hist, show="headings", height=20)
        self.hist_tree.heading("id", text="ID")
        self.hist_tree.heading("status", text="Status")
        self.hist_tree.heading("range", text="Last R (m)")
        self.hist_tree.heading("velocity", text="Last V (m/s)")
        self.hist_tree.column("id", anchor=tk.CENTER, width=50)
        self.hist_tree.column("status", anchor=tk.CENTER, width=80)
        self.hist_tree.column("range", anchor=tk.CENTER, width=100)
        self.hist_tree.column("velocity", anchor=tk.CENTER, width=100)
        self.hist_tree.pack(fill=tk.BOTH, expand=True)

        # --- Setup Navigation Sidebar ---
        self.nav_title = ttk.Label(self.sidebar, text="Display Modes", font=("Segoe UI", 14, "bold"))
        self.nav_title.pack(pady=(0, 20))

        ttk.Button(self.sidebar, text="Range-Doppler Map (FFT)", command=self.show_range_doppler_view).pack(pady=10,
                                                                                                            fill=tk.X,
                                                                                                            ipady=10)
        ttk.Button(self.sidebar, text="Spatial View (PPI)", command=self.show_spatial_view_view).pack(pady=10,
                                                                                                      fill=tk.X,
                                                                                                      ipady=10)
        ttk.Button(self.sidebar, text="Toggle Dark Mode", command=self.toggle_dark_mode).pack(pady=(50, 10), fill=tk.X,
                                                                                              ipady=10)

        # Initialize visual state
        self.apply_theme()
        self.current_view = "spatial_view"

        # Boot up the asynchronous rendering loop
        self.animate()

    def update_dashboard_data(self, rd_matrix, peak_power, targets_list):
        """
        Ingests fresh data from the backend IPC Queue.
        Does NOT trigger a render directly, only updates the background state.
        """
        self.rd_matrix = rd_matrix
        self.peak_power = peak_power
        self.backend_targets = targets_list

    def animate(self):
        """
        The core decoupled rendering loop.
        Calculates aesthetic effects (beam rotation, phosphor decay fading) and schedules
        the next frame to guarantee consistent UI performance.
        """
        # Advance the beam angle
        self.scan_angle = (self.scan_angle + self.degrees_per_frame) % 360.0
        sweep_width_deg = 60.0

        # ==========================================
        # TARGET MEMORY DECAY & REFRESH LOGIC
        # ==========================================
        # 1. Decay the visual lifespan of existing targets (fades out completely in 1.5 seconds)
        decay_rate = 1.0 / (self.fps * 1.5)
        for tid in list(self.target_memory.keys()):
            self.target_memory[tid]['life'] -= decay_rate
            if self.target_memory[tid]['life'] <= 0:
                del self.target_memory[tid]  # Purge from visual cache once fully faded

        active_ids_this_frame = set()

        # 2. Synchronize Backend data with Frontend Visual representations
        for tgt in self.backend_targets:
            tid = tgt['id']
            tsu = tgt['tsu']  # Time Since Update (Frames since actual radar detection)

            # Target is 'Active' if the Kalman filter recently measured it
            is_active = (tsu < 3)
            status_text = "Active" if is_active else "Lost"

            # Upsert into Global Historical Database
            if tid not in self.global_db:
                self.global_db[tid] = {'r': tgt['r'], 'v': tgt['v'], 'ang': tgt['ang'], 'status': status_text}
            else:
                self.global_db[tid]['r'] = tgt['r']
                self.global_db[tid]['v'] = tgt['v']
                self.global_db[tid]['ang'] = tgt['ang']
                self.global_db[tid]['status'] = status_text

            active_ids_this_frame.add(tid)

            # Update visual target memory ONLY if genuinely detected and currently inside the scanning beam
            if is_active:
                tgt_ang_norm = (tgt['ang'] + 360) % 360
                angle_diff = (self.scan_angle - tgt_ang_norm + 360) % 360

                if angle_diff <= sweep_width_deg:
                    self.target_memory[tid] = {
                        'data': tgt,
                        'life': 1.0  # Reset lifespan to 100% brightness
                    }

        # 3. Flag tracks that the backend Kalman filter has entirely deleted
        for tid in self.global_db:
            if tid not in active_ids_this_frame:
                self.global_db[tid]['status'] = "Dropped"

        # 4. Render the active selected View
        if self.current_view == "range_doppler":
            self.draw_range_doppler()
        else:
            self.draw_spatial_view()

        # 5. Update UI Tables and schedule next frame
        self.populate_target_list()
        self.populate_history_list()
        self.root.after(self.ms_per_frame, self.animate)

    # --- View Routing Methods ---
    def show_range_doppler_view(self):
        self.current_view = "range_doppler"

    def show_spatial_view_view(self):
        self.current_view = "spatial_view"

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()

    def apply_theme(self):
        """Applies dynamic tactical styling (Dark/Light mode) to UI components."""
        bg_color = '#1e1e1e' if self.dark_mode else '#f0f0f0'
        panel_bg = '#252526' if self.dark_mode else 'white'
        fg_color = '#d4d4d4' if self.dark_mode else 'black'
        accent_color = '#007acc' if self.dark_mode else '#0058a3'

        self.root.configure(bg=bg_color)
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabel', background=bg_color, foreground=fg_color)
        self.style.configure('TButton', font=("Segoe UI", 10))

        self.style.configure('TNotebook', background=bg_color, borderwidth=0)
        self.style.configure('TNotebook.Tab', background=panel_bg, foreground=fg_color, padding=[10, 5],
                             font=("Segoe UI", 10, "bold"))
        self.style.map('TNotebook.Tab', background=[('selected', accent_color)], foreground=[('selected', 'white')])

        self.style.configure('Treeview', background=panel_bg, foreground=fg_color, fieldbackground=panel_bg,
                             borderwidth=0, font=("Segoe UI", 10))
        self.style.map('Treeview', background=[('selected', accent_color)])
        self.style.configure('Treeview.Heading', background=bg_color, foreground=fg_color, relief='flat',
                             font=("Segoe UI", 10, "bold"))
        self.style.map('Treeview.Heading', background=[('active', panel_bg)])

        self.toolbar.config(background=bg_color)
        for child in self.toolbar.winfo_children():
            child.config(background=bg_color)
            if isinstance(child, tk.Label):
                child.config(foreground=fg_color)
            elif isinstance(child, tk.Button):
                child.config(background=panel_bg, foreground=fg_color, activebackground=accent_color,
                             activeforeground='white', relief=tk.FLAT, borderwidth=1)

    def get_plot_colors(self):
        return ('#1e1e1e', '#d4d4d4') if self.dark_mode else ('white', 'black')

    def apply_axes_theme(self, ax, bg_color, fg_color):
        """Applies current theme colors to Matplotlib axes, spines, and ticks."""
        self.fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=fg_color)
        ax.xaxis.label.set_color(fg_color)
        ax.yaxis.label.set_color(fg_color)
        ax.title.set_color(fg_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(fg_color)

    def populate_target_list(self):
        """Updates the Live Target Sector treeview based on visual memory caches."""
        for row in self.tree.get_children():
            self.tree.delete(row)

        for tid, mem in self.target_memory.items():
            tgt = mem['data']
            color_hex = self.target_colors[tid % len(self.target_colors)]
            tag_name = f"tgt_{tid}"

            self.tree.tag_configure(tag_name, foreground=color_hex, font=("Segoe UI", 10, "bold"))
            self.tree.insert("", tk.END, values=(
                f"TGT #{tid}",
                f"{tgt['r']:.2f}",
                f"{tgt['v']:.2f}",
                f"{tgt['ang']:.2f}"
            ), tags=(tag_name,))

        self.table_title.config(text=f"Live Tracking: {len(self.target_memory)} Targets")

    def populate_history_list(self):
        """Updates the Global Database treeview with full target lifecycle logs."""
        for row in self.hist_tree.get_children():
            self.hist_tree.delete(row)

        for tid, data in self.global_db.items():
            color_hex = self.target_colors[tid % len(self.target_colors)]

            # Apply typographic styling based on tactical status
            if data['status'] == 'Active':
                font_style = ("Segoe UI", 10, "bold")
                item_color = color_hex
            elif data['status'] == 'Lost':
                font_style = ("Segoe UI", 10, "italic")
                item_color = '#888888'
            else:
                font_style = ("Segoe UI", 10, "overstrike")
                item_color = '#555555'

            tag_name = f"hist_{tid}"
            self.hist_tree.tag_configure(tag_name, foreground=item_color, font=font_style)

            self.hist_tree.insert("", tk.END, values=(
                f"TGT #{tid}",
                data['status'],
                f"{data['r']:.2f}",
                f"{data['v']:.2f}"
            ), tags=(tag_name,))

        self.hist_title.config(text=f"Total Unique Targets Discovered: {len(self.global_db)}")

    def draw_range_doppler(self):
        """
        Renders the FFT spectral signature map of incoming targets.
        Visualizes Target Range against Radial Velocity.
        """
        self.ax_ppi.set_visible(False)
        self.ax_rd.set_visible(True)
        self.ax_rd.clear()
        bg_color, fg_color = self.get_plot_colors()

        # Render 2D-FFT Heatmap (if data exists)
        if self.rd_matrix.size > 0:
            cax = self.ax_rd.imshow(self.rd_matrix, aspect='auto', origin='lower',
                                    extent=[self.range_axis[0], self.range_axis[-1], self.velocity_axis[0],
                                            self.velocity_axis[-1]],
                                    cmap='jet', interpolation='bilinear', vmin=-100, vmax=self.peak_power)

            # Lazy-load colorbar to prevent duplicate UI elements
            if self.cb is None:
                self.cb = self.fig.colorbar(cax, ax=self.ax_rd)

            self.cb.set_label('Magnitude (dB)', color=fg_color)
            self.cb.ax.yaxis.set_tick_params(color=fg_color)
            plt.setp(plt.getp(self.cb.ax.axes, 'yticklabels'), color=fg_color)

        # Render Target markers over the heatmap, utilizing the fade-out alpha logic
        for tid, mem in self.target_memory.items():
            tgt = mem['data']
            life = mem['life']
            color_hex = self.target_colors[tid % len(self.target_colors)]
            self.ax_rd.plot(tgt['r'], tgt['v'], marker='x', color=color_hex, markersize=12, markeredgewidth=2,
                            alpha=max(0.1, life))

        self.ax_rd.set_title("I/Q Radar Range-Doppler Map (Active Sector Only)", fontsize=12)
        self.ax_rd.set_xlabel("Distance (Meters)")
        self.ax_rd.set_ylabel("Velocity (m/s) [Positive = Moving Away]")
        self.ax_rd.grid(color='orange', linestyle='--', alpha=0.6)

        self.apply_axes_theme(self.ax_rd, bg_color, fg_color)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def draw_spatial_view(self):
        """
        Renders the tactical Plan Position Indicator (PPI) spatial map.
        Simulates legacy CRT display behavior using non-linear phosphor decay trails.
        """
        self.ax_rd.set_visible(False)
        self.ax_ppi.set_visible(True)

        self.ax_ppi.clear()
        self.ax_ppi.set_theta_zero_location("N")
        self.ax_ppi.set_theta_direction(-1)

        bg_color, fg_color = self.get_plot_colors()

        # ==========================================
        # UPGRADED: Phosphor Decay Sweep Rendering
        # ==========================================
        sweep_width_deg = 60.0
        scan_rad = np.deg2rad(self.scan_angle)

        # Draw the bright leading edge of the radar beam
        self.ax_ppi.plot([scan_rad, scan_rad], [0, 100], color='#00ff00', linewidth=2.5, alpha=0.9)

        # Draw the decaying phosphor tail using a vectorized Bar plot for max performance
        num_segments = 40  # Break the 60 degrees into 40 tiny slices
        angles_deg = np.linspace(0, sweep_width_deg, num_segments, endpoint=False)
        width_rad = np.deg2rad(sweep_width_deg / num_segments)

        # Calculate theta for each segment, shifting backward from the leading edge
        theta_array = np.deg2rad(self.scan_angle - angles_deg) - width_rad
        radii = np.full(num_segments, 100)

        # Non-linear (quadratic) decay for a realistic glowing fade-out effect
        alphas = 0.25 * ((1.0 - (angles_deg / sweep_width_deg)) ** 2)

        # Build RGBA Color array for vectorized plotting
        colors = np.zeros((num_segments, 4))
        colors[:, 1] = 1.0  # Pure Green
        colors[:, 3] = alphas  # Apply fading alpha array

        self.ax_ppi.bar(theta_array, radii, width=width_rad, bottom=0, color=colors, edgecolor='none', align='edge')

        # ==========================================
        # Render Targets with Visual Persistence
        # ==========================================
        for tid, mem in self.target_memory.items():
            tgt = mem['data']
            life = mem['life']

            tgt_ang_norm = (tgt['ang'] + 360) % 360
            angle_diff = (self.scan_angle - tgt_ang_norm + 360) % 360

            # Targets currently inside the beam glow brightly; targets outside decay gradually
            if angle_diff <= sweep_width_deg:
                alpha_val = max(0.5, 1.0 - (angle_diff / sweep_width_deg))
            else:
                alpha_val = max(0.0, life * 0.4)

            angle_rad = np.deg2rad(tgt['ang'])
            color_hex = self.target_colors[tid % len(self.target_colors)]
            self.ax_ppi.scatter(angle_rad, tgt['r'], s=120, marker='o', color=color_hex, alpha=alpha_val)

        self.ax_ppi.set_title("Spatial View (PPI) - True 360° Radar Visualization", fontsize=12, pad=20)
        self.ax_ppi.set_ylim(0, 100)
        self.ax_ppi.grid(color='orange', linestyle='--', alpha=0.6)

        self.apply_axes_theme(self.ax_ppi, bg_color, fg_color)
        self.fig.tight_layout()
        self.canvas.draw_idle()