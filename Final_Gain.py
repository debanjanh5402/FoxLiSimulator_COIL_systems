# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------
# Name:         LASER_Cavity_Simulation_using_FOXLI.py
# Purpose:      A GUI for simulating optical cavity modes using the Fox-Li method,
#               with the capability to import and apply an experimental gain profile.
# Author:       Debanjan Halder
# Created:      18-Aug-2025
# Modified:     28-Aug-2025
# -----------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------

import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QMessageBox, QTabWidget, QFileDialog, QInputDialog, QFormLayout, QComboBox)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# MAIN APPLICATION CLASS DEFINITION
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Main class for the Fox-Li GUI, inheriting from QWidget.
# QWidget is the base class for all user interface objects in PyQt5, providing the basic window.
class FoxLiGUI(QWidget):
    def __init__(self):
        # The constructor for the main window.
        super().__init__() # Call the constructor of the parent class (QWidget).
        self.setWindowTitle("Fox-Li Cavity Simulation with Gain Profile")
        self.showFullScreen() # Setting title & maximizing the application window on startup.
        self.simulation_running = False # A boolean flag to control the simulation loop.
        self.gain_filepath = None # Variable to store the path to the gain data file.
        self.initUI() # Call the master method to construct the entire user interface.


    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # GUI SETUP AND INITIALIZATION
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def initUI(self):
        """
        This is the master method that builds the entire GUI by calling other specialized
        initializer methods. It sets up the main layout, creates the tabs, and initializes
        the timers that drive the animations.
        """
        # The main layout is a horizontal box (QHBoxLayout), which arranges widgets side-by-side.
        # It splits the screen into a left control panel and a right display area.
        main_layout = QHBoxLayout()
        # The right display area is a QTabWidget, allowing for multiple pages (views).
        self.tabs = QTabWidget()

        #################################
        # Tab Widget Creation
        #################################
        # Create the individual pages (tabs) for the QTabWidget. Each is a plain QWidget
        # that will later have its own layout and widgets.
        self.tab_visual = QWidget(); self.tab_simulation = QWidget()
        self.tab_results = QWidget(); self.tab_far_field = QWidget()
        # self.tab_diffraction = QWidget() # Diffraction tab is removed as per request.

        # Add the created pages as tabs to the tab widget, giving each a descriptive title.
        self.tabs.addTab(self.tab_visual, "Setup Visualisation"); self.tabs.addTab(self.tab_simulation, "Cavity Simulation")
        self.tabs.addTab(self.tab_results, "Simulation Results"); self.tabs.addTab(self.tab_far_field, "Far-Field analysis")
        # self.tabs.addTab(self.tab_diffraction, "Diffraction") # Diffraction tab is removed as per request.

        #################################
        # Component Initializers
        #################################
        # Call the dedicated initializer methods for each major component of the UI.
        self.init_param_panel()      # Builds the left-side panel for all user inputs.
        self.init_visual_tab(); self.init_simulation_tab(); self.init_results_tab()
        self.init_far_field_tab()
        # self.init_diffraction_tab() # Diffraction tab is removed as per request.

        #################################
        # Main Layout Assembly
        #################################
        # Add the control panel layout and the tab widget to the main horizontal layout.
        # The numbers '2' and '8' are stretch factors. They allocate horizontal space proportionally,
        # giving 20% to the control panel and 80% to the tabs, making the display area larger.
        main_layout.addLayout(self.control_layout, 2); main_layout.addWidget(self.tabs, 8)
        self.setLayout(main_layout) # Set this as the main layout for the FoxLiGUI window.

        #################################
        # Animation Timers
        #################################
        """
        QTimers are crucial for creating responsive GUIs with animations. Instead of a blocking
        `for` or `while` loop that would freeze the application, a QTimer emits a `timeout` signal
        at regular intervals. We connect this signal to a method that performs one step of the
        calculation/animation. This allows the GUI's main event loop to remain active, processing
        user inputs (like button clicks) between timer events.
        """
        # Timer for the main Fox-Li cavity simulation.
        self.simulation_timer = QTimer(); self.simulation_timer.timeout.connect(self.run_iteration)

    def init_param_panel(self):
        # This method constructs a vertical layout at top-center to stack parameter entries.
        self.control_layout = QVBoxLayout(); self.control_layout.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        self.control_layout.addSpacing(20)

        # A dictionary to hold references to all QLineEdit widgets.
        # This provides a convenient way to access user input later using string keys (e.g., self.inputs['N']).
        self.inputs = {}

        #################################
        # Simulation Parameters Section
        #################################
        simulation_label = QLabel("Simulation Parameters")
        font = simulation_label.font(); font.setPointSize(15); font.setBold(True) # Set a larger bold font size for the heading.
        simulation_label.setFont(font); simulation_label.setAlignment(Qt.AlignCenter) # Center the label text.
        self.control_layout.addWidget(simulation_label); self.control_layout.addSpacing(10)

        # Define the labels, default values, and dictionary keys for the general parameters.
        labels = ["Grid size (N)", "Wavelength (μm)", "Pixel size (μm)", "Propagation distance z (m)",
                  "R1 (m)", "R2 (m)", "D1 (mm)", "D2 (mm)", "Max Iter"]
        defaults = [1501, 1.315, 20.0, 2.5, 12.0, 7.0, 20.0, 11.76, 500]
        keys = ['N', 'wav', 'p', 'z', 'R1', 'R2', 'D1', 'D2', 'max_iter']

        # A QFormLayout is ideal for creating a two-column form of labels and input fields.
        general_layout = QFormLayout(); general_layout.setLabelAlignment(Qt.AlignLeft); general_layout.setFormAlignment(Qt.AlignLeft)

        # Loop through the definitions to programmatically create each row in the form.
        for label, default, key in zip(labels, defaults, keys):
            le = QLineEdit(str(default)) # Create the input field with the default value.
            self.inputs[key] = le        # Store the widget in our dictionary for later access.
            general_layout.addRow(QLabel(label), le) # Add the label and input field as a new row.

        self.control_layout.addLayout(general_layout); self.control_layout.addSpacing(15)

        #################################
        # Misalignment Parameters Section
        #################################
        misalign_label = QLabel("Misalignment Parameters")
        font = misalign_label.font(); font.setPointSize(15); font.setBold(True)
        misalign_label.setFont(font); misalign_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(misalign_label); self.control_layout.addSpacing(10)

        # A QGridLayout is used for more complex, grid-based arrangements like this one.
        mirror_layout = QGridLayout(); mirror_layout.setAlignment(Qt.AlignCenter)

        # Define labels, keys, and defaults for misalignment parameters.
        mis_labels = ["x (μm)", "y (μm)", "θx (μ rad)", "θy (μ rad)"]
        mis_keys_m1 = ['x1', 'y1', 'theta_x1', 'theta_y1'] # Keys for Mirror 1
        mis_keys_m2 = ['x2', 'y2', 'theta_x2', 'theta_y2'] # Keys for Mirror 2
        mis_defaults = [0.0, 0.0, 0.0, 0.0]

        # Create the grid for Mirror 1 inputs.
        mirror_layout.addWidget(QLabel("<b>Mirror 1</b>"), 0, 0, 1, 2, Qt.AlignCenter) # Header spans 2 columns.
        for i, (label, key, default) in enumerate(zip(mis_labels, mis_keys_m1, mis_defaults), start=1):
            mirror_layout.addWidget(QLabel(label), i, 0) # Label in column 0.
            le = QLineEdit(str(default))
            self.inputs[key] = le
            mirror_layout.addWidget(le, i, 1) # Input field in column 1.

        # Create the grid for Mirror 2 inputs.
        mirror_layout.addWidget(QLabel("<b>Mirror 2</b>"), 0, 2, 1, 2, Qt.AlignCenter) # Header spans 2 columns.
        for i, (label, key, default) in enumerate(zip(mis_labels, mis_keys_m2, mis_defaults), start=1):
            mirror_layout.addWidget(QLabel(label), i, 2) # Label in column 2.
            le = QLineEdit(str(default))
            self.inputs[key] = le
            mirror_layout.addWidget(le, i, 3) # Input field in column 3.

        self.control_layout.addLayout(mirror_layout); self.control_layout.addSpacing(15)

        #################################
        # Gain Profile Section
        #################################
        gain_label = QLabel("Gain Profile"); gain_label.setFont(font); gain_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(gain_label); self.control_layout.addSpacing(10)
        
        gain_layout = QFormLayout()
        self.gain_combo = QComboBox(); self.gain_combo.addItems(["No Gain", "Load from File..."])
        gain_layout.addRow(QLabel("Gain Mode:"), self.gain_combo)
        
        self.btn_browse_gain = QPushButton("Browse..."); self.btn_browse_gain.clicked.connect(self.select_gain_file)
        self.gain_filepath_label = QLabel("No file selected."); self.gain_filepath_label.setWordWrap(True)
        gain_layout.addRow(self.btn_browse_gain, self.gain_filepath_label)
        self.control_layout.addLayout(gain_layout); self.control_layout.addSpacing(20)

        #################################
        # Control Buttons Section
        #################################
        # Create the main action buttons.
        # The `.clicked.connect(...)` line is the event handling mechanism in PyQt. It connects the
        # 'clicked' signal of the button to a specific method (a "slot") that will be executed.
        self.btn_visualize = QPushButton("Visualize Setup"); self.btn_visualize.clicked.connect(self.visualize_setup)
        self.control_layout.addWidget(self.btn_visualize, alignment=Qt.AlignTop)

        self.btn_run = QPushButton("Run Simulation"); self.btn_run.clicked.connect(self.initialize_simulation)
        self.control_layout.addWidget(self.btn_run, alignment=Qt.AlignTop)

        self.btn_save = QPushButton("Save Results"); self.btn_save.clicked.connect(self.save_results)
        self.control_layout.addWidget(self.btn_save, alignment=Qt.AlignTop)

    def init_visual_tab(self):
        # Sets up the "Visualize Setup" tab. It contains a single Matplotlib canvas.
        layout = QVBoxLayout()
        self.visual_figure = Figure(figsize=(16, 12)) # Create a Matplotlib figure object.
        self.visual_canvas = FigureCanvas(self.visual_figure) # Create a special Qt widget to display the figure.
        layout.addWidget(self.visual_canvas) # Add the canvas to the layout.
        self.tab_visual.setLayout(layout) # Set the layout for this tab.

    def init_simulation_tab(self):
        # Sets up the "Simulation" tab, which is structurally identical to the visualize tab.
        layout = QVBoxLayout()
        self.sim_figure = Figure(figsize=(16, 12))
        self.sim_canvas = FigureCanvas(self.sim_figure)
        layout.addWidget(self.sim_canvas)
        self.tab_simulation.setLayout(layout)

    def init_results_tab(self):
        # Sets up the "View Results" tab.
        layout = QVBoxLayout()
        self.result_figure = Figure(figsize=(16, 16))
        self.result_canvas = FigureCanvas(self.result_figure)
        layout.addWidget(self.result_canvas)
        self.tab_results.setLayout(layout)

    def init_far_field_tab(self):
        # Sets up the "Far-Field Analysis" tab, which has controls and a plot.
        main_layout = QVBoxLayout(); top_layout = QHBoxLayout() # A horizontal layout for the button and result label.
        self.btn_calculate_ff = QPushButton("Calculate Far-Field & M²"); self.btn_calculate_ff.clicked.connect(self.calculate_far_field)
        top_layout.addWidget(self.btn_calculate_ff, 1)
        
        # Create a form layout for the results to keep them aligned.
        results_layout = QFormLayout()
        self.m2_label = QLabel("Not calculated"); self.dr_label = QLabel("Not calculated")
        self.dr_gauss_label = QLabel("Not calculated"); self.drho_label = QLabel("Not calculated")
        self.drho_gauss_label = QLabel("Not calculated")
        
        results_layout.addRow("<b>M² Factor:</b>", self.m2_label)
        results_layout.addRow("<b>Dr (simulated):</b>", self.dr_label)
        results_layout.addRow("<b>D_rho (simulated):</b>", self.drho_label)
        results_layout.addRow("<b>Dr_gauss (Gaussian):</b>", self.dr_gauss_label)
        results_layout.addRow("<b>D_rho_gauss (Gaussian):</b>", self.drho_gauss_label)

        top_layout.addLayout(results_layout, 2)
        main_layout.addLayout(top_layout)
        
        self.far_field_figure = Figure(figsize=(16, 12))
        self.far_field_canvas = FigureCanvas(self.far_field_figure)
        main_layout.addWidget(self.far_field_canvas)
        self.tab_far_field.setLayout(main_layout)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # EVENT HANDLERS & LOGIC
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def select_gain_file(self):
        # This method is called when the "Browse..." button for the gain profile is clicked.
        # It opens a standard file dialog to let the user select a text file.
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Gain Profile Data File", "", 
                                                  "Text Files (*.txt);;All Files (*)", options=options)
        if filepath:
            # If a file was successfully selected, store its path and update the label.
            self.gain_filepath = filepath
            self.gain_filepath_label.setText(os.path.basename(filepath))
            # Automatically switch the dropdown to "Load from File..."
            self.gain_combo.setCurrentIndex(1)

    def visualize_setup(self):
        """
        This method is an event handler for the "Visualize Setup" button. It plots the phase
        profiles of the cavity mirrors and, if provided, the gain profile. This is a sanity check
        to ensure all input parameters are creating the expected physical setup before running
        a time-consuming simulation.
        """
        try:
            self.get_inputs() # First, update all simulation parameters from the GUI.
        except Exception as e:
            QMessageBox.critical(self, "Input Error", f"Error reading parameters: {e}")
            return

        self.visual_figure.clf() # Clear any previous plots from the figure.
        
        # Use GridSpec for a more complex layout: 4 plots on top, 2 on the bottom.
        gs = GridSpec(2, 4, figure=self.visual_figure)
        ax_m1_nom = self.visual_figure.add_subplot(gs[0, 0])
        ax_m2_nom = self.visual_figure.add_subplot(gs[0, 1])
        ax_m1_mis = self.visual_figure.add_subplot(gs[0, 2])
        ax_m2_mis = self.visual_figure.add_subplot(gs[0, 3])
        ax_gain_raw = self.visual_figure.add_subplot(gs[1, 0:2])
        ax_gain_interp = self.visual_figure.add_subplot(gs[1, 2:4])
        
        self.visual_figure.subplots_adjust(wspace=0.4, hspace=0.5)

        #################################
        # Top Row: Mirror Phase Profiles
        #################################
        """
        The phase profile of a spherical mirror in the paraxial approximation is quadratic.
        The complex reflectance is given by R(x, y) = exp(-i * k * (x^2 + y^2) / R_c), where
        R_c is the radius of curvature. We plot the phase (the argument of this complex number)
        to visualize the curvature. A tilted mirror adds a linear phase ramp, and a shifted
        mirror displaces the center of the quadratic profile.
        """
        # Nominal (perfectly aligned) mirrors
        quad1_nominal = np.exp(-1j*self.k*((self.x)**2 + (self.y)**2)/self.R1)
        Mirror1_nominal = quad1_nominal * self.circ1
        quad2_nominal = np.exp(+1j*self.k*((self.x)**2 + (self.y)**2)/self.R2)
        Mirror2_nominal = quad2_nominal * self.circ2

        # Helper function for plotting phase to avoid code repetition and ensure consistency.
        def plot_phase(ax, data, title, circ_mask):
            ax.set_title(title, fontsize=8, fontweight='bold')
            # The phase is the argument of the complex field.
            phase_data = np.angle(data) * circ_mask
            im = ax.imshow(phase_data, cmap='jet')
            # Use make_axes_locatable for robust colorbar placement.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            self.visual_figure.colorbar(im, cax=cax)
            ax.axis('off') # Turn off axis ticks and labels for a cleaner look.

        # Plot all four mirror phase profiles using the helper function.
        plot_phase(ax_m1_nom, Mirror1_nominal, "M1 Phase-Nominal", self.circ1)
        plot_phase(ax_m2_nom, Mirror2_nominal, "M2 Phase-Nominal", self.circ2)
        plot_phase(ax_m1_mis, self.Mirror1, "M1 Phase-Misaligned", self.circ1)
        plot_phase(ax_m2_mis, self.Mirror2, "M2 Phase-Misaligned", self.circ2)

        #################################
        # Bottom Row: Gain Profiles
        #################################
        if hasattr(self, 'raw_gain_profile') and self.raw_gain_profile is not None:
            im4 = ax_gain_raw.imshow(self.raw_gain_profile/np.max(self.raw_gain_profile), cmap='viridis', origin='lower', aspect='auto')
            ax_gain_raw.set_title("Raw 2D Projected Gain", fontsize=8, fontweight='bold')
            self.visual_figure.colorbar(im4, ax=ax_gain_raw)
        else:
            ax_gain_raw.text(0.5, 0.5, "No Gain Data Loaded", ha='center', va='center')
            ax_gain_raw.set_title("Raw 2D Projected Gain", fontsize=8, fontweight='bold')

        if hasattr(self, 'gain_profile') and self.gain_profile is not None and self.gain_combo.currentText() == "Load from File...":
            # We plot log(gain) because the gain itself is exponential and hard to visualize.
            # Add a small epsilon to avoid log(0).
            exp_gain = np.log(self.gain_profile)
            im5 = ax_gain_interp.imshow(exp_gain/np.max(exp_gain), cmap='hot', origin='lower', aspect='auto')
            ax_gain_interp.set_title(f"Interpolated Exponential Gain", fontsize=8, fontweight='bold')
            self.visual_figure.colorbar(im5, ax=ax_gain_interp)
        else:
            ax_gain_interp.text(0.5, 0.5, "No Gain Applied", ha='center', va='center')
            ax_gain_interp.set_title(f"Interpolated Gain", fontsize=8, fontweight='bold')

        self.visual_canvas.draw() # Redraw the canvas to display the new plots.
        self.tabs.setCurrentWidget(self.tab_visual) # Automatically switch to this tab.

    def get_inputs(self):
        """
        This function is a centralized place to read all user inputs from the QLineEdit widgets.
        It converts the string inputs to appropriate numerical types (int, float), performs necessary
        unit conversions (e.g., μm to m), and calculates derived physical parameters. It also
        handles the loading and processing of the gain profile.
        """
        #################################
        # Read and Convert Inputs
        #################################
        self.N = int(self.inputs['N'].text())
        self.wav = float(self.inputs['wav'].text()) * 1e-6  # Convert wavelength from μm to m
        self.p = float(self.inputs['p'].text()) * 1e-6      # Convert pixel size from μm to m
        self.z = float(self.inputs['z'].text())
        self.R1 = float(self.inputs['R1'].text())
        self.R2 = float(self.inputs['R2'].text())
        self.D1 = float(self.inputs['D1'].text()) * 1e-3    # Convert mirror diameter from mm to m
        self.D2 = float(self.inputs['D2'].text()) * 1e-3
        self.max_iter = int(self.inputs['max_iter'].text())
        self.k = 2 * np.pi / self.wav # Optical wavenumber (k)

        # Read misalignment parameters and convert them to meters (for shifts) and radians (for angles).
        self.x1 = float(self.inputs['x1'].text()) * 1e-6 # Convert shift from μm to m
        self.y1 = float(self.inputs['y1'].text()) * 1e-6
        self.x2 = float(self.inputs['x2'].text()) * 1e-6
        self.y2 = float(self.inputs['y2'].text()) * 1e-6

        self.theta_x1 = float(self.inputs['theta_x1'].text()) * 1e-6
        self.theta_y1 = float(self.inputs['theta_y1'].text()) * 1e-6
        self.theta_x2 = float(self.inputs['theta_x2'].text()) * 1e-6
        self.theta_y2 = float(self.inputs['theta_y2'].text()) * 1e-6

        #################################
        # Create Computational Grids
        #################################
        # `x0` is a 1D array of pixel indices centered at zero.
        x0 = np.linspace(-self.N/2+0.5, self.N/2-0.5, self.N, endpoint=True)
        # `meshgrid` creates 2D arrays (matrices) of x and y coordinates from the 1D array.
        x_coords, y_coords = np.meshgrid(x0, x0)
        # Scale by pixel size to get spatial coordinates in meters.
        self.x = x_coords * self.p
        self.y = y_coords * self.p

        # Create frequency coordinate grids (fx, fy) corresponding to the spatial grid.
        fx0 = np.linspace(-0.5, 0.5, self.N, endpoint=True)
        fx_coords, fy_coords = np.meshgrid(fx0, fx0)
        self.fx = fx_coords / self.p # Spatial frequency in cycles per meter.
        self.fy = fy_coords / self.p
        
        #################################
        # Create Aperture Masks
        #################################
        # Create binary masks (2D arrays of 1s and 0s) that represent the finite apertures of the mirrors.
        self.circ0 = np.zeros((self.N, self.N)); self.circ0[self.x**2 + self.y**2 < (self.N*self.p/2)**2] = 1
        self.circ1 = np.zeros((self.N, self.N)); self.circ1[self.x**2 + self.y**2 < (self.D1/2)**2] = 1
        self.circ2 = np.zeros((self.N, self.N)); self.circ2[self.x**2 + self.y**2 < (self.D2/2)**2] = 1
        
        #################################
        # Define Mirror Properties
        #################################
        """
        The complex reflectance of each mirror is constructed by multiplying three components:
        1. `quad`: The quadratic phase factor of the spherical surface, centered at (x_shift, y_shift).
        2. `tilt`: A linear phase ramp representing the tilt (theta_x, theta_y).
        3. `circ`: The binary aperture mask representing the mirror's finite size.
        """
        # Mirror 1
        quad1 = np.exp(-1j*self.k*((self.x - self.x1)**2 + (self.y - self.y1)**2)/self.R1)
        tilt1 = np.exp(1j*self.k*(self.x*self.theta_x1 + self.y*self.theta_y1))
        self.Mirror1 = quad1 * tilt1 * self.circ1
        # Mirror 2
        quad2 = np.exp(+1j*self.k*((self.x - self.x2)**2 + (self.y - self.y2)**2)/self.R2)
        tilt2 = np.exp(1j*self.k*(self.x*self.theta_x2 + self.y*self.theta_y2))
        self.Mirror2 = quad2 * tilt2 * self.circ2
        
        #################################
        # Process Gain Profile
        #################################
        if self.gain_combo.currentText() == "Load from File..." and self.gain_filepath:
            self.process_gain_file(self.gain_filepath)
        else:
            # If "No Gain" is selected or no file is provided, create a uniform gain of 1 (no effect).
            self.gain_profile = np.ones((self.N, self.N))
            self.raw_gain_profile = None # No raw profile to display.

        #################################
        # Initialize Electric Field
        #################################
        """
        The Fox-Li method starts with an arbitrary electric field. A field of random amplitude and
        phase is a good choice because it contains components of all possible spatial modes that the
        cavity could support. During the iterative process, modes with high diffraction loss will be
        filtered out, and after many round trips, only the single lowest-loss resonant mode will
        survive and converge.
        """
        self.E0 = np.random.rand(self.N, self.N) * np.exp(1j * 2 * np.pi * np.random.rand(self.N, self.N))
        self.iter = 0 # Reset the iteration counter.

    def initialize_simulation(self):
        # This is the event handler for the "Run Simulation" button.
        try:
            self.get_inputs() # Load all parameters from the GUI.
            self.simulation_running = True # Set the flag to allow the iteration loop to run.
            self.tabs.setCurrentWidget(self.tab_simulation) # Switch to the simulation tab.
            self.simulation_timer.start(5) # Start the timer to call `run_iteration` every 5 ms.
        except Exception as e:
            # Display a user-friendly error message if something goes wrong (e.g., invalid text input).
            QMessageBox.critical(self, "Initialization Error", str(e))

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # PHYSICS & COMPUTATION CORE
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def process_gain_file(self, filepath):
        """
        This function handles the entire backend processing of the gain data file. It reads the
        raw 3D data, projects it onto a 2D plane, symmetrizes it, and finally interpolates it
        onto the simulation's computational grid.
        """
        # --- Configuration Parameters from original script ---
        Y_SYMMETRY_OFFSET = 0.102
        X_CENTER_OFFSET = 0.1205
        Y_CENTER_OFFSET = 0.102805
        GAIN_SCALING_FACTOR = 6.0 # (30 * 0.2)

        # --- Step 1: Load and Project to 2D ---
        data = pd.read_csv(filepath, sep="\s+")
        if not {'x', 'y', 'z', 'gain'}.issubset(data.columns):
            raise ValueError("Gain file must contain 'x', 'y', 'z', and 'gain' columns.")
        
        # IMPORTANT: Match the column mapping from the standalone script
        cumulative_gain_df = data.groupby(['x', 'z'])['gain'].sum().reset_index()
        
        gain_profile_df = cumulative_gain_df.pivot(index='z', columns='x', values='gain')
        
        x_unique = gain_profile_df.columns.values
        y_unique = gain_profile_df.index.values
        
        gain_profile_2d = np.nan_to_num(gain_profile_df.values, nan=0.0)

        # --- Step 2: Symmetrize the Profile ---
        mirrored_gain = np.flipud(gain_profile_2d)
        symmetrical_gain = np.concatenate((gain_profile_2d, mirrored_gain), axis=0)
        self.raw_gain_profile = symmetrical_gain # Store for visualization.
        
        # IMPORTANT: Match the y-coordinate creation from the standalone script
        mirrored_y = y_unique + Y_SYMMETRY_OFFSET
        symmetrical_y_unique = np.concatenate((y_unique, mirrored_y), axis=0)
        X_source, Y_source = np.meshgrid(x_unique, symmetrical_y_unique)

        # --- Step 3: Interpolate to Simulation Grid ---
        """
        This is the most critical step. We have sparse gain data at coordinates (X_source, Y_source)
        and we need to estimate the gain values at the coordinates of our simulation grid (self.x, self.y).
        `scipy.griddata` is the perfect tool for this, acting like a sophisticated resampling algorithm.
        """
        source_points = np.vstack((X_source.ravel() - X_CENTER_OFFSET, Y_source.ravel() - Y_CENTER_OFFSET)).T
        source_values = symmetrical_gain.ravel() * GAIN_SCALING_FACTOR
        
        # --- CORRECTION STARTS HERE ---
        # To exactly match the standalone script, we must interpolate onto a fixed grid
        # from -0.15 to 0.15, using the simulation's grid size (N).
        grid_size = self.N
        x_grid_target = np.linspace(-0.15, 0.15, grid_size)
        y_grid_target = np.linspace(-0.15, 0.15, grid_size)
        X_target, Y_target = np.meshgrid(x_grid_target, y_grid_target)
        
        target_points = (X_target, Y_target)
        # --- CORRECTION ENDS HERE ---
        
        gain_grid_fine = griddata(
            points=source_points,
            values=source_values,
            xi=target_points,
            method='linear',
            fill_value=0
        )

        # --- Step 4: Apply Physical Transformation ---
        """
        The cumulative gain often relates exponentially to the amplification of a field
        (e.g., in a laser amplifier, Intensity_out = Intensity_in * exp(gain)).
        This step calculates the final amplification factor per round trip.
        """
        self.gain_profile = np.exp(gain_grid_fine)

    def angspec_prop(self, u, dz):
        """
        Implements beam propagation using the Angular Spectrum method. This method is a direct
        solution to the Helmholtz equation and is highly accurate for near-field diffraction.
        
        Physics Concept:
        Any complex field `u(x, y)` can be decomposed into a sum (or integral) of plane waves,
        each with a different propagation direction. This is its "angular spectrum," which is
        found by taking the 2D Fourier Transform of `u(x, y)`.
        
        Propagation in free space over a distance `dz` is simple in this domain: each plane wave
        component is just multiplied by a phase factor `exp(i * kz * dz)`, where `kz` is the
        z-component of its wavevector. The final propagated field is found by taking the
        inverse Fourier Transform of the modified spectrum.
        """

        # The propagation phase depends on kz, which is calculated as:
        # kz = sqrt(k² - kx² - ky²) = sqrt(k² - (2πfx)² - (2πfy)²).
        # The term inside the sqrt is calculated here. We use np.sqrt on a complex argument,
        # which correctly handles evanescent waves (where the argument is negative).
        alpha = np.sqrt(self.k**2 - 4 * np.pi**2 * (self.fx**2 + self.fy**2))
        
        """
        A low-pass filter is applied in the frequency domain. This is crucial to prevent a numerical
        artifact called aliasing. Aliasing occurs when spatial frequencies are too high to be
        resolved by the discrete grid spacing (`p`). The filter removes these high-frequency components
        (which correspond to evanescent waves or waves at very steep angles) before the inverse FFT.
        The cutoff frequency `f0` depends on the propagation distance `dz`.
        """
        f0 = (1/self.wav) * 1/np.sqrt(1 + (2*dz/(self.N*self.p))**2)
        LP = np.zeros((self.N, self.N)); LP[self.fx**2 + self.fy**2 <= f0**2] = 1
        # The complete transfer function combines the propagation phase and the low-pass filter.
        H = np.exp(1j * dz * alpha) * LP
        
        """
        The `ifftshift`/`fftshift` functions are necessary because `numpy.fft` expects the
        zero-frequency component to be at the start of the array (index 0,0). For physics and
        visualization, it's more intuitive to have it in the center. These functions shift the
        data between these two conventions.
        """
        # Step 1: Go to frequency domain.
        U = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u)))
        # Step 2 & 3: Apply transfer function and return to spatial domain.
        u_out = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U * H)))
        return u_out

    def run_iteration(self):
        """
        This is the core of the Fox-Li simulation, executed by the `simulation_timer`.
        It models one complete round trip of the light field inside the optical cavity.
        The cavity acts as a filter; the iterative process converges to the cavity's
        lowest-loss eigenmode, which is the stable laser beam profile.
        """
        if not self.simulation_running: return # Stop if the flag is false.
        
        # Normalize the field at the start of each iteration to prevent numerical overflow/underflow.
        self.E0 /= np.max(np.abs(self.E0[:]))
        
        #################################
        # The Cavity Round Trip
        #################################
        # 1. The field `E0` reflects off Mirror 1.
        E1 = self.E0 * self.Mirror1
        # 2. The field propagates the cavity length `z` to Mirror 2.
        E1_prop = self.angspec_prop(E1, self.z)
        # 3. The field reflects off Mirror 2.
        E2 = E1_prop * self.Mirror2
        # 4. The field propagates back to Mirror 1.
        E2_prop = self.angspec_prop(E2, self.z)
        
        # 5. The field is amplified by the gain medium. This is the key modification.
        # The field that completes the round trip becomes the input field for the next iteration.
        self.E0 = E2_prop * self.gain_profile
        
        # The output of the laser is the portion of the field transmitted through Mirror 2.
        # This is modeled as the field incident on Mirror 2, masked by an inverted aperture.
        E_out = E1_prop * (1 - self.circ2)
        
        #################################
        # Live Plotting
        #################################
        intensity = np.abs(E_out)**2 * self.circ0 * (1 - self.circ2); phase = np.angle(E_out) * self.circ0 * (1 - self.circ2)

        self.sim_figure.clf()
        axs = self.sim_figure.subplots(1, 2)
        self.sim_figure.subplots_adjust(wspace=0.5)

        divider0 = make_axes_locatable(axs[0]); cax0 = divider0.append_axes("right", size="2%", pad=0.05)
        # Normalize intensity for visualization purposes only
        intensity_norm = intensity / (np.max(intensity) + 1e-16)
        im0 = axs[0].imshow(intensity_norm, cmap='viridis'); axs[0].set_title("Intensity", fontsize=12, fontweight='bold')
        self.sim_figure.colorbar(im0, cax=cax0)

        divider1 = make_axes_locatable(axs[1]); cax1 = divider1.append_axes("right", size="2%", pad=0.05)
        im1 = axs[1].imshow(phase, cmap='jet'); axs[1].set_title("Phase", fontsize=12, fontweight='bold')
        self.sim_figure.colorbar(im1, cax=cax1)

        self.sim_figure.suptitle(f"Iteration {self.iter+1}", fontsize=20, fontweight='bold')
        self.sim_canvas.draw()
        
        self.iter += 1
        
        
        # Check for the stopping condition.
        if self.iter >= self.max_iter:
            self.last_E_out = E_out # Store the most recent output field for final analysis.
            self.simulation_running = False
            self.simulation_timer.stop()
            QMessageBox.information(self, "Simulation Complete", f"{self.max_iter} iterations completed!")
            self.plot_results() # Plot the final, converged results.

    def plot_results(self):
        # This method generates the summary plots on the "View Results" tab after the simulation is complete.
        intensity = np.abs(self.last_E_out)**2 * self.circ0 * (1 - self.circ2); 
        phase = np.angle(self.last_E_out) * self.circ0 * (1 - self.circ2)
        self.last_intensity = intensity; self.last_phase = phase; self.last_center_row = self.N // 2

        self.result_figure.clf()
        axs = self.result_figure.subplots(2, 2)
        self.result_figure.subplots_adjust(wspace=0.5, hspace=0.6)

        divider00 = make_axes_locatable(axs[0, 0]); cax00 = divider00.append_axes("right", size="2%", pad=0.05)
        im00 = axs[0, 0].imshow(intensity / np.max(intensity), cmap='viridis', interpolation='nearest')
        axs[0, 0].set_title("Final Intensity", fontsize=12, fontweight='bold')
        self.result_figure.colorbar(im00, cax=cax00)

        divider01 = make_axes_locatable(axs[0, 1]); cax01 = divider01.append_axes("right", size="2%", pad=0.05)
        im01 = axs[0, 1].imshow(phase, cmap='jet', vmin=-np.pi, vmax=np.pi, interpolation='nearest')
        axs[0, 1].set_title("Final Phase", fontsize=12, fontweight='bold')
        self.result_figure.colorbar(im01, cax=cax01)

        cr = self.last_center_row
        axs[1, 0].plot(intensity[cr, :]/np.max(intensity), linewidth=1)
        axs[1, 0].plot(intensity[:, cr]/np.max(intensity), linewidth=1)
        axs[1, 0].set_title("Central Intensities", fontsize=12, fontweight='bold')
        axs[1, 0].grid(True, linestyle='--', alpha=0.25)

        axs[1, 1].plot(phase[cr, :], linewidth=1)
        axs[1, 1].plot(phase[:, cr], linewidth=1)
        axs[1, 1].set_title("Central Phases", fontsize=12, fontweight='bold')
        axs[1, 1].grid(True, linestyle='--', alpha=0.25)

        self.result_canvas.draw()
        self.tabs.setCurrentWidget(self.tab_results)

    def calculate_far_field(self):
        """
        This method handles the "Far-Field Analysis" tab. It calculates the far-field
        diffraction pattern and the M-squared beam quality factor.
        """
        if not hasattr(self, 'last_E_out'):
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return
        
        #################################
        # Far-Field Calculation
        #################################
        """
        Physics Concept: The Fraunhofer diffraction pattern, which describes the field distribution
        very far from an aperture (the "far-field"), is mathematically equivalent to the Fourier
        Transform of the field at the aperture (the "near-field").
        """
        E_out = self.last_E_out; I_out = np.abs(E_out)**2
        E_far = np.fft.fftshift(np.fft.fft2(E_out)); I_far = np.abs(E_far)**2
        
        #################################
        # M² Beam Quality Calculation
        #################################
        """
        Physics Concept: The M² factor (or "beam quality factor") quantifies how closely a real
        laser beam resembles an ideal, diffraction-limited Gaussian beam. It is defined by the
        "method of second moments," which measures the variance (σ²) of the beam's intensity
        profile in both the spatial domain (near-field) and the angular domain (far-field).
        An ideal Gaussian beam has the minimum possible space-bandwidth product, and M²=1.
        Higher M² values indicate a less focused, more divergent beam.
        """
        
        # Calculate second moment (variance) of the spatial intensity distribution (Dr).
        total_power_out = np.sum(I_out); x_c = np.sum(self.x * I_out) / total_power_out; y_c = np.sum(self.y * I_out) / total_power_out
        r_c = np.sqrt(x_c**2 + y_c**2); Dr = np.sum(((np.sqrt(self.x**2 + self.y**2) - r_c)**2 * I_out)) / total_power_out

        # Calculate second moment of the angular intensity distribution (Drho) in frequency space.
        total_power_far = np.sum(I_far); fx_c = np.sum(self.fx * I_far) / total_power_far; fy_c = np.sum(self.fy * I_far) / total_power_far
        f_c = np.sqrt(fx_c**2 + fy_c**2); Drho = np.sum(((np.sqrt(self.fx**2 + self.fy**2) - f_c)**2) * I_far) / total_power_far

        # Create a reference Gaussian beam for comparison.
        w0 = np.mean(np.array([self.D1, self.D2])); E_gauss = np.exp(-(self.x**2 + self.y**2) / w0**2) * self.circ1
        I_gauss = np.abs(E_gauss)**2; E_far_gauss = np.fft.fftshift(np.fft.fft2(E_gauss)); I_far_gauss = np.abs(E_far_gauss)**2
        total_power_gauss = np.sum(I_gauss); Dr_gauss = np.sum((self.x**2 + self.y**2) * I_gauss) / total_power_gauss
        total_power_far_gauss = np.sum(I_far_gauss); Drho_gauss = np.sum((self.fx**2 + self.fy**2) * I_far_gauss) / total_power_far_gauss

        # Calculate M-squared factor.
        M2 = np.sqrt(Drho / Drho_gauss)
        
        #################################
        # Plotting Far-Field Results
        #################################
        self.far_field_figure.clf(); axs = self.far_field_figure.subplots(2, 2)
        self.far_field_figure.subplots_adjust(wspace=0.5, hspace=0.5)

        divider00 = make_axes_locatable(axs[0, 0]); cax00 = divider00.append_axes("right", size="2%", pad=0.05)
        im00 = axs[0, 0].imshow(I_out/np.max(I_out), cmap='viridis')
        axs[0, 0].set_title("Simulated Beam Near-Field (Intensity)", fontsize=5, fontweight='bold')
        self.far_field_figure.colorbar(im00, cax=cax00)

        divider01 = make_axes_locatable(axs[0, 1]); cax01 = divider01.append_axes("right", size="2%", pad=0.05)
        im01 = axs[0, 1].imshow(I_gauss/np.max(I_gauss), cmap='viridis')
        axs[0, 1].set_title("Reference Gaussian Near-Field (Intensity)", fontsize=5, fontweight='bold')
        self.far_field_figure.colorbar(im01, cax=cax01)
        
        divider10 = make_axes_locatable(axs[1, 0]); cax10 = divider10.append_axes("right", size="2%", pad=0.05)
        im10 = axs[1, 0].imshow(I_far[self.N//2-50:self.N//2+50, self.N//2-50:self.N//2+50]**(0.5), cmap='jet')
        axs[1, 0].set_title("Simulated Beam Far-Field (amplitude)", fontsize=5, fontweight='bold')
        self.far_field_figure.colorbar(im10, cax=cax10)

        divider11 = make_axes_locatable(axs[1, 1]); cax11 = divider11.append_axes("right", size="2%", pad=0.05)
        im11 = axs[1, 1].imshow(I_far_gauss[self.N//2-50:self.N//2+50, self.N//2-50:self.N//2+50]**(0.5), cmap='jet')
        axs[1, 1].set_title("Reference Gaussian Far-Field (amplitude)", fontsize=5, fontweight='bold')

        self.far_field_figure.colorbar(im11, cax=cax11); self.far_field_canvas.draw()
        
        # Update all the result labels with the calculated values, formatted to 4 decimal places.
        self.m2_label.setText(f"{M2:.4f}")
        self.dr_label.setText(f"{Dr:.4e}")
        self.drho_label.setText(f"{Drho:.4f}")
        self.dr_gauss_label.setText(f"{Dr_gauss:.4e}")
        self.drho_gauss_label.setText(f"{Drho_gauss:.4f}")

        self.tabs.setCurrentWidget(self.tab_far_field)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # UTILITY METHODS
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def save_results(self):
        # This function provides the functionality to save the generated plots as PNG images.
        options = ["Setup Visualisation", "Cavity Simulation", "Simulation Results", "Far-Field analysis", "All"]
        # Use a QInputDialog to get the user's choice from a dropdown list.
        choice, ok = QInputDialog.getItem(self, "Select Figure", "Choose which plot to save:", options, 0, False)
        if ok and choice: # Proceed only if the user clicked "OK".
            if choice == "All":
                # If "All" is selected, open a dialog to choose a directory.
                dir_path = QFileDialog.getExistingDirectory(self, "Select Folder to Save All Figures")
                if dir_path:
                    # Save each figure to the selected directory with a high DPI for quality.
                    self.visual_figure.savefig(os.path.join(dir_path, "01_setup_visualisation.png"), dpi=300)
                    self.sim_figure.savefig(os.path.join(dir_path, "02_cavity_simulation.png"), dpi=300)
                    self.result_figure.savefig(os.path.join(dir_path, "03_simulation_results.png"), dpi=300)
                    self.far_field_figure.savefig(os.path.join(dir_path, "04_far_field_analysis.png"), dpi=300)
                    QMessageBox.information(self, "Saved", f"All figures saved in {dir_path}")
            else:
                # If a single figure is chosen, open a dialog to specify a filename.
                file_path, _ = QFileDialog.getSaveFileName(self, f"Save {choice} Figure", "", "PNG Images (*.png)")
                if file_path:
                    # Use a dictionary to map the string choice to the actual figure object.
                    figure_map = {
                        "Setup Visualisation": self.visual_figure, 
                        "Cavity Simulation": self.sim_figure,
                        "Simulation Results": self.result_figure, 
                        "Far-Field analysis": self.far_field_figure
                    }
                    figure_map[choice].savefig(file_path, dpi=300)

# This is the standard entry point for a Python script.
# The code inside this block will only run when the script is executed directly,
# not when it is imported as a module into another script.
if __name__ == '__main__':
    app = QApplication(sys.argv) # Create the PyQt application object.
    window = FoxLiGUI() # Create an instance of our main GUI class.
    window.show() # Make the window visible.
    sys.exit(app.exec_()) # Start the application's main event loop. This call is blocking.