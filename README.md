# GUI for 

A Python-baased GUI application for simulating unstable resonator modes using the Fox-Li iterative method for COIL (Chemical Oxygen-Iodine LASER) systems. This tool models the physics of beam propagation inside a laser cavity, including mirror misalignments and, most notably, the ability to import and apply an experimentally derived, non-uniform gain profile.



## Table of Contents

- [About The Project](#about-the-project)
- [Core Physics Concepts](#core-physics-concepts)
- [Features](#features)
- [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Code Architecture](#code-architecture)
- [License](#license)
- [Author](#author)

---

## About The Project

This simulator provides a powerful numerical environment to solve for the eigenmodes of an optical cavity. The traditional Fox-Li method involves starting with an arbitrary optical field and propagating it back and forth between the cavity mirrors. After many iterations, the field converges to the lowest-loss mode of the resonator.

This project extends that fundamental concept by:
1.  Implementing a highly accurate **Angular Spectrum propagation method** to solve the Helmholtz equation.
2.  Allowing for detailed **mirror misalignments**, including both transverse shifts (decenter) and angular tilts.
3.  A key feature: the ability to load a **3D experimental gain data**, project it to 2D, and apply it to the optical field during each round trip. This is crucial for accurately modeling high-power lasers where the gain medium significantly influences the final mode shape.
4.  Providing post-simulation analysis, including the calculation of the **M² beam quality factor**.

The entire simulation is wrapped in a user-friendly GUI built with PyQt5, offering real-time visualization of the mode convergence.

---

## Core Physics Concepts

The simulation is built on several key principles of Fourier optics and laser physics.

### 1. The Fox-Li Method
This is an iterative numerical technique to find the resonant modes of an optical cavity. The process is as follows:
- **Initialization**: Start with an arbitrary field, often random noise, on one of the mirrors (`self.E0`).
- **Iteration**: Simulate a complete round trip of the field within the cavity:
    1. Reflection from Mirror 1.
    2. Propagation to Mirror 2.
    3. Reflection from Mirror 2.
    4. Propagation back to Mirror 1.
    5. Application of the gain profile.
- **Convergence**: Repeat the process. With each round trip, higher-order modes with greater diffraction losses are filtered out. The field eventually converges to the fundamental, lowest-loss eigenmode of the specific cavity configuration.

### 2. Angular Spectrum Propagation (`angspec_prop`)
This method solves the Helmholtz wave equation for beam propagation. It's more accurate than the Fresnel approximation, especially for short propagation distances.
- The 2D Fourier Transform of the optical field $u(x, y)$ gives its "angular spectrum" $U(f_x, f_y)$. This decomposes the field into a superposition of plane waves.
- Propagation over a distance $z$ in Fourier space is achieved by multiplying the spectrum by a transfer function, $H(f_x, f_y)$:
$$H(f_x, f_y) = e^{i z k_z} = \exp\left(i z \sqrt{k^2 - (2\pi f_x)^2 - (2\pi f_y)^2}\right)$$
- An inverse Fourier Transform of the modified spectrum yields the propagated field $u(x, y, z)$.

### 3. M² Beam Quality Factor (`calculate_far_field`)
The M² factor quantifies how close a laser beam is to an ideal Gaussian beam. It is calculated using the method of second moments, which measures the beam's variance in both the near-field (spatial domain) and the far-field (angular/frequency domain).
$$M^2 = \frac{\pi}{4\lambda} \cdot (2\sigma_x) \cdot (2\sigma_{f_x})$$
Where $\sigma_x$ is the standard deviation of the beam's spatial intensity profile and $\sigma_{f_x}$ is the standard deviation of its angular intensity profile. In the code, this is calculated as `M2 = np.sqrt(Drho / Drho_gauss)`.

---

## Features

-   ⚡ **Interactive GUI**: A responsive and intuitive interface built with PyQt5.
-   🔬 **Detailed Physics**: Input parameters for wavelength, mirror curvature, cavity length, and aperture size.
-   🔧 **Misalignment Modeling**: Introduce tilts (μrad) and shifts (μm) for both mirrors to study their effects on the mode.
-   🔥 **Custom Gain Profiles**: Load experimental gain data from a `.txt` file. The application processes, symmetrizes, and interpolates this data onto the simulation grid.
-   🎬 **Live Simulation**: Watch the intensity and phase of the optical mode converge in real-time.
-   📊 **Comprehensive Analysis**:
    -   Visualize final intensity and phase in 2D and 1D cross-sections.
    -   Calculate the far-field pattern using FFT.
    -   Compute the M² beam quality factor to assess mode purity.
-   💾 **Save Results**: Export any of the generated plots as high-quality PNG images.

---

## Code Architecture

The code is organized within a single `FoxLiGUI(QWidget)` class, which manages the UI, event handling, and the physics simulation.

-   `__init__()`: Initializes the main window, flags, and calls `initUI()`.
-   **UI Initialization (`init_...`)**:
    -   `initUI()`: The master method that sets up the main layout and calls other initializers.
    -   `init_param_panel()`: Builds the entire left-side control panel for all user inputs.
    -   `init_..._tab()`: Each of these methods sets up a specific tab with its Matplotlib canvas and controls.
-   **Event Handlers & Core Logic**:
    -   `get_inputs()`: A critical function that reads all `QLineEdit` values, converts them to the correct units (e.g., μm -> m), and computes derived parameters like the wavevector `k`, coordinate grids (`x`, `y`), and mirror reflectance profiles.
    -   `process_gain_file()`: Reads the specified data file, projects the 3D gain to a 2D profile, symmetrizes it, and uses `scipy.griddata` to interpolate it onto the simulation grid.
    -   `initialize_simulation()`: Triggered by the "Run" button. It calls `get_inputs()` and starts a `QTimer` that repeatedly calls `run_iteration()`. Using a `QTimer` is essential for keeping the GUI responsive during the simulation.
    -   `run_iteration()`: The heart of the simulation. It performs one full round trip of the beam inside the cavity and updates the live plots.
    -   `calculate_far_field()`: Performs the Fourier transform to get the far-field pattern and computes the M² factor.
-   **Physics & Computation Core**:
    -   `angspec_prop(u, dz)`: Implements the Angular Spectrum method for free-space propagation. It's a self-contained function that handles all the necessary FFTs and applications of the propagation kernel.

---

## Author

**Debanjan Halder**
