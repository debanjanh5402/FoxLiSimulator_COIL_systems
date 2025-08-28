CavitySIM: A High-Fidelity Optical Resonator Simulator
An advanced numerical simulation suite for determining the transverse eigenmodes and beam quality of stable and unstable laser resonators using the Fox-Li iterative method coupled with an Angular Spectrum propagation engine.


!(https://img.shields.io/badge/License-MIT-yellow.svg)
!(https://img.shields.io/badge/build-passing-brightgreen.svg)
!(https://img.shields.io/badge/DOI-10.5281/zenodo.XXXXXXX-blue.svg)

1. Overview
CavitySIM is a powerful computational tool designed for the analysis and simulation of optical resonators, which form the core of all laser systems. The fundamental challenge in laser design is to determine the self-consistent electromagnetic field distribution—the laser mode—that can exist within a given cavity geometry. This software provides a robust and physically accurate solution to this problem, enabling researchers and engineers to predict the spatial properties, diffraction losses, and beam quality of laser outputs.

The simulation engine is built upon the foundational Fox-Li iterative method, which numerically mimics the physical process of mode formation within a laser. By simulating the propagation of an optical field for numerous round-trips between the resonator mirrors, the software iteratively converges on the stable, lowest-loss transverse eigenmode. This process is visualized below, where an initial, arbitrary field distribution evolves over hundreds of iterations into the fundamental TEM₀₀ Gaussian mode of a stable confocal resonator.


(An animated GIF showing an initial random field distribution on a mirror. With each frame representing multiple iterations, the field smooths out, high-frequency components die away, and the distribution converges into a stable, fundamental Gaussian mode (TEM₀₀).)

This package is intended for graduate students, researchers, and engineers in the fields of optics, photonics, and laser physics who require a precise and versatile tool for resonator design and analysis.

2. Table of Contents
1. Overview
-(#2-table-of-contents)
-(#3-scientific-foundation)
-(#31-the-physics-of-optical-resonators)
-(#32-the-fox-li-iterative-method)
-(#33-the-propagation-engine-the-angular-spectrum-method-asm)



-(#6-usage-and-walkthrough)
-(#61-example-1-simulating-a-stable-confocal-resonator)
-(#7-technical-deep-dive-m-beam-quality-factor)


-(#9-citing-this-work)



3. Scientific Foundation
This section details the physical and mathematical principles that form the basis of the CavitySIM software. A thorough understanding of these concepts is essential for the correct application of the tool and interpretation of its results.

3.1 The Physics of Optical Resonators
An optical resonator, or resonant cavity, is an arrangement of mirrors that confines light, allowing it to travel multiple round-trips. This confinement is the essential mechanism for providing optical feedback in a laser. The geometry of the resonator—specifically the curvature and separation of its mirrors—determines the spatial characteristics of the laser beam it produces.

The central concept in resonator theory is the principle of self-consistency. Not just any electromagnetic field can be sustained within a cavity. A stable field distribution, known as an eigenmode (or simply a mode), is one that reproduces itself after a single round-trip through the resonator. Mathematically, if 

E 
in
​
 (x,y) is the complex amplitude of the field at a reference plane (e.g., one of the mirrors), it must satisfy the following integral equation after one round-trip propagation:

γE 
in
​
 (x,y)=∫∫K(x,y;x 
′
 ,y 
′
 )E 
in
​
 (x 
′
 ,y 
′
 )dx 
′
 dy 
′
 
Here, K(x,y;x 
′
 ,y 
′
 ) is the round-trip propagation kernel, which encapsulates all the effects of diffraction and reflection within the cavity. The complex constant 

γ is the eigenvalue corresponding to the eigenmode E 
in
​
 . The eigenvalue's physical significance is twofold:

Magnitude (∣γ∣): The magnitude of the eigenvalue represents the fraction of the field's amplitude remaining after one round-trip. The round-trip power loss, primarily due to diffraction at the mirror apertures, is therefore given by 1−∣γ∣ 
2
 . The fundamental mode (e.g., TEM₀₀) is the eigenmode with the largest eigenvalue, meaning it has the lowest loss and will be the first to lase.

Phase (arg(γ)): The phase of the eigenvalue represents the phase shift accumulated by the mode during one round-trip. This phase shift determines the exact resonant frequencies of the cavity mode.

Resonators are broadly classified as stable or unstable based on whether geometric rays remain confined within the cavity over many bounces. Stable resonators, such as the confocal cavity, have low diffraction losses and typically produce high-quality beams, while unstable resonators have higher losses but can efficiently extract energy from large-volume gain media.

CavitySIM is capable of modeling both configurations.

3.2 The Fox-Li Iterative Method
For all but the simplest resonator geometries, the self-consistency integral equation cannot be solved analytically. The Fox-Li iterative method is the canonical numerical technique developed to solve this problem. First proposed by A. G. Fox and T. Li in 1961, this method elegantly simulates the physical process of mode competition and formation inside a laser cavity.

The algorithm proceeds as follows:

Initialization: An arbitrary complex field distribution, E 
0
​
 (x,y), is assumed to exist on one of the resonator mirrors. This initial field can be a simple plane wave of uniform amplitude or even random noise, as the method is generally insensitive to the starting conditions.

Iteration: The field is numerically propagated through the cavity for one full round-trip. For a two-mirror cavity, this involves propagating the field from mirror 1 to mirror 2, applying the reflection and phase transformation of mirror 2, propagating back to mirror 1, and applying the reflection of mirror 1. This yields a new field distribution, E 
1
​
 (x,y).

Convergence: This process is repeated for many iterations (q=1,2,3,...). With each round-trip, modes that have higher diffraction losses (smaller eigenvalues) decay more rapidly than the fundamental, lowest-loss mode. After a sufficient number of iterations (typically several hundred), the field distribution E 
q
​
 (x,y) converges to a steady-state solution that no longer changes shape from one iteration to the next. This stable field is the fundamental eigenmode of the resonator.

Eigenvalue Extraction: Once the mode has stabilized, the complex eigenvalue γ is calculated from the ratio of the fields in two successive iterations: γ=E 
q+1
​
 (x,y)/E 
q
​
 (x,y).

While other numerical techniques exist, such as the Eigenvector Method (EM) which solves for all modes at once by diagonalizing the propagation matrix , the Fox-Li method was deliberately chosen for 

CavitySIM. Its pass-by-pass simulation approach provides a more physically intuitive model and, critically, creates a flexible framework. This iterative structure allows for the straightforward future inclusion of nonlinear, intensity-dependent phenomena such as gain saturation, thermal lensing in the gain medium, or the effects of intracavity diffractive optical elements—scenarios where a simple matrix approach would be insufficient.

3.3 The Propagation Engine: The Angular Spectrum Method (ASM)
The accuracy of the Fox-Li simulation hinges on the numerical method used to model the diffraction of the field as it propagates between the mirrors. CavitySIM employs the Angular Spectrum Method (ASM), a rigorous and highly accurate technique based on Fourier optics. Unlike more common approximations (e.g., the Fresnel diffraction integral), ASM is not bound by the paraxial approximation, which assumes that light rays travel at small angles to the optical axis. This makes 

CavitySIM a non-paraxial simulator capable of accurately modeling a wide range of modern resonators, including those with short cavity lengths, high numerical apertures, or significant off-axis components.

The ASM algorithm treats wave propagation as a linear filtering operation in the spatial frequency domain and is implemented as follows:

Fourier Transform: The complex field at a source plane, E(x,y,z=0), is decomposed into a superposition of infinite plane waves, each with a unique transverse spatial frequency (k 
x
​
 ,k 
y
​
 ). This decomposition is efficiently performed by taking the 2D Fast Fourier Transform (FFT) of the field to obtain its angular spectrum,  
E
~
 (k 
x
​
 ,k 
y
​
 ).

Propagation in Fourier Space: Each plane wave component propagates independently. To propagate the entire field by a distance Δz, the angular spectrum is multiplied by a transfer function, H(k 
x
​
 ,k 
y
​
 ), often called the propagator in reciprocal space :

E
~
 (k 
x
​
 ,k 
y
​
 ;z=Δz)= 
E
~
 (k 
x
​
 ,k 
y
​
 ;z=0)×H(k 
x
​
 ,k 
y
​
 )

The propagator is given by:
$$ H(k_x, k_y) = e^{i k_z \Delta z} \quad \text{where} \quad k_z = \sqrt{k^2 - k_x^2 - k_y^2} \quad \text{and} \quad k = \frac{2\pi}{\lambda} $$
This step is a computationally efficient element-wise multiplication.

Inverse Fourier Transform: The complex field at the destination plane, E(x,y,z=Δz), is reconstructed by applying the 2D Inverse FFT to the propagated angular spectrum.

An important physical aspect captured by ASM is the distinction between propagating and evanescent waves. For spatial frequencies where k 
x
2
​
 +k 
y
2
​
 ≤k 
2
 , k 
z
​
  is real, and the corresponding plane waves propagate. However, for high spatial frequencies where k 
x
2
​
 +k 
y
2
​
 >k 
2
 , k 
z
​
  becomes imaginary. These components correspond to evanescent waves, which decay exponentially with distance and do not propagate into the far-field. They carry sub-wavelength information about the source field that is typically lost during propagation. While ASM is highly accurate, users should be aware that numerical sampling can lead to aliasing errors when simulating propagation over very long distances or with rapidly diverging beams.

4. Key Features
Versatile Resonator Geometries: Simulate arbitrary stable and unstable two-mirror resonators defined by mirror radii of curvature, aperture sizes, and separation.

High-Fidelity Propagation: Employs the non-paraxial Angular Spectrum Method for accurate diffraction modeling, suitable for both simple and complex cavity designs.

Comprehensive Analysis: Calculates the final steady-state mode intensity and phase profiles, round-trip diffraction loss, and the resonant frequency shift of the fundamental mode.

Advanced Beam Quality Metrics: Includes a robust implementation for calculating the M² beam quality factor according to the rigorous second-moment method defined in the ISO 11146 standard.

Extensible Framework: The iterative architecture is designed to be adaptable for future inclusion of active gain media, thermal lensing effects, and arbitrary intracavity optical elements.

Rich Visualization: Generates publication-quality plots of 2D field distributions, convergence behavior, and beam caustic measurements for M² analysis.

5. Installation
To get CavitySIM running on your local machine, please follow these steps.

Prerequisites
Python (version 3.8 or higher)

Git version control system

Setup Instructions
Clone the repository:
Open your terminal or command prompt and clone this repository to your local machine.

Create a virtual environment (recommended):
It is highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects.

Install dependencies:
Install the required Python packages using the provided requirements.txt file.

This will install necessary libraries such as NumPy, SciPy, and Matplotlib. You are now ready to run simulations.

6. Usage and Walkthrough
This section provides a practical guide to setting up and running a basic simulation.

6.1 Example 1: Simulating a Stable Confocal Resonator
A confocal resonator is a stable cavity where two identical spherical mirrors are separated by a distance equal to their radius of curvature (L=R). This configuration is known to support a fundamental Gaussian mode.

Step 1: Configure the Simulation
Create a Python configuration file (e.g., config_confocal.py) to define the physical and numerical parameters of the simulation.

A detailed description of the key simulation parameters is provided in the table below. This table serves as a quick reference for understanding and modifying the simulation inputs.

Step 2: Run the Simulation
Execute the main simulation script from your terminal, passing the configuration file as an argument.

The script will initialize the field, perform the Fox-Li iterations, and save the results (plots and data) to an output directory.

Step 3: Analyze the Output
Upon completion, the simulation will generate several output plots:

Final Mode Intensity: A 2D plot of the intensity profile (∣E(x,y)∣ 
2
 ) of the converged mode on the mirror surface. For a stable confocal resonator, this will show a clean, circular Gaussian spot (the TEM₀₀ mode).

Final Mode Phase: A 2D plot of the phase profile (arg(E(x,y))) of the converged mode. This will show a curved wavefront matching the curvature of the mirror.

Convergence Plot: A plot of the calculated round-trip loss as a function of the iteration number. This plot is critical for verifying that the simulation has reached a stable, converged solution, which is indicated by the loss value flattening out to a constant level.

7. Technical Deep Dive: M² Beam Quality Factor
Beyond simply determining the mode profile, CavitySIM provides a quantitative measure of the beam's quality using the M² (M-squared) factor. This dimensionless parameter is the industry and academic standard for quantifying how closely a real laser beam's propagation characteristics match those of an ideal, diffraction-limited Gaussian beam. An M² value of 1 signifies a perfect Gaussian beam, while higher values indicate poorer beam quality, meaning the beam will diverge more rapidly and cannot be focused to as small a spot.

CavitySIM implements the M² calculation according to the rigorous second-moment (D4σ) method outlined in the ISO 11146 standard. This method is robust and accurate even for non-Gaussian or complex beam profiles. The implementation follows these steps:

Beam Propagation: After finding the stable mode inside the cavity, the software propagates this field distribution out of the resonator through free space.

Caustic Measurement: The software measures the beam's width at multiple positions (z) along the propagation axis, particularly around the location of the beam waist (the point of minimum size). The beam width is calculated using the D4σ definition, which is four times the standard deviation of the beam's intensity distribution in the x and y directions.

Hyperbolic Fit: The measured beam widths, W(z), are fitted to the following hyperbolic equation, which describes the propagation of a real laser beam :

W(z) 
2
 =W 
0
2
​
 +(M 
2
  
πW 
0
​
 
λ
​
 ) 
2
 (z−z 
0
​
 ) 
2
 

This fitting procedure yields the key beam parameters: the beam waist radius (W 
0
​
 ), the waist location (z 
0
​
 ), and, most importantly, the beam quality factor (M 
2
 ).

Divergence Calculation: The far-field divergence half-angle, Θ, is also determined from the fit parameters as Θ=M 
2
 λ/(πW 
0
​
 ).

The software generates a beam caustic plot to visualize this process. This plot shows the measured D4σ beam radii at various positions along the propagation axis, overlaid with the best-fit hyperbola. This visualization serves as a direct confirmation that the software is performing the multi-point measurement and fitting procedure required by the ISO standard, ensuring a reliable and accurate M² value.

8. Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Reporting Bugs
If you encounter a bug, please open an issue on the GitHub repository. Please include a clear title, a detailed description of the problem, and, if possible, steps to reproduce the issue.

Suggesting Enhancements
If you have an idea for a new feature or an improvement to an existing one, please open an issue to start a discussion.

Pull Request Process
Fork the Project.

Create your Feature Branch (git checkout -b feature/NewFeature).

Commit your Changes (git commit -m 'Add some NewFeature').

Push to the Branch (git push origin feature/NewFeature).

Open a Pull Request.

Please ensure your code adheres to the existing style and includes relevant documentation.

9. Citing This Work
If you use CavitySIM in your academic research or publications, please cite it to ensure reproducibility and to give credit to the development effort. Please use the following BibTeX entry.

10. License
This project is licensed under the MIT License. See the LICENSE.md file for details.

11. Acknowledgments
This project relies heavily on the exceptional scientific computing libraries provided by the NumPy and SciPy development teams.

All visualizations are generated using the versatile Matplotlib library.

This work stands on the shoulders of giants, and we acknowledge the foundational contributions of A. G. Fox and T. Li, whose pioneering work made this field of computational optics possible.
