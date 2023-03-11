# Solar System simulation
## About
An N-body simulation project completed during my Physics degree.

This project made use of the SciPy Python library to numerically compute the trajectories of bodies in the Solar System. A feature of the software is that it interfaces with the NASA JPL Horizons System via its HTTP API to download ephemeris data from which to set the initial conditions for the integrations and to compare integration results with observed data.

## Results
Results of the project (presented as figures with extended captions as per course requirements): [results.pdf](results.pdf)

Project marks and feedback: [project-feedback.pdf](project-feedback.pdf)

## Description of software
Project guidance provided by teaching staff describing the background and some requirements: [project-guidance.pdf](project-guidance.pdf)

The main functionality of the tool is contained in the files model_module.py and simulation_module.py. The different simulations described in the figures PDF ([results.pdf](results.pdf)) each use a different file (e.g. validation_test-Sun_Earth.py) that imports one or both of these modules. All but one of the simulations (first validation test) have the functionality to save the integrated data and reload it. Most of the simulations take less than or about a minute to run, except the second validation test (validation_test-Burrau.py, which is set to run for a full t=70 time interval and takes more than 5 minutes), and one other simulation (Sun's trajectory, barycenter_sun_simulation.py). Therefore, these are set to reload the data. A variable at the top of each file can be set to instruct whether to re-run the integration or reload saved data -- the data is provided and set up to load automatically if the variable is set to RERUN_INTEGRATION=False.

model_module.py contains two classes that define the physical models used in the simulation -- Model and ModelApproximate. ModelApproximate inherits from Model and overrides solve_system().

simulation_module.py contains one class, Simulation, which contains variables and methods that organise the data and provide various functionality. Each method is commented, and the simulation files illustrate how they are used.

A main feature of the software is that it interfaces with the NASA JPL Horizons system via its HTTP API to download ephemeris data from which to set the initial conditions for the integrations and to compare integration results with observed data.

JPL Horizons system: https://ssd.jpl.nasa.gov/

JPL Horizons API documentation: https://ssd-api.jpl.nasa.gov/doc/horizons.html

## To reproduce the figures
1. Figure 1: Validation test.
Run validation_test-Sun_Earth.py

2. Figures 2 and 3: Burrau's problem.
Run validation_test-Burrau.py. Loads saved data due to computation time.

3. Figures 4 and 5: Solar System simulation.
Run solar_system_simulation.py. Runs for about a minute.

4. Figures 6 and 7: Solar System: Sun's trajectory about barycenter.
Run barycenter_sun_simulation.py. Loads saved data.

5. Figures 8 and 9: NEOs.
Run NEOs_simulation.py. Runs less than a minute.

6. Figures 10 and 11: Small-body simulation.
Run small_body_simulation.py. Runs less than a minute.
