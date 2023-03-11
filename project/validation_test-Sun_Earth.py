#%%
import numpy as np
import matplotlib.pyplot as plt

import model_module as mdl

# Basic test 1 - Sun and Earth
# ============================

# Define the system
# -----------------
G = (mdl.Model.G_SI/mdl.Model.L**3)*mdl.Model.M*mdl.Model.T**2
body_ids = [10, 399]
body_names = ['Sun', 'Earth']

mass_Sun = 1.989e30
mass_Earth = 5.972e24
x_Sun, y_Sun, z_Sun = 0, 0, 0
x_Earth, y_Earth, z_Earth = 1, 0, 0
vx_Sun, vy_Sun, vz_Sun = 0, 0, 0
vx_Earth, vy_Earth, vz_Earth = 0, np.sqrt((G*mass_Sun)/x_Earth), 0

masses = np.array([mass_Sun, mass_Earth], dtype=np.float64)
R0 = np.array([ (x_Sun, y_Sun, z_Sun), 
                (x_Earth, y_Earth, z_Earth)], dtype=np.float64)
V0 = np.array([ (vx_Sun, vy_Sun, vz_Sun), 
                (vx_Earth, vy_Earth, vz_Earth)], dtype=np.float64)

# the Model object has methods to integrate the system and calculate the total energy
system = mdl.Model(body_ids, body_names, masses, R0, V0)

# Integration
# -----------
t_start = 0
t_end = 5*365 # days
t_step = 1 # day
num_points = int(((t_end-t_start)/t_step)+1)
rtol = 1e-8
atol = 1e-10
hmax = 0.0 # maximum step, 0.0 default (integrator chooses)

solution, info = system.solve_system(t_start, t_end, num_points, rtol=rtol, 
        atol=atol, hmax=hmax, use_vect_ops_func=True, use_solve_ivp_API=True)

# Plots
# -----
# Figure 1
# --------
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=18)

fig, ax1 = plt.subplots(figsize=(10, 4))
fig.tight_layout()
fig.subplots_adjust(bottom=0.18)
ax1.plot(system.position('Earth', 'x'), system.position('Earth', 'y'), 'k')
ax1.plot(system.position('Sun', 'x'), system.position('Sun', 'y'), 'r')
ax1.set_xlabel('$x_\oplus$ [AU]')
ax1.set_ylabel('$y_\oplus$ [AU]')
ax1.set_aspect('equal', 'box')

fig, ax2 = plt.subplots(figsize=(10, 4))
fig.tight_layout()
fig.subplots_adjust(bottom=0.18, left=0.1)
ax2.plot(system.times/365, system.position('Earth', 'x'), 'k')
ax2.set_xlabel('t [yr]')
ax2.set_ylabel('$x_\oplus$ [AU]')

# Energy relative error
# ---------------------
E_total_rel_error = system.E_total_rel_error()
plt.figure(figsize=(10, 8))
plt.plot(system.times/365, E_total_rel_error)
plt.xlabel('t [yr]')
plt.ylabel('Total energy relative error')

