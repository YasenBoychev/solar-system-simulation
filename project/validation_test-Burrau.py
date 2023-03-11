#%%
import os
import numpy as np
import matplotlib.pyplot as plt

import model_module as mdl

# Whether to re-run the simulation or load the data from a saved text file.
RERUN_INTEGRATION = False
SAVE_SIMULATION_DATA = False
simulation_save_folder = 'Burrau_problem_data'

# Basic test 2 - Burrau's problem
# ===============================
body_ids = [1, 2, 3]
body_names = ['Body 1', 'Body 2', 'Body 3']
masses = np.array((3, 4, 5), dtype=np.float64)
R0 = np.array([ (1, 3, 0),
                (-2, -1, 0),
                (1, -1, 0)], dtype=np.float64)
V0 = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0)], dtype=np.float64)
system2 = mdl.Model(body_ids, body_names, masses, R0, V0, G=1.)

t_start = 0
t_end = 70
t_step = 0.01
num_points = int(((t_end-t_start)/t_step)+1)
rtol = 1e-13
atol = 1e-15
hmax = 0.0

if RERUN_INTEGRATION:
    solution2, info2 = system2.solve_system(t_start, t_end, num_points, 
            rtol=rtol, atol=atol, hmax=hmax, use_vect_ops_func=True, 
            use_solve_ivp_API=True)
    
    if SAVE_SIMULATION_DATA:
        # save the simulation data
        for i, id in enumerate(body_ids):
            # R_i/V_i shape (num times, 3)
            R_i = np.swapaxes(system2.position(id, 'all'),0,1)
            V_i = np.swapaxes(system2.velocity(id, 'all'),0,1)
            times_i = system2.times[:,np.newaxis]
            out_arr = np.hstack((times_i, R_i, V_i))
            fpath = os.path.join(simulation_save_folder, 
                    'ID'+str(id)+'_'+body_names[i]+'-Burrau.txt')
            if not os.path.isdir(simulation_save_folder):
                os.mkdir(simulation_save_folder)
            np.savetxt(fpath, out_arr, fmt='%.18e', delimiter=',')
else:
    # Load the data from file
    R = []
    V = []
    for i, id in enumerate(body_ids):
        fpath = os.path.join(simulation_save_folder, 
                'ID'+str(id)+'_'+body_names[i]+'-Burrau.txt')
        ephemeris = np.loadtxt(fpath, dtype=np.float64, delimiter=',')
        R.append(ephemeris[:,[1,2,3]])
        V.append(ephemeris[:,[4,5,6]])
        times = ephemeris[:,0]
    system2.R = np.swapaxes(np.stack(R, axis=2), 0,2)
    system2.V = np.swapaxes(np.stack(V, axis=2), 0,2)
    system2.times = times

# Plots
# -----
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=16)

# Figure 2
# --------
# from t = 0 to 10 and from t = 60 to 70
fig, ax = plt.subplots(1,2, figsize=(10, 8), sharey='all')
fig.tight_layout()
fig.subplots_adjust(bottom=0.1, left=0.1)
for i, times in enumerate({0:10, 60:70}.items()):
    t1, t2 = times
    t1_index = int((t1 - t_start) / t_step)
    t2_index = int((t2 - t_start) / t_step) - 1
    for id in system2.body_ids:
        x = system2.position(id, 'x')[t1_index+1:t2_index]
        y = system2.position(id, 'y')[t1_index+1:t2_index]
        ax[i].plot(x, y, label=str(id))
        # plot starting positions
        x = system2.position(id, 'x')[t1_index]
        y = system2.position(id, 'y')[t1_index]
        ax[i].plot(x, y, 'go')
        # plot ending positions
        x = system2.position(id, 'x')[t2_index]
        y = system2.position(id, 'y')[t2_index]
        ax[i].plot(x, y, 'ko')
    ax[i].set_xlabel('$x$')
    if i == 0:
        ax[i].set_ylabel('$y$')
    ax[i].set_ylim((-5,5))
    ax[i].set_xlim((-3,3))
    ax[i].set_aspect('equal', 'box')
    ax[i].legend(loc='upper left', fontsize=14)

# Figure 3
# --------
# plot the total energy relative error
E_total_rel_error = system2.E_total_rel_error()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(system2.times, E_total_rel_error)
ax.set_xlabel('$t$')
ax.set_ylabel('Total energy relative error')

