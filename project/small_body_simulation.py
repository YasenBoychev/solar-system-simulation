#%%
import os
import numpy as np
import matplotlib.pyplot as plt

import simulation_module
import model_module as mdl

# Whether to run integration or load the data from the saved text file
RERUN_SMALL_BODY_INTEGRATION = True
# ==============================================================================
# Folder where the simulation data (integrated ephemerides) will be saved. Each 
# ephemeris will be saved in a separate text file whose name includes the body 
# ID and name.
simulation_save_folder = 'small_body_simulation'
SAVE_SIMULATION_DATA = False

# ------------------------------------------------------------------------------
# Whether to re-download ephemerides (from JPL Horizons via the URL GET API, 
# documentation here: https://ssd-api.jpl.nasa.gov/doc/horizons.html) for the 
# bodies defined in the simulation (with conditions like start/end time 
# specified later in the script). The data will be downloaded in the folders 
# described below. Any ephemeris where the body ID matches will be overwritten.
# If REDOWNLOAD_REQUIRED_DATA is False, the script assumes that the required 
# data is already in the specified folders.
REDOWNLOAD_REQUIRED_DATA = False

# This folder stores text files with tabulated data about the system bodies,
# as well as folders with the (observed) body ephemerides.
data_main_folder = 'small_body_simulation'

# Folders within 'data_main_folder' for the (observed) body ephemerides.
planet_ephemerides = 'planet_ephemerides' # n.b. includes the Sun ID#10
small_body_ephemerides = 'small_body_ephemerides'

# The body names, Horizons IDs, and masses (or GM values) are tabulated in the 
# text files (named below) in the folder defined by 'data_main_folder'.
planets_file_name = 'planetary_physical_parameters.txt' # n.b. includes the Sun
# ==============================================================================

def simulation_main():
    '''Main function for the simulation.'''

    # Instantiate new Simulation object
    sim1 = simulation_module.Simulation()

    # Define the system bodies and load their physical parameters (Major Bodies)
    # --------------------------------------------------------------------------
    # The body names, Horizons IDs, and masses (or GM values) are tabulated in 
    # the text files in the folder defined by 'data_main_folder'

    loaded_planets = sim1.load_physical_params_from_file(data_main_folder, 
            planets_file_name, lf_usecols=(0,1,2), lf_skiprows=2, delimiter=',')
    if loaded_planets is False:
        print('SIMULATION MAIN: Unable to load planets.')
        return False

    # Load/download the (observed) ephemerides for the defined bodies
    # --------------------------------------------------------------------------
    # 1. Define the ephemeris parameters:
    # -----------------------------------
    # Start and stop time as Julian dates or calendar dates (c.f. Horizons API)
    # JD/time converter at: https://ssd.jpl.nasa.gov/tools/jdc/#/cd
    # n.b. there is a 'JD' string concatenated to this in the 
    # download_ephemerides() arguments (required by API).
    start_time = '2451544.5000000' # Julian date
    stop_time =  '2452244.5000000' # 700 day interval
    t_step = 0.1 # 0.1 day
    # The interval is divided into step_size points if step_size has no unit
    step_size=str(int((float(stop_time)-float(start_time))/t_step))
    t_precision = 'MINUTES'
    out_units='AU-D'

    # 2. If the ephemerides are not in the specified folders, download them
    # ---------------------------------------------------------------------
    if REDOWNLOAD_REQUIRED_DATA:
        # Ephemerides for the planets
        status = sim1.download_ephemerides(loaded_planets, data_main_folder, 
                planet_ephemerides, 'JD'+start_time, 'JD'+stop_time, 
                step_size=step_size, t_precision=t_precision, 
                out_units=out_units)
        if status is False:
            print('SIMULATION MAIN: Download of planet ephemerides failed.')
            return False
                
        # Small body ephemerides
        small_bodies = {25143 : 'Itokawa'}
        status = sim1.download_ephemerides(small_bodies, data_main_folder, 
                small_body_ephemerides, 'JD'+start_time, 'JD'+stop_time, 
                step_size=step_size, t_precision=t_precision, 
                out_units=out_units)
        if status is False:
            print('SIMULATION MAIN: Download of SB ephemerides failed.')
            return False
        if not RERUN_SMALL_BODY_INTEGRATION:
            print('This run was to download data only')
            return True

    # 3. Load the ephemerides into the Simulation object
    # --------------------------------------------------
    # Planets
    # ephemeris load order -- Horizons body IDs
    planets_load_order = ['10','199','299','399','499','599',
                            '699','799','899','999']
    status = sim1.load_ephemerides(data_main_folder, planet_ephemerides, 
            planets_load_order)
    if status is False:
        print('SIMULATION MAIN: Loading planet ephemerides into memory failed.')
        return False
    
    # Small Body Ephemerides
    # ----------------------
    status = sim1.load_ephemerides(data_main_folder, small_body_ephemerides, 
                    small_body=True)
    if status is False:
        print('SIMULATION MAIN: Loading planet ephemerides into memory failed.')
        return False

    # Small-body simulation
    # ==========================================================================

    # Object data
    # -----------
    object_id = 25143
    object_name = 'Itokawa'
    orbit_period = 567 # days
    G_val = mdl.Model.G_SI / 1e9 # units km^3 kg^-1 s^-2
    mass_object = 2.1e-9 / G_val # mass in kg
    if object_id in sim1.bodies['id']:
        object_index = sim1.bodies['id'].index(object_id)
        sm_ephem = sim1.bodies['ephemeris'][object_index]
        R0_object = sm_ephem[0,[1,2,3]] # shape (1,3)
        V0_object = sm_ephem[0,[4,5,6]] # R0/V0 shape (1,3)
        times = sm_ephem[:,0]
    else:
        print('SIMULATION MAIN: No data for small body')
        return False

    # Simulation parameters
    # ---------------------
    model_class = mdl.ModelApproximate
    t_start = float(start_time) # Julian date
    t_end = float(stop_time)

    # The step sizes have been set up to match the ephemeride data
    rtol = None
    atol = None
    h0=0.1 # first step
    hmax=0.1 # max step
    min_step = 0.1

    # Add the data to a ModelApproximate object
    sim1.model_obj_from_data(model_class, approx_model=True, 
            integrated_data=False)
    sim1.model_approx.object_id = object_id
    sim1.model_approx.object_name = object_name
    sim1.model_approx.mass_object = mass_object
    sim1.model_approx.R0_object = R0_object # shape (1,3)
    sim1.model_approx.V0_object = V0_object # R0/V0 shape (1,3)
    sim1.model_approx.times = times # shape (n, )

    if RERUN_SMALL_BODY_INTEGRATION:
        sim1.model_approx.solve_system(t_start, t_end, rtol=rtol, atol=atol, 
                h0=h0, hmax=hmax, min_step=min_step)
        
        # R/V shape (size(times), 3)
        R = np.swapaxes(sim1.model_approx.position(object_id, 'all'), 0,1)
        V = np.swapaxes(sim1.model_approx.velocity(object_id, 'all'), 0,1)

        # save the data to a text file
        if SAVE_SIMULATION_DATA:
            out_arr = np.hstack((times[:,np.newaxis], R, V))
            fpath = os.path.join(simulation_save_folder, 
                    'ID'+str(object_id)+'_'+object_name+'-sim.txt')
            np.savetxt(fpath, out_arr, fmt='%.18e', delimiter=',')
    else:
        # Load the data from file
        fpath = os.path.join(simulation_save_folder, 
                'ID'+str(object_id)+'_'+object_name+'-sim.txt')
        ephemeris = np.loadtxt(fpath, dtype=np.float64, delimiter=',')
        R = ephemeris[:,[1,2,3]]
        V = ephemeris[:,[4,5,6]]
        sim1.model_approx.R = R
        sim1.model_approx.V = V
    
    # Calculate the position relative errors
    # --------------------------------------
    R_diff = R - sim1.obs_pos(object_id, 'all')
    R_relative_error = R_diff / sim1.obs_pos(object_id, 'all')

    # Plots
    # ======================================================================
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=16)
        
    # Figure 10
    # ---------
    plot_planets = [10,199,299,399,499] # Horizons IDs

    # Plot the INTEGRATED small body trajectory in 2D projection
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0, left=0.125)
    
    id = object_id
    body_index = sim1.bodies['id'].index(id)
    x = R[:,0]
    y = R[:,1]
    ax.plot(x, y, 'r-')

    # Plot OBSERVED trajectories
    # --------------------------
    # Small body observed trajectory plot point step
    # index points per time unit (day)
    points_per_unit = int(float(step_size)/(float(stop_time)-float(start_time)))
    step = points_per_unit * 10

    # The observed small body trajectory
    if sim1.bodies['ephemeris'][body_index] is not None:
        # plot the observed positions except initial and last
        x_obs = sim1.obs_pos(id, 'x')[1:-1:step]
        y_obs = sim1.obs_pos(id, 'y')[1:-1:step]
        ax.plot(x_obs, y_obs, 'b.')
        # plot the intial observed positions
        x_obs = sim1.obs_pos(id, 'x')[0]
        y_obs = sim1.obs_pos(id, 'y')[0]
        ax.plot(x_obs, y_obs, 'go')
        # plot the last observed positions
        x_obs = sim1.obs_pos(id, 'x')[-1]
        y_obs = sim1.obs_pos(id, 'y')[-1]
        ax.plot(x_obs, y_obs, 'ko')

    # Plot other bodies on the same plot
    for i, id in enumerate(sim1.bodies['id']):
        if id in plot_planets:
            if sim1.bodies['ephemeris'][i] is not None:
                if sim1.bodies['obj_type'][i] != 'SB':
                    label = sim1.bodies['name'][i]
                    x_obs = sim1.obs_pos(id, 'x')[1:-1]
                    y_obs = sim1.obs_pos(id, 'y')[1:-1]
                    if id != 10:
                        ax.plot(x_obs, y_obs, label=label)
                    else:
                        ax.plot(x_obs, y_obs)
                    x_obs = sim1.obs_pos(id, 'x')[0]
                    y_obs = sim1.obs_pos(id, 'y')[0]
                    ax.plot(x_obs, y_obs, 'go')
                    x_obs = sim1.obs_pos(id, 'x')[-1]
                    y_obs = sim1.obs_pos(id, 'y')[-1]
                    ax.plot(x_obs, y_obs, 'ko')
    ax.set_xlabel('$x$ [AU]')
    ax.set_ylabel('$y$ [AU]')
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper left', fontsize=14)


    # Plot the small body trajectory in 3D
    # ------------------------------------
    # Plot the INTEGRATED small body trajectory
    # -----------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    id = object_id
    body_index = sim1.bodies['id'].index(id)
    x = R[:,0]
    y = R[:,1]
    z = R[:,2]
    ax.plot(x, y, z ,'r-')

    # Plot OBSERVED trajectories
    # --------------------------
    if sim1.bodies['ephemeris'][body_index] is not None:
        # plot the observed positions except initial and last
        x_obs = sim1.obs_pos(id, 'x')[1:-1:step]
        y_obs = sim1.obs_pos(id, 'y')[1:-1:step]
        z_obs = sim1.obs_pos(id, 'z')[1:-1:step]
        ax.plot(x_obs, y_obs, z_obs, 'b.')
        # plot the intial observed positions
        x_obs = sim1.obs_pos(id, 'x')[0]
        y_obs = sim1.obs_pos(id, 'y')[0]
        z_obs = sim1.obs_pos(id, 'z')[0]
        ax.plot(x_obs, y_obs, z_obs, 'go')
        # plot the last observed positions
        x_obs = sim1.obs_pos(id, 'x')[-1]
        y_obs = sim1.obs_pos(id, 'y')[-1]
        z_obs = sim1.obs_pos(id, 'z')[-1]
        ax.plot(x_obs, y_obs, z_obs, 'ko')
    # Plot other bodies on the same plot
    for i, id in enumerate(sim1.bodies['id']):
        if id in plot_planets:
            if sim1.bodies['ephemeris'][i] is not None:
                if sim1.bodies['obj_type'][i] != 'SB':
                    label = sim1.bodies['name'][i]
                    x_obs = sim1.obs_pos(id, 'x')[1:-1]
                    y_obs = sim1.obs_pos(id, 'y')[1:-1]
                    z_obs = sim1.obs_pos(id, 'z')[1:-1]
                    if id != 10:
                        ax.plot(x_obs, y_obs, z_obs, label=label)
                    else:
                        ax.plot(x_obs, y_obs, z_obs)
                    x_obs = sim1.obs_pos(id, 'x')[0]
                    y_obs = sim1.obs_pos(id, 'y')[0]
                    z_obs = sim1.obs_pos(id, 'z')[0]
                    ax.plot(x_obs, y_obs, z_obs, 'go')
                    x_obs = sim1.obs_pos(id, 'x')[-1]
                    y_obs = sim1.obs_pos(id, 'y')[-1]
                    z_obs = sim1.obs_pos(id, 'z')[-1]
                    ax.plot(x_obs, y_obs, z_obs, 'ko')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.zaxis.set_ticks([])
    aspect = (np.abs(np.ptp(x)), np.abs(np.ptp(y)), np.abs(np.ptp(z)))
    ax.set_box_aspect(aspect)
    #ax.legend(loc='upper right')

    # Figure 11
    # ---------
    # Plot the position relative error
    # --------------------------------
    x_err = R_relative_error[:,0]
    y_err = R_relative_error[:,1]
    z_err = R_relative_error[:,2]
    
    fig, ax1 = plt.subplots(figsize=(8,5))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15, left=0.1)
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('Position relative error')
    ax1.plot(times-float(start_time), x_err, '-', ms=4, label='$x$')
    ax1.plot(times-float(start_time), y_err, '-', ms=4, label='$y$')
    ax1.plot(times-float(start_time), z_err, '-', ms=4, label='$z$')
    ax1.legend(loc='upper left', fontsize=14)

    return sim1

sim1 = simulation_main()