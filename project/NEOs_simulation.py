#%%
import numpy as np
import matplotlib.pyplot as plt

import simulation_module
import model_module as mdl

# Whether to re-run the simulation or load the data from saved text files.
RERUN_INTEGRATION = True
# ==============================================================================
# Folder where the simulation data (integrated ephemerides) will be saved, only 
# if RERUN_INTEGRATION is set to True; if False then the script will look for 
# the required data in the folder. Each ephemeris will be saved in a separate 
# text file whose name includes the body ID and name.
simulation_save_folder = 'NEOs_simulation'
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
data_main_folder = 'NEOs_simulation'

# Folders within 'data_main_folder' for the (observed) body ephemerides.
planet_ephemerides = 'planet_ephemerides' # n.b. includes the Sun ID#10
NEO_ephemerides = 'NEO_ephemerides'

# The body names, Horizons IDs, and masses (or GM values) are tabulated in the 
# text files (named below) in the folder defined by 'data_main_folder'.
planets_file_name = 'planetary_physical_parameters.txt' # n.b. includes the Sun#
NEOs_file_name = 'NEOs.csv'

# For small bodies GM values are listed instead of masses
# Value of G corresponding to GM units of km^3 s^-2 (as in data).
G_val = mdl.Model.G_SI / 1e9 # units km^3 kg^-1 s^-2
# ==============================================================================

def simulation_main():
    '''Main function for the simulation.'''

    # Instantiate new Simulation object
    sim1 = simulation_module.Simulation(name='Solar System Simulation')

    # Define the system bodies and load their physical parameters
    # --------------------------------------------------------------------------
    # The body names, Horizons IDs, and masses (or GM values) are tabulated in 
    # the text files in the folder defined by 'data_main_folder'

    loaded_planets = sim1.load_physical_params_from_file(data_main_folder, 
            planets_file_name, lf_usecols=(0,1,2), lf_skiprows=2, delimiter=',')
    if loaded_planets is False:
        print('SIMULATION MAIN: Unable to load planets.')
        return False
    
    loaded_NEOs = sim1.load_physical_params_from_file(data_main_folder, 
            NEOs_file_name, lf_usecols=(0,1,2), lf_skiprows=1, delimiter=',',
            GM_instead_of_mass=True, G_val=G_val)
    if loaded_planets is False:
        print('SIMULATION MAIN: Unable to load small bodies.')
        return False
    
    # Load/download the (observed) ephemerides for the defined bodies
    # --------------------------------------------------------------------------
    # 1. Define the ephemeris parameters:
    # -----------------------------------
    # Start and stop time as Julian dates or calendar dates (c.f. Horizons API)
    # JD/time converter at: https://ssd.jpl.nasa.gov/tools/jdc/#/cd
    start_time = '2000-Jan-01' # Using cal. date as API required it this time
    stop_time =  '2001-Dec-01' # 700 day interval
    start_time_JD = float(2451544.5000000) # Julian date of start_time
    stop_time_JD =  float(2452244.5000000) # Julian date of stop_time
    step_size = int((stop_time_JD - start_time_JD)/10)
    step_size_api = str(step_size)+'d'
    t_precision = 'MINUTES'
    out_units='AU-D'

    # 2. If the ephemerides are not in the specified folders, download them
    # ---------------------------------------------------------------------
    if REDOWNLOAD_REQUIRED_DATA:
        # Ephemerides for the planets
        status = sim1.download_ephemerides(loaded_planets, data_main_folder, 
                planet_ephemerides, start_time, stop_time, 
                step_size=step_size_api, t_precision=t_precision, 
                out_units=out_units)
        if status is False:
            print('SIMULATION MAIN: Download of planet ephemerides failed.')
            return False
        
        # Ephemerides for the small bodies
        status = sim1.download_ephemerides(loaded_NEOs, data_main_folder, 
                NEO_ephemerides, start_time, stop_time, 
                step_size=step_size_api, t_precision=t_precision, 
                out_units=out_units)
        if status is False:
            print('SIMULATION MAIN: Download of small body ephemerides failed.')
            return False
        if not RERUN_INTEGRATION:
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
    
    # Small bodies
    status = sim1.load_ephemerides(data_main_folder, NEO_ephemerides,
            small_body=True)
    if status is False:
        print('SIMULATION MAIN: Loading SB ephemerides into memory failed.')
        return False
    

    # Begin the simulation
    # --------------------------------------------------------------------------
    # Define the initial conditions for the simulation
    # ------------------------------------------------
    # Julian Date of the initial conditions
    t_initial = float(start_time_JD)

    # get the position and velocity values from the ephemerides
    status = sim1.set_ICs_from_ephemerides(t_initial)
    if status is False:
        print('SIMULATION MAIN: Unable to set initial conditions from ephemerides.')
        return False
    
    # Simulation parameters
    # ---------------------
    model_class = mdl.Model
    t_start = 0
    t_end = stop_time_JD - start_time_JD # days
    t_step = 1 # day
    rtol = 1e-12
    atol = 1e-14
    tcrit=None # critical values (integration care)
    h0=0.0 # first step, default value
    hmax=0.0 # max step, default value

    # Run the simulation
    # ------------------
    if RERUN_INTEGRATION:
        print('INTEGRATION STARTED')
        status, info = sim1.run_simulation(model_class, t_start, t_end, 
                t_step, rtol=rtol, atol=atol, tcrit=tcrit, h0=h0, hmax=hmax, 
                use_vect_ops_func=True, use_solve_ivp_API=True, 
                save_data=SAVE_SIMULATION_DATA, save_dir=simulation_save_folder)
        if status is False:
            print('SIMULATION MAIN: Simulation failed to run.')
            return False, info
        sim1.times = sim1.model.times
        sim1.num_points = np.size(sim1.times)
    else:
        if not REDOWNLOAD_REQUIRED_DATA:
            status = sim1.load_sim_from_dir(simulation_save_folder)
            info = None
            if status is False:
                print('SIMULATION MAIN: Simulation failed to load.')
                return False, info

            # instantiate a Model object from the loaded data to allow, e.g. energy calculations
            sim1.model_obj_from_data(mdl.Model)
            sim1.times = sim1.model.times
            sim1.num_points = np.size(sim1.times)
        else:
            # This run was only to download data
            return True
    
    # Calculate the position relative errors of Itokawa
    # -------------------------------------------------
    id = 25143 # Itokawa Horizons ID
    R = sim1.sim_pos(id, 'all')[0::step_size]
    R_obs = sim1.obs_pos(id, 'all')[0:]
    R_relative_error = (R-R_obs)/R_obs
    times_R_rel = sim1.times[0::step_size]


    # PLOTS
    # =====
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=16)
    
    # Figure 8    
    # Plot the trajectories in 2D projection
    # --------------------------------------
    plot_planets = [10,199,299,399,499] # Horizons IDs
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1, left=0.15)
    for i, id in enumerate(sim1.bodies['id']):
        if id in plot_planets:
            if sim1.bodies['integrated_ephem'][i] is not None:
                # Plot the integrated positions
                x = sim1.sim_pos(id, 'x')[1:-1]
                y = sim1.sim_pos(id, 'y')[1:-1]
                ax.plot(x, y, 'k-', linewidth=3)
                x = sim1.sim_pos(id, 'x')[0]
                y = sim1.sim_pos(id, 'y')[0]
                ax.plot(x, y, 'go')
                x = sim1.sim_pos(id, 'x')[-1]
                y = sim1.sim_pos(id, 'y')[-1]
                ax.plot(x, y, 'ro')

    # Plot the small bodies on the same plot
    for i, id in enumerate(sim1.bodies['id']):
        if id not in plot_planets:
            if sim1.bodies['integrated_ephem'][i] is not None:
                if sim1.bodies['obj_type'][i] == 'SB':
                    name = sim1.bodies['name'][i]
                    if not name:
                        label = 'ID# '+str(sim1.bodies['id'][i])
                    else:
                        label = sim1.bodies['name'][i]
                    x = sim1.sim_pos(id, 'x')[1:-1]
                    y = sim1.sim_pos(id, 'y')[1:-1]
                    ax.plot(x, y, '-', label=label)
                    x = sim1.sim_pos(id, 'x')[0]
                    y = sim1.sim_pos(id, 'y')[0]
                    ax.plot(x, y, 'go')
                    x = sim1.sim_pos(id, 'x')[-1]
                    y = sim1.sim_pos(id, 'y')[-1]
                    ax.plot(x, y, 'ko')
                    # plot the observed positions except initial and last
                    x_obs = sim1.obs_pos(id, 'x')[1:-1]
                    y_obs = sim1.obs_pos(id, 'y')[1:-1]
                    ax.plot(x_obs, y_obs, 'b.')
                    
    ax.set_xlabel('$x$ [AU]')
    ax.set_ylabel('$y$ [AU]')
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper left', fontsize=14)


    # Plot the trajectories in 3D
    # ---------------------------
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i, id in enumerate(sim1.bodies['id']):
        if id in plot_planets:
            if sim1.bodies['integrated_ephem'][i] is not None:
                # Plot the integrated positions
                x = sim1.sim_pos(id, 'x')[1:-1]
                y = sim1.sim_pos(id, 'y')[1:-1]
                z = sim1.sim_pos(id, 'z')[1:-1]
                ax.plot(x, y, z, 'k-')
                x = sim1.sim_pos(id, 'x')[0]
                y = sim1.sim_pos(id, 'y')[0]
                z = sim1.sim_pos(id, 'z')[0]
                ax.plot(x, y, z, 'go')
                x = sim1.sim_pos(id, 'x')[-1]
                y = sim1.sim_pos(id, 'y')[-1]
                z = sim1.sim_pos(id, 'z')[-1]
                ax.plot(x, y, z, 'ro')

    # Plot the small bodies on the same plot
    for i, id in enumerate(sim1.bodies['id']):
        if id not in plot_planets:
            if sim1.bodies['integrated_ephem'][i] is not None:
                if sim1.bodies['obj_type'][i] == 'SB':
                    name = sim1.bodies['name'][i]
                    if not name:
                        label = 'ID# '+str(sim1.bodies['id'][i])
                    else:
                        label = sim1.bodies['name'][i]
                    x = sim1.sim_pos(id, 'x')[1:-1]
                    y = sim1.sim_pos(id, 'y')[1:-1]
                    z = sim1.sim_pos(id, 'z')[1:-1]
                    ax.plot(x, y, z, '-', label=label)
                    x = sim1.sim_pos(id, 'x')[0]
                    y = sim1.sim_pos(id, 'y')[0]
                    z = sim1.sim_pos(id, 'z')[0]
                    ax.plot(x, y, z, 'go')
                    x = sim1.sim_pos(id, 'x')[-1]
                    y = sim1.sim_pos(id, 'y')[-1]
                    z = sim1.sim_pos(id, 'z')[-1]
                    ax.plot(x, y, z, 'ko')
                    # plot the observed positions except initial and last
                    x_obs = sim1.obs_pos(id, 'x')[1:-1]
                    y_obs = sim1.obs_pos(id, 'y')[1:-1]
                    z_obs = sim1.obs_pos(id, 'z')[1:-1]
                    ax.plot(x_obs, y_obs, z_obs, 'b.')

    ax.set_xlabel('$x$ [AU]')
    ax.set_ylabel('$y$ [AU]')
    ax.set_zlabel('$z$ [AU]')
    # ax.zaxis.set_ticks([])
    aspect = (  np.abs(np.ptp(sim1.sim_pos('Earth', 'x'))), 
                np.abs(np.ptp(sim1.sim_pos('Earth', 'y'))), 
                np.abs(np.ptp(sim1.sim_pos('Earth', 'z'))))
    ax.set_box_aspect(aspect)
    ax.legend(loc='upper right', fontsize=14)

    # Figure 9
    # Plot total energy relative error
    # --------------------------------
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 8))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1, left=0.15)
    
    E_total_rel_error = sim1.model.E_total_rel_error()
    if E_total_rel_error is not False:
        ax1.plot(sim1.model.times, E_total_rel_error)
        ax1.set_ylabel('Total energy relative error')

    # Plot Itokawa position relative errors
    # -------------------------------------
    x_err = R_relative_error[:,0]
    y_err = R_relative_error[:,1]
    z_err = R_relative_error[:,2]

    width = 15
    ax2.bar(times_R_rel - width, x_err, width, label='$x$')
    ax2.bar(times_R_rel, y_err, width, label='$y$')
    ax2.bar(times_R_rel + width, z_err, width, label='$z$')
    ax2.set_xticks(times_R_rel)
    ax2.set_xlabel('Time [days]')
    ax2.set_ylabel('Position relative error')
    ax2.legend(fontsize=14)

    return sim1

sim1 = simulation_main()
