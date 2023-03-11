#%%
import numpy as np
import matplotlib.pyplot as plt

import simulation_module
import model_module as mdl

# Whether to re-run the simulation or load the data from saved text files.
RERUN_INTEGRATION = False
# ==============================================================================
# Folder where the simulation data (integrated ephemerides) will be saved, only 
# if RERUN_INTEGRATION is set to True; if False then the script will look for 
# the required data in the folder. Each ephemeris will be saved in a separate 
# text file whose name includes the body ID and name.
simulation_save_folder = 'barycenter_sun_simulation'
SAVE_SIMULATION_DATA = False

# ------------------------------------------------------------------------------
# Whether to re-download ephemerides (from JPL Horizons via the URL GET API, 
# documentation here: https://ssd-api.jpl.nasa.gov/doc/horizons.html) for the 
# bodies defined in the simulation (with conditions like start/end time 
# specified later in the script). The data will be downloaded in the folders 
# described below. Any ephemeris where the body ID matches will be overwritten.
# If REDOWNLOAD_REQUIRED_DATA is False, the script assumes that the required 
# data is already in the specified folders.
REDOWNLOAD_REQUIRED_DATA = False # must reset to False to run simulation

# This folder stores text files with tabulated data about the system bodies,
# as well as folders with the (observed) body ephemerides.
data_main_folder = 'barycenter_sun_simulation'

# Folders within 'data_main_folder' for the (observed) body ephemerides.
planet_ephemerides = 'planet_ephemerides' # n.b. includes the Sun ID#10
sun_ephemeris = 'sun_ephemeris'

# The body names, Horizons IDs, and masses (or GM values) are tabulated in the 
# text files (named below) in the folder defined by 'data_main_folder'.
planets_file_name = 'planetary_physical_parameters.txt' # n.b. includes the Sun
# ==============================================================================

def simulation_main():
    '''Main function for the simulation.'''

    # Download ephemerides
    # ====================
    # 1. Define the ephemeris parameters:
    # -----------------------------------
    # Ephemerides for the starting conditions only
    # --------------------------------------------
    # Start and stop time as Julian dates or calendar dates (c.f. Horizons API)
    # JD/time converter at: https://ssd.jpl.nasa.gov/tools/jdc/#/cd
    start_time = '1950-Jan-01'
    stop_time =  '1950-Jan-02'
    start_time_JD = float(2433282.5000000) # Julian date of start_time
    step_size = 1
    step_size_api = str(step_size)+'d'
    t_precision = 'MINUTES'
    out_units='AU-D'

    # Longer interval for the Sun ephemeris to compare with simulation
    # ----------------------------------------------------------------
    start_time_sun = '1950-Jan-01'
    stop_time_sun =  '2004-Oct-04' # JD 2453282.5; 20000 day range
    start_time_JD_sun = float(2433282.5000000) # Julian date of start_time
    step_size_sun = 100
    step_size_api_sun = str(step_size_sun)+'d'
    t_precision_sun = 'MINUTES'
    out_units_sun='AU-D'

    if REDOWNLOAD_REQUIRED_DATA:
        # Instantiate new Simulation object
        sim1 = simulation_module.Simulation()

        # 2. Define the system bodies
        # ----------------------------------------------------------------------
        # The body names, Horizons IDs, and masses (or GM values) are tabulated 
        # in the text files in the folder defined by 'data_main_folder'
        loaded_planets = sim1.load_physical_params_from_file(data_main_folder, 
                planets_file_name, lf_usecols=(0,1,2), lf_skiprows=2, delimiter=',')
        if loaded_planets is False:
            print('SIMULATION MAIN: Unable to load planets.')
            return False
        
        # 3. Download ephemerides
        # ---------------------------------------------------------------------
        # Ephemerides for the planets
        status = sim1.download_ephemerides(loaded_planets, data_main_folder, 
                planet_ephemerides, start_time, stop_time, 
                step_size=step_size_api, t_precision=t_precision, 
                out_units=out_units)
        if status is False:
            print('SIMULATION MAIN: Download of planet ephemerides failed.')
            return False
        
        # Longer ephemeris for the Sun
        status = sim1.download_ephemerides({10:'Sun'}, data_main_folder, 
                sun_ephemeris, start_time_sun, stop_time_sun, 
                step_size=step_size_api_sun, t_precision=t_precision_sun, 
                out_units=out_units_sun)
        if status is False:
            print('SIMULATION MAIN: Download of Sun ephemeris failed.')
            return False
        print('DATA DOWNLOADED. Set REDOWNLOAD_REQUIRED_DATA to False to run simulation.')
        return True

    # Begin simulation
    # ================
    # Instantiate/Reset Simulation object
    sim1 = simulation_module.Simulation()

    # Define/Redefine the system bodies
    # ----------------------------------------------------------------------
    # The body names, Horizons IDs, and masses (or GM values) are tabulated 
    # in the text files in the folder defined by 'data_main_folder'
    loaded_planets = sim1.load_physical_params_from_file(data_main_folder, 
            planets_file_name, lf_usecols=(0,1,2), lf_skiprows=2, delimiter=',')
    if loaded_planets is False:
        print('SIMULATION MAIN: Unable to load planets.')
        return False
        
    # Load the ephemerides into the Simulation object
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

    # Change units if needed
    # -------------------------
    CHANGE_UNITS = False
    if CHANGE_UNITS:
        # change the units in the ephemerides
        pos_cf = 1 # conversion factor for the position
        vel_cf = 1 # conversion factor for the velocity
        status = sim1.change_ephemerides_units(pos_cf, vel_cf)
        if status is False:
            print('SIMULATION MAIN: Changing units failed.')
            return False
        # change the units of the body masses from kg to those used in Model
        for i, mass in enumerate(sim1.bodies['mass']):
            if mass is not None:
                sim1.bodies['mass'][i] = mass / mdl.Model.M
    
    # Begin the simulation
    # ----------------------------------------------------------------------
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
    t_end = 20000 # days
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
                t_step, rtol=rtol, atol=atol, tcrit=tcrit, h0=h0, 
                hmax=hmax, use_vect_ops_func=True, use_solve_ivp_API=True, 
                save_data=SAVE_SIMULATION_DATA, 
                save_dir=simulation_save_folder)
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

            # instantiate a Model object from the loaded data
            sim1.model_obj_from_data(mdl.Model)
            sim1.times = sim1.model.times
            sim1.num_points = np.size(sim1.times)
        else:
            # This run was only to download data
            return True

    # Plots
    # =====
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=16)
    
    # Figure 6
    # --------
    # Sun trajectory wrt barycenter
    # -----------------------------
    sun_id = 10
    # sun_index = sim1.bodies['id'].index(sun_id)
    x = sim1.sim_pos(sun_id, 'x')
    y = sim1.sim_pos(sun_id, 'y')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1, left=0.1)
    
    # Plot the integrated positions
    x = sim1.sim_pos(sun_id, 'x')
    y = sim1.sim_pos(sun_id, 'y')
    ax.plot(x, y, 'r-', label='simulated')

    # Get the observed Sun trajectory from the long ephemeris
    # -------------------------------------------------------
    sun_obj = simulation_module.Simulation()
    status = sun_obj.load_ephemerides(data_main_folder, sun_ephemeris)
    if status is False:
        print('SIMULATION MAIN: Failed loading Sun ephemeris into memory.')
        return False
    # plot the observed positions except initial and last
    x_obs = sun_obj.obs_pos(sun_id, 'x')[1:-1]
    y_obs = sun_obj.obs_pos(sun_id, 'y')[1:-1]
    ax.plot(x_obs, y_obs, 'b.', label='observed')
    # plot the intial observed position
    x_obs = sun_obj.obs_pos(sun_id, 'x')[0]
    y_obs = sun_obj.obs_pos(sun_id, 'y')[0]
    ax.plot(x_obs, y_obs, 'go')
    # plot the last observed position
    x_obs = sun_obj.obs_pos(sun_id, 'x')[-1]
    y_obs = sun_obj.obs_pos(sun_id, 'y')[-1]
    ax.plot(x_obs, y_obs, 'ko')
    ax.set_xlabel('$x$ [AU]')
    ax.set_ylabel('$y$ [AU]')
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper left', fontsize=14)

    # Figure 7
    # --------
    # Relative error plots
    # --------------------------------------------------------------------------
    # Calculate the position relative errors
    # --------------------------------------
    id = 10 # Sun Horizons ID
    step_size = step_size_sun
    R = sim1.sim_pos(id, 'all')[0::step_size]
    R_obs = sun_obj.obs_pos(id, 'all')[0:]
    R_relative_error = (R-R_obs)/R_obs
    times_R_rel = sim1.times[0::step_size]

    # Plot total energy relative error
    # --------------------------------
    fig, (ax1, ax2) = plt.subplots(2,1, sharex='all', figsize=(10, 8))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1, left=0.1)
    
    E_total_rel_error = sim1.model.E_total_rel_error()
    if E_total_rel_error is not False:
        ax1.plot(sim1.model.times, E_total_rel_error)
        ax1.set_ylabel('Total energy relative error')
    
    # Plot position relative errors
    # -----------------------------
    x_err = R_relative_error[:,0]
    y_err = R_relative_error[:,1]
    z_err = R_relative_error[:,2]

    ax2.plot(times_R_rel, x_err, '.-', ms=4, label='$x$')
    ax2.plot(times_R_rel, y_err, '.-', ms=4, label='$y$')
    ax2.plot(times_R_rel, z_err, '.-', ms=4, label='$z$')
    ax2.set_xlabel('Time [days]')
    ax2.set_ylabel('Position relative error')
    ax2.legend(fontsize=14)

    return sim1

sim1 = simulation_main()
