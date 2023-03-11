import os
import requests
import numpy as np
from pandas import read_csv


class Simulation:
    '''
    Defines properties and method used in most of the simulations.
    Does not define the physical models -- these are defined in the
    model_module module.
    
    Includes methods to make HTTP GET queries to the JPL Horizons system
    in order to download required ephemeris data. Stores the data returned
    by the Model/ModelApproximate object defined in model_module for easy 
    plotting, and saving to a file.
    '''
    def __init__(self, name=None):
        self.sim_name = name # simulation name
        self.model = None # this stores the Model object
        self.model_approx = None # this stores the ModelApproximate object
        self.integration_info = None # integration info returned by the solver
        # this dictionary holds info about the system and the ephemeris data
        # R0/V0 and t0 store the initial conditions for the simulation
        self.bodies = {
            'id' : [],
            'name' : [],
            'mass' : [],
            'obj_type' : [],
            'ephemeris' : [],
            'R0' : [],
            'V0' : [],
            't0' : None,
            'integrated_ephem' : []
        }
        self.t_start = None # integration start time
        self.t_end = None # integration end time (included)
        self.t_step = None # integration time step
        self.num_points = None # evenly spaced between t_start/end
        self.times = None # the times defined by the above four vars


    def load_physical_params_from_file(self, data_main_folder, list_fname, 
            lf_usecols=(0,1,2), delimiter='\t', lf_skiprows=1, lf_skipfooter=0, comment='#', GM_instead_of_mass=False, G_val=None):
        '''
        Loads body data, ID, name and mass from data stored in a file.

        Arguments:
        - data_main_folder (str): name of the directory
        - list_fname (str): file name of data file
        - lf_usecols (tuple/list): columns to use. The data should be ordered
        as so: col 0: body name, col 1: body id (unique), col 2: mass
        - delimiter (str): delimiter, e.g. ','
        - lf_skiprows (int): rows to skip from the top of the file. Includes comments (#) but not blank lines.
        - lf_skipfooter (int): rows to skip from the bottom of the file
        - comment (str): comment line; skipped
        - GM_instead_of_mass (bool): if GM values are listed instead of masses,
        set to True and provide a value for G_val
        - G_val (float): if GM values used, conversion factor to apply to the 
        masses column

        Effect: Defines a unique body in the self.bodies dictionary and copies
        the available data into it, setting None for the rest of the fields.

        Returns: dict(zip(loaded_body_ids, loaded_body_names)). A dictionary
        with body IDs as keys and body names as values. Useful for passing on
        to the download_ephemerides() method which takes a dictionary.
        '''

        fpath = os.path.join(data_main_folder, list_fname)
        if not os.path.isfile(fpath):
            print('load_physical_params_from_file call: File does not exist.')
            return False

        loaded_params = read_csv(fpath, usecols=lf_usecols, delimiter=delimiter,
            skiprows=lf_skiprows, skipfooter=lf_skipfooter, na_filter=False, 
            index_col=False, header=None, engine='python', comment=comment)

        inx_names = lf_usecols[0]
        inx_ids = lf_usecols[1]
        inx_masses = lf_usecols[2]
        loaded_body_names = loaded_params[inx_names].tolist()
        loaded_body_ids = loaded_params[inx_ids].tolist()
        loaded_masses = loaded_params[inx_masses]

        if GM_instead_of_mass and G_val is not None:
            loaded_masses = loaded_masses / G_val
        loaded_masses = loaded_masses.to_numpy()

        # check that there are no missing values
        sizes_set = set([len(loaded_body_ids), len(loaded_body_names), 
                len(loaded_masses)])
        if len(sizes_set) != 1:
            print('load_physical_params_from_file call: There were missing values in the data. Data not loaded.')
            return False
        if len(loaded_body_ids) == 0:
            print('ALERT: load_physical_params_from_file call: Body list is empty. Returned False')
            return False

        for i, id in enumerate(loaded_body_ids):
            if int(id) not in self.bodies['id']:
                self.bodies['id'].append(int(loaded_body_ids[i]))
                self.bodies['name'].append(loaded_body_names[i])
                self.bodies['mass'].append(loaded_masses[i])
                self.bodies['obj_type'].append(None)
                self.bodies['ephemeris'].append(None)
                self.bodies['R0'].append(None)
                self.bodies['V0'].append(None)
                self.bodies['integrated_ephem'].append(None)
            else:
                body_index = self.bodies['id'].index(int(id))
                self.bodies['mass'].append(loaded_masses[i])

        # return the loaded IDs and names for later use
        return dict(zip(loaded_body_ids, loaded_body_names))


    def get_body_list_from_file(self, data_main_folder, list_fname, 
            lf_usecols=(0,1), delimiter=',', lf_skiprows=1, lf_skipfooter=0, 
            comment='#'):
        '''
        Produces a dictionary of body IDs and names from a list in a file.
        Useful for passing on to the download_ephemerides() method which takes 
        a dictionary.

        Arguments:
        - data_main_folder (str): name of the directory
        - list_fname (str): file name of data file
        - lf_usecols (tuple/list): columns to use. The data should be ordered
        as so: col 0: body name, col 1: body id (unique)
        - delimiter (str): delimiter, e.g. ','
        - lf_skiprows (int): rows to skip from the top of the file. Includes comments (#) but not blank lines.
        - lf_skipfooter (int): rows to skip from the bottom of the file
        - comment (str): comment line; skipped

        Returns: dict(zip(loaded_body_ids, loaded_body_names)). A dictionary
        with body IDs as keys and body names as values. Useful for passing on
        to the download_ephemerides() method which takes a dictionary.        
        '''

        fpath = os.path.join(data_main_folder, list_fname)
        if not os.path.isfile(fpath):
            print('load_physical_params_from_file call: File does not exist.')
            return False

        loaded_data = read_csv(fpath, usecols=lf_usecols, delimiter=delimiter,
            skiprows=lf_skiprows, skipfooter=lf_skipfooter, na_filter=False, 
            index_col=False, header=None, engine='python', comment=comment)

        inx_names = lf_usecols[0]
        inx_ids = lf_usecols[1]
        loaded_body_names = loaded_data[inx_names].tolist()
        loaded_body_ids = loaded_data[inx_ids].tolist()

        # return the loaded IDs and names for later use
        return dict(zip(loaded_body_ids, loaded_body_names))


    def get_body_list_loaded(self):
        '''
        As get_body_list_from_file(), but produces the dictionary from the 
        bodies that are already loaded in self.bodies.
        '''
        body_ids = self.bodies['id']
        body_names = self.bodies['name']
        if len(body_ids) != len(body_names):
            print('INTERNAL ERROR: get_body_list_loaded call: Sizes of self.bodies[\'id\'] and self.bodies[\'name\'] do not match')
            return False
        if len(body_ids) == 0:
            print('ALERT: get_body_list_loaded call: Body list is empty. Returned None')
            return None
        return dict(zip(body_ids, body_names))


    def make_Horizons_API_query(self, api_parameters):
        '''
        Method used internally by download_MBs_list_to_file() and
        download_ephemerides() to connect to the JPL Horizons system
        via the HPPT GET API and make the respective requests. However,
        may be called manually by passing api_parameters which defines
        the Horizons API query as a dictionary in the format:
            Horizons API parameter : value
        The URL string is URL-encoded by the requests module.
        JPL Horizons API documentation at:
        https://ssd-api.jpl.nasa.gov/doc/horizons.html

        Returns: The API response
        '''
        # Horizons API URL
        api_url = 'https://ssd.jpl.nasa.gov/api/horizons.api'

        # make the GET request
        response = requests.get(api_url, params=api_parameters)

        # check that the request was OK
        if not response.status_code == requests.codes.ok:
            # if bad response print status and exit
            print(response.raise_for_status())
            return False

        # copy the response content into a python dictionary
        api_response = response.json()

        # check the current version of the Horizons API
        if api_response['signature']['version'] != '1.1':
            print('ALERT: Horizons API version has changed!')
            print('Current version: '+str(api_response['signature']['version']))
            print('This app was built using version 1.1. Check that the parameters used are still OK.')
        
        # check that the API did not return an error
        if 'error' in api_response:
            print('ALERT: make_Horizons_API_query call: The Horizons API returned an error:')
            print(api_response['error'])
            return False
        
        # All is OK
        return api_response


    def download_MBs_list_to_file(self, data_main_folder):
        '''
        Uses the method make_Horizons_API_query() to a list of all 'Major 
        Bodies' in the JPL Horizons system. N.B includes dynamical points 
        (system barycenters, Lagrangian points).
        '''
        major_bodies = self.make_Horizons_API_query({'COMMAND':'MB'})
        if not major_bodies:
            return False
        
        # write list of all MB objects to a text file
        # -------------------------------------------
        if not os.path.isdir(data_main_folder):
            os.mkdir(data_main_folder)

        fname = 'all_Horizons_major_bodies.txt'

        fpath = os.path.join(data_main_folder, fname)
        with open(fpath, 'w') as f:
            f.write(major_bodies['result'])
        return True


    def download_ephemerides(self, download_bodies, data_main_folder, 
            data_subfolder, start_time, stop_time, step_size='1d', 
            t_precision='MINUTES', out_units='AU-D'):
        '''
        Make an API query to the JPL Horizons system to download ephemeris
        data for the bodies defined in download_bodies between the start and 
        stop times specified. Uses make_Horizons_API_query() to connect to the
        API.

        Some ephemeris parameters are specified within the method in the 
        dictionary api_parameters -- such as reference frame, coordinate origin,
        etc. c.f. API documentation.

        JPL Horizons API documentation at:
        https://ssd-api.jpl.nasa.gov/doc/horizons.html

        Arguments:
        - download_bodies (dict): Horizons body ID : body name
        The value in key should be the Horizons body ID but in practice 
        could be any value to pass to the API COMMAND parameter,
        for example an objects primary designation if querying for small
        bodies.
        - data_main_folder (str): the name of the data folder where ephemerides
        will be downloaded within a sub-folder data_subfolder
        - data_subfolder (str)
        - start_time (str): The ephemeris start time. Could be a calendar data
        yyyy-mon-dd or a Julian date. c.f. the API docs
        - stop_time (str): as start_time
        - step_size (str): the time between the data points returned.
        Could be 'days'/'d', 'months'/'mo', etc. If no unit is specified, the
        stop-start period is divided evenly between the step_size value
        - t_precision (str): time precision of the data
        - out_units (str): the position and velocity units. AU and days by 
        default.

        Effect: Downloads ephemerides for the specified bodies, between the 
        requested times. Some ephemeris parameters are specified within the
        method in the dictionary api_parameters.
        '''

        for body_id, body_name in download_bodies.items():
            # Download the ephemeris (and mass) data from JPL Horizons
            # --------------------------------------------------------
            # set up the API parameters passed with the GET request
            command = str(body_id)

            api_parameters = {
                'format' : 'json',
                'COMMAND' : command,
                'OBJ_DATA' : 'NO',
                'MAKE_EPHEM' : 'YES',
                'EPHEM_TYPE' : 'VECTORS',
                'CENTER' : '500@0',
                'REF_PLANE' : 'ECLIPTIC',
                'START_TIME' : start_time,
                'STOP_TIME' : stop_time,
                'STEP_SIZE' : step_size,
                'REF_SYSTEM' : 'ICRF',
                'OUT_UNITS' : out_units,
                'VEC_TABLE' : '2',
                'VEC_CORR' : 'NONE',
                'TIME_DIGITS' : t_precision,
                'CSV_FORMAT' : 'YES',
                'VEC_LABELS' : 'NO'
            }

            api_response = self.make_Horizons_API_query(api_parameters)
            if not api_response:
                return False

            # Write the ephemeris data to a text file
            # ---------------------------------------
            dpath = os.path.join(data_main_folder, data_subfolder)
            if not os.path.isdir(dpath):
                os.makedirs(dpath)

            fname = 'ID'+str(body_id)+'_'+str(body_name)+'.txt'

            fpath = os.path.join(data_main_folder, data_subfolder, fname)
            with open(fpath, 'w') as f:
                f.write(api_response['result'])
            print('DOWNLOADED: ID# '+str(body_id)+' '+str(body_name))
        return True


    def load_ephemerides(self, data_main_folder, data_subfolder=None, load_order=[], small_body=False, satellite=False):
        '''load_order is IDs'''

        if data_subfolder is not None:
            ephemerides_path = os.path.join(data_main_folder, data_subfolder)
        else:
            ephemerides_path = data_main_folder

        # list of file names
        if not os.path.isdir(ephemerides_path):
            print('load_ephemerides call: The folder ', end='')
            print(ephemerides_path, end='')
            print(' does not exist')
            return False
        ephemerides_fnames = os.listdir(ephemerides_path)

        # number of bodies
        N = len(ephemerides_fnames)
        if N == 0:
            print('load_ephemerides call: No data was found in the specified folder: '+str(ephemerides_path))
            return False

        # test for uniquenes: file names in ephemerides folder
        if not len(set(ephemerides_fnames)) == N:
            print('load_ephemerides call: The file names in the ephemerides folder are not unique.')
            return False
        
        # test for uniquenes: body IDs in load_order list
        if not len(set(load_order)) == len(load_order):
            print('load_ephemerides call: The body IDs in the requested load_order list are not unique.')
            return False
        
        # produce list of body IDs in the requested order
        # -----------------------------------------------
        # get the body IDs and names from file names in the ephemerides folder
        unordered_ids = []
        unordered_names = []
        for fname in ephemerides_fnames:
            id = fname.split('_')[0]
            id = id.split('ID')[1]
            name = fname.split('.')[0]
            name = name.split('_')[1]
            unordered_ids.append(int(id))
            unordered_names.append(name)

        # n.b. load_order may also be empty
        ordered_names = [None]*len(load_order) # initialize
        for index_i, id_i in enumerate(load_order):
            id_i = int(id_i)
            if id_i not in unordered_ids:
                print('load_ephemerides call: A body in the load_order list is missing from the ephemerides folder.')
                return False
            for index_j, id_j in enumerate(unordered_ids):
                if id_i == id_j:
                    unordered_ids.pop(index_j)
                    name = unordered_names.pop(index_j)
                    ordered_names[index_i] = name
        ordered_ids = load_order + unordered_ids
        ordered_names = ordered_names + unordered_names
        ordered_fnames = list(map(lambda a,b: 'ID'+str(a)+'_'+str(b)+'.txt', 
                                            ordered_ids, ordered_names))
        
        # get the number of state vectors
        # -------------------------------
        '''The number of state vectors (for the same times), between the tags $$SOE and $$EOE (format used by JPL Horizons), must be the same in all the ephemerides. These may be positioned at different lines in the text depending on the size of the headers (summary info) at the start of the files.'''
        # use one of the files to find the number of state vectors
        fpath = os.path.join(ephemerides_path, ordered_fnames[0])
        if os.path.isfile(fpath):
            with open(fpath, 'r') as f:
                SOE_line_number = 0
                EOE_line_number = 0
                current_line_num = 0
                for line in f:
                    current_line_num += 1
                    if line in ['$$SOE\n']:
                        SOE_line_number = current_line_num
                    if line in ['$$EOE\n']:
                        EOE_line_number = current_line_num
            num_times = (EOE_line_number-1) - SOE_line_number

        # initialize arrays
        loaded_body_ids = []
        loaded_body_names = []
        loaded_ephemerides = np.zeros((num_times, 7, N), dtype=np.float64)

        # load the data into the initialized arrays
        for i, fname in enumerate(ordered_fnames):
            fpath = os.path.join(ephemerides_path, fname)
            if os.path.isfile(fpath):
                id = fname.split('_')[0]
                id = id.split('ID')[1]
                name = fname.split('.')[0]
                name = name.split('_')[1]
                loaded_body_ids.append(int(id))
                loaded_body_names.append(name)
                
                # read the ephemeris
                # ------------------
                # the position of the $$SOE tag may vary depending on the size # of the header (mainly due object summary) so a fixed size of 
                # 'skiprows' is not reliable
                with open(fpath, 'r') as f:
                    SOE_line_number = 0
                    EOE_line_number = 0
                    current_line_num = 0
                    for line in f:
                        current_line_num += 1
                        if line in ['$$SOE\n']:
                            SOE_line_number = current_line_num
                        if line in ['$$EOE\n']:
                            EOE_line_number = current_line_num
                skip_rows = SOE_line_number
                max_rows = (EOE_line_number-1) - skip_rows

                # check that the number of times is the same as those assumed
                if max_rows != num_times:
                    # stop and exit the method call
                    print('load_ephemerides call: The number of state vectors in one of the ephemerides is different from those in the others.')
                    return False

                loaded_ephemerides[:,:,i] = np.loadtxt(fpath, dtype=np.float64, 
                    delimiter=',', skiprows=skip_rows, usecols=(0,2,3,4,5,6,7), 
                    max_rows=max_rows)
        
        # check that the times between all the bodies are the same
        for i in range(num_times):
            if not np.all(loaded_ephemerides[i,0,:]):
                # stop and exit the method call
                    print('load_ephemerides call: The times (dates) for the data does not match between all the ephemerides.')
                    return False

        # Add the ephemeris data to the bodies dictionary
        # -----------------------------------------------
        # check that the existing data is of the right size and the times match # up, and reset the existing ephemeris data if not
        if len(self.bodies['ephemeris']) != 0:
            # compare any two (existing and loaded)
            ephemeris_1 = None
            for eph in self.bodies['ephemeris']:
                if eph is not None:
                    ephemeris_1 = eph
                    break
            # if ephemeris_1 is None then there is no data
            if ephemeris_1 is not None:
                ephemeris_2 = loaded_ephemerides[:,:,0]
                # case: data exits but wrong size (number of state vectors)
                size_1 = (np.size(ephemeris_1,0), np.size(ephemeris_1,1))
                size_2 = (np.size(ephemeris_2,0), np.size(ephemeris_2,1))
                if size_1 != size_2:
                    print('ALERT: load_ephemerides call: Cannot join ephemerides with those loaded before (due to different numbers of state vectors). The old ephemerides were deleted from memory (but not from the text files).')
                    self.bodies['ephemeris'] = [None]*len(self.bodies['id'])
            
                # case: data exists and same numbers of state vectors but the times do not match up
                times_1 = ephemeris_1[:,0]
                times_2 = ephemeris_2[:,0]
                if not np.all(times_1 == times_2):
                    print('ALERT: load_ephemerides call: Cannot join ephemerides with those loaded before (the times do not match up). The old ephemerides were deleted from memory (but not from the text files).')
                    self.bodies['ephemeris'] = [None]*len(self.bodies['id'])

        # case: no data OR data exits and times match up. Add the data
        # n.b. overides if body ID# matches
        for i, id in enumerate(loaded_body_ids):
            if int(id) not in self.bodies['id']:
                self.bodies['id'].append(int(loaded_body_ids[i]))
                self.bodies['name'].append(loaded_body_names[i])
                self.bodies['mass'].append(None)
                if small_body:
                    self.bodies['obj_type'].append('SB')
                elif satellite:
                    self.bodies['obj_type'].append('SAT')
                else:
                    self.bodies['obj_type'].append(None)
                self.bodies['ephemeris'].append(loaded_ephemerides[:,:,i])
                self.bodies['R0'].append(None)
                self.bodies['V0'].append(None)
                self.bodies['integrated_ephem'].append(None)
            else:
                body_index = self.bodies['id'].index(int(id))
                self.bodies['ephemeris'][body_index] = loaded_ephemerides[:,:,i]
                if small_body:
                    self.bodies['obj_type'][body_index] = 'SB'
                elif satellite:
                    self.bodies['obj_type'][body_index] = 'SAT'
                else:
                    self.bodies['obj_type'][body_index] = None
        return True


    def change_ephemerides_units(self, pos_cf=1, vel_cf=1):
        '''
        Change the units of the imported data to those used in the simulation. 
        N.B. This change applies to all the data from all calls of load_ephemerides(), so only run once. All ephemerides are assumed 
        to have the same units.

        Arguments:
        pos_cf (float/int): Position conversion factor for the new units
        vel_cf (float/int): Velocity conversion factor for the new units
        '''
        # checks
        input_types = set([type(pos_cf), type(vel_cf)])
        if not input_types <= set([float, int]):
            print('change_ephemerides_units call: Arguments have to be type float or int.')
            return False
        
        for ephemeris in self.bodies['ephemeris']:
            if ephemeris is not None:
                ephemeris[:,[1,2,3]] *= pos_cf
                ephemeris[:,[1,2,3]] *= vel_cf
        return True


    def set_ICs_from_ephemerides(self, t_initial):
        '''
        Set the initial conditions for the simulation from ephemerides data, 
        where the time matches t_initial.

        Arguments:
        t_initial (float/int): The time in the ephemeris from which to set the
        initial conditions. The precision must match that in the ephemeris.
        
        Effect: Sets the initial conditions for the simulation, saving to
        self.bodies['R0]/self.bodies['V0] and self.bodies['t0]
        '''
        # checks
        if not isinstance(t_initial, (float, int)):
            print('set_ICs_from_ephemerides call: t_initial has to be a float or int. The precision must match that in the ephemeris.')
            return False
        if len(self.bodies['ephemeris']) == 0:
            print('set_ICs_from_ephemerides call: Run load_ephemerides() first')
            return False

        # get index for the requested time
        # n.b. checks that the times match up between different bodies are made
        # in the load_ephemerides() calls
        # get any ephemeris
        for i, id in enumerate(self.bodies['id']):
            if self.bodies['mass'][i] is not None:
                if self.bodies['ephemeris'][i] is not None:
                    ephemeris = self.bodies['ephemeris'][i] # any is OK
                    break
        times_slice = ephemeris[:,0]
        t_index = np.argwhere(times_slice == t_initial)
        if np.shape(t_index) != (1, times_slice.ndim):
            print('set_ICs_from_ephemerides call: requested t_initial does not exist or is not unique. t_initial must be as recorded in the ephemerides.')
            return False
        t_index = t_index[0,0]

        # set the initial conditions for bodies where the mass and ephemeris
        # are defined
        for i, id in enumerate(self.bodies['id']):
            if self.bodies['mass'][i] is not None:
                if self.bodies['ephemeris'][i] is not None:
                    ephemeris = self.bodies['ephemeris'][i]
                    self.bodies['R0'][i] = ephemeris[t_index,[1,2,3]]
                    self.bodies['V0'][i] = ephemeris[t_index,[4,5,6]]
        self.bodies['t0'] = t_initial
        return True


    def add_body_to_ICs(self, id, name, mass, R0, V0, t_initial):
        '''
        Additional method for adding initial conditions to the system.
        Run method after running set_ICs_from_ephemerides() or on its own to 
        build up the system.

        Arguments:
        - id (int)): body id
        - name (str): body name
        - R0/V0: Numpy array, shape (3,). The initial conditions for the body,
        as np.array([x,y,z])
        - mass: Numpy array, shape (1,) (float) or scalar float. Body mass as
        np.array([m]). This structure is needed to ensure merge with existing
        data is OK.
        '''
        # checks
        if not isinstance(t_initial, (float, int)):
            print('set_ICs_from_ephemerides call: t_initial has to be a float or int. The precision of the requested value must match that in the data.')
            return False
        if not isinstance(self.bodies['t0'], type(None)):
            if self.bodies['t0'] != t_initial:
                print('add_body_to_ICs call: The times (dates) must match for all R0 and V0. t_initial is different to the recorded value.')
                return False
        # add to initial conditions
        if int(id) not in self.bodies['id']:
            self.bodies['id'].append(int(id))
            self.bodies['name'].append(name)
            self.bodies['mass'].append(mass)
            self.bodies['obj_type'].append(None)
            self.bodies['ephemeris'].append(None)
            self.bodies['R0'].append(R0)
            self.bodies['V0'].append(V0)
            self.bodies['integrated_ephem'].append(None)
        else:
            index = self.bodies['id'].index(int(id))
            self.bodies['R0'][index] = R0
            self.bodies['V0'][index] = R0
            print('ALERT: add_body_to_ICs call: Body ID# exists. Updating R0 and V0 only.')
        self.bodies['t0'] = t_initial
        return True


    def run_simulation(self, model_class, t_start, t_end, t_step, rtol=None, 
            atol=None, tcrit=None, h0=0.0, hmax=0.0, use_vect_ops_func=True, 
            use_solve_ivp_API=False, save_data=True, save_dir=None):
        '''
        This method runs the simulation from the initial conditions defined
        in the self.bodies['R0']/self.bodies['V0'] between the start and 
        end times provided. The integration itself is done by the Model
        object, which returns the integrated data to be stored in the
        self.bodies['integrated_ephem] field.

        Two implementations of integrator are available -- the older odeint
        or solve_ivp which are both API to the same FORTRAN libraries but one
        may be more convenient than the other sometimes. Scipy documentation
        recommends using solve_ivp for new code.

        Arguments:
        - model class (reference): a reference to the Model class which is
        defined in model_module.py and has the functions defining the physical
        model
        - save_data (bool): whether to save the integration data (within
        self.bodies['integrated_ephem']) to data files within a folder
        'integrated ephemerides' within the folder save_dir
        - save_dir (str): folder name
        - t_step (float): the time step. This is used to define
        self.num_points = int(((t_end-t_start)/t_step)+1), which is the variable
        actually passed to the integrator and defines the spacing of the time 
        points via numpy.linspace

        These descriptions are the same as in Model class solve_system()
        in model_module.py:
        - t_start (scalar): integration start time
        - t_end (scalar): integration end time (included in the range)
        - rtol (scalar): the relative error tolerance limit. Default, None, 
        means it is set by the integrator (different for odeint and solve_ivp)
        - atol (scalar): the absolute error tolerance limit. Default - as above
        - tcrit (array like): Used by the odeint integrator only. Critical times
        where integration care should be taken.
        - h0 (scalar): the (time) size of the first integration step. Default 
        zero means the integrator chooses freely.
        - hmax (scalar): the maximum (time) size of the integration steps. 
        Default zero means that the integrator chooses this dynamically based 
        on the set error (rtol, atol) tolerances.
        - use_vect_ops_func (bool): whether to use the implementation of 
        model_func that uses numpy vectorized operations or that which uses 
        Python loops.
        - use_solve_ivp_API (bool): whether to use the odeint or solve_ivp API.
        Both have been set-up to use the LSODA method which calls the same 
        FORTRAN libraries, and both should produce the same results, but one 
        may be more convenient that the other in certain circumstances.
        Scipy documentation recommends using solve_ivp for new code.

        Effect: Saves the integrated data within self.bodies['integrated_ephem']
        for each body. Save the data a seried of files (ID----_name.txt) within
        the folder specified.
        '''

        # re-format the data to pass on to the Model object
        # -------------------------------------------------
        # Required format
        # body_ids/body_names: Python list
        # masses: Numpy array, shape (N,)
        # R0/V0: Numpy array, shape (N, 3)
        body_ids = []
        body_names = []
        masses = []
        R0 = []
        V0 = []
        for i, id in enumerate(self.bodies['id']):
            if self.bodies['mass'][i] is not None:
                if self.bodies['R0'][i] is not None:
                    if self.bodies['V0'][i] is not None:
                        body_ids.append(int(self.bodies['id'][i]))
                        body_names.append(self.bodies['name'][i])
                        masses.append(self.bodies['mass'][i])
                        R0.append(self.bodies['R0'][i])
                        V0.append(self.bodies['V0'][i])
        if len(R0) == 0:
            print('run_simulation call: Initial conditions have not been defined')
            return False, None
                
        masses = np.array(masses, dtype=np.float64)
        if masses.ndim == 2:
            masses = np.squeeze(masses)
        R0 = np.array(R0, dtype=np.float64)
        V0 = np.array(V0, dtype=np.float64)
        if np.shape(R0) != (np.size(masses), 3):
            print('INTERNAL ERROR: run_simulation call: The shape of R0 ('+str(np.shape(R0))+') is not the expected shape (N, 3) (required by Model object)')
            return False, None

        # initialize the Model object
        self.model = model_class(body_ids, body_names, masses, R0, V0)

        # run the simulation using the solve_system() method of Model object
        # see model_module.py
        self.t_start = t_start
        self.t_end = t_end
        self.t_step = t_step
        self.num_points = int(((t_end-t_start)/t_step)+1)

        solution, info = self.model.solve_system(t_start, t_end, 
            self.num_points, rtol=rtol, atol=atol, tcrit=tcrit, h0=h0, 
            hmax=hmax, use_vect_ops_func=use_vect_ops_func, 
            use_solve_ivp_API=use_solve_ivp_API)
        
        if not isinstance(solution, np.ndarray):
            print('INTEGRATION FAILED')
            return False, info

        print('INTEGRATION ENDED')
        self.integration_info = info
        del solution # the data is in the Model object

        if save_data:
            print('SAVING DATA')

        # copy the data to the Simulation object (having the data in the Sim. 
        # object will make it easier to reload the data from a saved text file)
        for i, id in enumerate(self.model.body_ids):
            position = np.swapaxes(self.model.position(int(id), 'all'), 0,1)
            velocity = np.swapaxes(self.model.velocity(int(id), 'all'), 0,1)
            times = self.model.times[:, np.newaxis]
            # shapes: position/velocity (size(times), 3), times (num times, 1)

            index_body = self.bodies['id'].index(int(id))
            self.bodies['integrated_ephem'][index_body] = np.concatenate(
                (times, position, velocity), axis=1, dtype=np.float64)

            # save the data to a text file
            if save_data:
                if save_dir is not None:
                    save_dir = str(save_dir)
                    dir_path = os.path.join(save_dir, 'integrated_ephemerides')
                    if not os.path.isdir(dir_path):
                        os.makedirs(dir_path)
                else:
                    dir_path = 'integrated_ephemerides'
                    if not os.path.isdir(dir_path):
                        os.mkdir(dir_path)
                                    
                fname = 'ID'+str(id)+'_'+str(self.model.body_names[i])+'.txt'
                save_path = os.path.join(dir_path, fname)

                # n.b. the decimal places in fmt
                np.savetxt(save_path, 
                    self.bodies['integrated_ephem'][index_body], fmt='%.18e', 
                    delimiter=',')
        return True, info


    def model_obj_from_data(self, model_class, approx_model=False, 
            integrated_data=True):
        '''
        If loading saved data from a file using load_sim_from_dir() (below)
        use this method to create a Model object for the data, in order to
        allow energy calculations. Model/ModelApproximate classes are in
        model_module.py

        Arguments:
        - model class (reference): a reference to the Model class which is
        defined in model_module.py and has the functions defining the physical
        model
        - approx_model (bool): whether this data was created using the 
        ModelApproximate object rather than Model object
        - integrated_data (bool): whether this is integrated data or an
        observed ephemeris downloaded from, e.g. Horizons

        Effect: Creates the appropriate model object and saves the data to it
        '''
        # re-format the data to pass on to the Model object
        # -------------------------------------------------
        # Required format
        # body_ids/body_names: Python list
        # masses: Numpy array, shape (N,)
        # R0/V0: Numpy array, shape (N, 3)
        body_ids = []
        body_names = []
        masses = []
        R = []
        V = []
        times = []
        for i, id in enumerate(self.bodies['id']):
            if self.bodies['mass'][i] is not None:
                if approx_model:
                    if self.bodies['obj_type'] != 'SM':
                        if integrated_data:
                            ephemeris = self.bodies['integrated_ephem'][i]
                        else:
                            ephemeris = self.bodies['ephemeris'][i]

                        body_ids.append(int(self.bodies['id'][i]))
                        body_names.append(self.bodies['name'][i])
                        masses.append(self.bodies['mass'][i])
                        R_body = ephemeris[:,[1,2,3]]
                        V_body = ephemeris[:,[4,5,6]]
                        R.append(np.swapaxes(R_body[:,:,np.newaxis], 0,2))
                        V.append(np.swapaxes(V_body[:,:,np.newaxis], 0,2))
                        times.append(ephemeris[:,0])
                else:
                    if integrated_data:
                        ephemeris = self.bodies['integrated_ephem'][i]
                    else:
                        ephemeris = self.bodies['ephemeris'][i]

                    body_ids.append(int(self.bodies['id'][i]))
                    body_names.append(self.bodies['name'][i])
                    masses.append(self.bodies['mass'][i])
                    R_body = ephemeris[:,[1,2,3]]
                    V_body = ephemeris[:,[4,5,6]]
                    R.append(np.swapaxes(R_body[:,:,np.newaxis], 0,2))
                    V.append(np.swapaxes(V_body[:,:,np.newaxis], 0,2))
                    times.append(ephemeris[:,0])

        masses = np.array(masses, dtype=np.float64)
        if masses.ndim == 2:
            masses = np.squeeze(masses)
        R = np.concatenate(R, axis=0)
        V = np.concatenate(V, axis=0)

        # initialize the Model object
        if approx_model:
            self.model_approx = model_class(body_ids, body_names, masses)
            self.model_approx.R = R
            self.model_approx.V = V
            self.model_approx.times = times[-1]
        else:
            self.model = model_class(body_ids, body_names, masses)
            self.model.R = R
            self.model.V = V
            self.model.times = times[-1]
        return True


    def load_sim_from_dir(self, folder_name):
        '''
        Loads simulation data only for bodies that have been defined in the 
        Simulation object already (where there is a body ID). The data is loaded
        from the folder, folder_name. This assumes folder_name includes a folder
        named integrated_ephemerides. This would have been created automatically
        when saving the data.
        '''

        # get a list of body IDs and names in the files folder
        dir_path = os.path.join(folder_name, 'integrated_ephemerides')
        file_names = os.listdir(dir_path)
        body_ids = []
        body_names = []
        for file_name in file_names:
            id = file_name.split('_')[0]
            id = id.split('ID')[1]
            name = file_name.split('.')[0]
            name = name.split('_')[1]
            body_ids.append(int(id))
            body_names.append(name)

        # load the data into the 'bodies' dictionary where the ID#s match
        for i, id_i in enumerate(self.bodies['id']):
            for j, id_j in enumerate(body_ids):
                fname = file_names[j]
                path = os.path.join(dir_path, fname)
                if id_i == id_j:
                    ephemeris = np.loadtxt(path, dtype=np.float64, 
                                        delimiter=',')
                    self.bodies['integrated_ephem'][i] = ephemeris
        return True


    def get_quantity(self, body, eph='sim', quantity=None, coordinate='all'):
        '''Used internally, returns position and velocity (passed to it as the 
        argument quantity) for arg. body.
        Arguments:
        - body is body name or ID; unique if ID; if name and not unique return false
        - eph={'sim'/'obs'}: from simulation or observed ephemeris
        - quantity={'pos'/'vel'/None}: for position or velocity. None returns both
        - coordinate ("x"/"y"/"z"/"all"): the returned coordinates
        '''

        # input checks
        if coordinate not in ['all', 'x', 'y', 'z']:
            print("get_quantity call: Invalid input for argument 'coordinate'")
            print("coordinate={'all' (default)/'x'/'y'/'z'}")
            return False
        if eph not in ['sim', 'obs']:
            print("get_quantity call: Invalid input for argument eph={'sim'/obs'}")
            return False
        
        # get the body index
        if type(body) is str:
            if body not in self.bodies['name']:
                print('get_quantity call: Body name, '+str(body)+', is not in the list')
                return False
            if self.bodies['name'].count(body) > 1:
                print('get_quantity call: Body name not unique.')
                return False
            body_index = self.bodies['name'].index(body)
        elif type(body) is int:
            # Get index from the body IDs
            if body not in self.bodies['id']:
                print('get_quantity call: Body ID#, '+str(body)+', is not in the list')
                return False
            body_index = self.bodies['id'].index(body)
        else:
            print("get_quantity call: Invalid input for argument 'body'")
            print("Must be either the body ID (unique) or body name (unique)")
            return False
        
        if eph == 'sim':
            if self.bodies['integrated_ephem'][body_index] is not None:
                ephemeris = self.bodies['integrated_ephem'][body_index]
            else:
                print("get_quantity call: No simulation data for this body")
                return False
        elif eph == 'obs':
            if self.bodies['ephemeris'][body_index] is not None:
                ephemeris = self.bodies['ephemeris'][body_index]
            else:
                print("get_quantity call: No (observed) ephemeris data for this body.")
                return False
        
        if quantity == 'pos':
            result = ephemeris[:,[1,2,3]]
        elif quantity == 'vel':
            result = ephemeris[:,[4,5,6]]
        elif quantity is None:
            result = ephemeris
        else:
            print("get_quantity call: Invalid input for argument 'quantity'")
            return False
        
        if coordinate == 'all':
            return result # returned shape (size(times), 3)
        elif coordinate == 'x':
            return result[:,0] # returned shape (size(times), )
        elif coordinate == 'y':
            return result[:,1]
        elif coordinate == 'z':
            return result[:,2]

    def sim_pos(self, body, coordinate='all'):
        '''See get_quantity()
        Returns: the simulated positions for body. Shape (size(times), 3)
        '''
        return self.get_quantity(body, 'sim', 'pos', coordinate)
    
    def obs_pos(self, body, coordinate='all'):
        '''See get_quantity()
        Returns: the observed positions for body. Shape (size(times), 3)
        '''
        return self.get_quantity(body, 'obs', 'pos', coordinate)

    def sim_vel(self, body, coordinate='all'):
        '''See get_quantity()
        Returns: the simulated velocities for body. Shape (size(times), 3)
        '''
        return self.get_quantity(body, 'sim', 'vel', coordinate)

    def obs_vel(self, body, coordinate='all'):
        '''See get_quantity()
        Returns: the observed velocities for body. Shape (size(times), 3)
        '''
        return self.get_quantity(body, 'obs', 'vel', coordinate)
