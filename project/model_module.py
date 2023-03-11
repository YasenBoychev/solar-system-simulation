import numpy as np
from scipy.integrate import odeint, solve_ivp

# N.B. this module includes two classes: Model and ModelApproximate
# ModelApproximate inherits from Model and overrides solve_system()

class Model:
    '''
    Defines the physical model used in the simulation, as well as methods
    for calculating the total energy relative error, and retrieving data from
    the object.
    
    The physical model is defined in the function, model_func, 
    (within the solve_system() method) which is given as two different 
    implementations -- one using vectorized operations (default) 
    and one using Python loops. The vectorized operations function is preferred
    but is less easy to read.

    There are two integrator options, both using the LSODA method -- the older
    scipy odeint and newer solve_ivp API. The solve_ivp API set with the LSODA
    method is just a wrapper to the same FORTRAN code but is preferred here as
    it worked better for one of the simulations, but both should give the same
    results.
    '''
    G_SI = 6.67430e-11 # m^3 kg^-1 s^-2
    # Conversion factors for the units used in simulation
    # Units used: mass (kg), distance (au), time (day)
    M = 1 # kg / kg
    L = 1.495978707e11 # meters / au
    T = 86400 # sec / day

    def __init__(self, body_ids, body_names, masses, R0=None, V0=None, G=None):
        if G == None:
            # G units: L^3 M^-1 T^-2
            self.G = (Model.G_SI/Model.L**3)*Model.M*Model.T**2
        else:
            self.G = G
        self.body_ids = body_ids
        self.body_names = body_names
        self.masses = masses # shape: (N,)
        self.R0 = R0 # R0, V0 shape: (N, 3)
        self.V0 = V0
        self.R = None # shape (self.N, 3, np.size(self.times))
        self.V = None # shape (self.N, 3, np.size(self.times))
        self.times = None # shape: (N,)
        self.N = np.size(masses)
    
    def solve_system(self, t_start, t_end, num_points, rtol=None, atol=None,
                        tcrit=None, h0=0.0, hmax=0.0, use_vect_ops_func=True,
                        use_solve_ivp_API=False):
        '''
        Integrate the system from the initial conditions defined in self.R0
        and self.V0 from t_start to t_end.

        Arguments:
        - t_start (scalar): integration start time
        - t_end (scalar): integration end time (included in the range)
        - num_points (scalar): number of time points, evenly spaced between
        t_start and t_end, to return in the solution. The integrator may work
        to a higher precision than this.
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
        '''

        # the times at which the integrator will return a soluiton
        self.times = np.linspace(t_start, t_end, num=num_points, 
                                 endpoint=True, dtype=np.float64)
        N = self.N # system size
        
        if use_vect_ops_func:
            # masses broadcast to the correct shape
            mass_j = np.broadcast_to(np.reshape(self.masses, (1,N))[:,:,np.newaxis], (N,N,3))

            def model_func(t, Y, N, masses, mass_j, G):
                '''
                Physical model implemented with vectorized operations. 
                The format of this function is that required by the solver.

                The state vector, Y, is ndarray, shape (6*N, ). The first
                half (size 3*N) is body positions as (x1,y1,z1,x2,y2,z2,...), 
                and the second half is velocities in the same format.

                masses and mass_j are both passed but which is used depends
                on which definition of model_func is passed to the solver. 
                masses is the variable self.masses, shape (N,)

                Other arguments:
                t is the time parameter, G is the gravitational constant, N is
                system size

                Returns: Y_dot: ndarray, shape (6*N, ): (velocity, acceleration)
                '''
                R, V = np.split(Y, 2)
                R_i = np.reshape(R, (N, 1, 3))
                R_j = np.reshape(R, (1, N, 3))

                # R_ij is shape (N, N, 3). Rows correspond to bodies i,
                # cols to bodies j. Axis 2 lists x, y and z coorinates.
                # Along the diagonal R_ij is zero.
                R_ij = R_j - R_i
                modulus_R_ij_cubed = np.sqrt(np.sum(R_ij**2, axis=2, keepdims=True))**3

                # The acceleration term requires that the sum excludes 
                # j equals i which corresponds to the diagonal of R_ij,
                # which is zero. Dividing with the codition:
                # where=(modulus_R_ij_cubed != 0) takes care of this.
                A_i_term = np.zeros(np.shape(R_ij), dtype=np.float64)
                np.divide(mass_j*R_ij, modulus_R_ij_cubed, out=A_i_term,
                    where=(modulus_R_ij_cubed != 0))
                V_dot = G * np.sum(A_i_term, axis=1)

                # return the time derivative of the state vector Y
                return np.concatenate((V, np.reshape(V_dot, (3*N,))))
        else:
            mass_j = None
            def model_func(t, Y, N, masses, mass_j, G):
                '''
                Physical model implemented with Python loops. 
                The format of this function is that required by the solver.

                The arguments are the same as those in the vectorized ops
                medel_func. The implementation is also similar but with 
                Python loops rather than numpy ops.
                Returns: Y_dot: ndarray, shape (6*N, ): (velocity, acceleration)
                '''
                
                Y_dot = np.zeros(6*N, dtype=np.float64)
                for i in range(N):
                    # Y_dot velocity is Y velocity
                    Y_dot[int(0+3*i)] = Y[int(3*N+0+3*i)]
                    Y_dot[int(1+3*i)] = Y[int(3*N+1+3*i)]
                    Y_dot[int(2+3*i)] = Y[int(3*N+2+3*i)]

                    # calculate Y_dot acceleration 
                    for j in range(N):
                        if i != j:
                            Rx_ij = Y[int(0+3*j)] - Y[int(0+3*i)]
                            Ry_ij = Y[int(1+3*j)] - Y[int(1+3*i)]
                            Rz_ij = Y[int(2+3*j)] - Y[int(2+3*i)]
                            R_ij = np.sqrt(Rx_ij**2 + Ry_ij**2 + Rz_ij**2)

                            Y_dot[int(3*N+0+3*i)] += (G*masses[j]*Rx_ij)/R_ij**3
                            Y_dot[int(3*N+1+3*i)] += (G*masses[j]*Ry_ij)/R_ij**3
                            Y_dot[int(3*N+2+3*i)] += (G*masses[j]*Rz_ij)/R_ij**3
                return Y_dot
        
        Y0 = np.concatenate( (np.ravel(self.R0), np.ravel(self.V0)) )
        args = (N, self.masses, mass_j, self.G)

        if use_solve_ivp_API:
            method = 'LSODA'
            t_span = (t_start, t_end)
            t_eval = self.times
            first_step = None if not h0 else h0
            max_step = np.inf if not hmax else hmax
            rtol = 1e-3 if not rtol else rtol
            atol = 1e-6 if not atol else atol

            sol_obj = solve_ivp(model_func, t_span, Y0, method=method, 
                                t_eval=t_eval, args=args, first_step=first_step,
                                max_step=max_step, rtol=rtol, atol=atol)
            # format the solution to be the same shape as odeint
            solution = np.swapaxes(sol_obj.y, 0, 1)
            info = {'t':sol_obj.t, 'nfev':sol_obj.nfev, 'njev':sol_obj.njev,
                    'nlu':sol_obj.nlu, 'status':sol_obj.status, 'message':sol_obj.message, 'success':sol_obj.success}
        else:
            solution, info = odeint(model_func, Y0, self.times, args=args, 
                                    rtol=rtol, atol=atol, tcrit=tcrit, h0=h0, 
                                    hmax=hmax, printmessg=True, tfirst=True,
                                    full_output=True)
        
        R, V = np.split(solution, 2, axis=1)

        # self.R/V shapes: (self.N, 3, np.size(self.times))
        self.R = np.reshape(np.swapaxes(R[:,:,np.newaxis], 0, 2), 
            (self.N, 3, np.size(self.times)))
        self.V = np.reshape(np.swapaxes(V[:,:,np.newaxis], 0, 2), 
            (self.N, 3, np.size(self.times)))
        return solution, info
    
    
    def calculate_U_tot(self):
        '''
        Calculates the total potential energy of the system by the formula:
        U = 1/2 * sum_i(sum_j(-G*m_j*m_i/r_ij)) where j is not equal to i.
        Uses vectorized opperations in a way similar to that in model_func.
        '''
        N = self.N
        mass_i = np.broadcast_to(np.reshape(self.masses, (N,1))[:,:,np.newaxis], (N,N,np.size(self.times)))
        mass_j = np.broadcast_to(np.reshape(self.masses, (1,N))[:,:,np.newaxis], (N,N,np.size(self.times)))

        R_i = self.R[:,np.newaxis,:,:] # shape: N, 1, 3, np.size(self.times)
        R_j = np.reshape(R_i, (1, N, 3, np.size(self.times)))
        R_ij = R_j - R_i # shape: N, N, 3, np.size(self.times)
        modulus_R_ij = np.sqrt(np.sum(R_ij**2, axis=2, keepdims=False)) # shape: N, N, np.size(self.times)

        U_ij_term = np.zeros(np.shape(mass_j), dtype=np.float64)
        np.divide(mass_j * mass_i, modulus_R_ij, out=U_ij_term,
            where=(modulus_R_ij != 0))
        U_tot = 0.5 * -self.G * np.sum(U_ij_term, axis=(0,1))
        return U_tot
    
    def calculate_T_tot(self):
        '''Calculates total kinetic energy.'''
        N = self.N
        mass_i = np.broadcast_to(np.reshape(self.masses, (N,1)), (N,np.size(self.times)))

        V_i = self.V # shape: N, 3, np.size(self.times)
        modulus_V_i_squared = np.sum(V_i**2, axis=1, keepdims=False) # shape: N, np.size(self.times)

        T_i_term = 0.5 * mass_i * modulus_V_i_squared
        T_tot =  np.sum(T_i_term, axis=0)
        return T_tot

    def E_total_rel_error(self):
        '''Uses calculate_U_tot() and calculate_T_tot() to return the total
        energy relative error from the initial total energy.'''

        # checks
        if self.R is None:
            print('E_total_rel_error call: Run solve_system() method first')
            return False
        
        U_tot = self.calculate_U_tot()
        T_tot = self.calculate_T_tot()
        E_tot = U_tot + T_tot
        print('Model: Initial total energy: '+str(E_tot[0]))
        return (E_tot - E_tot[0]) / E_tot[0]
    

    def get_quantity(self, quantity, body, coordinate='all'):
        '''Used internally, returns position and velocity (passed to it as the 
        argument quantity) for arg. body.
        Arguments:
        - body is body name or ID; unique if ID; if name and not unique return false
        - quantity (variable reference): position, self.R, or velocity, self.V
        - coordinate ("x"/"y"/"z"/"all"): the returned coordinates
        '''

        # input checks
        if coordinate not in ['all', 'x', 'y', 'z']:
            print("get_quantity call: Invalid input for argument 'coordinate'")
            print("coordinate={'all' (default)/'x'/'y'/'z'}")
            return False
        if self.R is None:
            print('get_quantity call: Run solve_system() method first')
            return False
        
        # get the body index
        if type(body) is str:
            if body not in self.body_names:
                print('get_quantity call: Body name, '+str(body)+', is not in the list')
                return False
            if self.body_names.count(body) > 1:
                print('get_quantity call: Body name not unique.')
                return False
            body_index = self.body_names.index(body)
        elif type(body) is int:
            # Get index from the body IDs
            if body not in self.body_ids:
                print('get_quantity call: Body ID#, '+str(body)+', is not in the list')
                return False
            body_index = self.body_ids.index(body)
        else:
            print("get_quantity call: Invalid input for argument 'body'")
            print("Must be either the body ID (unique) or body name (unique)")
            return False
        
        if coordinate == 'all':
            return quantity[body_index,:,:] # returned shape (3, size(times))
        elif coordinate == 'x':
            return quantity[body_index,0,:] # returned shape (size(times), )
        elif coordinate == 'y':
            return quantity[body_index,1,:]
        elif coordinate == 'z':
            return quantity[body_index,2,:]

    def position(self, body, coordinate='all'):
        '''Uses get_quantity() internally to return the body position'''
        return self.get_quantity(self.R, body, coordinate)
    
    def velocity(self, body, coordinate='all'):
        '''Uses get_quantity() internally to return the body velocity'''
        return self.get_quantity(self.V, body, coordinate)


class ModelApproximate(Model):
    '''Inherits from class Model and overrides the solve_system method
    to define a new model which is an approximation to the full system
    of differential equations when considering a single small body.
    '''
    def __init__(self, body_ids, body_names, masses, R0=None, V0=None, G=None):
        super().__init__(body_ids, body_names, masses, R0, V0, G)
        self.object_id = None # int, object id
        self.object_name = None # str
        self.mass_object = None # scalar
        self.R0_object = None # shape (1,3)
        self.V0_object = None # R0/V0 shape (1,3)
        self.R_object = None
        self.V_object = None
        self.A_object = None
        self.times = None

    def solve_system(self, t_start, t_end, rtol=None, atol=None, h0=0.0,
                hmax=0.0, min_step=None):
        '''Overrides solve_system of the Model class to define a new
        approximate model. Introduces min_step as an argument which
        sets the minimum step for the solver.'''
        
        # Check that there are R/V values in the object
        if self.R is None or self.V is None:
            print('ModelApproximate solve_system call: No R/V values in object')
            return False

        def model_func(t, Y, self):
            # Y includes the coordinates for a single particle only
            Y_dot = np.zeros(6, dtype=np.float64)
            Y_dot[0] = Y[3]
            Y_dot[1] = Y[4]
            Y_dot[2] = Y[5]

            calc_new_acc = False

            # truncate t to same precision as 'times'
            precision = 1 # d.p.
            t = (lambda a: int(a*10**precision)/(10**precision))(t)
            if t in self.times:
                t_index = np.argwhere(self.times == t)
                t_index = t_index[0,0]
                calc_new_acc = True

            if calc_new_acc:
                for j in range(self.N):
                    Rx_ij = self.R[j, 0, t_index] - Y[0]
                    Ry_ij = self.R[j, 1, t_index] - Y[1]
                    Rz_ij = self.R[j, 2, t_index] - Y[2]
                    R_ij = np.sqrt(Rx_ij**2 + Ry_ij**2 + Rz_ij**2)
                    Y_dot[3] += (self.G*self.masses[j]*Rx_ij)/R_ij**3
                    Y_dot[4] += (self.G*self.masses[j]*Ry_ij)/R_ij**3
                    Y_dot[5] += (self.G*self.masses[j]*Rz_ij)/R_ij**3
                # store the last acceleration values
                self.A_object = Y_dot[[3,4,5]]
            else:
                    # else use the last recorded acceleration values
                    Y_dot[3] = self.A_object[0]
                    Y_dot[4] = self.A_object[1]
                    Y_dot[5] = self.A_object[2]
            return Y_dot
        
        Y0 = np.concatenate((np.ravel(self.R0_object),np.ravel(self.V0_object)))
        args = (self,)

        method = 'LSODA'
        t_span = (t_start, t_end)
        t_eval = self.times
        first_step = None if not h0 else h0
        max_step = np.inf if not hmax else hmax
        min_step = 0 if not min_step else min_step
        rtol = 1e-3 if not rtol else rtol
        atol = 1e-6 if not atol else atol

        sol_obj = solve_ivp(model_func, t_span, Y0, method=method, 
                            t_eval=t_eval, args=args, first_step=first_step,
                            max_step=max_step, min_step=min_step, 
                            rtol=rtol, atol=atol)
        # format the solution to be the same shape as odeint
        solution = np.swapaxes(sol_obj.y, 0, 1)
        info = {'t':sol_obj.t, 'nfev':sol_obj.nfev, 'njev':sol_obj.njev,
                'nlu':sol_obj.nlu, 'status':sol_obj.status, 'message':sol_obj.message, 'success':sol_obj.success}
    
        R_object, V_object = np.split(solution, 2, axis=1)

        self.R_object = np.reshape(np.swapaxes(R_object[:,:,np.newaxis], 0, 2), 
            (1, 3, np.size(self.times)))
        self.V_object = np.reshape(np.swapaxes(V_object[:,:,np.newaxis], 0, 2), 
            (1, 3, np.size(self.times)))

        # concatenate the solution and body info to the rest of the data
        self.body_ids.append(self.object_id)
        self.body_names.append(self.object_name)
        self.masses = np.append(self.masses, self.mass_object)
        self.R = np.vstack((self.R, self.R_object))
        self.V = np.vstack((self.V, self.V_object))
        self.N = np.size(self.masses)
        return solution, info
