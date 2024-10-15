import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import cflib.crtp 
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
from vicon_receiver.msg import Position
from casadi import *
import time
import numpy as np
import math


# Uri for the radio connection for the crazyflie! Will be changed depending upon the configuration!
uri = uri_helper.uri_from_env(default='radio://0/90/2M/E7E7E7E7E7')

class cf_publisher(Node):

    def __init__(self):

        # Call the constructor of the parent class: Node!
        super().__init__('crazyflie_publisher')

        # Calling optimization_problem function to initialze the the optimization_problem
        self.dt = 10 #msec to call the callback for getting data from vicon!
        self.optimization_problem()

        # Connect some callbacks from the Crazyflie API
        self.cf = Crazyflie(rw_cache='./cache')
        self.cf.close_link()
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)

        print('Connecting to %s' % uri)

        # Try to connect to the Crazyflie
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.cf.open_link(uri)

        # Variable used to keep main loop occupied until disconnect
        self.is_connected = True

        # Send an initial command of zeros to start the quadcopter!
        self.cf.commander.send_setpoint(0,0,0,0)

        # Variables initialization that will come in handy later!
        self.flag = True

        # Variable for checking if the timer is called or not. 
        self.convergance = False

        # Initializing varaibels to store the position,orientation, and velocities of the crazyflie. 
        self.cf_state_pos = np.array([0, 0, 0])
        self.cf_state_orientation = np.array([0, 0, 0])
        self.cf_state_vel = np.array([0, 0, 0])
        self.cf_rot_vel = np.array([0, 0, 0])
        self.x0 = DM([0,0,0,0,0,0,0,0,0,0,0,0])
        # Initializing variables to store the final position of the target!
        self.target_pos = np.array([0, 0, 0])

        # "kk_fly" and "rccar" are the custom names. Must be changed to your vicon configuration!!!
        self.cf_vicon_subscriber = self.create_subscription(Position, "vicon/kk_fly/kk_fly", self.cf_vicon_callback, self.dt)
        self.target_subscriber = self.create_subscription(Position, "vicon/rccar/rccar", self.target_vicon_callback, self.dt)
        self.X0 = np.tile(self.x0.full(), (1, self.horizon_steps + 1))
        self.timer = self.create_timer(.05,self.timer_callback)

    def optimization_problem(self):
        
        """The parameters that are needed to solve the optimization problem"""

        # Declaring parameters for the crazyflie:
        self.gravity = 9.81  # gravity
        #self.step_time = 0.2  # sec, time step for MPC
        self.drone_radius = 0.1 # m

        # Drone parameters!
        self.I_x = 2.4*10**(-5)  # moment of inertia along x-axis
        self.I_y = self.I_x # moment of inertia along y-axis
        self.I_z = 3.2*10**(-5)  # moment of inertia along z-axis
        self.m = .033 # mass (kg)
        self.bounds = inf
        self.v_max = 1
        self.v_min = -1
        self.w_max = 10.47
        self.w_min = -10.47
        self.a_max = 4
        self.a_min = -4
        self.w_dot_max = 17.45
        self.w_dot_min = -17.45
        self.thrust_max = 1.9*self.m*self.gravity # 1.9 is the thrust to weight
        self.thrust_min = 0
        self.tau_max = 0.0097
        self.tau_min = -self.tau_max

        # Number of horizon steps to look in the future!
        self.horizon_steps = 10  # MPC horizon

        # State variables: x, y, z, psi, phi, theta, x_dot, y_dot, z_dot, r, p, q
        self.x = SX.sym('x')       # Position x
        self.y = SX.sym('y')       # Position y
        self.z = SX.sym('z')       # Position z
        self.psi = SX.sym('psi')   # Yaw
        self.phi = SX.sym('phi')   # Roll
        self.theta = SX.sym('theta')  # Pitch
        self.x_dot = SX.sym('x_dot')  # Velocity x
        self.y_dot = SX.sym('y_dot')  # Velocity y
        self.z_dot = SX.sym('z_dot')  # Velocity z
        self.r = SX.sym('r')       # Yaw rate
        self.p = SX.sym('p')       # Roll rate
        self.q = SX.sym('q')       # Pitch rate

        # Control inputs: thrust, tau_phi (roll torque), tau_theta (pitch torque), tau_psi (yaw torque)
        self.thrust = SX.sym('thrust')  
        self.tau_phi = SX.sym('tau_phi')  # Roll torque
        self.tau_theta = SX.sym('tau_theta')  # Pitch torque
        self.tau_psi = SX.sym('tau_psi')  # Yaw torque

        # State and control vectors
        state_vector = vertcat(self.x, self.y, self.z, self.psi, self.phi, self.theta, self.x_dot, self.y_dot, self.z_dot, self.p, self.q, self.r)
        control_vector = vertcat(self.thrust, self.tau_phi, self.tau_theta, self.tau_psi)
        self.n_states = state_vector.size1()
        self.n_controls = control_vector.size1()

        # Define the state and control prediction matrices!

        self.state_prediction = SX.sym('X', self.n_states,self.horizon_steps+1)
        self.control_prediction = SX.sym('U',self.n_controls,self.horizon_steps)

        # Define the parameter vector which will contain the final and initial state!
        self.p = SX.sym('p',self.n_states*2) # Multipled by 2 because you are storing the initial and final state!

        # Define the constraints and cost function!!!
        self.g = (self.state_prediction[:,0] - self.p[0:self.n_states]) # Constraints on the functions of optimal variables.
        self.obj = 0 # Cost function!!!

        # Define the weighting matrices!
        Q = 100*DM.eye(self.n_states)
        R = .1*DM.eye(self.n_controls)

        # Start the for loop to build up the constraint vector and the cost function!!!
        for i in range(self.horizon_steps):
            curr_state = self.state_prediction[:,i]
            curr_input = self.control_prediction[:,i]
            self.obj = self.obj + ((curr_state - self.p[self.n_states:self.n_states*2]).T @ Q @ (curr_state - self.p[self.n_states:2*self.n_states])) + curr_input.T @ R @ curr_input
            next_state_estimate = self.runge_kutta_4(curr_state,curr_input)
            self.g = vertcat(self.g,self.state_prediction[:,i+1] - next_state_estimate) # Adding up the constraints in the g function!

        self.opt_vars = vertcat(reshape(self.state_prediction,self.n_states*(self.horizon_steps + 1),1),reshape(self.control_prediction,self.n_controls*self.horizon_steps,1)) # You are not just optimizing v and w but all instances of them!
        self.nlp_prob = {}
        self.nlp_prob['x'] = self.opt_vars
        self.nlp_prob['f'] = self.obj
        self.nlp_prob['g'] = self.g
        self.nlp_prob['p'] = self.p # This will change so might have to redefine the nlp_prob!

        # Initializing the solver!
        self.solver = nlpsol('solver','ipopt',self.nlp_prob)

        #Populate the constraints and the dynamics with their upper/lower bounds!Defining constraints for the NLP!

        self.lbg = [0] * self.g.size1() # Constraints on the dynamics and obstacle condition!
        self.ubg = [0] * self.g.size1() # Constraints on the dynamics and obstacle condition!

        # Constraints on the optimization variable: STATE VARIABLES. 
        self.lbx = [-self.bounds,-self.bounds,-self.bounds,self.v_min,self.v_min,self.v_min,-inf,-inf,-inf,self.w_min,self.w_min,self.w_min] * ((1+self.horizon_steps))
        self.ubx = [self.bounds,self.bounds,self.bounds,self.v_max,self.v_max,self.v_max,inf,inf,inf,self.w_max,self.w_max,self.w_max] * ((self.horizon_steps+1))

        # Constraints on the optimization variable: Control vector
        self.lbx = self.lbx + [self.thrust_min,self.tau_min,self.tau_min,self.tau_min]*(self.horizon_steps)
        self.ubx = self.ubx + [self.thrust_max,self.tau_max,self.tau_max,self.tau_max]*(self.horizon_steps)
        
        # Creating a control prediction matrix which will be used later for the optimization!!
        self.u0 = DM.zeros(self.n_controls, self.horizon_steps)  # Initial control guess

    def calc_velocities(self,x,y,z,roll,theta,psi):

        """Do dirty differentiation for the calculation of velocities: translation and rotation!"""

        new_pos = np.array([x,y,z])
        new_orientation = np.array([roll,theta,psi])
        self.cf_state_vel = (new_pos - self.cf_state_pos)/(self.dt/1000)
        self.cf_rot_vel = (new_orientation - self.cf_state_orientation)/(self.dt/1000)

    def cf_vicon_callback(self,msg_in):

        """ Populating the position and orientation of the crazyflie"""

        new_orientation = self.quat2euler(msg_in.x_rot,msg_in.y_rot,msg_in.z_rot,msg_in.w)
        self.calc_velocities(msg_in.x_trans, msg_in.y_trans, msg_in.z_trans,new_orientation[0],new_orientation[1],new_orientation[2])
        self.cf_state_pos = np.array([msg_in.x_trans, msg_in.y_trans, msg_in.z_trans])
        self.cf_state_orientation = new_orientation
        self.x0 = DM([self.cf_state_pos[0], self.cf_state_pos[1], self.cf_state_pos[2], self.cf_state_vel[0], self.cf_state_vel[1], self.cf_state_vel[2],self.cf_state_orientation[0], self.cf_state_orientation[1], self.cf_state_orientation[2], self.cf_rot_vel[0], self.cf_rot_vel[1], self.cf_rot_vel[2]])  # state update!

    def target_vicon_callback(self,msg_in):

        """ Populating the position of the target"""

        self.target_pos = np.array([msg_in.x_trans, msg_in.y_trans, msg_in.z_trans])
        self.xf = DM([self.target_pos[0], self.target_pos[1], self.cf_state_pos[2], 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Target state update
    
    def timer_callback(self):

        """ This is where the optimization problem will be solved! """

        if self.convergance == False:

            # Start the optimization sovler!

            # Define the parameters for the MPC
            p_numeric = vertcat(self.x0, self.xf)

            # Initial guess for optimization variables
            opt_init = vertcat(reshape(self.X0, self.n_states * (self.horizon_steps + 1), 1), reshape(self.u0, self.n_controls * self.horizon_steps, 1))

            # Solve the NLP problem
            sol = self.solver(x0=opt_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p_numeric)
            solution = sol['x'].full()

            # Extract states and control inputs from the solution
            state_values = reshape(solution[:self.n_states * (self.horizon_steps + 1)], self.n_states, self.horizon_steps + 1)
            control_values = reshape(solution[self.n_states * (self.horizon_steps + 1):], self.n_controls, self.horizon_steps)

            # Extract the commands that are to be sent to the crazyflie!
            first_state = state_values[:,0]
            u = control_values[:, 0]

            # Send the commands to the crazyflie
            self.cf.commander.send_setpoint(math.degrees(first_state[6]),math.degrees(first_state[7]),math.degrees(first_state[11]),int(u[0].full().item()*65535/self.thrust_max))

            print("sending//////////////////////////////")
            print(math.degrees(first_state[6]),math.degrees(first_state[7]),math.degrees(first_state[11]),int(u[0].full().item()*(49999/self.thrust_max)+10001))
            # Update the state with first control input
            #x0 = vertcat(self.cf_state_pos,self.cf_state_vel,self.cf_state_orientation,self.cf_rot_vel)

            # Reinitialize guesses for next iteration
            self.u0 = horzcat(control_values[:, 1:], control_values[:, -1])

            self.X0 = horzcat(state_values[:, 1:], state_values[:, -1])
            
            #time.sleep(.2)
            #self.cf.commander.send_notify_setpoint_stop(remain_valid_milliseconds=10)

            if norm_2(self.x0[0:3] - self.xf[0:3]) < 0.01:
                self.convergance = True
    def _connected(self, link_uri):

        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""

        print('Connected to %s' % link_uri)

        # The definition of the logconfig can be made before connecting
        self.lg_stab = LogConfig(name='Stabilizer', period_in_ms=100)
        self.lg_stab.add_variable('stateEstimate.x', 'float')
        self.lg_stab.add_variable('stateEstimate.y', 'float')
        self.lg_stab.add_variable('stateEstimate.z', 'float')
        self.lg_stab.add_variable('stabilizer.roll', 'float')
        self.lg_stab.add_variable('stabilizer.pitch', 'float')
        self.lg_stab.add_variable('stabilizer.yaw', 'float')
        # The fetch-as argument can be set to FP16 to save space in the log packet
        self.lg_stab.add_variable('pm.vbat', 'FP16')

        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
        try:
            self.cf.log.add_config(self.lg_stab)
            # This callback will receive the data
            self.lg_stab.data_received_cb.add_callback(self._stab_log_data)
            # This callback will be called on errors
            self.lg_stab.error_cb.add_callback(self._stab_log_error)
            # Start the logging
            self.lg_stab.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')

    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):

        # yaw = data['stabilizer.yaw']
        # pitch = data['stabilizer.pitch']
        # roll = data['stabilizer.roll']

        #self.state_pos = np.array([data['stateEstimate.x'], data['stateEstimate.y'], data['stateEstimate.z']])
        
        #self.state_orientation = np.array([data['stabilizer.roll'], data['stabilizer.pitch'], data['stabilizer.yaw']])

        # print("state", self.state_pos, self.state_quat)

        """Callback from a the log API when data arrives"""
        """print(f'[{timestamp}][{logconf.name}]: ', end='')
        for name, value in data.items():
            print(f'{name}: {value:3.3f} ', end='')
        print()"""


    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)
        self.cf.commander.send_setpoint(0,0,0,0)
        self.is_connected = False

    def _disconnect(self):
        self.is_connected = False
        self.cf.commander.send_setpoint(0,0,0,0)
        self.cf.close_link()

    def quat2euler(self,x,y,z,w):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        
        Parameters:
        q (list or numpy array): A list or numpy array containing the quaternion in the format [w, x, y, z].
        
        Returns:
        numpy array: Euler angles in radians, in the format [roll, pitch, yaw].
        """
        
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Return the euler angles!
        return np.array([roll, pitch, yaw])

    def runge_kutta_4(self,state,forces_moments):
        step_time = self.dt/1000
        k1 = self.dynamics(state, forces_moments)
        k2 = self.dynamics(state + step_time/2.*k1, forces_moments)
        k3 = self.dynamics(state + step_time/2.*k2, forces_moments)
        k4 = self.dynamics(state + step_time*k3, forces_moments)
        state += step_time/6 * (k1 + 2*k2 + 2*k3 + k4)
        return state

    def dynamics(self,state,control):
        x = state[0]  # Extract x
        y = state[1]  # Extract y
        z = state[2]  # Extract z
        x_dot = state[3]
        y_dot = state[4]
        z_dot = state[5]
        phi = state[6]
        theta = state[7]
        psi = state[8]
        p = state[9]
        q = state[10]
        r = state[11]

        thrust = control[0]
        tau_phi = control[1]
        tau_theta = control[2]
        tau_psi = control[3]

        # Linear dynamics
        R11 = cos(theta) * cos(psi)
        R12 = -sin(psi)
        R13 = sin(theta) * cos(psi)

        R21 = cos(theta) * sin(psi)
        R22 = cos(psi)
        R23 = sin(theta) * sin(psi)

        R31 = -sin(theta)
        R32 = 0
        R33 = cos(theta)

        # Translational dynamics in the inertial frame
        x_dot = R11 * x_dot + R12 * y_dot + R13 * z_dot
        y_dot = R21 * x_dot + R22 * y_dot + R23 * z_dot
        z_dot = R31 * x_dot + R32 * y_dot + R33 * z_dot
        
        # Rotational kinematics
        phi_dot = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
        theta_dot = q * cos(phi) - r * sin(phi)
        psi_dot = (q * sin(phi) + r * cos(phi)) / cos(theta)
        
        # Translational accelerations considering thrust and angular velocities
        acc_x = (thrust / self.m) * (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)) - (q * z - r * y)
        acc_y = (thrust / self.m) * (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)) + (q * x - r * z)
        acc_z = (thrust / self.m) * (cos(phi) * cos(theta)) - self.gravity + (p * y - q * x)
        
        # Rotational dynamics
        p_dot = (self.I_y - self.I_z) / self.I_x * q * r + tau_phi / self.I_x
        q_dot = (self.I_z - self.I_x) / self.I_y * p * r + tau_theta / self.I_y
        r_dot = (self.I_x - self.I_y) / self.I_z * p * q + tau_psi / self.I_z

        
        return vertcat(x_dot, y_dot, z_dot, acc_x, acc_y, acc_z, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot)

def main(args=None):

    try:
        rclpy.init(args=args)
        pub = cf_publisher()
        rclpy.spin(pub)
    except Exception as ki:
        print(ki)
    finally:
        pub._disconnect()
        pub.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()