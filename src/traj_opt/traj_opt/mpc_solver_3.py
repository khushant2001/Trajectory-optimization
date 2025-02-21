"""
Author: Khushant Khurana

Formulating the MPC optimization problem as a nonlinear (QP) problem. Multiple
shooting is used as the transcription technique to discretize the dynamics. Also creating 
custom publishers for recording state information and the transmission of the MPC solution. 
Using RK4 as the integrator of the dynamics in MPC. 

"""
import rclpy
from rclpy.node import Node
from vicon_receiver.msg import Position
from casadi import *
import numpy as np
import math
from custom_msgs.msg import Actuation, StateInfo, OptimizedInput, LiftOff

class solve_mpc(Node):

    def __init__(self):

        # Call the constructor of the parent class: Node!
        super().__init__('mpc_solver')
        
        """
        Initial parameters declaration!
        """
        # TIme step for the integration of the model in the MPC!
        self.dt = 50 # msec. For the model integration

        # Time step for the update of the vicon topics to get velocities through dirty differentiation!
        self.vicon_topic_update_time = 0.01 # sec! Got this from the vicon topic information!!!

        # Calling optimization_problem function to initialze the the optimization_problem
        self.optimization_problem()

        """
        Declaration of flags that assist in the state machine!
        """
        # Variables initialization that will come in handy later!
        self.flag = True

        # Variable for checking if the timer is called or not. 
        self.convergance = False

        # Variable to skip the initialization of the drone velocities for the first time step!
        self.first_step_check = True

        # Variable to update the initial values for the optimization variables but just for the first time step!
        self.opt_var_update_first_step = True

        # Variable to check if the liftoff has happened or not
        self.liftoff_flag = False

        """
        Initialize Variables that will be used later in the program!
        """
        # Initializing varaibels to store the position,orientation, and velocities of the crazyflie. 
        self.cf_state_pos = np.array([0, 0, 0])
        self.cf_state_orientation = np.array([1, 0, 0, 0])
        self.cf_state_vel = np.array([0, 0, 0])
        self.cf_rot_vel = np.array([0, 0, 0])

        # Declaring state arrays for the initial and final state which will be used later for the optimization problem!
        self.x0 = DM([0,0,0,0,0,0,1,0,0,0,0,0,0])
        self.xf = DM([0,0,0,0,0,0,1,0,0,0,0,0,0])
        self.X0 = None

        # Initializing variables to store the final position of the target!
        self.target_pos = np.array([0, 0, 0])

        # Alpha coefficient for the low pass filter for dirty differentiation of the velocities!
        self.alpha = 0.3

        """ 
        Declare all the message instances and publishers/subscribers for the corresponding topics

        Note => "kk_fly" and "rccar" are the custom names. Must be changed according to your vicon configuration!!!
        """

        # Create subscription to get the target pose from the vicon!
        self.target_subscriber = self.create_subscription(Position, "vicon/rccar/rccar", self.target_vicon_callback, 10)

        # Create publisher to get the target message
        self.target_publisher = self.create_publisher(StateInfo, "/Target", 10)

        # Create subscription to get the crazyflie pose from the vicon! 
        self.cf_vicon_subscriber = self.create_subscription(Position, "vicon/kk_fly/kk_fly", self.cf_vicon_callback, 10)

        # Creating instance of the message that will be published by the mpc_solver node. 
        self.mpc_sol_msg = Actuation()

        # Create publisher to send the MPC solution to the cf_publisher!
        self.mpc_publisher = self.create_publisher(Actuation, "/mpc_solution", 10)

        # Create publisher to record the true state of the crazyflie (Used later in the plotjuggler)
        self.true_state_publisher = self.create_publisher(StateInfo, "/true_state", 10)

        # Subscribe to the liftoff topic!
        self.liftoff_check = self.create_subscription(LiftOff, '/liftoff', self.liftoff_callback, 10)

        # Create publisher to record the estimated state of the crazyflie (for all horizon steps) derived from the model in the MPC!
        self.model_state_publisher0 = self.create_publisher(StateInfo, "/model_state_calcs0", 10)
        self.model_state_publisher1 = self.create_publisher(StateInfo, "/model_state_calcs1", 10)
        self.model_state_publisher2 = self.create_publisher(StateInfo, "/model_state_calcs2", 10)
        self.model_state_publisher3 = self.create_publisher(StateInfo, "/model_state_calcs3", 10)
        self.model_state_publisher4 = self.create_publisher(StateInfo, "/model_state_calcs4", 10)
        self.model_state_publisher5 = self.create_publisher(StateInfo, "/model_state_calcs5", 10)
        
        # Creating a message instance to record the state info of the crazyflie from the vicon!
        self.true_state_msg = StateInfo()

        # Creating instance of the message that will be published by the mpc_solver node. 
        self.optimized_solution = OptimizedInput()

        # Create publisher to send the MPC solution to the cf_publisher!
        self.optimized_solution_publisher = self.create_publisher(OptimizedInput, "/optimized_input", 10)

        # Creating message instances to record the state info (for all horizon steps) of the crazyflie model!
        self.model_state_msg0 = StateInfo()
        self.model_state_msg1 = StateInfo()
        self.model_state_msg2 = StateInfo()
        self.model_state_msg3 = StateInfo()
        self.model_state_msg4 = StateInfo()
        self.model_state_msg5 = StateInfo()
        
        # Create message instance for the target state!
        self.target_msg = StateInfo()

        """
        Creating a timer that contains the MPC solver code. This will be called at a set rate. 
        """
        # Create timer that solves the MPC!
        self.timer = self.create_timer(.05, self.timer_callback)    

    def optimization_problem(self):
        
        """Formulating the optimization problem and declaring the corresponding parameters!"""

        # State vector => [x, y, z, x_dot, y_dot, z_dot, e0, e1, e2, e3, p, q, r]
        # Control vector => [thrust, tau_x, tau_y, tau_z]

        self.get_logger().info("Initializing the optimization problem")

        # Declaring world parameters!
        self.gravity = 9.81  # m/sec^2 gravity
        
        # Drone parameters!
        self.drone_radius = 0.1 # m. Required for obstacle avoidance!
        self.I_x = 2.4*10**(-5)  # kg*m^2 moment of inertia along x-axis
        self.I_y = self.I_x # kg*m^2 moment of inertia along y-axis
        self.I_z = 3.2*10**(-5)  # kg*m^2 moment of inertia along z-axis
        self.m = .033 # mass (kg). Barebone + markers + hot glue. Measured on a precise scale. 
        self.bounds = 1000 # These are the bounds for the x, y, z in the 3D space. Could be made tighter 
        self.v_max = .3 # 1 m/sec. 
        self.v_min = -self.v_max
        self.w_max = 2.0#10.47 # rad/sec. Got from the official documentation of Bitcraze. 
        self.w_min = -self.w_max # rad/sec
        self.a_max = 4 # m/sec^2
        self.a_min = -self.a_max # m/sec^2
        self.w_dot_max = 17.45 # rad/sec^2
        self.w_dot_min = -self.w_dot_max # rad/sec^2
        self.thrust_max = 1.9*self.m*self.gravity # 1.9 is the thrust to weight ratio
        self.thrust_min = self.m*self.gravity # No reversible props so no reversible thrust!
        self.tau_max = 0.001#0.0097 # Nm. Thrust produced by each motor = 0.294 N and the armlength is about 33 m. Torque = F x r
        self.tau_min = -self.tau_max

        # Number of horizon steps to look in the future!
        self.horizon_steps = 5  # MPC horizon

        # State and control vectors lengths
        self.n_states = 13
        self.n_controls = 4

        # Define the state and control prediction matrices!
        self.state_prediction = SX.sym('X', self.n_states,self.horizon_steps+1)
        self.control_prediction = SX.sym('U',self.n_controls,self.horizon_steps)

        # Define the parameter vector which will contain the final and initial state!
        self.p = SX.sym('p',self.n_states*2) # Multipled by 2 because you are storing the initial and final state!

        # Define the constraints and cost function!!!
        self.g = (self.state_prediction[:,0] - self.p[0:self.n_states]) # Constraints on the functions of optimal variables.
        self.obj = 0 # Cost function!!!

        # Define the weighting matrices!
        Q = 100*DM.eye(3)
        Q[2:2] *= 10
        R = 100*DM.eye(self.n_controls) # Adding more weight on the z

        # Start the for loop to build up the constraint vector and the cost function!!!
        for i in range(self.horizon_steps):
            curr_state = self.state_prediction[:,i]
            curr_input = self.control_prediction[:,i]
            # self.obj = self.obj + ((curr_state[0:3] - self.p[self.n_states:self.n_states+3]).T @ Q @ (curr_state[0:3] - self.p[self.n_states:3+self.n_states])) + curr_input.T @ R @ curr_input
            self.obj = self.obj + curr_input.T @ R @ curr_input
            next_state_estimate = self.runge_kutta_4(curr_state,curr_input)
            self.g = vertcat(self.g,self.state_prediction[:,i+1] - next_state_estimate) # Adding up the constraints in the g function!

        self.obj = self.obj + ((self.state_prediction[0:3,-1] - self.p[self.n_states:self.n_states+3]).T @ Q @ (curr_state[0:3,-1] - self.p[self.n_states:3+self.n_states])) 
        
        # Formulate the optimization problem and the corresponding optimization variables!
        self.opt_vars = vertcat(reshape(self.state_prediction,self.n_states*(self.horizon_steps + 1),1),reshape(self.control_prediction,self.n_controls*self.horizon_steps,1)) # You are not just optimizing v and w but all instances of them!
        self.nlp_prob = {}
        self.nlp_prob['x'] = self.opt_vars
        self.nlp_prob['f'] = self.obj
        self.nlp_prob['g'] = self.g
        self.nlp_prob['p'] = self.p # This will change so might have to redefine the nlp_prob!

        # Initializing the solver!
        self.solver = nlpsol('solver','ipopt',self.nlp_prob, {"print_time": True, "record_time": True})

        #Populate the constraints and the dynamics with their upper/lower bounds! Defining constraints for the NLP!
        self.lbg = [0] * self.g.size1() # Constraints on the dynamics and obstacle condition!
        self.ubg = [0] * self.g.size1() # Constraints on the dynamics and obstacle condition!

        # Constraints on the optimization variable: STATE VARIABLES. 
        self.lbx = [-self.bounds,-self.bounds,0,self.v_min,self.v_min,self.v_min,-1,-1,-1,-1,self.w_min,self.w_min,self.w_min] * ((1+self.horizon_steps))
        self.ubx = [self.bounds,self.bounds,self.bounds,self.v_max,self.v_max,self.v_max,1,1,1,1,self.w_max,self.w_max,self.w_max] * ((self.horizon_steps+1))

        # Constraints on the optimization variable: Control vector
        self.lbx = self.lbx + [self.thrust_min,self.tau_min,self.tau_min,self.tau_min]*(self.horizon_steps)
        self.ubx = self.ubx + [self.thrust_max,self.tau_max,self.tau_max,self.tau_max]*(self.horizon_steps)
        
        # Creating a control prediction matrix which will be used later for the optimization!!
        self.u0 = DM.zeros(self.n_controls, self.horizon_steps)  # Initial control guess
        #self.u0[0,:] = 40000 + self.u0[0,:]

    def liftoff_callback(self,msg):
        self.get_logger().info('Checking if the liftoff has happened!')
        self.liftoff_flag = msg.liftoff

    def calc_velocities(self,x,y,z,new_orientation):

        """Do dirty differentiation for the calculation of velocities: translation and rotation!"""

        # If the velocities are being calculated for the first time then skip because the state initialization is 0
        # but the actual position in the 3D space is not 0!

        if self.first_step_check == False:

            # Concatenate the positions in an array for ease of calculation and storage later!
            new_pos = np.array([x,y,z])

            # Calculate the velocities using finite difference method!
            cf_state_vel_temp = (new_pos - self.cf_state_pos)/(self.vicon_topic_update_time)

            # Converting quaternion to euler to get velocities. 
            roll, pitch, yaw = self.quat2euler(new_orientation[0], new_orientation[1], new_orientation[2], new_orientation[3])
            roll1, pitch1, yaw1 = self.quat2euler(self.cf_state_orientation[0], self.cf_state_orientation[1], self.cf_state_orientation[2], self.cf_state_orientation[3])

            cf_rot_vel_temp = (np.array([roll, pitch, yaw]) - np.array([roll1, pitch1, yaw1]))/(self.vicon_topic_update_time)
            self.get_logger().info(str(cf_rot_vel_temp))
            # Pass the velocities through a discrete low pass filter!
            self.cf_state_vel = self.alpha*cf_state_vel_temp + (1-self.alpha)*self.cf_state_vel
            self.cf_rot_vel = self.alpha*cf_rot_vel_temp + (1-self.alpha)*self.cf_rot_vel
            
            # Concatenate the true states and publish them to plotjuggler for analysis!
            self.get_logger().info(str( [new_pos[0],new_pos[1],new_pos[2],self.cf_state_vel[0],self.cf_state_vel[1],self.cf_state_vel[2],
                                        new_orientation[0],new_orientation[1],new_orientation[2], new_orientation[3], self.cf_rot_vel[0],
                                        self.cf_rot_vel[1],self.cf_rot_vel[2]]))
            
            # Publish the message for the Plotjuggler to record it!
            self.true_state_publisher.publish(self.true_state_msg)
        
        else:
            self.first_step_check = False


    def cf_vicon_callback(self,msg_in):

        """ Populating the position and orientation of the crazyflie"""

        self.get_logger().info("Updating Crazyflie's pose")

        # Dividing the translation measurements of x, y, z from vicon by 1000 to convert from mm to m! 
        new_orientation = np.array([msg_in.w, msg_in.x_rot, msg_in.y_rot, msg_in.z_rot])

        # Calculate the velocities and update state variables!
        self.calc_velocities(msg_in.x_trans/1000, msg_in.y_trans/1000, msg_in.z_trans/1000,new_orientation)
        self.cf_state_pos = np.array([msg_in.x_trans/1000, msg_in.y_trans/1000, msg_in.z_trans/1000])
        self.cf_state_orientation = new_orientation/sqrt(sumsqr(new_orientation))

        self.get_logger().info(str(self.cf_state_orientation[0]))
        
        # Update the state of the crazyflie!
        self.x0 = DM([self.cf_state_pos[0], self.cf_state_pos[1], self.cf_state_pos[2],
                    self.cf_state_vel[0], self.cf_state_vel[1], self.cf_state_vel[2],
                    self.cf_state_orientation[0], self.cf_state_orientation[1], self.cf_state_orientation[2], self.cf_state_orientation[3],
                    self.cf_rot_vel[0], self.cf_rot_vel[1], self.cf_rot_vel[2]])  # state update!
        
        # Update the guess for the next optimization run only for the first initialization though!
        if self.opt_var_update_first_step == True:
            self.X0 = np.tile(self.x0.full(), (1, self.horizon_steps + 1))

            # VERY IMPORTANT!!!!! Improve the guess by a simple linear interpolation between the final and target position!
            delta_x = (self.xf - self.x0)/self.horizon_steps
            for i in range(self.horizon_steps):
                self.X0[0:3,i+1] = (delta_x[0:3]*i + self.x0[0:3]).full().squeeze()
            self.opt_var_update_first_step = False 

    def target_vicon_callback(self,msg_in):

        """ Populating the position of the target"""

        self.get_logger().info("Updating target's pose")

        # Adding 1 meter elevation to the altitude of the target so the drone doesn't collide with the rccar (target)
        self.target_pos = np.array([msg_in.x_trans/1000, msg_in.y_trans/1000, 1+msg_in.z_trans/1000])
        self.xf = DM([self.target_pos[0], self.target_pos[1], self.target_pos[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Target state update
        
        # Publish the target pose to the plotjuggler!
        self.target_msg.state = [self.target_pos[0],self.target_pos[1],self.target_pos[2], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.target_publisher.publish(self.target_msg)

    def timer_callback(self):

        """ This is where the optimization problem will be solved! """
        
        # Running the criteria for which the MPC is solved!

        if self.convergance == False and self.liftoff_flag == True:

            """ Start the optimization sovler! """
            self.get_logger().info("Solving the MPC!")

            # Define the parameters for the MPC
            p_numeric = vertcat(self.x0, self.xf)

            # Initial guess for optimization variables
            opt_init = vertcat(reshape(self.X0, self.n_states * (self.horizon_steps + 1), 1), reshape(self.u0, self.n_controls * self.horizon_steps, 1))

            # Solve the NLP problem
            sol = self.solver(x0=opt_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p_numeric)

            # Extract the solution
            solution = sol['x'].full()

            # Print the solver stats: Solution status & time took to get the solution
            stats = self.solver.stats()
            self.get_logger().info(f"Solver status: {stats['return_status']}")
            self.get_logger().info(f"Sol time: {stats['t_wall_total']}")

            # Extract states and control inputs from the solution
            state_values = reshape(solution[:self.n_states * (self.horizon_steps + 1)], self.n_states, self.horizon_steps + 1)
            control_values = reshape(solution[self.n_states * (self.horizon_steps + 1):], self.n_controls, self.horizon_steps)

            # Extract the commands that are to be sent to the crazyflie!
            first_state = state_values[:,1].full()
            u = control_values[:, 0]

            # Converting quaternions to euler angles:
            phi, theta, _ = self.quat2euler(first_state[6], first_state[7], first_state[8], first_state[9])
            
            # Update the custom message with mpc solution!
            self.mpc_sol_msg.roll = math.degrees(phi)
            self.mpc_sol_msg.pitch = math.degrees(theta)
            self.mpc_sol_msg.yaw_rate = math.degrees(first_state[12])
            self.mpc_sol_msg.thrust = int(u[0].full().item()*(65535/self.thrust_max))

            # Publish the mpc_solution!
            self.mpc_publisher.publish(self.mpc_sol_msg)

            # Update the custom message with the optimized thrust and torques!
            self.optimized_solution.thrust = u[0].full().item()
            self.optimized_solution.tau_x = u[1].full().item()
            self.optimized_solution.tau_y = u[2].full().item()
            self.optimized_solution.tau_z = u[3].full().item()

            # Publish the optimized solution
            self.optimized_solution_publisher.publish(self.optimized_solution)

            # Record the estimated state of the crazyflie from the model in MPC!
            self.model_state_msg0.state = (state_values[:,0].full()).flatten().tolist()
            self.model_state_msg1.state = first_state.flatten().tolist()
            self.model_state_msg2.state = (state_values[:,2].full()).flatten().tolist()
            self.model_state_msg3.state = (state_values[:,3].full()).flatten().tolist()
            self.model_state_msg4.state = (state_values[:,4].full()).flatten().tolist()
            self.model_state_msg5.state = (state_values[:,5].full()).flatten().tolist()

            # Publish the message so that the plotjuggler can record it. 
            self.model_state_publisher0.publish(self.model_state_msg0)
            self.model_state_publisher1.publish(self.model_state_msg1)
            self.model_state_publisher2.publish(self.model_state_msg2)
            self.model_state_publisher3.publish(self.model_state_msg3)
            self.model_state_publisher4.publish(self.model_state_msg4)
            self.model_state_publisher5.publish(self.model_state_msg5)

            # Reinitialize guesses (both state and control input) for next iteration
            self.u0 = horzcat(control_values[:, 1:], control_values[:, -1])
            self.X0 = horzcat(state_values[:, 1:], state_values[:, -1])
            
            # Criteria for convergance of the MPC solution!
            if norm_2(self.x0[0:3] - self.xf[0:3]) < 0.01:

                # Turn on the convergance flag for the MPC to stop solving!
                self.convergance = True

                # Sending a constant thrust command for the crazyflie to hover in place!
                self.mpc_sol_msg.roll = 0.0
                self.mpc_sol_msg.pitch = 0.0
                self.mpc_sol_msg.yaw_rate = 0.0
                self.mpc_sol_msg.thrust = 40000
                self.mpc_publisher.publish(self.mpc_sol_msg)
        else:
            # Sending a constant thrust command for the crazyflie to hover in place!
            self.mpc_sol_msg.roll = 0.0
            self.mpc_sol_msg.pitch = 0.0
            self.mpc_sol_msg.yaw_rate = 0.0
            self.mpc_sol_msg.thrust = 40000
            self.mpc_publisher.publish(self.mpc_sol_msg)

    def quaternion_to_rotation(self, e0, e1, e2, e3):
        """
        converts a quaternion attitude to a rotation matrix
        """

        R = np.array([[e1 ** 2.0 + e0 ** 2.0 - e2 ** 2.0 - e3 ** 2.0, 2.0 * (e1 * e2 - e3 * e0), 2.0 * (e1 * e3 + e2 * e0)],
                    [2.0 * (e1 * e2 + e3 * e0), e2 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e3 ** 2.0, 2.0 * (e2 * e3 - e1 * e0)],
                    [2.0 * (e1 * e3 - e2 * e0), 2.0 * (e2 * e3 + e1 * e0), e3 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e2 ** 2.0]])
        R = R/det(R)

        return R

    def quat2euler(self,e0,e1,e2,e3):
        """
        converts a quaternion attitude to an euler angle attitude
        :param quaternion: the quaternion to be converted to euler angles in a np.matrix
        :return: the euler angle equivalent (phi, theta, psi) in a np.array
        """
        
        phi = np.arctan2(2.0 * (e0 * e1 + e2 * e3), e0**2.0 + e3**2.0 - e1**2.0 - e2**2.0)
        theta = np.arcsin(2.0 * (e0 * e2 - e1 * e3))
        psi = np.arctan2(2.0 * (e0 * e3 + e1 * e2), e0**2.0 + e1**2.0 - e2**2.0 - e3**2.0)

        # Return the euler angles!
        return phi, theta, psi

    def runge_kutta_4(self,state,forces_moments):

        # Look at semi-implicit euler!

        """ Integration routine (Rk4) to solve the differential equations"""
        
        step_time = self.dt/1000
        k1 = self.dynamics(state, forces_moments)
        k2 = self.dynamics(state + step_time/2.*k1, forces_moments)
        k3 = self.dynamics(state + step_time/2.*k2, forces_moments)
        k4 = self.dynamics(state + step_time*k3, forces_moments)

        # Updated state after integration!
        state += step_time/6 * (k1 + 2*k2 + 2*k3 + k4)

        # Normalize the quaternion!
        e0 = state[6]
        e1 = state[7]
        e2 = state[8]
        e3 = state[9]

        normE = sqrt(sumsqr(state[6:10]))#(e0**2 + e1**2 + e2**2 + e3**2)**0.5 # Finding the norm!

        state[6] = state[6]/normE
        state[7] = state[7]/normE
        state[8] = state[8]/normE
        state[9] = state[9]/normE

        return state

    def dynamics(self,state,control):

        """ This is the full 12 state dynamics used for the model in MPC declaration"""
        
        # Extract the state vector!
        u = state[3] # X velocity
        v = state[4] # Y velocity ...
        w = state[5]
        e0 = state[6]
        e1 = state[7]
        e2 = state[8]
        e3 = state[9]
        p = state[10]
        q = state[11]
        r = state[12]

        # Extract the control actuation!
        thrust = control[0]
        tau_phi = control[1]
        tau_theta = control[2]
        tau_psi = control[3]

        # Get the position derivatives
        pos_dot = self.quaternion_to_rotation(e0, e1, e2, e3) @ np.array([
            [u],[v],[w]
            ])
        
        # Rotational kinematics
        orientation_dot = .5*np.array([
            [0, -p, -q, -r],
            [p, 0, r, -q],
            [q, -r, 0, p],
            [r, q, -p, 0]
        ]) @ np.array([
            [e0],[e1],[e2],[e3]
        ])
        
        # Get the euler angles from the present quaternion!
        phi, theta, psi = self.quat2euler(e0, e1, e2, e3)

        # Translational accelerations considering thrust and angular velocities
        acc_x = (thrust / self.m) * (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)) + (r * v - q * w)
        acc_y = (thrust / self.m) * (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)) + (p * w - r * u)
        acc_z = (thrust / self.m) * (cos(phi) * cos(theta)) - self.gravity + (q * u - p * v)
        
        # Rotational dynamics
        p_dot = (self.I_y - self.I_z) / self.I_x * q * r + tau_phi / self.I_x
        q_dot = (self.I_z - self.I_x) / self.I_y * p * r + tau_theta / self.I_y
        r_dot = (self.I_x - self.I_y) / self.I_z * p * q + tau_psi / self.I_z

        # Return the 12 differential equations!
        return vertcat(pos_dot[0], pos_dot[1], pos_dot[2], acc_x, acc_y, acc_z, orientation_dot[0], orientation_dot[1], orientation_dot[2], orientation_dot[3], p_dot, q_dot, r_dot)

# Define the main function to run the node!
def main(args=None):

    rclpy.init(args=args)
    node = solve_mpc()
    rclpy.spin(node)
    node.destroy_node()
    node.shutdown()
        
if __name__ == '__main__':
    main()