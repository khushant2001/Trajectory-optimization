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

class hjb_traj_jen(Node):

    def optimization_problem(self):
        
        """Formulating the optimization problem and declaring the corresponding parameters!"""

        self.get_logger().info("Initializing the optimization problem")

        # Horizon steps to look into the future!
        self.horizon_steps = 20

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

        # State and control vectors lengths
        self.n_states = 12
        self.n_controls = 4

        # Define the state and control prediction matrices!
        self.state_prediction = SX.sym('X', self.n_states,self.horizon_steps+1)
        self.control_prediction = SX.sym('U',self.n_controls,self.horizon_steps)

        # Populating the state_prediction matrix!
        for i in range(self.horizon_steps-1):
            self.state_prediction[:,i+1] = self.runge_kutta_4(self.state_prediction[:,i], self.control_prediction[:,i])

        # Define the parameter vector which will contain the final and initial state!
        self.p = SX.sym('p',self.n_states*2) # Multipled by 2 because you are storing the initial and final state!

        self.obj = 0 # Cost function!!!

        # Define the weighting matrices!
        Q = 100*DM.eye(self.n_states)
        R = 100*DM.eye(self.n_controls)

        # Weight matrix for the terminal cost!
        T = 100*DM.eye(self.n_states)

        # Actual Terminal cost
        l_terminal = ((self.state_prediction[self.horizon_steps-1] - self.p[3:6]).T @ T @ (self.state_prediction[self.horizon_steps-1] - self.p[3:6]))

        # Making a dictionary to store the cost at individual . This will help for recursive cost propagation!
        V = dict()
        V[self.steps-1] = l_terminal # Cost at the terminal step!

        # Initiating the cost function!
        obj = l_terminal

        # Summing over the steps to build up the cost function!
        for i in range(self.horizon_steps-2, -1, -1):
            curr_state = self.state_prediction[:,i]
            curr_input = self.control_prediction[:,i]
            cost = ((curr_state - self.p[3:6]).T @ Q @ (curr_state - self.p[3:6])) + curr_input.T @ R @ curr_input + V[i+1]
            V[i] = cost
            obj = obj + cost
        
        # Defining the cost function! Need to fix this!
        self.g = reshape(self.state_prediction[0:2, :], -1, 1)

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
        self.lbx = [-self.bounds,-self.bounds,-self.bounds,self.v_min,self.v_min,self.v_min,-inf,-inf,-inf,self.w_min,self.w_min,self.w_min] * ((1+self.horizon_steps))
        self.ubx = [self.bounds,self.bounds,self.bounds,self.v_max,self.v_max,self.v_max,inf,inf,inf,self.w_max,self.w_max,self.w_max] * ((self.horizon_steps+1))

        # Constraints on the optimization variable: Control vector
        self.lbx = self.lbx + [self.thrust_min,self.tau_min,self.tau_min,self.tau_min]*(self.horizon_steps)
        self.ubx = self.ubx + [self.thrust_max,self.tau_max,self.tau_max,self.tau_max]*(self.horizon_steps)
        
        # Creating a control prediction matrix which will be used later for the optimization!!
        self.u0 = DM.zeros(self.n_controls, self.horizon_steps)  # Initial control guess
        self.u0[0,:] = 40000 + self.u0[0,:]

    def quat2euler(self,w,x,y,z):

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

        # Look at semi-implicit euler!

        """ Integration routine (Rk4) to solve the differential equations"""
        
        step_time = self.dt/1000
        k1 = self.dynamics(state, forces_moments)
        k2 = self.dynamics(state + step_time/2.*k1, forces_moments)
        k3 = self.dynamics(state + step_time/2.*k2, forces_moments)
        k4 = self.dynamics(state + step_time*k3, forces_moments)

        # Updated state after integration!
        state += step_time/6 * (k1 + 2*k2 + 2*k3 + k4)
        return state

    def dynamics(self,state,control):

        """ This is the full 12 state dynamics used for the model in MPC declaration"""
        
        # Extract the state vector!
        x = state[0]  # Extract x
        y = state[1]  # Extract y
        z = state[2]  # Extract z ...
        x_dot = state[3]
        y_dot = state[4]
        z_dot = state[5]
        phi = state[6]
        theta = state[7]
        psi = state[8]
        p = state[9]
        q = state[10]
        r = state[11]

        # Extract the control actuation!
        thrust = control[0]
        tau_phi = control[1]
        tau_theta = control[2]
        tau_psi = control[3]

        # Define the entries of the rotation matrices!
        R11 = cos(theta) * cos(psi)
        R12 = sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)
        R13 = cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)

        R21 = cos(theta) * sin(psi)
        R22 = sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)
        R23 = cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)

        R31 = -sin(theta)
        R32 = sin(phi) * cos(theta)
        R33 = cos(phi)* cos(theta)
        
        # R11 = cos(theta) * cos(psi)
        # R12 = -sin(psi)
        # R13 = sin(theta) * cos(psi)

        # R21 = cos(theta) * sin(psi)
        # R22 = cos(psi)
        # R23 = sin(theta) * sin(psi)

        # R31 = -sin(theta)
        # R32 = 0
        # R33 = cos(theta)

        # Translational dynamics in the inertial frame
        x_dot = R11 * x_dot + R12 * y_dot + R13 * z_dot
        y_dot = R21 * x_dot + R22 * y_dot + R23 * z_dot
        z_dot = R31 * x_dot + R32 * y_dot + R33 * z_dot
        
        # Rotational kinematics
        phi_dot = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
        theta_dot = q * cos(phi) - r * sin(phi)
        psi_dot = (q * sin(phi) + r * cos(phi)) / cos(theta)
        
        # Translational accelerations considering thrust and angular velocities
        acc_x = (thrust / self.m) * (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)) - (q * z_dot - r * y_dot)
        acc_y = (thrust / self.m) * (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)) + (p * z_dot - r * x_dot)
        acc_z = (thrust / self.m) * (cos(phi) * cos(theta)) - self.gravity + (p * y_dot - q * x_dot)
        
        # Rotational dynamics
        p_dot = (self.I_y - self.I_z) / self.I_x * q * r + tau_phi / self.I_x
        q_dot = (self.I_z - self.I_x) / self.I_y * p * r + tau_theta / self.I_y
        r_dot = (self.I_x - self.I_y) / self.I_z * p * q + tau_psi / self.I_z

        # Return the 12 differential equations!
        return vertcat(x_dot, y_dot, z_dot, acc_x, acc_y, acc_z, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot)

# Define the main function to run the node!
def main(args=None):

    rclpy.init(args=args)
    node = hjb_traj_jen()
    rclpy.spin(node)
    node.destroy_node()
    node.shutdown()
        
if __name__ == '__main__':
    main()