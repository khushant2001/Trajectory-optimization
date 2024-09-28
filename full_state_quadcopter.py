from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Setting the initial parameters. 

gravity = 9.81  # gravity
step_time = 0.2  # sec, time step for MPC
drone_radius = 0.1 # m

# Drone parameters!
I_x = 2.4*10**(-5)  # moment of inertia along x-axis
I_y = I_x # moment of inertia along y-axis
I_z = 3.2*10**(-5)  # moment of inertia along z-axis
m = .027 # mass (kg)
bounds = 2
v_max = 1
v_min = -1
w_max = 10.47
w_min = -10.47
a_max = 4
a_min = -4
w_dot_max = 17.45
w_dot_min = -17.45
thrust_max = 1.9*m*gravity # 1.9 is the thrust to weight
thrust_min = -thrust_max
tau_max = 0.0097
tau_min = -tau_max

def full_state_quad():
    
    horizon_steps = 25  # MPC horizon

    # State variables: x, y, z, psi, phi, theta, x_dot, y_dot, z_dot, r, p, q
    x = SX.sym('x')       # Position x
    y = SX.sym('y')       # Position y
    z = SX.sym('z')       # Position z
    psi = SX.sym('psi')   # Yaw
    phi = SX.sym('phi')   # Roll
    theta = SX.sym('theta')  # Pitch
    x_dot = SX.sym('x_dot')  # Velocity x
    y_dot = SX.sym('y_dot')  # Velocity y
    z_dot = SX.sym('z_dot')  # Velocity z
    r = SX.sym('r')       # Yaw rate
    p = SX.sym('p')       # Roll rate
    q = SX.sym('q')       # Pitch rate

    # Control inputs: thrust, tau_phi (roll torque), tau_theta (pitch torque), tau_psi (yaw torque)
    thrust = SX.sym('thrust')  
    tau_phi = SX.sym('tau_phi')  # Roll torque
    tau_theta = SX.sym('tau_theta')  # Pitch torque
    tau_psi = SX.sym('tau_psi')  # Yaw torque

    # State and control vectors
    state_vector = vertcat(x, y, z, psi, phi, theta, x_dot, y_dot, z_dot, r, p, q)
    control_vector = vertcat(thrust, tau_phi, tau_theta, tau_psi)
    n_states = state_vector.size1()
    n_controls = control_vector.size1()

    # Define the state and control prediction matrices!

    state_prediction = SX.sym('X', n_states,horizon_steps+1)
    control_prediction = SX.sym('U',n_controls,horizon_steps)

    # Define the dynamics for differential flatness!
    R11 = cos(theta) * cos(psi)
    R12 = -sin(psi)
    R13 = sin(theta) * cos(psi)

    R21 = cos(theta) * sin(psi)
    R22 = cos(psi)
    R23 = sin(theta) * sin(psi)

    R31 = -sin(theta)
    R32 = 0
    R33 = cos(theta)

    # Define the differential equations
    rhs = vertcat(
        # Linear dynamics
        # Translational dynamics in the inertial frame
        R11 * x_dot + R12 * y_dot + R13 * z_dot,
        R21 * x_dot + R22 * y_dot + R23 * z_dot,
        R31 * x_dot + R32 * y_dot + R33 * z_dot,
        # Translational accelerations considering thrust and angular velocities
        (thrust / m) * (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)) - (q * z - r * y),
        (thrust / m) * (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)) + (q * x - r * z),
        (thrust / m) * (cos(phi) * cos(theta)) - gravity + (p * y - q * x),
        p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta),
        q * cos(phi) - r * sin(phi),
        (q * sin(phi) + r * cos(phi)) / cos(theta),
        # Rotational dynamics
        (I_y - I_z) / I_x * q * r + tau_phi / I_x,
        (I_z - I_x) / I_y * p * r + tau_theta / I_y,
        (I_x - I_y) / I_z * p * q + tau_psi / I_z
    ) 

    # Define the function that will calculate the dynamics
    dynamics = Function('dynamics',[state_vector,control_vector],[rhs])

    # Define the parameter vector which will contain the final and initial state!
    p = SX.sym('p',n_states*2) # Multipled by 2 because you are storing the initial and final state!

    # Define the constraints and cost function!!!
    g = (state_prediction[:,0] - p[0:n_states]) # Constraints on the functions of optimal variables.
    obj = 0 # Cost function!!!

    # Define the weighting matrices!
    Q = 100*DM.eye(n_states)
    R = .1*DM.eye(n_controls)

    # Start the for loop to build up the constraint vector and the cost function!!!
    for i in range(horizon_steps):
        curr_state = state_prediction[:,i]
        curr_input = control_prediction[:,i]
        obj = obj + ((curr_state - p[n_states:n_states*2]).T @ Q @ (curr_state - p[n_states:2*n_states])) + curr_input.T @ R @ curr_input
        next_state_estimate = runge_kutta_4(curr_state,curr_input)
        g = vertcat(g,state_prediction[:,i+1] - next_state_estimate) # Adding up the constraints in the g function!


    #########################################################
    # Adding constraints for the obstacle avoidance
    
    obs_x = [.3,.5,.7]
    obs_y = [.3,.5,.7]
    obs_z = [0,-.8,-.7]
    r0 = [.1,.1,.1]
    for i in range(horizon_steps+1):
        for j in range(len(obs_x)):           
            constraint = -((state_prediction[0,i]-obs_x[j])**2 + (state_prediction[1,i]-obs_y[j])**2 + (state_prediction[2,i]-obs_z[j])**2) + (drone_radius+r0[j])**2
            g = vertcat(g,constraint)
    #########################################################

    # Now lets define the non linear program structure!
    opt_vars = vertcat(reshape(state_prediction,n_states*(horizon_steps + 1),1),reshape(control_prediction,n_controls*horizon_steps,1)) # You are not just optimizing v and w but all instances of them!
    nlp_prob = {}
    nlp_prob['x'] = opt_vars
    nlp_prob['f'] = obj
    nlp_prob['g'] = g
    nlp_prob['p'] = p # This will change so might have to redefine the nlp_prob!

    # Initializing the solver!
    solver = nlpsol('solver','ipopt',nlp_prob)

    #Populate the constraints and the dynamics with their upper/lower bounds!Defining constraints for the NLP!

    lbg = [0] * g.size1() # Constraints on the dynamics and obstacle condition!
    ubg = [0] * g.size1() # Constraints on the dynamics and obstacle condition!

    # Constraints for the obstacle avoidance. The upper bound is 0 and the lower bound -inf.
    for i in range(n_states*(1+horizon_steps),(n_states+len(obs_x))*(1+horizon_steps)):
        lbg[i] = -inf

    # Constraints on the optimization variable: STATE VARIABLES. 
    lbx = [-bounds,-bounds,-bounds,v_min,v_min,v_min,-inf,-inf,-inf,w_min,w_min,w_min] * ((1+horizon_steps))
    ubx = [bounds,bounds,bounds,v_max,v_max,v_max,inf,inf,inf,w_max,w_max,w_max] * ((horizon_steps+1))

    # Constraints on the optimization variable: Control vector
    lbx = lbx + [thrust_min,tau_min,tau_min,tau_min]*(horizon_steps)
    ubx = ubx + [thrust_max,tau_max,tau_max,tau_max]*(horizon_steps)

    # SIMULATION START! Simulation parameters
    t0 = 0
    t = []  # Time history
    x_hist, y_hist, z_hist = [], [], []
    x0 = DM([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Initial state
    xf = DM([1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Target state
    u0 = DM.zeros(n_controls, horizon_steps)  # Initial control guess
    X0 = np.tile(x0.full(), (1, horizon_steps + 1))
    optimized_inputs = np.array([[0,0,0,0]])
    angles = np.array([[0,0,0]])

    # Setup plot
    plt.ion()
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    
    while norm_2(x0[0:3] - xf[0:3]) > 0.01 and t0 < 20:
        # Define the parameters for the MPC
        p_numeric = vertcat(x0, xf)

        # Initial guess for optimization variables
        opt_init = vertcat(reshape(X0, n_states * (horizon_steps + 1), 1), reshape(u0, n_controls * horizon_steps, 1))

        # Solve the NLP problem
        sol = solver(x0=opt_init, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_numeric)
        solution = sol['x'].full()

        # Extract states and control inputs from the solution
        state_values = reshape(solution[:n_states * (horizon_steps + 1)], n_states, horizon_steps + 1)
        control_values = reshape(solution[n_states * (horizon_steps + 1):], n_controls, horizon_steps)

        # Update the state with first control input
        u = control_values[:, 0]
        x0 = runge_kutta_4(x0.full(),u.full())
        
        # Reinitialize guesses for next iteration
        u0 = horzcat(control_values[:, 1:], control_values[:, -1])
        X0 = horzcat(state_values[:, 1:], state_values[:, -1])

        # Store the positions for plotting
        x_hist.append(float(x0[0]))  # Convert symbolic expression to numerical value
        y_hist.append(float(x0[1]))
        z_hist.append(float(x0[2]))
        optimized_inputs = np.vstack((optimized_inputs,u.T))
        angles = np.vstack((angles,x0[6:9].T))
        t0 += step_time
        t.append(t0)

        # Inside your while loop, replace the plotting section with the following:

        # Plot the drone's trajectory
        ax.clear()
        # Draw the obstacles
        for xi, yi, zi,ri in zip(obs_x, obs_y, obs_z,r0):
            plot_sphere(ax,xi,yi,zi,ri)

        ax.plot(x_hist, y_hist, z_hist, label='Trajectory')

        ax.scatter(xf[0], xf[1], xf[2], label='Target')

        # Plot the predicted trajectory
        ax.scatter(np.array(state_values[0, :].full()).flatten(),
                np.array(state_values[1, :].full()).flatten(),
                np.array(state_values[2, :].full()).flatten(), 
                label='Predicted trajectory', color='green')

        # Represent the drone's current position with an 'X'
        ax.text(float(x0[0]), float(x0[1]), float(x0[2]), 'X', color='r', fontsize=18, ha='center', va='center')

        # Set plot limits and labels
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')

        ax.legend()
        plt.pause(0.3)
    

    # Plottingt the optimized input!
    optimized_inputs = np.delete(optimized_inputs,0,axis = 0)
    plt.ioff()  # Disable interactive mode for final plot
    plt.figure()
    #plt.plot(t,optimized_inputs[:,0],label = 'Thrust')
    plt.plot(t,optimized_inputs[:,1],label = 'Torque phi')
    plt.plot(t,optimized_inputs[:,2],label = 'Torque theta')
    plt.plot(t,optimized_inputs[:,3],label = 'Torque psi')
    plt.grid()
    plt.legend()
    plt.title('Optimized control inputs')

    angles = np.delete(angles,0,axis = 0)
    plt.figure()
    plt.plot(t,(180/pi)*angles[:,0],label = 'Phi')
    plt.plot(t,(180/pi)*angles[:,1],label = 'Theta')
    plt.plot(t,(180/pi)*angles[:,2],label = 'Psi')
    plt.grid()
    plt.legend()
    plt.title('Orientation [deg]')
    plt.show()

def plot_sphere(ax, center, radius, color='b', alpha=0.5):
    # Create a grid of points
    u = np.linspace(0, 2 * np.pi, 100)  # Angle around the z-axis
    v = np.linspace(0, np.pi, 100)      # Angle from the z-axis down

    # Parametric equations for the sphere
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface of the sphere
    ax.plot_surface(x, y, z, color=color, alpha=alpha)
         
def dynamics(state,control):
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
    acc_x = (thrust / m) * (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)) - (q * z - r * y)
    acc_y = (thrust / m) * (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)) + (q * x - r * z)
    acc_z = (thrust / m) * (cos(phi) * cos(theta)) - gravity + (p * y - q * x)
    
    # Rotational dynamics
    p_dot = (I_y - I_z) / I_x * q * r + tau_phi / I_x
    q_dot = (I_z - I_x) / I_y * p * r + tau_theta / I_y
    r_dot = (I_x - I_y) / I_z * p * q + tau_psi / I_z

    
    return vertcat(x_dot, y_dot, z_dot, acc_x, acc_y, acc_z, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot)

def plot_sphere(ax, x_center, y_center, z_center, radius):
    """ Function to plot a sphere given a center and a radius """
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = x_center + radius * np.cos(u) * np.sin(v)
    y = y_center + radius * np.sin(u) * np.sin(v)
    z = z_center + radius * np.cos(v)
    ax.plot_surface(x, y, z, color="r", alpha=1)

def runge_kutta_4(state,forces_moments):
    k1 = dynamics(state, forces_moments)
    k2 = dynamics(state + step_time/2.*k1, forces_moments)
    k3 = dynamics(state + step_time/2.*k2, forces_moments)
    k4 = dynamics(state + step_time*k3, forces_moments)
    state += step_time/6 * (k1 + 2*k2 + 2*k3 + k4)
    return state

full_state_quad()