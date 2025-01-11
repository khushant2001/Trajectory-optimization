"""
This is an example script of trajectory optimization using MPC and dynamic programming on a toy car!
"""

from casadi import *
import matplotlib.pyplot as plt
import numpy as np

def toy_car_single_shooting():

    # Define the dynamics!!!
    # x_dot = vcos(theta)
    # y_dot = vsin(theta)
    # theta_dot = omega
    
    step_time = 0.2 #sec when does the observation come!
    horizon_steps = 6 # how many steps should the controller look into the future.
    v_min = -.6
    v_max = .6
    omega_min = -np.pi/4
    omega_max = np.pi/4
    # Define the variables!
    x = SX.sym('x')
    y = SX.sym('y')
    theta = SX.sym('theta')
    v = SX.sym('v')
    w = SX.sym('w')

    # Make up the vectors!
    state_vector = vertcat(x,y,theta)
    control_vector = vertcat(v,w)
    n_states = state_vector.size1()
    n_control = control_vector.size1()

    # Define the dynamics using a CasADi function!
    rhs = vertcat(v*cos(theta),v*sin(theta),w)
    dynamics = Function('dynamics',[state_vector,control_vector],[rhs])

    # Define the prediction vectors!
    state_prediction = SX.sym('X',n_states,horizon_steps+1) # 3*4 matrix where each column is the state vector and each row is the horizon step!
    control_prediction = SX.sym('U',n_control,horizon_steps)
    
    # Define the parameters vector: initial and final point!
    p = SX.sym('p',n_states*2)

    # Populate the first entry of the state_prediction matrix with the initial condition.
    state_prediction[:,0] = p[0:3]

    # Populating the state_prediction matrix!
    for i in range(horizon_steps):
        state_prediction[:,i+1] = state_prediction[:,i] + step_time*dynamics(state_prediction[:,i],control_prediction[:,i])

    # Defining a function which automates the filling up of the state_prediction vector
    state_prediction_f = Function('state_prediction_f',[control_prediction,p],[state_prediction])

    # Define the objective function: which is basically a quadratic cost function
    Q = DM([
        [.1,0,0],   # Increased weight on x and y for faster convergence
        [0,10,0],
        [0,0,.1]
    ])

    R = DM([
        [0.01,0],   # Lowered penalty on control inputs
        [0,0.01]
    ])

    # Sum over the entire cost function!
    obj = 0
    for i in range(horizon_steps):
        curr_state = state_prediction[:,i]
        curr_input = control_prediction[:,i]
        obj = obj + ((curr_state - p[3:6]).T @ Q @ (curr_state - p[3:6])) + curr_input.T @ R @ curr_input

    # Define the constraint vector g: which is the constraint on the function of input variables: dynamics
    g = reshape(state_prediction[0:2, :], -1, 1)
    
    # Now lets define the non linear program structure!
    opt_vars = vertcat(reshape(control_prediction, -1, 1)) # You are not just optimizing v and w but all instances of them!
    nlp_prob = {}
    nlp_prob['x'] = opt_vars
    nlp_prob['f'] = obj
    nlp_prob['g'] = g
    nlp_prob['p'] = p # This will change so might have to redefine the nlp_prob!

    # Initializing the solver!
    solver = nlpsol('solver','ipopt',nlp_prob)

    # Defining constraints for the NLP!
    lbg = -2 # so that the x and y are below -2!
    ubg = 2 # so that the x and y are below 2!

    lbx = [v_min if i % 2 == 0 else omega_min for i in range(n_control * horizon_steps)]
    ubx = [v_max if i % 2 == 0 else omega_max for i in range(n_control * horizon_steps)]
    
    # SET UP THE SIMULATION!!!
    t0 = 0 # sec
    t = [] # time history
    x_1 = [] # state history for x
    x_2 = [] # state history for y
    x0 = DM([0, 0, 0])  # Initial state
    xf = DM([1, 1, pi])  # Target state
    
    # Update the arrays with initializations above.
    x_1.append(x0.full()[0])
    x_2.append(x0.full()[1])
    t.append(t0)
    u0 = DM([0, 0] * horizon_steps).T  # Initial control guess
    v_history = [[0]]
    w_history = [[0]]

    # Set up the figure for animation
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    car_radius = 0.1
    car_circle = plt.Circle((x0.full()[0], x0.full()[1]), car_radius, color='blue', alpha=0.5)
    ax.add_patch(car_circle)

    # Initial perpendicular line to represent heading
    line_length = 0.2
    line_x_end = x0[0].full()[0] + line_length * np.cos(x0[2].full()[0])
    line_y_end = x0[1].full()[0] + line_length * np.sin(x0[2].full()[0])
    car_heading, = ax.plot([x0[0].full()[0], line_x_end], [x0[1].full()[0], line_y_end], color='black')

    # Run the simulation untill the time converges to final sim time or the car reaches its final state within some bounds.
    while norm_2(x0 - xf) > 0.01 and t0 < 20:
    
        # Update the parameters! 
        p_numeric = vertcat(x0, xf)

        # Give an initial estimate for the optimization variables!
        u_0 = reshape(u0, 2*horizon_steps, 1) # initial value of the optimization variables!

        # Solve the optimization problem!
        sol = solver(x0=u_0, lbx=lbx, ubx=ubx, ubg=ubg, lbg=lbg, p=p_numeric)

        # Extract the optimal output!
        u = reshape(sol['x'], 2, horizon_steps) # Optimal solution
        ff_value = state_prediction_f(u, p_numeric)

        # Save control inputs
        v_history.append(u[:,0].full()[0])
        w_history.append(u[:,0].full()[1])

        # Update the state
        x0 = ff_value[:,1]

        # Re-initialize the guess for optimization variables!
        u0 = horzcat(u[:, 1:], u[:, -1])

        # Append updated state and time. Also record them in their corresponding list!
        x_1.append(x0.full()[0])
        x_2.append(x0.full()[1])
        t0 += step_time
        t.append(t0)

        # Clear the plot and update the car's trajectory
        plt.clf()
        plt.plot(x_1, x_2, label='Trajectory', color='blue')
        plt.scatter(ff_value[0, :].full(), ff_value[1, :].full(), label='Predicted states', color='green')  # Predicted points

        # Plot the car as a circle with a line for heading (orientation)
        car_x = x0[0].full()[0]
        car_y = x0[1].full()[0]
        car_theta = x0[2].full()[0]
        
        # Car circle
        plt.gca().add_patch(plt.Circle((car_x, car_y), 0.05, color='blue', alpha=0.5))

        # Line extending from the center of the car circle to represent orientation
        line_length = 0.1  # Length of the orientation line
        line_x_end = car_x + line_length * np.cos(car_theta)
        line_y_end = car_y + line_length * np.sin(car_theta)
        plt.plot([car_x, line_x_end], [car_y, line_y_end], color='black')

        # Plot the reference state as a circle with a line for its heading
        ref_x = xf.full()[0]
        ref_y = xf.full()[1]
        ref_theta = xf.full()[2]  # Assuming that xf contains the reference orientation

        # Reference circle
        plt.gca().add_patch(plt.Circle((ref_x, ref_y), 0.05, color='red', alpha=0.5))

        # Line extending from the reference circle to represent the reference orientation
        ref_line_x_end = ref_x + line_length * np.cos(ref_theta)
        ref_line_y_end = ref_y + line_length * np.sin(ref_theta)
        plt.plot([ref_x, ref_line_x_end], [ref_y, ref_line_y_end], color='red',label = 'Reference')

        plt.legend()
        plt.grid()
        plt.title('Car trajectory')
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)
        plt.pause(0.3)  # Slows down the loop for visualization


    # Plot inputs (v and omega)
    plt.ioff()  # Disable interactive mode for final plot
    plt.figure()
    plt.plot(t, v_history, label='Velocity')
    plt.plot(t, w_history, label='Omega')
    plt.grid()
    plt.legend()
    plt.title('Inputs')
    plt.show()