"""
Another transcription method which treats the dyanmics as polynomial fittings. Making the solver faster.
Did not try this on the hardware yet! The HIL testing uses the multiple shooting transcription method!
"""
from casadi import *
import matplotlib.pyplot as plt
import numpy as np

def toy_car_obstacle_avoidance_direct_collocation():
    # Define the dynamics!!!
    # x_dot = vcos(theta)
    # y_dot = vsin(theta)
    # theta_dot = omega
    
    step_time = 0.2  # sec when does the observation come!
    horizon_steps = 100  # how many steps should the controller look into the future.
    n_collocation_points = 4  # Number of collocation points per time step

    # Limits on the control actions!
    v_min = -0.6
    v_max = 0.6
    omega_min = -np.pi / 2
    omega_max = np.pi / 2

    # For drawing the car!
    car_radius = 0.1

    # Define the variables!
    x = SX.sym('x')
    y = SX.sym('y')
    theta = SX.sym('theta')
    v = SX.sym('v')
    w = SX.sym('w')

    # Make up the vectors!
    state_vector = vertcat(x, y, theta)
    control_vector = vertcat(v, w)
    n_states = state_vector.size1()
    n_control = control_vector.size1()

    # Define the dynamics using a CasADi function!
    rhs = vertcat(v * cos(theta), v * sin(theta), w)
    dynamics = Function('dynamics', [state_vector, control_vector], [rhs])

    # Define the prediction vectors for states and controls
    state_prediction = SX.sym('X', n_states, horizon_steps + 1)
    control_prediction = SX.sym('U', n_control, horizon_steps)

    # Define the parameters vector: initial and final point!
    p = SX.sym('p', n_states * 2)

    # Define the constraint vector g
    g = (state_prediction[:, 0] - p[0:3])  # Initial constraint!

    # Define the objective function: which is basically a quadratic cost function
    Q = DM([[100, 0, 0], [0, 100, 0], [0, 0, 0.1]])
    R = DM([[0.01, 0], [0, 0.01]])

    # Objective function initialization
    obj = 0
    # Loop over time steps
    for i in range(horizon_steps):
        for j in range(n_collocation_points):
            # Compute collocation time
            tau = (j + 1) / n_collocation_points
            
            # Estimate the next state using a simple Euler integration
            if j == 0:
                curr_state = state_prediction[:, i]  # Current state at the start of the time step
            else:
                curr_state = state_prediction[:, i] + (step_time / n_collocation_points) * dynamics(curr_state, control_prediction[:, i])
            
            # Add state prediction for next collocation point
            g = vertcat(g, state_prediction[:, i + 1] - curr_state)

        # Objective accumulation
        obj += ((state_prediction[:, i] - p[3:6]).T @ Q @ (state_prediction[:, i] - p[3:6])) + (control_prediction[:, i].T @ R @ control_prediction[:, i])

    #########################################################
    # Adding constraints for the obstacle avoidance
    obs_x = [0.3, 0.5, 0.7, 0.7]
    obs_y = [0.3, 0.7, 0.2, 0.5]
    r0 = [0.1, 0.1, 0.1, 0.1]
    for i in range(horizon_steps + 1):
        for j in range(len(obs_x)):
            constraint = -((state_prediction[0, i] - obs_x[j]) ** 2 + (state_prediction[1, i] - obs_y[j]) ** 2) + (car_radius + r0[j]) ** 2
            g = vertcat(g, constraint)
    #########################################################

    # Now let's define the non-linear program structure!
    opt_vars = vertcat(reshape(state_prediction, 3 * (horizon_steps + 1), 1), reshape(control_prediction, 2 * horizon_steps, 1))
    nlp_prob = {}
    nlp_prob['x'] = opt_vars
    nlp_prob['f'] = obj
    nlp_prob['g'] = g
    nlp_prob['p'] = p  # This will change so might have to redefine the nlp_prob!

    # Initializing the solver!
    solver = nlpsol('solver', 'ipopt', nlp_prob)

    # Defining constraints for the NLP!
    lbg = [0] * g.size1()  # Constraints on the dynamics and obstacle condition!
    ubg = [0] * g.size1()  # Constraints on the dynamics and obstacle condition!

    # Set up the input bounds
    lbx = []
    ubx = []

    # Set up constraints on the states!
    for i in range(3 * (horizon_steps + 1)):
        # Constraint on first state: x!
        if i % 3 == 0:
            lbx.append(-2)
            ubx.append(2)
        # Constraint on second state: y!
        elif i % 3 == 1:
            lbx.append(-2)
            ubx.append(2)
        # Constraint on third state: theta!
        else:
            lbx.append(-np.inf)
            ubx.append(np.inf)

    # Set up constraints on the inputs!
    for i in range(2 * horizon_steps):
        if i % 2 == 0:
            lbx.append(v_min)
            ubx.append(v_max)
        else:
            lbx.append(omega_min)
            ubx.append(omega_max)

    # SET UP THE SIMULATION!!!
    t0 = 0  # sec
    t = []  # time history
    x_1 = []  # state history for x
    x_2 = []  # state history for y
    x0 = DM([0, 0, 0])  # Initial state
    xf = DM([1, 1, np.pi])  # Target state
    
    # Update the arrays with initializations above.
    x_1.append(x0.full()[0])
    x_2.append(x0.full()[1])
    t.append(t0)
    u0 = DM.zeros((2, horizon_steps))  # Initial control guess
    X0 = np.tile(x0.full(), (1, horizon_steps + 1))
    v_history = [[0]]
    w_history = [[0]]

    # Set up the figure for animation
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    car_circle = plt.Circle((x0.full()[0], x0.full()[1]), car_radius, color='blue', alpha=0.5)
    ax.add_patch(car_circle)

    # Initial perpendicular line to represent heading
    line_length = 0.2
    line_x_end = x0[0].full()[0] + line_length * np.cos(x0[2].full()[0])
    line_y_end = x0[1].full()[0] + line_length * np.sin(x0[2].full()[0])
    car_heading, = ax.plot([x0[0].full()[0], line_x_end], [x0[1].full()[0], line_y_end], color='black')

    # Run the simulation until the time converges to final sim time or the car reaches its final state within some bounds.
    while norm_2(x0 - xf) > 0.01 and t0 < 20:
        # Update the parameters! 
        p_numeric = vertcat(x0, xf)

        # Give an initial estimate for the optimization variables!
        u_0 = vertcat(reshape(X0, 3 * (1 + horizon_steps), 1), reshape(u0, 2 * horizon_steps, 1))  # initial value of the optimization variables!

        # Solve the optimization problem!
        sol = solver(x0=u_0, lbx=lbx, ubx=ubx, ubg=ubg, lbg=lbg, p=p_numeric)

        # Get the optimal output!
        full_solution = sol['x'].full().flatten()
        # Get the optimized states and control variables
        optimized_states = full_solution[:3 * (horizon_steps + 1)].reshape(3, horizon_steps + 1)
        optimized_controls = full_solution[3 * (horizon_steps + 1):].reshape(2, horizon_steps)

        # Update the current state
        x0 = optimized_states[:, 1]  # take the next state from the optimization
        u0 = optimized_controls[:, 0]  # take the first control from the optimization

        # Update history for plotting
        x_1.append(x0[0])
        x_2.append(x0[1])
        t.append(t0)

        # Update the graphical representation
        car_circle.set_center((x0[0], x0[1]))
        car_heading.set_xdata([x0[0], line_x_end])
        car_heading.set_ydata([x0[1], line_y_end])

        plt.draw()
        plt.pause(0.01)

        # Advance the time
        t0 += step_time

    plt.ioff()
    plt.show()

# Call the function to run the simulation
toy_car_obstacle_avoidance_direct_collocation()