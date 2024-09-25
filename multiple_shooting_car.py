from casadi import *
import matplotlib.pyplot as plt
import numpy as np

# Similar toy car problem but with multiple shooting!
### Difference: The dynamics are described as equality constraints at every single computation. Rather than non linear propagation of previous states!
## Meaning: now we are optimizing for not only control but also for states. This allows us to solve the MPC problem much faster!
def toy_car_multiple_shooting():
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

    # # Populating the state_prediction matrix! DONT NEED THIS NOW! because THE DYNAMICS ARE CONSTRAINTS NOW!
    # for i in range(horizon_steps):
    #     state_prediction[:,i+1] = state_prediction[:,i] + step_time*dynamics(state_prediction[:,i],control_prediction[:,i])

    # Defining a function which automates the filling up of the state_prediction vector
    #state_prediction_f = Function('state_prediction_f',[control_prediction,p],[state_prediction])

    # Define the constraint vector g: which is the constraint on the function of input variables: dynamics
    g = (state_prediction[:,0] - p[0:3]) # Initial constraint!

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
        next_state_estimate = curr_state + step_time*dynamics(curr_state,curr_input)
        g = vertcat(g,state_prediction[:,i+1] - next_state_estimate) # Adding up the constraints in the g function!

    # Now lets define the non linear program structure!
    opt_vars = vertcat(reshape(state_prediction,3*(horizon_steps + 1),1),reshape(control_prediction,2*horizon_steps,1)) # You are not just optimizing v and w but all instances of them!
    nlp_prob = {}
    nlp_prob['x'] = opt_vars
    nlp_prob['f'] = obj
    nlp_prob['g'] = g
    nlp_prob['p'] = p # This will change so might have to redefine the nlp_prob!

    # Initializing the solver!
    solver = nlpsol('solver','ipopt',nlp_prob)

    # Defining constraints for the NLP!

    lbg = [0] * 3*(horizon_steps+1) # Constraints on the dynamics!
    ubg = [0] * 3*(horizon_steps+1) # Constraints on the dynamics!

    lbx = []
    ubx = [] 

    # Set up constraints on the states!
    for i in range(3*(horizon_steps+1)):
        
        # Constraint on second state: y!
        if i % 3 == 1:
            lbx.append(-2)
            ubx.append(2)

        # Constraint on the third state: theta!
        elif i % 3 == 2:
            lbx.append(-np.inf)
            ubx.append(np.inf)
        
        # Constraint on the first state: x!
        else:
            lbx.append(-2)
            ubx.append(2)
    
    # Set up constraints on the inputs!
    for i in range(2*horizon_steps):
        if i % 2 == 0:
            lbx.append(v_min)
            ubx.append(v_max)
        else:
            lbx.append(omega_min)
            ubx.append(omega_max)

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
    u0 = DM.zeros((2,horizon_steps)) # Initial control guess
    X0 = np.tile(x0.full(),(1,horizon_steps+1))
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
        u_0 = vertcat(reshape(X0,3*(1+horizon_steps),1),reshape(u0, 2*horizon_steps, 1)) # initial value of the optimization variables!

        # Solve the optimization problem!
        sol = solver(x0=u_0, lbx=lbx, ubx=ubx, ubg=ubg, lbg=lbg, p=p_numeric)

        # Get the optimal output!
        full_solution = sol['x'].full()#reshape(sol['x'].full(),opt_vars.size1(),1) # Optimal solution
        
        # Extract the predicted states:
        ff_value = reshape(full_solution[0:3*(horizon_steps+1)],n_states,horizon_steps+1)
        
        # Extracted the actual output:
        u = reshape(full_solution[3*(horizon_steps+1):],n_control,horizon_steps)

        # Save control inputs
        v_history.append(u[:,0].full()[0])
        w_history.append(u[:,0].full()[1])

        # Update the state
        x0 = x0 + step_time*dynamics(x0,u[:,0])

        # Re-initialize the guess for optimization variables!
        u0 = horzcat(u[:, 1:], u[:, -1])
        X0 = horzcat(ff_value[:, 1:], ff_value[:, -1])
        
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
        plt.xlim(0,1.1)
        plt.ylim(0,1.1)
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