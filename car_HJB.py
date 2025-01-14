"""
This is an Example script that performs trajectory optimization by solving the HJB equation rather than a MPC!

Discrete HJB equation: 
u_optimal = argmin{u}[l(x,u) + V*(f(x),n+1)] => recursive over the optimal cost function!

Strategy: TREAT THIS AS A COST FUNCTION!

PROS: Can be done offline and then hope for the best for the hardware to perform like the simulated model!
CONS: Needs a longer time horizon. Doesn't matter if you increase the terminal cost. The car wouldn't reach the final spot. 
Also for some cases, it is giving some weird results (BLATANTLY WRONG)! Like you need specific combinations of the horizon time
and step time to make the HJB converge. Weird!
"""

from casadi import *
import matplotlib.pyplot as plt
import numpy as np

# Defining the function that optimizes the toy car trajectory
def toy_car_HJB():
    
    # Define the dynamics!!!
    # x_dot = vcos(theta)
    # y_dot = vsin(theta)
    # theta_dot = omega
    
    time_horizon = 5 # The total time given for the trajectory!
    step_time = 0.1 #sec when does the observation come!
    steps = int(time_horizon/step_time)
    v_min = -.6 # Min linear velocity
    v_max = .6 # Max linear velocity
    omega_min = -np.pi/4 # Min rotational vel
    omega_max = np.pi/4
    
    # Define the STATE variables!
    x = SX.sym('x')
    y = SX.sym('y')
    theta = SX.sym('theta')

    # Define the CONTROL variables
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

    # Building the state and control prediction matrices! PAY Attention to the size. Since indexing starts at 0, the 
    # state size is steps but control size is steps-1. This convention is followed later in the code!
    state_prediction = SX.sym('X',n_states,steps) # The recorded states over the trajectory!
    control_prediction = SX.sym('U',n_control,steps-1) # The optimized control inputs!

    # Define the parameters vector: initial and final point!
    p = SX.sym('p',n_states*2)

    # Populate the first entry of the state_prediction matrix with the initial condition.
    state_prediction[:,0] = p[0:3]

    # Populating the state_prediction matrix!
    for i in range(steps-1):
        state_prediction[:,i+1] = state_prediction[:,i] + step_time*dynamics(state_prediction[:,i],control_prediction[:,i])

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

    # Weight matrix for the terminal cost!
    T = 100*DM.eye(n_states)

    # Actual Terminal cost
    l_terminal = ((state_prediction[steps-1] - p[3:6]).T @ T @ (state_prediction[steps-1] - p[3:6]))

    # Making a dictionary to store the cost at individual . This will help for recursive cost propagation!
    V = dict()
    V[steps-1] = l_terminal # Cost at the terminal step!

    # Initiating the cost function!
    obj = l_terminal

    # Summing over the steps to build up the cost function!
    for i in range(steps-2, -1, -1):
        curr_state = state_prediction[:,i]
        curr_input = control_prediction[:,i]
        cost = ((curr_state - p[3:6]).T @ Q @ (curr_state - p[3:6])) + curr_input.T @ R @ curr_input + V[i+1]
        V[i] = cost
        obj = obj + cost

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

    # Constraints for the optimized variables.
    lbx = [v_min if i % 2 == 0 else omega_min for i in range(n_control * (steps-1))]
    ubx = [v_max if i % 2 == 0 else omega_max for i in range(n_control * (steps-1))]

    # SET UP THE SIMULATION!!!
    x0 = DM([0, 0, 0])  # Initial state
    xf = DM([1, 1, pi])  # Target state

    u0 = DM([0, 0] * (steps-1)).T  # Initial control guess

    # Update the parameters! 
    p_numeric = vertcat(x0, xf)

    # Give an initial estimate for the optimization variables!
    u_0 = reshape(u0, 2*(steps-1), 1) # initial value of the optimization variables!

    # Solve the optimization problem!
    sol = solver(x0=u_0, lbx=lbx, ubx=ubx, ubg=ubg, lbg=lbg, p=p_numeric)

    # Extract the optimal output!
    u = reshape(sol['x'], 2, steps-1) # Optimal solution
    
    # True state
    true_state = [x0.full()]

    # Determine the true_state from the optimized control variables!
    for i in range(steps-1):
        updated_state = true_state[i] + step_time*dynamics(true_state[i],u[:,i])
        true_state.append(updated_state.full())

    # Converting the list to an array for assisting in plotting!
    true_state = np.array(true_state)

    # Plotting the state info over x vs y plot. 
    plt.figure()
    plt.scatter(true_state[:,0],true_state[:,1])
    plt.scatter(xf[1].full(),xf[0].full(),label = 'Final_destination')
    plt.scatter(x0[1].full(),x0[0].full(),label = 'Initial_position')
    plt.xlabel('X[m]')
    plt.ylabel("Y[m]")
    plt.title("Optimized Trajectory")
    plt.legend()
    plt.grid()

    # Defining a time array for plotting purposes!
    step_array = np.linspace(0,time_horizon,steps-1)
    plt.figure()
    plt.plot(step_array,u.full()[0:,].T, label = 'Linear Velocity [m/sec]')
    plt.plot(step_array,u.full()[1:,].T, label = 'Angular velocity [rad/sec]')
    plt.legend()
    plt.title("Optimized inputs")
    plt.xlabel("Time [sec]")
    plt.grid()
    plt.show()

# Call the function!
toy_car_HJB()