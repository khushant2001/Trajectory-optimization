"""
Following are the demo examples of using CasaDy to optimize stuff! 
"""

from casadi import *
import matplotlib.pyplot as plt
import numpy as np

## Example 1: Simple 1D optimization

def single_var_example():

    # Variable to optimize!
    x = MX.sym('w')

    # Cost function!
    obj = x**2 - 6*x + 13

    q = [] # optimization constraints!
    p = [] # optimization problem parameters!

    opt_vars = x # Define the variables that are being minimized or maximized!
    nlp_prob = {} # Define the problem
    nlp_prob['x'] = opt_vars # Pass the optimization variables
    nlp_prob['f'] = obj
    nlp_prob['g'] = p

    # Initiate the solver!
    solver = nlpsol('solver_name','ipopt',nlp_prob)

    # Define the arguments for the NLP problem
    x0 = -.5
    lbx = -100000 # constraints on optimization varaible
    ubx = 1000000 # constraints on optimization variable
    lbg = -1000000 # constraint on function of optimization varaible
    ubg = 1000000 # constraint on function of optimization varaible

    # Solve the problem
    solution = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    print(solution['x'])

# Example 2: Linear regression using optimization

def linear_regression():

    # The X and Y points for linear regression
    x = [0,45,90,135,180]
    y = [667,661,757,871,1210]

    # Optimization variables!
    m = MX.sym('m')
    b = MX.sym('b')

    # Defining the cost function which is basilly minimizing the variance!

    obj = 0
    for i in range(len(x)):
        obj = obj + (y[i] - (m*x[i]+b))**2
    
    q = [] # There are no constraints!
    p = [] # There are no parameters!

    # Defint the OCP!
    opt_var = vertcat(m,b) # For more than 1 variable, you need to make sure that 

    nlp_prob = {}
    nlp_prob['x'] = opt_var
    nlp_prob['f'] = obj
    nlp_prob['g'] = q

    # Initiate the solver!
    solver = nlpsol('solver_name','ipopt',nlp_prob)

    # Define the arguments for the NLP problem
    x0 = [100,100]
    lbx = -100000 # constraints on optimization varaible
    ubx = 1000000 # constraints on optimization variable
    lbg = -1000000 # constraint on function of optimization varaible
    ubg = 1000000 # constraint on function of optimization varaible

    # Solve the problem
    solution = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    print(solution['x'])

    # Plotting the linear regression. 
    plt.figure()
    plt.scatter(x,y,marker = 'x',label = 'Raw data')
    plt.plot(x,solution['x'][0]*x+solution['x'][1],label = 'Approximation')
    plt.legend()
    plt.show()