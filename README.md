As my Masters project, I aim to perform real-time trajectory optimization for obstacle avoidance on Crazyflie - a mini quadcopter. This experimental project seeks to integrate Crazyflie’s existing attitude control with a high-level model predictive controller for online optimization. CasADi is used to formulate the MPC as a non-linear programming problem. Both simulation and hardware testing are performed. For the latter, Robotic Operating System (Ros 2) is used to integrate Crazyflie’s firmware, motion capture system (Vicon), and the high-level controller. For the model itself, the full state 12 DOF model is used to increase the accuracy of the program. The mass, intertias, and the experimental limits to velocities and thrust are determined from the bitcraze documentation. While the updates to the crazyflie's position and orientation are derived from the motion capture system, the velocities are calculated using dirty differentiation. To transform the problem from an optimization problem to a non-linear programming problem, multiple shooting is used to discretize the dynamics. Doing so allows to use both the state and control variables as optimization variables (also allowing to give a better approximation for the starting solution). Other methods like single shooting and direct collocation are tested in simulation. Hardware testing is still in progress; initial results can be seen below. 

# 1. Simulation Example: Toy Car Trajectory Optimization
https://github.com/user-attachments/assets/137c417f-359c-431a-b6d0-a26160b05b8e

# 2. Simulation Example: Toy Car Trajectory Optimization While Avoiding Obstacles
https://github.com/user-attachments/assets/601fca65-f66f-42e9-b11f-3b51828654dc

# 3. Simulation Example: Crazyflie Trajectory Optimization While Avoiding Obstacles
https://github.com/user-attachments/assets/90b00b89-126e-4326-ad11-b87b683974eb

# 4. Vicon Setup
![full_scene](https://github.com/user-attachments/assets/7430d3a5-eedf-4953-b3f3-52dca80e9fbc)

![rccar](https://github.com/user-attachments/assets/f61cbc71-0cbc-4db9-9f09-a39edc1f6b10)
![crazyflie](https://github.com/user-attachments/assets/216ada0e-29f9-465d-94db-30acd821484a)

![drone](https://github.com/user-attachments/assets/741278aa-d79e-4d5f-bcab-b8ac27ede143)

![rccar_markers](https://github.com/user-attachments/assets/e95ade61-4bc0-435e-828c-496a0426f154)

# 5. Robotic Operating System Architecture
For Ros2 architecture, 4 packages are created: vicon_receiver (to convert the vicon radio messages to Ros2 topics), traj_opt (which contains the 2 nodes: mpc_solver and cf_publisher), custom_msgs (custom topic and message to have the 2 nodes interacting), launcher (to create the launch file). The cf_publisher node takes care of connecting with the Crayzflie over the radio and providing a constant heartbeat (command) after every 10 msec. The mpc_solver node takes care of getting measurments from the vicon, computing the state of the crazyflie, and solving the mpc to generate the next solution. This solution - roll, pitch, yaw_rate, and thrust - are then transmitted to the crazyflie node through the custom topic - mpc_solution. 

![traj_opt_rqt_graph](https://github.com/user-attachments/assets/cc044d90-6ab9-4292-8835-e9a6a7aea6ae)

# 6. Initial Hardware Testing
https://github.com/user-attachments/assets/4cac5d11-225e-4432-b04c-307508a24585

