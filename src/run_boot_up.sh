#!/bin/bash
colcon build
source install/setup.bash
ros2 launch vicon_receiver client.launch.py &

sleep 2

# Open a new terminal and run commands inside it
gnome-terminal --tab -- bash -c "
  source install/setup.bash;
  ros2 launch launcher cf.launch.py;
  exec bash"

sleep 2

# Open a new terminal and run commands inside it
gnome-terminal --tab -- bash -c "
  source install/setup.bash;
  ros2 run plotjuggler plotjuggler -l 'plotjuggler_template.xml';
  exec bash"

