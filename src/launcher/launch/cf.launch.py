from launch import LaunchDescription
from launch_ros.actions import Node 

def generate_launch_description():
    ld = LaunchDescription()

    cf_cmd_node = Node(
        package = "traj_opt",
        executable = "cf_publisher",
    )
    
    mpc_node = Node(
        package = "traj_opt",
        executable = "mpc_solver",
    )
    ld.add_action(cf_cmd_node)
    ld.add_action(mpc_node)
    return ld