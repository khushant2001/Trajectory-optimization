import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import cflib.crtp 
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
from custom_msgs.msg import Actuation
import numpy as np

# Uri for the radio connection for the crazyflie! Will be changed depending upon the configuration!
uri = uri_helper.uri_from_env(default='radio://0/90/2M/E7E7E7E7E7')

class cf_cmd_sender(Node):

    def __init__(self):

        # Call the constructor of the parent class: Node!
        super().__init__('cf_publisher')

        self.dt = 10 # msec to call the callback for getting the optimized input from the mpc!

        # Connect some callbacks from the Crazyflie API
        self.cf = Crazyflie(rw_cache='./cache')
        self.cf.close_link()
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)

        print('Connecting to %s' % uri)

        # Try to connect to the Crazyflie
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.cf.open_link(uri)

        # Variable used to keep main loop occupied until disconnect
        self.is_connected = True

        # Defining an array for the command that will get populated according to the MPC solver!
        self.command = np.array([0,0,0,0])

        # Send an initial command of zeros to start the quadcopter!
        self.cf.commander.send_setpoint(0,0,0,0)

        # Call the timer to send commands!
        self.timer = self.create_timer(self.dt/1000,self.timer_callback)

        # Call the custom topic to get the optimized commands from MPC!
        self.mpc_subscriber = self.create_subscription(Actuation, "/mpc_solution",self.mpc_callback, self.dt)
    
    def mpc_callback(self,msg_in):

        """ Get the updated MPC solution from the mpc_solver node """
        self.get_logger().info("MPC Solution UPDATED!")

        # Update the variable that stores the command that goes to the crazyflie!
        self.command = np.array([msg_in.roll,msg_in.pitch,msg_in.yaw_rate,msg_in.thrust])

    def timer_callback(self):

        """ Send commands to the crazyflie! """
        self.get_logger().info(f"Sending commands to Crazyflie! {self.command}")

        # Use the commander API from crazyflie library to send commands!
        self.cf.commander.send_setpoint(self.command[0],self.command[1],self.command[2],int(self.command[3]))
    
    def _connected(self, link_uri):

        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""

        self.get_logger().info(f'Connected to %s' % link_uri)

        # The definition of the logconfig can be made before connecting
        self.lg_stab = LogConfig(name='Stabilizer', period_in_ms=100)
        self.lg_stab.add_variable('stateEstimate.x', 'float')
        self.lg_stab.add_variable('stateEstimate.y', 'float')
        self.lg_stab.add_variable('stateEstimate.z', 'float')
        self.lg_stab.add_variable('stabilizer.roll', 'float')
        self.lg_stab.add_variable('stabilizer.pitch', 'float')
        self.lg_stab.add_variable('stabilizer.yaw', 'float')
        # The fetch-as argument can be set to FP16 to save space in the log packet
        self.lg_stab.add_variable('pm.vbat', 'FP16')

        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
        try:
            self.cf.log.add_config(self.lg_stab)
            # This callback will receive the data
            self.lg_stab.data_received_cb.add_callback(self._stab_log_data)
            # This callback will be called on errors
            self.lg_stab.error_cb.add_callback(self._stab_log_error)
            # Start the logging
            self.lg_stab.start()
        except KeyError as e:
            self.get_logger().info('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            self.get_logger().info('Could not add Stabilizer log config, bad configuration.')

    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        self.get_logger().info('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):

        # yaw = data['stabilizer.yaw']
        # pitch = data['stabilizer.pitch']
        # roll = data['stabilizer.roll']

        #self.state_pos = np.array([data['stateEstimate.x'], data['stateEstimate.y'], data['stateEstimate.z']])
        
        #self.state_orientation = np.array([data['stabilizer.roll'], data['stabilizer.pitch'], data['stabilizer.yaw']])

        # print("state", self.state_pos, self.state_quat)

        """Callback from a the log API when data arrives"""
        """print(f'[{timestamp}][{logconf.name}]: ', end='')
        for name, value in data.items():
            print(f'{name}: {value:3.3f} ', end='')
        print()"""


    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        self.get_logger().info(f'Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        self.get_logger().info(f'Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)
        self.cf.commander.send_setpoint(0,0,0,0)
        self.is_connected = False

def main(args=None):

    try:
        rclpy.init(args=args)
        pub = cf_cmd_sender()
        rclpy.spin(pub)
    except KeyboardInterrupt as ki:
        print(ki)
    finally:
        pub._disconnected(uri)
        pub.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()