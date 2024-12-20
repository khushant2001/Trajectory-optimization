import rclpy
from rclpy.node import Node
import cflib.crtp  # noqa
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
import numpy as np
import time

uri = uri_helper.uri_from_env(default='radio://0/50/2M/E7E7E7E705')

class cf_publisher(Node):

    def __init__(self):
        # self.pos = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 1)

        # Connect some callbacks from the Crazyflie API
        super().__init__('crazyflie_publisher')
        self.t0 = time.time()
        self.cf = Crazyflie(rw_cache='./cache')
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
        self.cf.commander.send_setpoint(0,0,0,0)
        self.timer = self.create_timer(.05,self.timer_callback)

    def timer_callback(self):
        if (time.time() - self.t0) < 5:
            print("Stage1")
            self.cf.commander.send_setpoint(15,0,0,20000)
        else:
            print("Step 2")
            self.t0 = self.t0 + time.time()
            self.cf.commander.send_setpoint(15,15,0,20000)

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        print('Connected to %s' % link_uri)

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
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')

    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        # print("state", self.state_pos, self.state_quat)

        """Callback from a the log API when data arrives"""
        """print(f'[{timestamp}][{logconf.name}]: ', end='')
        for name, value in data.items():
            print(f'{name}: {value:3.3f} ', end='')
        print()"""


    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)
        self.is_connected = False

def main(args=None):

    rclpy.init(args=args)
    pub = cf_publisher()
    rclpy.spin(pub)
    pub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()