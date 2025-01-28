
import rclpy
from rclpy.node import Node
from gesture_msgs.msg import Frame
import subprocess
import time
import threading

ALIVE_CHECK_PERIOD = 1 # s
STARTUP_PATIENCE = 10 # s

class LeapdManager(Node):
    """Runs `sudo leapd` and checks if hand data are published, if not, it restarts the leapd backend
    """
    def __init__(self):
        super().__init__('leapd_manager')
        self.subscriber = self.create_subscription(
            Frame,
            '/teleop_gesture_toolbox/hand_frame',
            self.hand_frame_callback,
            10
        )
        # Initial run
        self.last_received_time = 0.0
        self.start_leapd()
        time.sleep(STARTUP_PATIENCE)

        self.monitor_thread = threading.Thread(target=self.monitor_leapd)
        self.monitor_thread.start()

    def hand_frame_callback(self, msg):
        self.last_received_time = time.time()

    def start_leapd(self):
        self.get_logger().info("Starting sudo leapd...")
        subprocess.Popen(
            ['sudo', 'leapd'],  
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def stop_leapd(self):
        self.get_logger().info("Stopping sudo leapd...")
        subprocess.run(['sudo', 'pkill', '-f', 'leapd', "-9"])

    def monitor_leapd(self):
        while rclpy.ok():
            print(f"time diff: {time.time() - self.last_received_time}")
            time.sleep(ALIVE_CHECK_PERIOD)
            while (time.time() - self.last_received_time) > ALIVE_CHECK_PERIOD:
                self.restart()

    def restart(self):
        self.get_logger().warn("No hand frame received, restarting leapd...")
        self.stop_leapd()
        self.start_leapd()
        time.sleep(STARTUP_PATIENCE)
        while (time.time() - self.last_received_time) > STARTUP_PATIENCE:
            self.stop_leapd()
            self.start_leapd()
            time.sleep(STARTUP_PATIENCE)

    def destroy_node(self):
        self.stop_leapd()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    leapd_manager = LeapdManager()
    try:
        rclpy.spin(leapd_manager)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        leapd_manager.get_logger().error(f"Unexpected error: {e}")
    finally:
        leapd_manager.destroy_node()
        try:
            rclpy.shutdown()
        except rclpy._rclpy_pybind11.RCLError:
            print("Shutdown already called")

if __name__ == '__main__':
    main()
