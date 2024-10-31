import zmq
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from realsense2_camera_msgs.msg import RGBD  # Import the RGBD message type

TIAGO_IP = "10.35.127.230" # PUT HERE TIAGO IP!

class RGBDPublisher(Node):
    def __init__(self):
        super().__init__('rgbd_publisher')
        
        # Initialize ROS 2 publisher
        self.publisher_ = self.create_publisher(RGBD, '/camera/camera/rgbd', 10)
        
        # Initialize ZeroMQ subscriber
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect("tcp://10.35.127.230:5555")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # CvBridge for converting OpenCV images to ROS Image messages
        self.bridge = CvBridge()

        # Timer to check for incoming data
        self.timer = self.create_timer(0.01, self.timer_callback)

    def timer_callback(self):
        # Receive message over ZeroMQ
        try:
            message = self.socket.recv(zmq.NOBLOCK)  # Non-blocking receive
        except zmq.Again:
            return  # No new message; continue

        # Parse color image
        color_len = np.frombuffer(message[:4], dtype=np.int32)[0]
        color_data = message[4:4 + color_len]
        color_image = cv2.imdecode(np.frombuffer(color_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Parse depth image
        depth_len = np.frombuffer(message[4 + color_len:4 + color_len + 4], dtype=np.int32)[0]
        depth_data = message[4 + color_len + 4:]
        depth_image = cv2.imdecode(np.frombuffer(depth_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        # Convert OpenCV images to ROS 2 Image messages
        color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")  # Depth format is usually 16-bit

        # Create and populate RGBD message
        rgbd_msg = RGBD()
        rgbd_msg.rgb = color_msg
        rgbd_msg.depth = depth_msg

        # Publish RGBD message
        self.publisher_.publish(rgbd_msg)
        self.get_logger().info("Published RGBD message")

def main(args=None):
    rclpy.init(args=args)
    rgbd_publisher = RGBDPublisher()
    rclpy.spin(rgbd_publisher)
    rgbd_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

