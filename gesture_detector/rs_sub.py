# ros2_subscriber.py
import rclpy
import zmq
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://10.35.127.230:5555")
socket.setsockopt(zmq.SUBSCRIBE, b"")

bridge = CvBridge()

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.publisher_ = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.timer = self.create_timer(0.01, self.timer_callback)

    def timer_callback(self):
        
        # Receive image over ZeroMQ
        buffer = socket.recv()
        # Decode JPEG image
        np_img = np.frombuffer(buffer, dtype=np.uint8)
        cv_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        # Convert OpenCV image to ROS 2 Image message and publish
        ros_image = bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.publisher_.publish(ros_image)
        print("pub")

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
