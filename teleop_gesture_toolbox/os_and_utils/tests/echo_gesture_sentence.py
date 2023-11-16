
import rclpy
from rclpy.node import Node

from context_based_gesture_operation.msg import HRICommand

class EchoGestureSentence(Node):
    ''' ROS communication of main thread: Subscribers (init & callbacks) and Publishers
    '''
    def __init__(self):
        super().__init__('echo_gesture_sentence_node')
    
        self.examplesub = self.create_subscription(HRICommand, '/teleop_gesture_toolbox/gesture_sentence_original', self.callback, 10)

    def callback(self, msg):
        print(msg)


if __name__ == '__main__':
    rclpy.init()
    rn = EchoGestureSentence()
    rclpy.spin(rn)
