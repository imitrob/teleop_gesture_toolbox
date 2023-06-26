
import rclpy
from rclpy.node import Node
from crow_msgs.msg import GestureSentence
from std_msgs.msg import Bool
import threading
import time

rclpy.init()
rosnode = Node("mocked_gesture_sentence_publisher")

''' Publish instanteneous gesturing bool thread '''
class PG():
    def __init__(self, rosnode):
        self.ongoing_bool = False
        self.og_pub = rosnode.create_publisher(Bool, "/gesture/ongoing", 5)
    def pub_gesturing(self):
        self.og_pub.publish(Bool(data=self.ongoing_bool))
    def og_thread(self):
        while True:
            time.sleep(0.1)
            self.pub_gesturing()

pog = PG(rosnode)
t1 = threading.Thread(target=pog.og_thread)
t1.daemon = True
t1.start()

gs_pub = rosnode.create_publisher(GestureSentence, "/gesture/gesture_sentence", 5)

while True:
    ''' Gesturing '''
    pog.ongoing_bool = True
    time.sleep(3.)
    pog.ongoing_bool = False
    
    ''' Send gesture sentence '''
    gs = GestureSentence( \
        actions = ['PICK_TASK', 'POINT_TASK', 'FETCH'], \
        action_likelihoods = [1.0, 0.0, 0.0], \
        objects = ['red box', 'blue box', 'red peg'], \
        object_likelihoods = [1.0, 0.0, 0.0], \
        auxiliary_parameters = ['distance', 'angular'], \
        auxiliary_parameter_likelihoods = [] \
    )
    gs.header.stamp = rosnode.get_clock().now().to_msg()
    gs_pub.publish(gs)

    input("Press for new gesture sentence")

