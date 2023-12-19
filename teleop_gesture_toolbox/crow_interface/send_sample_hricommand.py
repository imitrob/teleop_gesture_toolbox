
import rclpy
from rclpy.node import Node
from teleop_msgs.msg import HRICommand

class PublishHRICommand(Node):
    def __init__(self):
        super().__init__("rosnodepublisherofhricommand")
        
        self.pub = self.create_publisher(HRICommand, "/hri/command", 5)
    
if __name__ == '__main__':
    rclpy.init()
    rosnode = PublishHRICommand()
    
    while True:
        n = str(input("Enter object od: "))
        
        cube_holes_od_xxx = 'cube_holes_od_' + n
        msg_to_send = HRICommand()
        s = "{'target_action': 'pick_up', 'target_object': '" + cube_holes_od_xxx + "', 'actions': ['move_up', 'move_left', 'move_down', 'move_right', 'pick_up', 'put', 'place', 'pour', 'push', 'replace'], 'action_probs': [0.012395880917197533, 0.014667431372768347, 0.0008680663118536268, 0.035168211530459945, 0.9984559292675215, 0.012854139530004692, 0.0068131722011598225, 0.04846120672655781, 0.0020918881693065285, 0.01454853390045828], 'action_timestamp': 0.0, 'objects': ['" + cube_holes_od_xxx + "'], 'object_probs': [1.0], 'object_classes': ['object'], 'parameters': ''}"
        s = s.replace("'", '"')
        msg_to_send.data = [s]
        
        rosnode.pub.publish(msg_to_send)
    