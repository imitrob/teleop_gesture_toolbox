import time
import rclpy
import zmq
import json
from rclpy.node import Node
import scene_msgs.msg as scene_ros
from scene_getter.scene_lib.scene import Scene
from scene_getter.scene_lib.scene_object import SceneObject
from tiago_merger.msg import GDRNSolution # make sure this import works

class TiagoScenePublisher(Node):
    def __init__(self):
        super().__init__("tiago_scene_publisher_node")

        self.scene_pub = self.create_publisher(scene_ros.Scene, "/scene", 5)
        self.gdrnet_pub = self.create_publisher(scene_ros.GDRNSolution, "/gdrn_from_tiago")
        
        self.gdrn = self.get_gdrnet()
        objects = [SceneObject(
          obj["name"],obj["position"],obj["orientation"]) for obj in self.gdrn
                   ]
        
        self.scene = Scene(name="scene_1", objects=objects)
        
    def __call__(self):
        self.scene_pub.publish(self.scene.to_ros())
        self.gdrnet_pub.publish(self.gdrn)
    
    def get_gdrnet(self):
      # currently gdrn from zmq
      self.context = zmq.Context()
      self.socket = self.context.socket(zmq.SUB)
      self.socket.connect("tcp://*:5557")
      self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
      
      received = False
      gdrn = []
      while not received or KeyboardInterrupt:
        try:
          msg_str = self.socket.recv_string()
          msg_data = json.loads(msg_str)
          if type(msg_data) == list:
            for obj in msg_data:
              if type(msg_data) == dict:
                received = True
                gdrn.append(obj)
              else:
                self.get_logger().warn(
                  "[Tester] object in msg_data is not dict"
                  )
          else:
            self.get_logger().error("[Tiago Scene] Received msg_str is not list")
        except:      
          self.get_logger().error(
            "[Tiago Scene] Error in receiving msg from gdrn ROS1"
            )
      return gdrn
      
      

def main():
    rclpy.init()
    sp = TiagoScenePublisher()
    
    while rclpy.ok():
        sp()
        time.sleep(1.0)

if __name__ == '__main__':
    main()