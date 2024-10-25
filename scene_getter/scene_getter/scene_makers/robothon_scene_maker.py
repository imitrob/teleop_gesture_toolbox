




from typing import Iterable
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Pose
import scene_msgs.msg as scene_msgs 
from scene_getter.scene_lib.scene import Scene
from scene_getter.scene_lib.scene_object import SceneObject

class RobothonSceneMakerNode(Node):
    #                        [ x    , y     , z  , x  , y  , z  , w  ]
    peg_hole_static_tf     = [ 0.02 ,  0.04 , 0.0, 0.0, 0.0, 0.0, 1.0]
    probe_hole_static_tf   = [-0.04 , -0.95 , 0.0, 0.0, 0.0, 0.0, 1.0]
    door_static_tf         = [ 0.015, -0.045, 0.0, 0.0, 0.0, 0.0, 1.0]
    cable_holder_static_tf = [ 0.85 ,  0.0  , 0.0, 0.0, 0.0, 0.0, 1.0]

    def __init__(self):
        Node.__init__(self, 'scene_getter_node')
        self.create_subscription(Pose, "/robothonbox", self.callback)
        self.pub_scene = self.create_publisher(scene_msgs.Scene, "/scene", 5)

    def callback(self, msg):
        pose = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        self.scene = self.generate_object_points_from_box_pose(pose)

        self.pub_scene.publish(self.scene.to_ros())

    def generate_object_points_from_box_pose(self, pose: Iterable[float]):
        
        assert len(pose) == 7

        peg_hole_position,     peg_hole_orientation     = self.apply_tf(pose, self.peg_hole_static_tf)
        probe_hole_position,   probe_hole_orientation   = self.apply_tf(pose, self.probe_hole_static_tf)
        door_position,         door_orientation         = self.apply_tf(pose, self.door_static_tf)
        cable_holder_position, cable_holder_orientation = self.apply_tf(pose, self.cable_holder_static_tf)

        return Scene(name="robothon_scene", objects=[
            SceneObject(name="peg_hole",     position=peg_hole_position,     orientation=peg_hole_orientation),
            SceneObject(name="probe_hole",   position=probe_hole_position,   orientation=probe_hole_orientation),
            SceneObject(name="door",         position=door_position,         orientation=door_orientation),
            SceneObject(name="cable_holder", position=cable_holder_position, orientation=cable_holder_orientation),
        ])

    def apply_tf(self, pose1: Iterable[float], pose2: Iterable[float]):
        assert len(pose1) == 7 and len(pose2) == 7
        pose1 = np.array(pose1)
        pose2 = np.array(pose2)

        return pose1[0:3] + pose2[0:3], quaternion_multiply(pose1[3:7], pose2[3:7])
    

def quaternion_multiply(quaternion1, quaternion2):
    w1, x1, y1, z1 = quaternion1
    w2, x2, y2, z2 = quaternion2
    # Calculate the product quaternion components.
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])