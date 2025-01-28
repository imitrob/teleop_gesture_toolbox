
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

import scene_msgs.msg as scene_msgs
from scene_getter.scene_lib.scene import Scene

class HandVisualizerRosNode(Node):
    def __init__(self):
        super(HandVisualizerRosNode, self).__init__('scene_visualizer')

class HandVisualizer(HandVisualizerRosNode):
    def __init__(self):
        """Publishes TF and markers for rViz visualization
        """        
        super(HandVisualizer, self).__init__()

        self.tf_broadcaster = TransformBroadcaster(self)
        self.scene_subscriber = self.create_subscription(scene_msgs.Scene, "/scene", self.scene_callback, 5)

    def scene_callback(self, msg):
        scene = Scene.from_ros(msg)

        for scene_object in scene.objects:
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "base"
            transform.child_frame_id = scene_object.name
            transform.transform.translation.x = scene_object.position[0]
            transform.transform.translation.y = scene_object.position[1]
            transform.transform.translation.z = scene_object.position[2]
            transform.transform.rotation.x = scene_object.orientation[0]
            transform.transform.rotation.y = scene_object.orientation[1]
            transform.transform.rotation.z = scene_object.orientation[2]
            transform.transform.rotation.w = scene_object.orientation[3]

            self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = HandVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
