import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

import yaml
import pointing_object_selection

class StaticTFPublisher(Node):
    """ See tf description: http://imitrob.ciirc.cvut.cz/demos/teleop_gesture_toolbox/teleop_toolbox_tf.png
    """
    def __init__(self, tf_file: str):
        """Publishes TF that represents static transformations for given setup
        """        
        super(StaticTFPublisher, self).__init__('static_tf_pub')

        # Static Transform Broadcaster
        self.static_broadcasters = []

        with open(pointing_object_selection.path+"/saved_setups/"+tf_file) as f:
            tf_dict = yaml.safe_load(f)

        for tf_name, tf_values in tf_dict.items():
            self.static_broadcasters.append(StaticTransformBroadcaster(self))

            static_transform = TransformStamped()
            static_transform.header.stamp = self.get_clock().now().to_msg()
            static_transform.header.frame_id = tf_values['parent']
            static_transform.child_frame_id = tf_name
            static_transform.transform.translation.x = tf_values['position'][0]
            static_transform.transform.translation.y = tf_values['position'][1]
            static_transform.transform.translation.z = tf_values['position'][2]
            static_transform.transform.rotation.x = tf_values['orientation'][0]
            static_transform.transform.rotation.y = tf_values['orientation'][1]
            static_transform.transform.rotation.z = tf_values['orientation'][2]
            static_transform.transform.rotation.w = tf_values['orientation'][3]
            self.static_broadcasters[-1].sendTransform(static_transform)

def a404_static_pub_main():
    rclpy.init(args=None)
    node = StaticTFPublisher(tf_file="a404.yaml")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def b300_static_pub_main():
    rclpy.init(args=None)
    node = StaticTFPublisher(tf_file="b300.yaml")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    a404_static_pub_main()