import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import numpy as np

class RosNode(Node):
    def __init__(self):
        super(RosNode, self).__init__("TransformUpdater")

class TransformUpdater():
    def __init__(self):
        super(TransformUpdater, self).__init__()
        
        # Set up the tf2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Store the target and source frames
        self.target_frame = None
        self.source_frame = None
        
        # Initialize transformation variable
        self.latest_transform = None
        
        # Timer to update the transform at a regular interval
        self.timer = self.create_timer(0.1, self.update_transform)  # Adjust polling interval as needed

    def update_transform(self):
        try:
            # Look up the latest transform
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time()
            )
            
            # Store position and orientation in a dictionary for easy access
            self.latest_transform = [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ]

            # Logging for debugging (optional)
            # self.get_logger().info(f"Updated transform: {self.latest_transform}")

        except Exception as e:
            self.get_logger().warn(f"Could not update transform: {e}")
            self.latest_transform = None

    def quaternion_multiply(self, q1, q2):
        """ Multiplies two quaternions (q1 * q2). """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([x, y, z, w])

    def apply_transform(self, pose, return_format="position"):
        # Extract position and quaternion from pose and self.latest_transform
        pose_position = np.array(pose[:3])
        if len(pose) <= 3:
            pose_orientation = np.array([1.,0.,0.,0.])
        else:
            pose_orientation = np.array(pose[3:])


        tf_position = np.array(self.latest_transform[:3])
        tf_orientation = np.array(self.latest_transform[3:])

        # Rotate pose position by self.latest_transform orientation (apply quaternion rotation)
        # Convert tf_orientation to rotation matrix
        x, y, z, w = tf_orientation
        rot_matrix = np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
        ])
        rotated_position = rot_matrix.dot(pose_position) + tf_position

        # Return the transformed pose
        if return_format=="position":
            return rotated_position
        else:
            # Combine orientations using quaternion multiplication
            new_orientation = self.quaternion_multiply(tf_orientation, pose_orientation)
            return np.concatenate((rotated_position, new_orientation))


class TFBaseLeapworld(TransformUpdater):
    def __init__(self):
        super(TFBaseLeapworld, self).__init__()
        self.target_frame = "base"
        self.source_frame = "leapworld"


class TransformUpdaterTestClass(TFBaseLeapworld, RosNode):
    pass


def main(args=None):
    rclpy.init(args=args)
    transform_updater = TransformUpdaterTestClass()

    try:
        rclpy.spin(transform_updater)
    except KeyboardInterrupt:
        pass
    finally:
        transform_updater.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
