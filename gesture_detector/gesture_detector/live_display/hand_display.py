import time
import rclpy
from rclpy.node import Node
from gesture_msgs.msg import Frame as FrameMSG
from geometry_msgs.msg import Point
from std_msgs.msg import String
from gesture_detector.hand_processing.frame_lib import Frame
from visualization_msgs.msg import Marker, MarkerArray  # Import for RViz visualization
from gesture_msgs.msg import DeicticSolution

from pointing_object_selection.transform import transform_to_world
from pointing_object_selection.transform_ros_getter import TransformUpdater

from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

class HandVisualizerRosNode(Node):
    def __init__(self):
        super(HandVisualizerRosNode, self).__init__('hand_visualizer')

class HandVisualizer(TransformUpdater, HandVisualizerRosNode):
    def __init__(self):
        super(HandVisualizer, self).__init__()

        # Subscriber for the hand frame
        self.create_subscription(FrameMSG, "/hand_frame", self.frame_callback, 10)

        # Publisher for RViz visualization
        self.marker_publisher = self.create_publisher(MarkerArray, '/hand_bone_markers', 10)

        # Static Transform Broadcaster
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # Define and send a static transform from "world" to "hand"
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = "map"
        static_transform.child_frame_id = "hand"
        static_transform.transform.translation.x = 0.0
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = 0.0
        static_transform.transform.rotation.x = 0.0
        static_transform.transform.rotation.y = 0.0
        static_transform.transform.rotation.z = 0.0
        static_transform.transform.rotation.w = 1.0

        self.static_broadcaster.sendTransform(static_transform)


    def frame_callback(self, msg):
        # Create a new instance of Frame from the received message
        frame = Frame()
        frame.import_from_ros(msg)

        # Visualize the hand using RViz MarkerArray
        if frame.l.visible and self.latest_transform is not None:
            marker_array = MarkerArray()
            marker_id = 0
            
            for finger in frame.l.fingers:
                for bone in finger.bones:
                    # Create a line marker for each bone
                    start = bone.prev_joint  # Start point (x, y, z)
                    end = bone.next_joint    # End point (x, y, z)

                    marker = Marker()
                    marker.header.frame_id = "hand"  # Replace with the actual frame you are working in (e.g., base_frame)
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = "hand_bones"
                    marker.id = marker_id
                    marker.type = Marker.LINE_STRIP
                    marker.action = Marker.ADD

                    position_start = [start.x, start.y, start.z]
                    position_end = [end.x, end.y, end.z]
                    # Set the start and end points of the line
                    # position_start = self.apply_transform(position_start)
                    # position_end = self.apply_transform(position_end)
                    
                    position_start = transform_to_world(position_start)
                    position_end = transform_to_world(position_end)
                    
                    start_point = Point(x=position_start[0],y=position_start[1],z=position_start[2])
                    end_point = Point(x=position_end[0],y=position_end[1],z=position_end[2])

                    print(position_start, position_end)
                    marker.points.append(start_point)
                    marker.points.append(end_point)

                    # Set marker properties
                    marker.scale.x = 0.005  # Thickness of the bone lines
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                    marker.color.a = 1.0  # Full opacity

                    # Increment marker ID for each bone
                    marker_id += 1

                    # Add the marker to the MarkerArray
                    marker_array.markers.append(marker)

            # Publish the MarkerArray
            
            self.marker_publisher.publish(marker_array)




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
