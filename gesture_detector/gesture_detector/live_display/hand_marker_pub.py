import time
import numpy as np
import rclpy
from rclpy.node import Node
from gesture_msgs.msg import Frame as FrameMSG
from geometry_msgs.msg import Point
from std_msgs.msg import String
from gesture_detector.hand_processing.frame_lib import Frame
from visualization_msgs.msg import Marker, MarkerArray  # Import for RViz visualization
from gesture_msgs.msg import DeicticSolution as DeicticSolutionMSG

class HandVisualizerRosNode(Node):
    def __init__(self):
        super(HandVisualizerRosNode, self).__init__('hand_visualizer')

class HandVisualizer(HandVisualizerRosNode):
    def __init__(self):
        """Publishes TF and markers for rViz visualization
        """        
        super(HandVisualizer, self).__init__()

        # Subscriber for the hand frame
        self.create_subscription(FrameMSG, "/teleop_gesture_toolbox/hand_frame", self.frame_callback, 10)
        self.create_subscription(DeicticSolutionMSG, "/teleop_gesture_toolbox/deictic_solution", self.point_callback, 10)

        # Publisher for RViz visualization
        self.marker_publisher = self.create_publisher(MarkerArray, '/teleop_gesture_toolbox/hand_bone_markers', 10)
        self.line_marker_publisher = self.create_publisher(MarkerArray, '/teleop_gesture_toolbox/line_marker', 10)

    def frame_callback(self, msg):
        # Create a new instance of Frame from the received message
        frame = Frame()
        frame.import_from_ros(msg)

        # Visualize the hand using RViz MarkerArray
        marker_array = MarkerArray()
        marker_id = 0
        for hand in [frame.l, frame.r]:
            if not hand.visible:
                continue
            
            for finger in hand.fingers:
                for bone in finger.bones:
                    marker = Marker()
                    marker.header.frame_id = "leapworld"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = "hand_bones"
                    marker.id = marker_id
                    marker.type = Marker.LINE_STRIP
                    marker.action = Marker.ADD

                    # Create a line marker for each bone
                    start = bone.prev_joint.world  # Start point (x, y, z)
                    end = bone.next_joint.world    # End point (x, y, z)

                    marker.points.append(Point(x=start[0],y=start[1],z=start[2]))
                    marker.points.append(Point(x=end[0],y=end[1],z=end[2]))

                    # Set marker properties
                    marker.scale.x = 0.01  # Thickness of the bone lines
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

    def point_callback(self, msg):
        """ Publish marker for pointing line """
        marker_array = MarkerArray()
        marker_id = 0

        marker = Marker()
        marker.header.frame_id = "base"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "hand_point"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.points.append(msg.line_point_1)
        marker.points.append(msg.line_point_2)

        # Set marker properties
        marker.scale.x = 0.01  # Thickness of the bone lines
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Full opacity

        # Increment marker ID for each bone
        marker_id += 1
        
        marker_array.markers.append(marker)
        self.line_marker_publisher.publish(marker_array)



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
