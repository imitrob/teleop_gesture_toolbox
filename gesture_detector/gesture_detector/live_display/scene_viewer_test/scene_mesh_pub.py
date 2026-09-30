#!/usr/bin/env python3
"""Test publisher: ask the scene viewer to show STL meshes.

    python3 scene_mesh_pub.py [taskboard|tabletop]

Latched, so a viewer that connects later still gets it. The viewer keeps the
meshes after this stops, until another set arrives or the page reloads.
"""

import sys

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

TOPIC = "/teleop_gesture_toolbox/scene_mesh"
MM = 0.001  # the STLs are in millimetres
IDENTITY = (0.0, 0.0, 0.0, 1.0)  # x, y, z, w in base
RED, GREEN, BLUE = (0.8, 0.1, 0.1), (0.2, 0.6, 0.25), (0.15, 0.35, 0.8)
YELLOW, PAPER, GREY = (0.95, 0.8, 0.15), (0.75, 0.6, 0.42), (0.6, 0.6, 0.6)

# (URL relative to the viewer page, position in base, orientation, scale, color)
CONFIGS = {
    "taskboard": [
        # trajectory_tools render_skill.py placement (--mesh-xyz, --mesh-yaw
        # 270), y +0.1; z lifts the mesh's -79.5 mm bottom onto the table.
        ("models/taskboard_2pegs.stl", (0.5, -0.05, 0.0795),
         (0.0, 0.0, -0.70710678, 0.70710678), MM, GREY),
    ],
    # x y copied from hri_benchmark data/study/scenes/study_tabletop.yaml.
    # models/tabletop/*.stl have their origin at the bottom centre, so z 0
    # stands them on the table. Poly Haven CC0 assets, see models/tabletop/.
    "tabletop": [
        ("models/tabletop/cup.stl", (0.65, 0.15, 0.0), IDENTITY, MM, RED),  # cup1
        ("models/tabletop/cup.stl", (0.5, 0.0, 0.0), IDENTITY, 1.4 * MM, RED),  # cup2, big
        ("models/tabletop/cup.stl", (0.35, 0.15, 0.0), IDENTITY, MM, RED),  # cup3
        ("models/tabletop/bowl.stl", (0.5, 0.25, 0.0), IDENTITY, MM, GREEN),
        ("models/tabletop/banana.stl", (0.5, -0.25, 0.0), IDENTITY, MM, YELLOW),
        ("models/tabletop/box.stl", (0.35, -0.15, 0.0), IDENTITY, MM, PAPER),
        ("models/tabletop/container.stl", (0.65, -0.15, 0.0), IDENTITY, MM, BLUE),
    ],
}


def make_marker(index, url, position, orientation, scale, color):
    marker = Marker()
    marker.header.frame_id = "base"
    marker.ns = "scene_mesh"
    marker.id = index
    marker.type = Marker.MESH_RESOURCE
    marker.action = Marker.ADD
    marker.mesh_resource = url
    pose = marker.pose
    pose.position.x, pose.position.y, pose.position.z = position
    o = pose.orientation
    o.x, o.y, o.z, o.w = orientation
    marker.scale.x = marker.scale.y = marker.scale.z = scale
    marker.color.r, marker.color.g, marker.color.b = color
    marker.color.a = 1.0
    return marker


def main():
    config = sys.argv[1] if len(sys.argv) > 1 else "taskboard"
    if config not in CONFIGS:
        sys.exit(f"unknown config {config!r}, choose from {', '.join(CONFIGS)}")
    rclpy.init()
    node = rclpy.create_node("scene_mesh_pub")
    publisher = node.create_publisher(
        MarkerArray, TOPIC,
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    publisher.publish(MarkerArray(markers=[
        make_marker(index, *mesh) for index, mesh in enumerate(CONFIGS[config])
    ]))
    node.get_logger().info(f"published {config}: {len(CONFIGS[config])} meshes")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
