#!/usr/bin/python

import rospy, zmq, struct

from sensor_msgs.msg import Image
from object_detector_msgs.srv import detectron2_service_server, estimate_poses
from message_filters import ApproximateTimeSynchronizer, Subscriber

# Set up ZeroMQ
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5557")

def detect_gdrn(rgb):
    rospy.wait_for_service('detect_objects')
    try:
        detect_objects = rospy.ServiceProxy('detect_objects', detectron2_service_server)
        response = detect_objects(rgb)
        return response.detections.detections
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def estimate_gdrn(rgd, depth, detection):
    rospy.wait_for_service('estimate_poses')
    try:
        estimate_pose = rospy.ServiceProxy('estimate_poses', estimate_poses)
        response = estimate_pose(detection, rgd, depth)
        return response.poses
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def process_gdrn_output(gdrn):
    """
    Function to process gdrn output from list -> dict
    """
    ret = dict()
    gdrn = str(gdrn)
    name_idx = gdrn.find("name")
    pose_idx = gdrn.find("pose")
    position_idx = gdrn.find("position")
    orient_idx = gdrn.find("orientation")
    conf_indx = gdrn.find("confidence")
    ret["name"] = gdrn[name_idx+len("name:  "):pose_idx-2]
    ret["score"] = float(gdrn[conf_indx+len("confidence: "):-1])
    position = gdrn[position_idx+len("position: "):orient_idx-1]
    orient = gdrn[orient_idx+len("orientation: "):conf_indx-2]
    p_x = position.find("x")
    p_y = position.find("y")
    p_z = position.find("z")
    o_x = orient.find("x")
    o_y = orient.find("y")
    o_z = orient.find("z")
    o_w = orient.find("w")
    ret["position"] = [float(position[p_x+len("x: "):p_y]), float(position[p_y+len("y: "):p_z]), float(position[p_z+len("z: "):-1])]
    ret["orientation"] = [float(orient[o_x+len("x: "):o_y]), float(orient[o_y+len("y: "):o_z]), float(orient[o_z+len("z: "):o_w]), float(orient[o_w+len("w: "):-1])]
    print("Detected %s" % ret["name"])
    return ret

def publish_gdrnet(event):
    # detects objects, estimate their poses and publishes then via ZeroMQ
    gdrnet_objects = []
    rgb = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
    depth = rospy.wait_for_message('/xtion/depth/image_raw', Image)
    detections = detect_gdrn(rgb)
    if detections is not None or len(detections) > 0:
        for detect in detections:
            pose = estimate_gdrn(rgb, depth, detect)
            gdrnet_objects.append(process_gdrn_output(pose))
    else:
        rospy.loginfo("Nothing detected by GDR-Net++")

    packed_data = struct.pack(f"{len(gdrnet_objects)}", *gdrnet_objects)
    socket.send(packed_data)

if __name__ == '__main__':
    rospy.init_node('gdrnet_publisher')
    rate = 1.0 # Hz
    rospy.Timer(rospy.Duration(1.0 / rate), publish_gdrnet)
    rospy.spin()

