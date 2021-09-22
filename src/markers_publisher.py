#!/usr/bin/env python2
''' rViz Markers library. Called from 'main.py' as a thread.
    'createMarker' function 'GenerateMarkers' and publishes it in a loop.
'''
import settings

from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String, Bool, Int8, Float64MultiArray
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Vector3Stamped, QuaternionStamped, Vector3
from moveit_msgs.srv import ApplyPlanningScene
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from moveit_msgs.msg import RobotTrajectory
from sensor_msgs.msg import JointState
from relaxed_ik.msg import EEPoseGoals, JointAngles

import tf
from copy import deepcopy
import numpy as np
import moveit_lib
import rospy

class MarkersPublisher():
    @staticmethod
    def markersThread():
        ''' Called as a new thread. Generates marker in a loop.
        '''
        publisher = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=5)
        markerArray = MarkerArray()
        while not rospy.is_shutdown():
            markerArray.markers = MarkersPublisher.generateMarkers()
            publisher.publish(markerArray)
            rospy.sleep(0.25)

    @staticmethod
    def generateMarkers():
        q_norm_to_dir = Quaternion(0.0, -np.sqrt(2)/2, 0.0, np.sqrt(2)/2)
        markers_array = []
        ## marker_interaction_box
        sx,sy,sz = settings.extv(settings.md.ENV['start'])
        m = Marker()
        m.header.frame_id = "/"+settings.BASE_LINK
        m.type = m.SPHERE
        m.action = m.ADD
        m.scale.x = 0.01
        m.scale.y = 0.01
        m.scale.z = 0.01
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.pose.orientation = settings.md.ENV['ori']
        m.pose.position = settings.mo.pointToScene(Point(+0.1175, +0.0735, +0.0825))
        m.id = 100
        markers_array.append(deepcopy(m))
        m.pose.position = settings.mo.pointToScene(Point(-0.1175, +0.0735, +0.0825))
        m.id += 1
        markers_array.append(deepcopy(m))
        m.pose.position = settings.mo.pointToScene(Point(-0.1175, -0.0735, +0.0825))
        m.id += 1
        markers_array.append(deepcopy(m))
        m.pose.position = settings.mo.pointToScene(Point(+0.1175, -0.0735, +0.0825))
        m.id += 1
        markers_array.append(deepcopy(m))
        m.pose.position = settings.mo.pointToScene(Point(+0.1175, +0.0735, +0.0825+0.3175))
        m.id += 1
        markers_array.append(deepcopy(m))
        m.pose.position = settings.mo.pointToScene(Point(-0.1175, +0.0735, +0.0825+0.3175))
        m.id += 1
        markers_array.append(deepcopy(m))
        m.pose.position = settings.mo.pointToScene(Point(-0.1175, -0.0735, +0.0825+0.3175))
        m.id += 1
        markers_array.append(deepcopy(m))
        m.pose.position = settings.mo.pointToScene(Point(+0.1175, -0.0735, +0.0825+0.3175))
        m.id += 1
        markers_array.append(deepcopy(m))

        ## Marker env + normal
        m = Marker()
        m.header.frame_id = "/"+settings.BASE_LINK
        m.type = m.ARROW
        m.action = m.ADD
        m.scale.x = 0.1
        m.scale.y = 0.01
        m.scale.z = 0.01
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.pose.position = settings.md.ENV['start']
        m.pose.orientation = settings.md.ENV['ori']
        m.id += 1
        markers_array.append(deepcopy(m))
        m.color.g = 0.0
        m.pose.orientation = Quaternion(*tf.transformations.quaternion_multiply(settings.extq(m.pose.orientation), settings.extq(q_norm_to_dir)))
        m.id += 1
        markers_array.append(deepcopy(m))

        ## Marker endeffector + normal + action to attach
        m = Marker()
        m.header.frame_id = "/"+settings.EEF_NAME
        m.type = m.ARROW
        m.action = m.ADD
        m.scale.x = 0.2
        m.scale.y = 0.01
        m.scale.z = 0.01
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = -np.sqrt(2)/2
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = np.sqrt(2)/2
        m.pose.position.x = 0
        m.pose.position.y = 0
        m.pose.position.z = 0
        m.id += 1
        markers_array.append(deepcopy(m))
        m.color.g = 0.0
        m.pose.orientation = Quaternion(*tf.transformations.quaternion_multiply(settings.extq(m.pose.orientation), settings.extq(q_norm_to_dir)))
        m.id += 1
        markers_array.append(deepcopy(m))
        if settings.md.ACTION:
            m.type = m.SPHERE
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.scale.z = 0.2
            m.id += 1
            markers_array.append(deepcopy(m))
        else:
            m.type = m.SPHERE
            m.scale.x = 0.01
            m.scale.y = 0.01
            m.scale.z = 0.01
            m.id += 1
            markers_array.append(deepcopy(m))

        ## Marker relaxit input
        if isinstance(settings.goal_pose, Pose):
            m = Marker()
            m.header.frame_id = "/"+settings.BASE_LINK
            m.type = m.ARROW
            m.action = m.ADD
            m.scale.x = 0.2
            m.scale.y = 0.01
            m.scale.z = 0.01
            m.color.a = 1.0
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.id += 1
            m.pose = Pose()
            m.pose = deepcopy(settings.goal_pose)
            markers_array.append(deepcopy(m))
            m.color.g = 0.0
            m.pose.orientation = Quaternion(*tf.transformations.quaternion_multiply(settings.extq(m.pose.orientation), settings.extq(q_norm_to_dir)))
            m.id += 1
            markers_array.append(deepcopy(m))


        ## Writing trajectory
        m = Marker()
        m.header.frame_id = "/"+settings.BASE_LINK
        m.type = m.SPHERE
        m.action = m.ADD
        m.scale.x = 0.01
        m.scale.y = 0.01
        m.scale.z = 0.01
        m.color.a = 1.0
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 1.0
        m.id = 1000
        if settings.print_path_trace:
            for frame in list(settings.goal_pose_array):
                m.pose = frame
                m.id += 1
                markers_array.append(deepcopy(m))
        else:
            for frame in list(settings.goal_pose_array):
                m.action = m.DELETE
                m.id += 1
                markers_array.append(deepcopy(m))

        #goal_target_pose = Pose()
        #goal_target_pose.orientation.w = 1.0
        #goal_target_pose.position.x = 0.7
        #goal_target_pose.position.y = -0.2
        #goal_target_pose.position.z = 0.0

        #m.pose = goal_target_pose
        #m.action = m.ADD
        #m.id += 1
        #m.type = m.CUBE
        #m.color.r = 0.0
        #m.color.g = 1.0
        #m.color.b = 0.0
        #m.scale.x = 0.2
        #m.scale.y = 0.2
        #m.scale.z = 0.001
        #markers_array.append(deepcopy(m))

        return markers_array
