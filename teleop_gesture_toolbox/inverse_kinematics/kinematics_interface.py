#!/usr/bin/env python2
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.

@author: Jan Behrens

This source code is derived from the dmp_gestures project
(https://github.com/awesomebytes/dmp_gestures)
Copyright (c) 2013, Willow Garage, Inc., licensed under the BSD license,
cf. 3rd-party-licenses.txt file in the root directory of this source tree.
"""

"""
Created on 12/08/14
@author: Sammy Pfeiffer
@email: sammypfeiffer@gmail.com
This file contains kinematics related classes to ease
the use of MoveIt! kinematics services.
"""
import time
import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetStateValidity
from sensor_msgs.msg import JointState
from moveit_msgs.msg import MoveItErrorCodes, Constraints
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

DEFAULT_FK_SERVICE = "/compute_fk"
DEFAULT_IK_SERVICE = "/compute_ik"
DEFAULT_SV_SERVICE = "/check_state_validity"

class ForwardKinematics(Node):
    """Simplified interface to ask for forward kinematics"""

    def __init__(self, frame_id=None):
        super().__init__("forward_kinematics")

        print("Loading ForwardKinematics class.")
        self.fk_srv = self.create_client(GetPositionFK, DEFAULT_FK_SERVICE)
        print("Connecting to FK service")
        while not self.fk_srv.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        print("Ready for making FK calls")

        self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)
        self.joint_states = None

        # Initilize global frame_id
        self.FRAME_ID = None
        if frame_id:
            self.FRAME_ID = frame_id

    def closeFK(self):
        self.fk_srv.close()

    def joint_states_callback(self, msg):
        self.joint_states = msg

    def getFK(self, fk_link_names, joint_names, positions, frame_id=None):
        """Get the forward kinematics of a joint configuration
        @fk_link_names list of string or string : list of links that we want to get the forward kinematics from
        @joint_names list of string : with the joint names to set a position to ask for the FK
        @positions list of double : with the position of the joints
        @frame_id string : the reference frame to be used"""
        gpfkr = GetPositionFK.Request()
        if type(fk_link_names) == type("string"):
            gpfkr.fk_link_names = [fk_link_names]
        else:
            gpfkr.fk_link_names = fk_link_names
        gpfkr.robot_state.joint_state.name = joint_names
        gpfkr.robot_state.joint_state.position = positions
        gpfkr.header.frame_id = frame_id
        # fk_result = GetPositionFKResponse()
        fk_result = self.fk_srv.call(gpfkr)
        return fk_result

    def getCurrentFK(self, fk_link_names, frame_id=None):
        """Get the forward kinematics of a set of links in the current configuration"""
        # if frame_id already specified when creating an object instance
        if self.FRAME_ID:
            frame_id = self.FRAME_ID
        # Subscribe to a joint_states
        #js = self.wait_for_message('/joint_states', JointState)
        while self.joint_states is None:
            rclpy.spin_once(self)
            print("Waiting for joint_states")
            time.sleep(1)
        # Call FK service
        fk_result = self.getFK(fk_link_names, self.joint_states.name, self.joint_states.position, frame_id=frame_id)
        return fk_result


#     def isJointConfigInCollision(self, fk_link_names, joint_names, positions): # This must use InverseKinematics as it has collision avoidance detection
#         """Given a joint config return True if in collision, False otherwise"""
#         fk_result = self.getFK(fk_link_names, joint_names, positions)
#         print fk_result
#         if fk_result.error_code != MoveItErrorCodes.SUCCESS:
#             return False
#         return True


class InverseKinematics(Node):
    """Simplified interface to ask for inverse kinematics"""

    def __init__(self, ik_srv=None):
        super().__init__("forward_kinematics")


        if ik_srv is None:
            print("Loading InverseKinematics class.")
            self.ik_srv = self.create_client(GetPositionIK, DEFAULT_IK_SERVICE)
        else:
            print("Loading InverseKinematics class.")
            self.ik_srv = self.create_client(GetPositionIK, ik_srv)
        print("Connecting to IK service")
        self.ik_srv.wait_for_service()
        print("Ready for making IK calls")

    def closeIK(self):
        self.ik_srv.close()

    def getIK(self, group_name, ik_link_name, pose, avoid_collisions=True, attempts=None, robot_state=None,
              constraints=None):
        """Get the inverse kinematics for a group with a link a in pose in 3d world.
        @group_name string group i.e. right_arm that will perform the IK
        @ik_link_name string link that will be in the pose given to evaluate the IK
        @pose PoseStamped that represents the pose (with frame_id!) of the link
        @avoid_collisions Bool if we want solutions with collision avoidance
        @attempts Int number of attempts to get an Ik as it can fail depending on what IK is being used
        @robot_state RobotState the robot state where to start searching IK from (optional, current pose will be used
        if ignored)"""
        gpikr = GetPositionIKRequest()
        gpikr.ik_request.group_name = group_name
        if robot_state != None:  # current robot state will be used internally otherwise
            gpikr.ik_request.robot_state = robot_state
        gpikr.ik_request.avoid_collisions = avoid_collisions
        gpikr.ik_request.ik_link_name = ik_link_name
        if type(pose) == type(PoseStamped()):
            gpikr.ik_request.pose_stamped = pose
        else:
            self.get_logger().error("pose is not a PoseStamped, it's: " + str(type(pose)) + ", can't ask for an IK")
            return
        if attempts != None:
            gpikr.ik_request.attempts = attempts
        else:
            gpikr.ik_request.attempts = 0
        if constraints != None:
            gpikr.ik_request.constraints = constraints
        ik_result = self.ik_srv.call(gpikr)
        self.get_logger().warning("Sent: " + str(gpikr))
        return ik_result

class StateValidity(Node):
    def __init__(self):
        super().__init__("state_validity")

        print("Initializing stateValidity class")

        self.sv_srv = self.create_client(GetStateValidity, DEFAULT_SV_SERVICE)
        print("Connecting to State Validity service")
        self.sv_srv.wait_for_service()
        if self.has_parameter('/play_motion/approach_planner/planning_groups'):
            list_planning_groups = self.get_parameter('/play_motion/approach_planner/planning_groups')
            # Get groups and joints here
            # Or just always use both_arms_torso...
        else:
            self.get_logger().warning("Param '/play_motion/approach_planner/planning_groups' not set. We can't guess controllers")
        print("Ready for making Validity calls")

    def close_SV(self):
        self.sv_srv.close()

    def getStateValidity(self, robot_state, group_name='both_arms_torso', constraints=None):
        """Given a RobotState and a group name and an optional Constraints
        return the validity of the State"""
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = robot_state
        gsvr.group_name = group_name
        if constraints != None:
            gsvr.constraints = constraints
        result = self.sv_srv.call(gsvr)
        # GetStateValidityResponse()
        # self.get_logger().warning("sent: " + str(gsvr))

    # JAN BEHRENS (jan.behrens@de.bosch.com - 2018-10-30): Return a bool instead of the full message
        return result.valid
    ################################################################################################


if __name__ == '__main__':
    rclpy.init(args=None)

    print("Initializing forward kinematics test.")
    fk = ForwardKinematics()
    print("Current FK:")
    print(str(fk.getCurrentFK('r1_ee')))
    print(str(fk.getCurrentFK('r1_gripper')))
    print(str(fk.getCurrentFK('r1_link_7')))
    #     print("isJointConfigInCollision with all left arm at 0.0 (should be False):")
    #     print( str(fk.isJointConfigInCollision('arm_left_7_link',
    #                                 ['arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint',
    #                                  'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint',
    #                                  'arm_left_7_joint'],
    #                                  [0.0, 0.0, 0.0,
    #                                   0.0, 0.0, 0.0,
    #                                   0.0]) ))
    #     print("isJointConfigInCollision with shoulder pointing inwards (should be True):")
    #     print( str(fk.isJointConfigInCollision('arm_left_7_link',
    #                                 ['arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint',
    #                                  'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint',
    #                                  'arm_left_7_joint'],
    #                                  [-2.0, 0.0, 0.0,
    #                                   0.0, 0.0, 0.0,
    #                                   0.0]) ))

    fk.closeFK()
    exit()
    print("Initializing inverse kinematics test.")
    ik = InverseKinematics()
    ps = PoseStamped()
    ps.header.frame_id = 'panda_link0'
    ps.pose.position = Point(0.3, -0.3, 1.1)
    ps.pose.orientation.w = 1.0
    print("IK for:\n" + str(ps))
    args = ["right_arm", "arm_right_7_link", ps]
    print(str(ik.getIK(*args)))

    ps.pose.position.x = 0.9
    print("IK for:\n" + str(ps))
    args = ["right_arm", "arm_right_7_link", ps]
    print(str(ik.getIK(*args)))
    ik.closeIK()
