#!/usr/bin/env python3
from threading import Thread
import numpy as np

import rclpy, sys
from rclpy.callback_groups import ReentrantCallbackGroup

from ament_index_python.packages import get_package_share_directory
package_share_directory = get_package_share_directory('pymoveit2')
sys.path.append(package_share_directory)

from pymoveit2 import MoveIt2
from pymoveit2.robots import panda
from pymoveit2 import MoveIt2Gripper

from sensor_msgs.msg import JointState
import time
''' Experimental '''
EXPERIMENTAL = False
if EXPERIMENTAL:
    try:
        import roboticstoolbox as rtb
        from spatialmath import UnitQuaternion
    except ModuleNotFoundError:
        rtb = None


class PyMoveIt2Interface():
    def __init__(self, rosnode):
        self.rosnode = rosnode
        self.rosnode.declare_parameter("joint_positions", [-2.23, 0.26, 2.44, -2.48, -0.20, 2.18, 1.13])
        self.rosnode.declare_parameter("position", [0.5, 0.0, 0.25])
        self.rosnode.declare_parameter("quat_xyzw", [1.0, 0.0, 0.0, 0.0])
        self.rosnode.declare_parameter("cartesian", False)

        self.callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self.rosnode,
            joint_names=panda.joint_names(),
            base_link_name=panda.base_link_name(),
            end_effector_name=panda.end_effector_name(),
            group_name=panda.MOVE_GROUP_ARM,
            callback_group=self.callback_group,
        )

        # Create MoveIt 2 gripper interface
        self.moveit2_gripper = MoveIt2Gripper(
            node=self.rosnode,
            gripper_joint_names=panda.gripper_joint_names(),
            open_gripper_joint_positions=panda.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=panda.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=panda.MOVE_GROUP_GRIPPER,
            callback_group=self.callback_group,
        )


        ''' Experimental '''
        if EXPERIMENTAL:
            self.rosnode.create_subscription(JointState, '/joint_states', self.save_joint_states, 5)
            self.joint_states = None



            while self.joint_states is None: print("waiting for js"); time.sleep(1)

            if rtb is not None:
                self.model = rtb.models.Panda()

    def save_joint_states(self, msg):
        self.joint_states = np.array(msg.position[:7])

    def go_to_pose(self, pose, wait=True, limit_per_call=999):
        '''
        limit_per_call: max meters/radians per fn call - useful, when publishing goal with frequency
        '''
        if isinstance(pose, (list,tuple,np.ndarray)) and len(pose) == 3:
            pose = pose[0:3] + [0.,1.,0.,0.]
        elif isinstance(pose, (list,tuple,np.ndarray)) and len(pose) == 7:
            pass #pose = pose
        else:
            pose = [pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        ''' Experimental '''
        if EXPERIMENTAL:
            if rtb is not None and self.joint_states is not None:
                EE = self.model.fkine(self.joint_states)
                q = UnitQuaternion(EE).vec_xyzs
                p = EE.t

                pose[0:3] = list(p + np.clip(np.array(pose[0:3]) - p, -limit_per_call, limit_per_call))
                pose[3:7] = list(q + np.clip(np.array(pose[3:7]) - q, -limit_per_call, limit_per_call))

        self.rosnode.undeclare_parameter("position")
        self.rosnode.undeclare_parameter("quat_xyzw")
        self.rosnode.undeclare_parameter("cartesian")
        self.rosnode.declare_parameter("position", pose[0:3])
        self.rosnode.declare_parameter("quat_xyzw", pose[3:7])
        self.rosnode.declare_parameter("cartesian", False)

        # Get parameters
        position = self.rosnode.get_parameter("position").get_parameter_value().double_array_value
        quat_xyzw = self.rosnode.get_parameter("quat_xyzw").get_parameter_value().double_array_value
        cartesian = self.rosnode.get_parameter("cartesian").get_parameter_value().bool_value

        self.moveit2.move_to_pose(position=position, quat_xyzw=quat_xyzw, cartesian=cartesian)
        if wait:
            self.moveit2.wait_until_executed()



    def executor_excepted(self, executor):
        try:
            executor.spin()
        except RuntimeError:
            pass

    def go_to_joints(self, j, wait=True, limit_per_call=999):
        '''
        limit_per_call: max radians per fn call - useful, when publishing goal with frequency
        '''
        #diff = np.clip(np.array(j) - self.joint_states, -limit_per_call, limit_per_call)
        #j = list(moveit.joint_states + diff)

        self.rosnode.undeclare_parameter("joint_positions")
        self.rosnode.declare_parameter("joint_positions", j)

        joint_positions = (self.rosnode.get_parameter("joint_positions").get_parameter_value().double_array_value)

        self.moveit2.move_to_configuration(joint_positions)
        if wait:
            self.moveit2.wait_until_executed()

    def set_gripper(self, position=-1, effort=0.4, eef_rot=-1, action="", object=""):
        '''
        Parameters:
            position (Float): 0. -> gripper closed, 1. -> gripper opened
            effort (Float): Range <0.0, 1.0>
            action (Str): 'grasp' attach object specified as 'object', 'release' will release previously attached object, '' (no attach/detach anything)
            object (Str): Name of object specified to attach
        Returns:
            success (Bool)
        '''
        # Spin the node in background thread(s)
        executor = rclpy.executors.MultiThreadedExecutor(2)
        executor.add_node(self.rosnode)
        executor_thread = Thread(target=executor.spin, daemon=True, args=())
        executor_thread.start()

        if position != -1:

            if position > 0.8:
                self.moveit2_gripper.open()
                self.moveit2_gripper.wait_until_executed()
            elif position < 0.2:
                self.moveit2_gripper.close()
                self.moveit2_gripper.wait_until_executed()


if __name__ == "__main__":
    rclpy.init()
    node = rclpy.node.Node("pymoveit2int")
    moveit = PyMoveIt2Interface(node)
    moveit.go_to_joints([-2.23, 0.26, 2.44, -2.48, -0.20, 2.18, 1.13])
    moveit.go_to_joints([-2.03, 0.26, 2.44, -2.48, -0.20, 2.18, 1.13])
    moveit.go_to_joints([-2.53, 0.26, 2.44, -2.48, -0.20, 2.18, 1.13])
