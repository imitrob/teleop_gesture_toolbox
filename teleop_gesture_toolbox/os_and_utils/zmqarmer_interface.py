#!/usr/bin/env python3
import numpy as np
import sys, zmq, time
import pickle
import json
import cv2
import pickle
import spatialmath as sm
from spatialmath import UnitQuaternion

class ZMQArmerInterface():
    def __init__(self):
        ## GoToPose
        context = zmq.Context()
        self.posesocket = context.socket(zmq.PUB)
        self.posesocket.bind("tcp://*:5557")

        self.grippersocket = context.socket(zmq.REQ)
        self.grippersocket.connect("tcp://192.168.88.146:5556")

        self.cosyposesocket = context.socket(zmq.REQ)
        self.cosyposesocket.connect("tcp://192.168.88.146:5559")

        self.getjoints_socket = context.socket(zmq.SUB)
        self.getjoints_socket.connect("tcp://192.168.88.146:5566")
        self.getjoints_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.getjoints_socket.setsockopt(zmq.LINGER, 100)


    def __exit__(self):
        self.posesocket.unbind("tcp://*:5557")
        self.posesocket.close()
        context.destroy()

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

        self.posesocket.send_string(f'{pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]}')



    def go_to_joints(self, j, wait=True, limit_per_call=999):
        '''
        limit_per_call: max radians per fn call - useful, when publishing goal with frequency
        '''
        raise Exception("Not implemented!")

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
        if position != -1:
            if position > 0.8:
                self.grippersocket.send(b'open')
                res = self.grippersocket.recv()
                print(res)
            elif position < 0.2:
                self.grippersocket.send(b'close')
                res = self.grippersocket.recv()
                print(res)

    def gripper_control(self, position, effort=0.4, eef_rot=-1, action="", object=""):
        self.set_gripper(position, effort=effort, eef_rot=eef_rot, action=action, object=object)

    def open_gripper(self):
        self.set_gripper(position=1.0)

    def close_gripper(self):
        self.set_gripper(position=0.0)

    def pick_object(self, object):
        self.set_gripper(position=0.0, action='grasp', object=object)

    def release_object(self):
        self.set_gripper(position=1.0, action='release')

    def get_object_positions(self):
        self.cosyposesocket.send_string("scene_camera")
        msg = self.cosyposesocket.recv()
        state = pickle.loads(msg)
        print(state['objects'])

        return state['objects']

        cv2.imshow('image', cv2.cvtColor(state['image'], cv2.COLOR_BGR2RGB))
        cv2.waitKey()
        sys.exit()

        cv2.imwrite(sys.argv[1], cv2.cvtColor(state['image'], cv2.COLOR_BGR2RGB))
        # json_object = json.dumps(state['objects'], indent = 4)
        with open(sys.argv[2], "wb") as outfile:

            pickle.dump(state['objects'], outfile)

    def get_joints(self):
        #self.getjoints_socket.send_string("enable_socket?")
        msg = self.getjoints_socket.recv()
        msg = pickle.loads(msg)
        #assert isinstance(msg, ())


        eef_pose = Tmat_to_pose(msg['O_T_EE'])

    @staticmethod
    def Tmat_to_pose(T):

        T = np.array(T)
        T.reshape((4,4))
        T_SE3 = sm.SE3(T)
        p = T_SE3.t
        q = UnitQuaternion(T_SE3.R).vec_xyzs
        return p, q


    def add_or_edit_object(self, *args, **kwargs):
        pass

    def remove_object(self, *args, **kwargs):
        pass

def test_Tmat_to_pose():
    T = np.eye(4)
    T.flatten()
    p,q = ZMQArmerInterface.Tmat_to_pose(T)
    assert np.allclose(p, np.zeros(3))
    assert np.allclose(q, np.array([0.,0.,0.,1.]))


if __name__ == "__main__":
    r = ZMQArmerInterface()
    print(r.get_joints())
    time.sleep(5)
    r.go_to_pose([0.5, 0.0, 0.4, -1.0, 0.0, 0.0, 0.0])
    r.go_to_pose([0.5, 0.0, 0.4, -1.0, 0.0, 0.0, 0.0])
    r.go_to_pose([0.5, 0.0, 0.4, -1.0, 0.0, 0.0, 0.0])
    r.go_to_pose([0.5, 0.0, 0.4, -1.0, 0.0, 0.0, 0.0])
    r.go_to_pose([0.5, 0.0, 0.4, -1.0, 0.0, 0.0, 0.0])
    r.go_to_pose([0.5, 0.0, 0.4, -1.0, 0.0, 0.0, 0.0])

    time.sleep(5)
    #r.close_gripper()
    '''r.go_to_pose([0.5, 0.1, 0.3, 0.0, 1.0, 0.0, 0.0])
    time.sleep(1)
    r.go_to_pose([0.5,-0.1, 0.3, 0.0, 1.0, 0.0, 0.0])
    time.sleep(1)
    r.set_gripper(0.0)
    time.sleep(1)
    r.set_gripper(1.0)
    time.sleep(1)'''
    #print("get joints")
    #print("get object positions")
    #print(r.get_object_positions())


class ZMQArmerInterfaceWithSem():
    '''
    '''
    def __init__(self, sem, *args, **kwargs):
        self.r = ZMQArmerInterface(*args, **kwargs)
        self.sem = sem

    def ready(self, *args, **kwargs):
        with self.sem:
            return self.r.ready(*args, **kwargs)

    def eef_pose_callback(self, *args, **kwargs):
        with self.sem:
            return self.r.eef_pose_callback(*args, **kwargs)

    def execute_trajectory_with_waypoints(self, *args, **kwargs):
        with self.sem:
            return self.r.execute_trajectory_with_waypoints(*args, **kwargs)

    def handle_waypoint_action(self, *args, **kwargs):
        with self.sem:
            return self.r.handle_waypoint_action(sem=self.sem, *args, **kwargs)

    def go_to_pose(self, *args, **kwargs):
        with self.sem:
            return self.r.go_to_pose(*args, **kwargs)

    def move_above_axis(self, *args, **kwargs):
        with self.sem:
            return self.r.move_above_axis(*args, **kwargs)

    def reset(self, *args, **kwargs):
        with self.sem:
            return self.r.reset(*args, **kwargs)

    def gripper_control(self, *args, **kwargs):
        with self.sem:
            return self.r.open_gripper(sem=self.sem, *args, **kwargs)

    def open_gripper(self, *args, **kwargs):
        with self.sem:
            return self.r.open_gripper(sem=self.sem, *args, **kwargs)

    def close_gripper(self, *args, **kwargs):
        with self.sem:
            return self.r.close_gripper(sem=self.sem, *args, **kwargs)

    def pick_object(self, *args, **kwargs):
        with self.sem:
            return self.r.pick_object(sem=self.sem, *args, **kwargs)

    def release_object(self, *args, **kwargs):
        with self.sem:
            return self.r.release_object(sem=self.sem, *args, **kwargs)

    def toggle_object(self, *args, **kwargs):
        with self.sem:
            return self.r.toggle_object(*args, **kwargs)

    def set_gripper(self, *args, **kwargs):
        with self.sem:
            return self.r.set_gripper(sem=self.sem, *args, **kwargs) # 1

    def add_line(self, *args, **kwargs):
        with self.sem:
            return self.r.add_line(sem=self.sem, *args, **kwargs) # 2

    def add_or_edit_object(self, *args, **kwargs):
        with self.sem:
            return self.r.add_or_edit_object(sem=self.sem, *args, **kwargs) # 3

    def remove_object(self, *args, **kwargs):
        with self.sem:
            return self.r.remove_object(sem=self.sem, *args, **kwargs) # 4
