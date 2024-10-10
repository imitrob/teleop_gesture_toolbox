
import panda_py
from panda_py import Panda
from panda_py.libfranka import Gripper
from panda_py import controllers
import numpy as np

# Panda hostname/IP and Desk login information of your robot
HOSTNAME = "192.168.89.140"
username = 'admin'
password = '123456789'

# panda-py is chatty, activate information log level
import logging
logging.basicConfig(level=logging.INFO)

class PandaPy():
    def __init__(self):
        super(PandaPy, self).__init__()

        self.desk = panda_py.Desk(HOSTNAME, username, password)
        self.desk.unlock()
        self.desk.activate_fci()

        self.panda = Panda(HOSTNAME)
        self.panda.move_to_start()

        self.gripper = Gripper(HOSTNAME)
        self.position = (0.4,0.0,0.4)
        self.orientation = (1.0,0.0,0.0,0.0)

    def ctrl_node(self):
        ctrl = controllers.CartesianImpedance(filter_coeff=1.0, impedance=np.diag([100, 100, 100, 50, 50, 50]))
        x0 = self.panda.get_position()
        q0 = self.panda.get_orientation()
        runtime = np.pi * 4.0
        self.panda.start_controller(ctrl)

        while True:
            self.panda.start_controller(ctrl)
            try:
                with self.panda.create_context(frequency=1e3, max_runtime=runtime) as ctx:
                    while ctx.ok():
                        ctrl.set_control(self.position, self.orientation)
            except:
                print("Ctrl aborted, trying again")

    def move_to_pose(self, position, orientation, speed_factor):
        self.position = tuple(position)
        self.orientation = tuple((1.0,0.0,0.0,0.0))
        # self.panda.move_to_pose(*args, **kwargs)

    def grasp(self, *args, **kwargs):
        self.gripper.grasp(*args, **kwargs)

    def move(self, *args, **kwargs):
        self.gripper.move(*args, **kwargs)
