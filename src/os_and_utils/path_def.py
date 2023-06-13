
class Waypoint():
    def __init__(self, p=None, v=None, gripper=None, eef_rot=None):
        self.p = p # position [x,y,z]
        self.v = v # velocity [x,y,z]
        self.gripper = gripper # open to close (0. to 1.) [-]
        self.eef_rot = eef_rot # last joint position (-2.8973 to 2.8973) [rad]
    def export(self):
        return (self.p, self.v, self.gripper, self.eef_rot)
