
import numpy as np
from os_and_utils.transformations import Transformations as tfm
try:
    import os_and_utils.ros_communication_main as rc
except:
    rc = None

class DeiticLib():
    def __init__(self):
        self.disabled = False

    ''' The Leap Motion base to Robot base '''
    # Note: Currently using function in file transformations.py
    # transformLeapToBase__CornerConfig()
    def calibrate_leap_motion(self):
        ''' Calibration procedure
        Returns:
            transformation matrix 4x4
        - Note: Measured by ruler
        '''
        return np.array([[0, 1, 0, 1.07],
                         [0, 0, 1,-0.40],
                         [1, 0, 0, 0.01],
                         [0, 0, 0, 1   ]])

    def leap_motion_to_world_link(self, p):
        T = calibrate_leap_motion()
        # TMP
        return [T[0,3], T[1,3], T[2,3]]

    def get_closest_point_to_line(self,line_points, test_point):
        ''' Line points
        '''
        p1, p2 = line_points
        p3 = test_point

        x1, y1, z1 = p1
        x2, y2, z2 = p2
        x3, y3, z3 = p3
        dx, dy, dz = x2-x1, y2-y1, z2-z1
        det = dx*dx + dy*dy + dz*dz
        a = (dy*(y3-y1)+dx*(x3-x1)+dz*(z3-z1))/det
        return x1+a*dx, y1+a*dy, z1+a*dz

    def get_id_of_closest_point_to_line(self, line_points, test_points, max_dist=np.inf):
        '''
        Embedding: line_points are start and end of last bone (distal) from pointing finger (2nd finger)
        test_points:
        '''
        distances_from_line = []
        for test_point in test_points:
            closest_point = self.get_closest_point_to_line(line_points, test_point)
            norm_distance = np.linalg.norm(np.array(closest_point)-np.array(test_point))
            distances_from_line.append(norm_distance)

        if np.min(distances_from_line) > max_dist: return None, np.min(distances_from_line)
        return np.argmin(distances_from_line), np.min(distances_from_line)

    def main_deitic_fun(self, f, h, object_poses):
        ''' has dependencies
        f (Frame): ml.md.frames[-1] (take last detection frame)
        h (string): hand - 'l' left, 'r', right
        object_poses (): sl.scene.object_poses
        '''
        hand = getattr(f, h)
        p1, p2 = np.array(hand.palm_position()), np.array(hand.palm_position())+np.array(hand.direction())
        #p1, p2 = np.array(hand.fingers[1].bones[3].prev_joint()), np.array(hand.fingers[1].bones[3].next_joint())
        p1s = np.array(tfm.transformLeapToBase__CornerConfig(p1))
        p2s = np.array(tfm.transformLeapToBase__CornerConfig(p2))
        v = 1000*(p2s-p1s)
        line_points = (p1s, p2s+v)

        object_positions = [[pose.position.x,pose.position.y,pose.position.z] for pose in object_poses]
        idobj, _ = self.get_id_of_closest_point_to_line(line_points, object_positions, max_dist=np.inf)

        #if self.set_focus_logic(hand):
        if True:
            if rc is not None:

                with rc.rossem:
                    rc.roscm.r.add_line(name='line1', points=line_points)
                    rc.roscm.r.add_or_edit_object(name="Focus_target", pose=object_positions[idobj])
        return

    def set_focus_logic(self, hand):
        '''
        Set focus action can be enabled again only when:
            1.) done grab gesture (hand.grab_strength > 0.9)
            2.) hand out of scene (triggered externally in main script)
        '''
        if not self.disabled:
            self.disabled = True
            return True
        else:
            if hand.grab_strength > 0.9:
                self.enable()
            return False

    def enable(self):
        self.disabled = False

def init():
    global dd
    dd = DeiticLib()


if __name__ == '__main__':
    dl = DeiticLib()
    # test 1
    line_points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    test_points = ((0.0, 0.0, 1.00000), (10.0, 0.0, 1.0), (10.0, 0.0, 2.0))
    closest_point = dl.get_id_of_closest_point_to_line(line_points, test_points)
    closest_point

    # test 2
    line_points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    test_points = ((0.0, 0.0, 0.99999), (10.0, 0.0, 1.0), (10.0, 0.0, 2.0))
    closest_point = dl.get_id_of_closest_point_to_line(line_points, test_points)
    closest_point

    # test 3
    line_points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    test_points = ((0.0, 0.0, 0.99999), (10.0, 0.0, 1.0), (10.0, 0.0, 2.0))
    closest_point = dl.get_id_of_closest_point_to_line(line_points, test_points, max_dist=0.3)
    closest_point
