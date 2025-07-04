
from __future__ import annotations
from pydantic import BaseModel, Field, conlist
from typing import List
from typing import Iterable, Optional
import numpy as np

class Point3D(BaseModel):
    """A single 3‑D point in metres, expressed in the robot/world frame."""
    x: float = Field(..., description="X‑coordinate (m)")
    y: float = Field(..., description="Y‑coordinate (m)")
    z: float = Field(..., description="Z‑coordinate (m)")

    def to_list(self):
        return [self.x, self.y, self.z]

class Line3D(BaseModel):
    """A directed line segment between *start* and *end* for one time sample."""
    start: Point3D
    end: Point3D

class PointingSequence(BaseModel):
    """Time‑ordered list of 3‑D line segments that make up the pointing gesture."""
    lines: List[DeicticSolution] = Field(
        default_factory=list,
        description="Sequence of line segments, one per time sample, oldest → newest.",
    )

    def get_contacts_with_ground(self):
        return [self.z0_hit(sol.line_points.start.to_list(), sol.line_points.end.to_list()) for sol in self.lines]

    @staticmethod
    def z0_hit(p0, p1):
        """Return (x, y, 0) where the infinite line through p0, p1 meets z=0, else None."""
        z0, z1 = p0[2], p1[2]
        if z0 == z1:                               # parallel to plane
            return p0[0:2] if z0 == 0 else (np.nan, np.nan) # either lies in plane or never intersects
        t = -z0 / (z1 - z0)                        # param where z(t)=0
        return (p0[0] + t*(p1[0]-p0[0]),
                p0[1] + t*(p1[1]-p0[1]))


class SelectionResult(BaseModel):
    """Just the ID of the object that the system thinks the user meant."""

    object_ids: List[int] = Field(..., min_length=1, description="")

    def to_bools(self, n):
        boolean_array = [False] * n
        for i in range(n):
            if i in self.object_ids:
                boolean_array[i] = True
        return boolean_array
        

class DeicticSolution(BaseModel):
    target_object_id: int = Field(..., description="Unique identifier of the selected object")
    target_object_name: str = Field(..., description="Unique name of the selected object")
    object_names: List[str] = Field(..., min_length=1, description="List of all object names.")
    object_distances: List[float] = Field(..., min_length=1, description="List of distances from human pointing ray to all objects.")
    object_likelihoods: List[float] = Field(..., min_length=1, description="List of likelihoods. One for each objects.")
    line_points: Line3D
    target_object_position: Point3D
    hand_velocity: float # palm velocity
    target_object_stamp: Optional[float] = None # ROS clock fills this




class DeiticLib():
    def __init__(self):
        super(DeiticLib, self).__init__()
        self.disabled = False

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

        Returns:
            id (int): Unique object id
            dist (float): Distance from the line to the chosen object
            all distances (List[float]): List of distances. Distance from the line towards each object on scene.
        '''
        assert test_points != [], "test_points empty"
        distances_from_line = []
        for test_point in test_points:
            closest_point = self.get_closest_point_to_line(line_points, test_point)
            norm_distance = np.linalg.norm(np.array(closest_point)-np.array(test_point))
            distances_from_line.append(norm_distance)

        if np.min(distances_from_line) > max_dist: 
            return None, np.min(distances_from_line), distances_from_line
        return int(np.argmin(distances_from_line)), np.min(distances_from_line), distances_from_line

    def compute_deictic_solution(self,
                                f, 
                                h: str,
                                object_poses: Iterable[list],
                                object_names: Iterable[str],
                                tf_fun,
        ):
        """
        Args:
            f (Frame): self.hand_frames[-1] (take last detection frame)
            h (str): hand - 'l' left, 'r', right
            object_poses (Iterable[list]): sl.scene.object_poses
            object_names (Iterable[str]): _description_

        Returns:
            dict: deictic_solution
        """
        if len(object_poses) == 0: return None

        if h == 'lr':
            if f.l.visible:
                h = 'l'
            elif f.r.visible:
                h = 'r'
            else:
                h = 'l'
        if not isinstance(h, str): 
            print(f"h is not string, h is {type(h)}")
        hand = getattr(f, h)

        if not hand.visible:
            return None

        p1, p2 = np.array(hand.palm_position.world), np.array(hand.palm_position.world)+np.array(hand.direction.world)
        #p1, p2 = np.array(hand.fingers[1].bones[3].prev_joint()), np.array(hand.fingers[1].bones[3].next_joint())
        p1s = np.array(tf_fun(p1))
        p2s = np.array(tf_fun(p2))
        v = 1000*(p2s-p1s) # extrapolate, make the line longer
        line_points = [list(p1s), list(p2s+v)]

        if isinstance(object_poses[0], (list, tuple, np.ndarray)):
            object_positions = [[pose[0],pose[1],pose[2]] for pose in object_poses]
        else:
            object_positions = [[pose.position.x,pose.position.y,pose.position.z] for pose in object_poses]
        idobj, _, distances_from_line = self.get_id_of_closest_point_to_line(line_points, object_positions, max_dist=np.inf)
        # print(f"[Deictic] Chosen: {object_names[int(np.argmin(distances_from_line))]}", flush=True)
        
        assert len(object_poses) == len(distances_from_line)
        
        deictic_solution = DeicticSolution(
            target_object_id = idobj,
            target_object_name = object_names[idobj],
            object_names = object_names,
            object_distances = distances_from_line,
            object_likelihoods= 1/np.array(distances_from_line),
            line_points = Line3D(start=Point3D(x=line_points[0][0], y=line_points[0][1], z=line_points[0][2]),
                                 end=Point3D(x=line_points[1][0], y=line_points[1][1], z=line_points[1][2])),
            target_object_position = Point3D(x=object_positions[idobj][0], y=object_positions[idobj][1], z=object_positions[idobj][2]),
            hand_velocity = np.linalg.norm(hand.palm_velocity()) / 1000 # from mm/s to m/s, make magnitude
        )
        return deictic_solution
    



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


