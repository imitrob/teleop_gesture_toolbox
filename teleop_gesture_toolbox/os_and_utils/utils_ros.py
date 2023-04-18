import numpy as np

from geometry_msgs.msg import Quaternion, Pose, PoseStamped, Point, Vector3
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Int8, Float64MultiArray

def samePoses(pose1, pose2, accuracy=0.05):
    ''' Checks if two type poses are near each other
        (Only for cartesian (xyz), not orientation wise)
    Parameters:
        pose1 (type Pose(), Point(), list or tuple)
        pose2 (type Pose(), Point(), list or tuple)
        accuracy (Float): threshold of return value
    Returns:
        same poses (Bool)
    '''
    assert isinstance(pose1,(Pose,Point,np.ndarray,list,tuple)), "Not right datatype, pose1: "+str(pose1)
    assert isinstance(pose2,(Pose,Point,np.ndarray,list,tuple)), "Not right datatype, pose2: "+str(pose2)

    if isinstance(pose1,(list,tuple,np.ndarray)):
        pose1 = pose1[0:3]
    elif isinstance(pose1,Point):
        pose1 = [pose1.x, pose1.y, pose1.z]
    elif isinstance(pose1,Pose):
        pose1 = [pose1.position.x, pose1.position.y, pose1.position.z]
    if isinstance(pose2,(list,tuple,np.ndarray)):
        pose2 = pose2[0:3]
    elif isinstance(pose2,Point):
        pose2 = [pose2.x, pose2.y, pose2.z]
    elif isinstance(pose2,Pose):
        pose2 = [pose2.position.x, pose2.position.y, pose2.position.z]

    if np.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2 + (pose1[2] - pose2[2])**2) < accuracy:
        return True
    return False

def sameJoints(joints1, joints2, accuracy=0.1):
    ''' Checks if two type joints are near each other
    Parameters:
        joints1 (type float[7])
        joints2 (type float[7])
        threshold (Float): sum of joint differences threshold
    '''
    assert isinstance(joints1[0],float) and len(joints1)==7, "Not datatype List w len 7, joints 1: "+str(joints1)
    assert isinstance(joints2[0],float) and len(joints2)==7, "Not datatype List w len 7, joints 2: "+str(joints2)

    if sum([abs(i[0]-i[1]) for i in zip(joints1, joints2)]) < accuracy:
        return True
    return False
