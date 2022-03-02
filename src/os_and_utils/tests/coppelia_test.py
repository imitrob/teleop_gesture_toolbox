import sys, os, argparse, time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import rospy
from itertools import combinations, product


sys.path.append('../..')
import settings; settings.init()

sys.path.append('leapmotion') # .../src/leapmotion
#from loading import HandDataLoader, DatasetLoader
from os_and_utils.loading import HandDataLoader, DatasetLoader

from os_and_utils.transformations import Transformations as tfm
from os_and_utils.utils_ros import extv
from os_and_utils.visualizer_lib import VisualizerLib
from os_and_utils.path_def import Waypoint

import gestures_lib as gl
from os_and_utils.nnwrapper import NNWrapper
gl.init()

# TEMP:
from geometry_msgs.msg import Pose, Point, Quaternion

## init Coppelia
from geometry_msgs.msg import Pose, Point, Quaternion

# Import utils.py from src folder

sys.path.append(settings.paths.mirracle_sim_path)
from utils import *

from mirracle_sim.srv import AddOrEditObject, AddOrEditObjectResponse, RemoveObject, RemoveObjectResponse, GripperControl, GripperControlResponse
from mirracle_sim.msg import ObjectInfo
from sensor_msgs.msg import JointState, Image

import os_and_utils.scenes as sl
if __name__ == '__main__': sl.init()
import os_and_utils.move_lib as ml
if __name__ == '__main__': ml.init()

from coppelia_sim_ros_lib import CoppeliaROSInterface
from os_and_utils.ros_communication_main import ROSComm

from os_and_utils.parse_yaml import ParseYAML
from os_and_utils.utils import get_object_of_closest_distance
from os_and_utils.utils_ros import samePoses

rospy.init_node("coppeliaSimPublisherTopic", anonymous=True)

pose = Pose()
pose.position = Point(0.3, 0.0, 0.0)
pose.orientation = Quaternion(0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
CoppeliaROSInterface.add_or_edit_object(name="object1",pose=pose, shape='sphere', color='b', dynamic='false', size=[0.02,0.02,0.02], collision='false')


pose = Pose()
pose.position = Point(*ml.md.mouse3d_position)
pose.orientation = Quaternion(0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
CoppeliaROSInterface.add_or_edit_object(name="mouse3d",pose=pose, shape='sphere', color='r', dynamic='false', size=[0.02,0.02,0.02], collision='false')

roscm = ROSComm()

sim = CoppeliaROSInterface()
pose.position = Point(0.4, 0.0, 0.2)
sim.go_to_pose(pose, blocking=True)
sl.scenes.make_scene(sim, 'pickplace')



def execute(pathwaypoints):
    pose = Pose()
    pose.position = Point(0.4, 0.0, 0.2)
    pose.orientation = Quaternion(0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
    sim.go_to_pose(pose, blocking=True)
    sl.scenes.make_scene(sim, 'pickplace')
    time.sleep(2)

    path, waypoints = pathwaypoints

    if path is not None:
        for point in path:
            pose.position = Point(*point)
            #%timeit sim.add_or_edit_object(name="object1",pose=pose)
            sim.go_to_pose(pose, blocking=False)

    for n, waypoint_key in enumerate(list(waypoints.keys())):
        waypoint = waypoints[waypoint_key]
        s = f"wp {n} "
        if waypoint.gripper is not None: s += f'(gripper {waypoint.gripper})'; sim.gripper_control(waypoint.gripper)
        if waypoint.eef_rot is not None: s += f'(eef_rot {waypoint.eef_rot})'
        print(s)

    time.sleep(2)

pathwaypoints = [[ 0.45660701, -0.03497654,  0.25430629],
[ 0.45656446, -0.03517538,  0.25439763],
[ 0.45652134, -0.03537437,  0.25448196],
[ 0.45647972, -0.03557179,  0.25456775],
[ 0.45643801, -0.03576861,  0.25465176],
[ 0.45639952, -0.03596453,  0.25472892],
[ 0.45636063, -0.03615952,  0.25480719],
[ 0.45632287, -0.03635061,  0.2548813 ],
[ 0.45628563, -0.03654133,  0.25495052],
[ 0.45625052, -0.03672939,  0.25502147],
[ 0.45621634, -0.03691609,  0.25508741],
[ 0.45618283, -0.03709941,  0.25514921],
[ 0.45615194, -0.0372792,   0.25520795],
[ 0.45612111, -0.03745696,  0.25526285],
[ 0.45609256, -0.03763114,  0.25531649],
[ 0.45606568, -0.03780115,  0.25536105],
[ 0.45604056, -0.03796791,  0.25540408],
[ 0.45601636, -0.03813031,  0.25544387],
[ 0.45599474, -0.03828889,  0.2554762 ],
[ 0.45597392, -0.03844254,  0.25550864],
[ 0.45595653, -0.0385928,   0.2555328 ],
[ 0.45593997, -0.03873713,  0.25555241],
[ 0.45592522, -0.03887686,  0.25556641],
[ 0.45591251, -0.03901199,  0.25557804],
[ 0.45590338, -0.03914005,  0.2555803 ],
[ 0.4558946,  -0.03926207,  0.25558205],
[ 0.45588838, -0.03938047,  0.25557585],
[ 0.45588453, -0.03949176,  0.25556509],
[ 0.45588345, -0.03959803,  0.25554787],
[ 0.45588402, -0.03969656,  0.25552778],
[ 0.45588658, -0.03979033,  0.2555007 ],
[ 0.45589251, -0.03987501,  0.2554642 ],
[ 0.45590108, -0.03995558,  0.25542659],
[ 0.4559105,  -0.04002817,  0.25537972],
[ 0.4559233,  -0.04009433,  0.25533059],
[ 0.45593901, -0.0401536,   0.25527452],
[ 0.45595631, -0.04020507,  0.25521126],
[ 0.45597711, -0.04024921,  0.25514586],
[ 0.45600021, -0.0402869,   0.25507073],
[ 0.45602605, -0.04031622,  0.25499046],
[ 0.45605459, -0.04033897,  0.25490608],
[ 0.45608499, -0.04035388,  0.25481358],
[ 0.45611818, -0.04036247,  0.25471977],
[ 0.45615429, -0.04036274,  0.25461635],
[ 0.45619335, -0.04035551,  0.25450859],
[ 0.4562357,  -0.04034137,  0.25439673],
[ 0.45627974, -0.04031824,  0.25427918],
[ 0.45632677, -0.040289,    0.25415817],
[ 0.45637541, -0.04025281,  0.25403076],
[ 0.45642785, -0.04020912,  0.25389618],
[ 0.45648274, -0.04015655,  0.25375777],
[ 0.45653958, -0.04009851,  0.2536157 ],
[ 0.45659991, -0.04003297,  0.25347013],
[ 0.45666219, -0.03996063,  0.25331894],
[ 0.45672619, -0.03988101,  0.25316805],
[ 0.4567938,  -0.0397942,   0.25300723],
[ 0.45686381, -0.03970173,  0.25284688],
[ 0.45693606, -0.03960199,  0.25268054],
[ 0.4570099,  -0.03949616,  0.2525138 ],
[ 0.45708658, -0.03938442,  0.25233938],
[ 0.45716438, -0.03926649,  0.25216737],
[ 0.45724633, -0.03914092,  0.25198911],
[ 0.45732988, -0.03901231,  0.25181071],
[ 0.45741464, -0.03887705,  0.25163009],
[ 0.45750197, -0.03873627,  0.25144634],
[ 0.45759122, -0.03858978,  0.25126288],
[ 0.45768259, -0.03843875,  0.25107567],
[ 0.45777604, -0.03828441,  0.25089128],
[ 0.45786923, -0.03812306,  0.25070384],
[ 0.45796553, -0.03795973,  0.25051233],
[ 0.45806338, -0.03779097,  0.25032519],
[ 0.4581634,  -0.0376185,   0.25013693],
[ 0.45826343, -0.03744311,  0.24994668],
[ 0.45836615, -0.03726366,  0.24976173],
[ 0.4584683,  -0.03708043,  0.24957418],
[ 0.45857288, -0.03689578,  0.2493874 ],
[ 0.45867869, -0.03670802,  0.24920107],
[ 0.45878507, -0.03651713,  0.24902003],
[ 0.45889272, -0.03632374,  0.24883827],
[ 0.45900095, -0.03612893,  0.24865628],
[ 0.45910965, -0.03593272,  0.24848094],
[ 0.45921929, -0.0357352,   0.24830418],
[ 0.45932902, -0.03553611,  0.2481336 ],
[ 0.45943897, -0.03533577,  0.24796425],
[ 0.45954895, -0.03513388,  0.24779672],
[ 0.45966007, -0.03493236,  0.24763395],
[ 0.45977173, -0.03472894,  0.24747528],
[ 0.45988229, -0.03452721,  0.24731637],
[ 0.45999315, -0.03432476,  0.24716297],
[ 0.46010307, -0.03412333,  0.24701781],
[ 0.46021327, -0.03392091,  0.2468747 ],
[ 0.46032374, -0.03371937,  0.24673568],
[ 0.46043319, -0.03351926,  0.24659791],
[ 0.46054265, -0.0333202,   0.24647143],
[ 0.46064957, -0.03312124,  0.24634296],
[ 0.46075777, -0.03292458,  0.24622395],
[ 0.46086348, -0.03272948,  0.24610323],
[ 0.46096831, -0.03253456,  0.24599742],
[ 0.46107324, -0.03234289,  0.24589181],
[ 0.46117658, -0.03215237,  0.24578983],
[ 0.461277,   -0.0319636,   0.24569592],
[ 0.46137886, -0.0317769,   0.24560419],
[ 0.46147763, -0.03159316,  0.24551844],
[ 0.46157507, -0.03141086,  0.24544223],
[ 0.46167191, -0.03123042,  0.24536677],
[ 0.46176523, -0.03105413,  0.24530008],
[ 0.46185817, -0.03087862,  0.24523249],
[ 0.46194832, -0.03070591,  0.24517495],
[ 0.46203674, -0.03053534,  0.24512374],
[ 0.4621241,  -0.03036848,  0.24507328],
[ 0.46220805, -0.03020285,  0.24503196],
[ 0.4622906,  -0.03004111,  0.24499261],
[ 0.46237034, -0.02988015,  0.24495839],
[ 0.46244939, -0.02972317,  0.24492815],
[ 0.46252431, -0.02956851,  0.24490379],
[ 0.46259802, -0.02941572,  0.24488726],
[ 0.46266842, -0.0292663,   0.24487288],
[ 0.4627368,  -0.0291183,   0.24486224],
[ 0.46280323, -0.02897377,  0.24485321],
[ 0.46286591, -0.02883112,  0.24485064],
[ 0.46292604, -0.02869038,  0.24485296],
[ 0.46298445, -0.02855238,  0.24485564],
[ 0.46303909, -0.02841677,  0.24486622],
[ 0.46309153, -0.02828249,  0.24487525],
[ 0.46314097, -0.02815131,  0.24488997],
[ 0.46318891, -0.02802015,  0.24490575],
[ 0.46323139, -0.02789196,  0.24492347],
[ 0.46327251, -0.02776388,  0.24494828],
[ 0.46331131, -0.02763923,  0.24497242],
[ 0.46334611, -0.0275157,   0.24499961],
[ 0.4633781,  -0.0273923,   0.24502667],
[ 0.4634079,  -0.0272712,   0.24505657],
[ 0.46343419, -0.0271508,   0.24508649],
[ 0.46345769, -0.02703151,  0.24511836],
[ 0.46347837, -0.02691128,  0.24515073],
[ 0.46349598, -0.02679331,  0.24518438],
[ 0.46351048, -0.02667689,  0.24521722],
[ 0.46352245, -0.02655952,  0.24525194],
[ 0.46353151, -0.02644279,  0.24528387],
[ 0.46353673, -0.02632518,  0.24531913],
[ 0.4635404,  -0.02620872,  0.24535064],
[ 0.46354018, -0.02609222,  0.24538531],
[ 0.46353802, -0.02597527,  0.24541477],
[ 0.46353157, -0.02585724,  0.24544453],
[ 0.46352344, -0.02574068,  0.24547457],
[ 0.46351267, -0.02562182,  0.24550231],
[ 0.46349841, -0.02550304,  0.24552665],
[ 0.4634827,  -0.02538261,  0.24554937],
[ 0.46346339, -0.02526169,  0.24557133],
[ 0.46344049, -0.02513969,  0.24559129],
[ 0.46341779, -0.0250167,   0.24560806],
[ 0.46338956, -0.0248924,   0.24562052],
[ 0.46336029, -0.0247671,   0.24563438],
[ 0.4633286,  -0.02463988,  0.24564285],
[ 0.46329379, -0.02451108,  0.24564857],
[ 0.46325746, -0.02438081,  0.24564945],
[ 0.46321723, -0.02424967,  0.24564896],
[ 0.46317646, -0.02411595,  0.24564634],
[ 0.46313308, -0.02398088,  0.2456377 ],
[ 0.46308686, -0.02384392,  0.24562572],
[ 0.46303875, -0.02370468,  0.24561246],
[ 0.46298876, -0.02356369,  0.24559136],
[ 0.46293711, -0.02342116,  0.24556812],
[ 0.46288395, -0.02327602,  0.24554457],
[ 0.46282826, -0.02312931,  0.24551193],
[ 0.46277002, -0.02298026,  0.24547744],
[ 0.46271201, -0.02282867,  0.24544057],
[ 0.46265077, -0.02267643,  0.24539756],
[ 0.46258869, -0.02252119,  0.2453491 ],
[ 0.46252501, -0.02236419,  0.24529937],
[ 0.46246065, -0.0222051,   0.2452433 ],
[ 0.4623926,  -0.0220433,   0.24518336],
[ 0.46232462, -0.0218807,   0.24512068],
[ 0.46225504, -0.0217152,   0.24505395],
[ 0.46218488, -0.02154781,  0.24498333],
[ 0.46211319, -0.02137809,  0.24490449],
[ 0.46204075, -0.02120668,  0.24482736],
[ 0.46196663, -0.02103406,  0.24474103],
[ 0.46189222, -0.02085955,  0.24465611],
[ 0.46181657, -0.02068309,  0.24456381],
[ 0.46174123, -0.02050528,  0.24447045],
[ 0.46166387, -0.0203241,   0.24437365],
[ 0.46158651, -0.02014302,  0.24426962],
[ 0.46150814, -0.0199607,   0.24416561],
[ 0.46142951, -0.01977661,  0.24405547],
[ 0.46135162, -0.01959016,  0.24394593],
[ 0.46127082, -0.01940438,  0.24382694],
[ 0.46119089, -0.0192158,   0.24370927],
[ 0.46111178, -0.01902657,  0.24358811],
[ 0.46103043, -0.0188371,   0.24346252],
[ 0.46095032, -0.01864568,  0.24333844],
[ 0.46086969, -0.01845306,  0.24320591],
[ 0.46078973, -0.01826044,  0.24307592],
[ 0.46070796, -0.01806775,  0.24293844],
[ 0.46062766, -0.01787443,  0.24280088],
[ 0.46054752, -0.01767947,  0.24265944],
[ 0.46046702, -0.01748528,  0.24252133],
[ 0.46038639, -0.01728995,  0.24237747],
[ 0.46030682, -0.01709411,  0.24223195],
[ 0.4602274,  -0.01689904,  0.24208619]]

execute((pathwaypoints, {}))







#
