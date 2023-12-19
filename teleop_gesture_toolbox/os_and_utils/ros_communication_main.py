import sys, os, time, json
import rclpy
from rclpy.node import Node
from os_and_utils import settings
import numpy as np
from copy import deepcopy
import os_and_utils.move_lib as ml
if __name__ == '__main__': ml.init()
from os_and_utils.parse_yaml import ParseYAML
#from inverse_kinematics.ik_lib import IK_bridge
import gesture_classification.gestures_lib as gl
if __name__ == '__main__': gl.init()
import os_and_utils.scenes as sl
if __name__ == '__main__': sl.init()
from os_and_utils.transformations import Transformations as tfm

from std_msgs.msg import Int8, Float64MultiArray, Int32, Bool, MultiArrayDimension, String
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from teleop_msgs.msg import EEPoseGoals, JointAngles
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import JointState

from hand_processing.frame_lib import Frame
import teleop_msgs.msg as rosm
from teleop_msgs.msg import DetectionSolution, DetectionObservations
import threading

from teleop_msgs.msg import Scene as SceneRos
from teleop_msgs.msg import HRICommand
from teleop_msgs.msg import Gestures as GesturesRos
from teleop_msgs.srv import BTreeSingleCall
from teleop_msgs.srv import ChangeNetwork, SaveHandRecord
from gesture_classification.sentence_creation import GestureSentence
from copy import deepcopy

DEBUGSEMAPHORE = False

try:
    from spatialmath import UnitQuaternion
    import roboticstoolbox as rtb
except ModuleNotFoundError:
    rtb = None

try:
    import coppelia_sim_ros_interface
    from coppelia_sim_ros_interface_msgs.msg import ObjectInfo
    sys.path.append(settings.paths.coppelia_sim_ros_interface_path)
    from coppelia_sim_ros_client import CoppeliaROSInterface, CoppeliaROSInterfaceWithSem
except:
    print("WARNING: coppelia_sim_ros_interface package was not found!")
    coppelia_sim_ros_interface = None

import ament_index_python
try:
    from os_and_utils.pymoveit2_interface import PyMoveIt2Interface
    from ament_index_python.packages import get_package_share_directory
    package_share_directory = get_package_share_directory('pymoveit2')
except ModuleNotFoundError and ament_index_python.packages.PackageNotFoundError:
    print("WARNING: pymoveit2 package was not found!")
    PyMoveIt2Interface = None

try:
    from os_and_utils.zmqarmer_interface import ZMQArmerInterface, ZMQArmerInterfaceWithSem
except ModuleNotFoundError:
    ZMQArmerInterface = None

from os_and_utils.utils import ordered_load, point_by_ratio, get_cbgo_path, cc
sys.path.append(get_cbgo_path())
import context_based_gesture_operation as cbgo

def withsem(func):
    def inner(*args, **kwargs):
        if DEBUGSEMAPHORE: print(f"ACQ, {args}, {kwargs}")
        with rossem:
            ret = func(*args, **kwargs)
        if DEBUGSEMAPHORE: print("---")
        return ret
    return inner

try:
    # Crow interface
    from rclpy.qos import QoSProfile
    from rclpy.qos import QoSReliabilityPolicy
    from rdflib.namespace import Namespace, RDF, RDFS, OWL, FOAF, XSD
    from crow_ontology.crowracle_client import CrowtologyClient
    from crow_msgs.msg import StampedString

    ONTO_IRI = "http://imitrob.ciirc.cvut.cz/ontologies/crow"
    CROW = Namespace(f"{ONTO_IRI}#")

    ACTION_TRANSL = {"seber":"lift", "pustit":"drop", "ukaž":"reach", "podej":"pass me"}
except ModuleNotFoundError:
    CrowtologyClient = False

class ROSComm(Node):
    ''' ROS communication of main thread: Subscribers (init & callbacks) and Publishers
    '''
    def __init__(self, robot_interface='no-interface'):
        super().__init__('ros_comm_main')
        self.robot_interface = robot_interface
        self.is_real = True
        print(f"[ROScomm] On {robot_interface}")
        # Saving relaxedIK output
        #self.create_subscription(JointAngles, '/joint_angle_solutions', self.ik, 10)
        # Goal pose publisher
        self.ee_pose_goals_pub = self.create_publisher(EEPoseGoals,'/ee_pose_goals', 5)

        self.examplesub = self.create_subscription(rosm.Frame, '/hand_frame', ROSComm.hand_frame_callback, 10)

        if settings.launch_gesture_detection:
            self.create_subscription(DetectionSolution, '/teleop_gesture_toolbox/static_detection_solutions', self.save_static_detection_solutions_callback, 10)
            self.static_detection_observations_pub = self.create_publisher(DetectionObservations,'/teleop_gesture_toolbox/static_detection_observations', 5)

            self.create_subscription(DetectionSolution, '/teleop_gesture_toolbox/dynamic_detection_solutions', self.save_dynamic_detection_solutions_callback, 10)
            self.dynamic_detection_observations_pub = self.create_publisher(DetectionObservations, '/teleop_gesture_toolbox/dynamic_detection_observations', 5)

        self.hand_mode = settings.get_hand_mode()
        #self.controller = self.create_publisher(Float64MultiArray, '/teleop_gesture_toolbox/target', 5)
        #self.ik_bridge = IK_bridge()

        self.gesture_solution_pub = self.create_publisher(String, '/teleop_gesture_toolbox/filtered_gestures', 5)
        self.gesture_sentence_publisher_original = self.create_publisher(HRICommand, '/teleop_gesture_toolbox/gesture_sentence_original', 5)
        
        # self.gesture_sentence_publisher_mapped  = self.create_publisher(HRICommand, '/teleop_gesture_toolbox/action_sentence_mapped', 5)
        self.gesture_sentence_publisher_mapped  = self.create_publisher(HRICommand, '/hri/command', 5)


        self.call_tree_singlerun_cli = self.create_client(BTreeSingleCall, '/btree_onerun')
        #while not self.call_tree_singlerun_cli.wait_for_service(timeout_sec=1.0):
        #    print('Behaviour tree not available, waiting again...')

        self.save_hand_record_cli = self.create_client(SaveHandRecord, '/save_hand_record')
        if not self.save_hand_record_cli.wait_for_service(timeout_sec=1.0):
            print('save_hand_record service not available!!!')
        self.save_hand_record_req = SaveHandRecord.Request()

        self.r = None
        if 'coppelia' in robot_interface:
            self.create_subscription(JointState, "joint_states_coppelia", self.joint_states_callback, 10)

            self.create_subscription(Pose, '/pose_eef', self.coppelia_eef, 10)
            self.create_subscription(Vector3, '/coppelia/camera_angle', self.camera_angle, 10)
            if coppelia_sim_ros_interface is not None:
                self.create_subscription(ObjectInfo, '/coppelia/object_info', self.object_info_callback, 10)

            self.init_coppelia_interface()
            self.is_real = False

        if 'real' in robot_interface:
            self.create_subscription(JointState, "joint_states", self.joint_states_callback, 10)

            self.init_real_interface__pymoveit2__panda()
            self.is_real = True

        if 'ros1armer' in robot_interface:
            self.create_subscription(JointState, "joint_states", self.joint_states_callback, 10)

            self.init_real_interface__zmqarmer__panda()
            self.is_real = True

        self.last_time_livin = time.time()

        if rtb is not None:
            self.rtb_model = rtb.models.Panda()
            
        self.all_states_pub = self.create_publisher(String, '/teleop_gesture_toolbox/all_states', 5)
        ''' Data about scene, is alive, etc. Sent as string dict '''

    @withsem
    def save_hand_record(self, dir):
        self.save_hand_record_req.directory = settings.paths.learn_path+dir
        self.save_hand_record_req.save_method = 'numpy'
        self.save_hand_record_req.recording_length = 1.0

        return self.save_hand_record_cli.call_async(self.save_hand_record_req)

    @withsem
    def call_tree_singlerun(self, req):

        future = self.call_tree_singlerun_cli.call_async(req)
        while future.result() is None:
            rossem.release()
            time.sleep(0.2)
            rossem.acquire()
        #self.spin_until_future_complete_(future)
        result = future.result().intent

        return result

    @withsem
    def change_network(self, network, type):
        change_network_cli = self.create_client(ChangeNetwork, f'/teleop_gesture_toolbox/change_{type}_network')

        while not change_network_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            response = change_network(data=network)
            Gs = [g.lower() for g in response.gs]
            settings.args = response.args
            print("[UI] Gestures & Network changed, new set of gestures: "+str(", ".join(Gs)))
        except rclpy.ServiceException as e:
            print("Service call failed: %s"%e)
        settings.paths.gesture_network_file = network

        gl.gd.gesture_change_srv(local_data=response)

    @withsem
    def create_rate_(self, rate):
        return self.create_rate(rate)

    @withsem
    def get_time(self):
        return self.get_clock().now().nanoseconds/1e9

    @withsem
    def gesture_solution_publish(self, msg):
        return self.gesture_solution_pub.publish(msg)

    def init_coppelia_interface(self):
        assert coppelia_sim_ros_interface is not None, "coppelia_sim_ros_interface failed to load!"

        self.r = CoppeliaROSInterfaceWithSem(sem=rossem, rosnode=self)

    def init_real_interface__pymoveit2__panda(self):
        assert PyMoveIt2Interface is not None, "pymoveit2_interface failed to load"

        self.r = PyMoveIt2Interface(rosnode=self)

    def init_real_interface__zmqarmer__panda(self):
        assert ZMQArmerInterface is not None, "ZMQArmerInterface failed to load!"

        self.r = ZMQArmerInterface()
        #self.r = ZMQArmerInterfaceWithSem(rossem)

    def spin_once(self, sem=True):
        if sem:
            if DEBUGSEMAPHORE: print("ACQ - spinner")
            with rossem:
                if DEBUGSEMAPHORE: print("*")
                self.last_time_livin = time.time()
                rclpy.spin_once(roscm)
            if DEBUGSEMAPHORE: print("---")
        else:
            self.last_time_livin = time.time()
            rclpy.spin_once(roscm)

    def publish_eef_goal_pose(self, goal_pose):
        ''' Publish goal_pose /ee_pose_goals to relaxedIK with its transform
            Publish goal_pose /ee_pose_goals the goal eef pose
        '''
        self.ee_pose_goals_pub.publish(goal_pose)
        #self.ik_bridge.relaxedik.ik_node_publish(pose_r = self.ik_bridge.relaxedik.relaxik_t(goal_pose))

    def object_info_callback(self, data):
        ''' Only handles pose
        '''
        if sl.scene:
            object_names = sl.scene.O
            if data.name in object_names:
                object_id = object_names.index(data.name)
                #sl.scene.object_poses[object_id] = data.pose
                o = sl.scene.get_object_by_name(data.name)
                o.position_real = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
                o.quaternion = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])

            else:
                print("Warning. Unwanted objects on the scene")

    @staticmethod
    def ik(data):
        joints = []
        for ang in data.angles:
            joints.append(ang.data)
        ml.md.goal_joints = joints

    @staticmethod
    def coppelia_eef(data):
        ml.md.eef_pose = data

    @staticmethod
    def camera_angle(data):
        ml.md.camera_orientation = data

    @staticmethod
    def hand_frame_callback(data):
        ''' Hand data received by ROS msg is saved
        '''
        #print("new data")
        f = Frame()
        f.import_from_ros(data)
        gl.gd.hand_frames.append(f)

    def joint_states_callback(self, data):
        ''' Saves joint_states to * append array 'settings.joint_in_time' circle buffer.
                                  * latest data 'ml.md.joints', 'ml.md.velocity', 'ml.md.effort'
            Topic used:
                - CoppeliaSim (PyRep) -> topic "/joint_states_coppelia"
                - Other (Gazebo sim / Real) -> topic "/joint_states"

        '''
        ml.md.joint_states.append(data)


    def custom_fkine(self):
        fk_se3 = self.rtb_model.fkine(self.joints)
        p = fk_se3.t
        q = UnitQuaternion(fk_se3).vec_xyzs
        return Pose(position = Point(x=p[0], y=p[1], z=p[2]), orientation = Quaternion(x=q[0],y=q[1],z=q[2],w=q[3]))

    def eef_pose_correct(self):
        pose = self.custom_fkine()
        p2 = np.array([self.eef_pose.position.x, self.eef_pose.position.y, self.eef_pose.position.z])
        print(p.round(2), p2.round(2), np.allclose(p, p2, atol=1e-2))


    @staticmethod
    def save_static_detection_solutions_callback(data):
        gl.gd.new_record(data, type='static')

    @staticmethod
    def save_dynamic_detection_solutions_callback(data):
        gl.gd.new_record(data, type='dynamic')

    @withsem
    def send_g_data(self, dynamic_detection_window=1.5):
        ''' Sends appropriate gesture data as ROS msg
            Launched node for static/dynamic detection.
        '''
        if len(gl.gd.hand_frames) == 0: return
        hand_mode = self.hand_mode

        msg = DetectionObservations()
        msg.observations = Float64MultiArray()
        msg.sensor_seq = gl.gd.hand_frames[-1].seq
        msg.header.stamp.sec = gl.gd.hand_frames[-1].sec
        msg.header.stamp.nanosec = gl.gd.hand_frames[-1].nanosec

        mad1 = MultiArrayDimension()
        mad1.label = 'time'
        mad2 = MultiArrayDimension()
        mad2.label = 'xyz'

        for key in hand_mode.keys():
            args = gl.gd.static_network_info

            send_static_g_data_bool = hand_mode is not None and 'static' in hand_mode[key] and gl.gd.static_network_info is not None
            if send_static_g_data_bool:
                if key == 'l':
                    if gl.gd.l_present():
                        msg.observations.data = gl.gd.hand_frames[-1].l.get_learning_data_static(definition=args['input_definition_version'])
                        msg.header.frame_id = 'l'
                        self.static_detection_observations_pub.publish(msg)
                elif key == 'r':
                    if gl.gd.r_present():
                        msg.observations.data = gl.gd.hand_frames[-1].r.get_learning_data_static(definition=args['input_definition_version'])
                        msg.header.frame_id = 'r'
                        self.static_detection_observations_pub.publish(msg)


            time_samples = settings.yaml_config_gestures['misc_network_args']['time_samples']
            send_dynamic_g_data_bool = hand_mode is not None and 'dynamic' in hand_mode[key] and len(gl.gd.hand_frames) > time_samples and gl.gd.dynamic_network_info is not None
            if send_dynamic_g_data_bool:
                args = gl.gd.dynamic_network_info
                if getattr(gl.gd, key+'_present')():# and getattr(gl.gd.hand_frames[-1], key).grab_strength < 0.5:
                    try:
                        '''  '''
                        n = 1
                        visibles = []
                        while True:
                            ttt = gl.gd.hand_frames[-1].stamp() - gl.gd.hand_frames[-n].stamp()
                            visibles.append( gl.gd.hand_frames[-n].visible )
                            if ttt > 1.5: break
                            n += 1
                        if not np.array(visibles).all():
                            return

                        ''' Creates timestamp indexes starting with [-1, -x, ...] '''
                        time_samples_series = [-1]
                        time_samples_series.extend((n * np.array(range(-1, -time_samples, -1))  / time_samples).astype(int))
                        time_samples_series.sort()

                        ''' Compose data '''
                        data_composition = []
                        for time_sample in time_samples_series:
                            data_composition.append(getattr(gl.gd.hand_frames[time_sample], key).get_single_learning_data_dynamic(definition=args['input_definition_version']))

                        ''' Transform to Leap frame id '''
                        data_composition_ = []
                        for point in data_composition:
                            data_composition_.append(tfm.transformLeapToBase(point, out='position'))
                        data_composition = data_composition_

                        ''' Check if the length of composed data is aorund 1sec '''
                        ttt = gl.gd.hand_frames[-1].stamp() - gl.gd.hand_frames[int(time_samples_series[0])].stamp()
                        if not (0.7 <= ttt <= 2.0):
                            print(f"WARNING: data frame composed is {ttt} long")
                        ''' Subtract middle path point from all path points '''

                        #if 'normalize' in args:
                        data_composition_ = []
                        data_composition0 = deepcopy(data_composition[len(data_composition)//2])
                        for n in range(0, len(data_composition)):
                            data_composition_.append(np.subtract(data_composition[n], data_composition0))
                        data_composition = data_composition_

                        ''' Fill the ROS msg '''
                        data_composition = np.array(data_composition, dtype=float)

                        mad1.size = data_composition.shape[0]
                        mad2.size = data_composition.shape[1]
                        data_composition = list(data_composition.flatten())
                        msg.observations.data = data_composition
                        msg.observations.layout.dim = [mad1, mad2]
                        msg.header.frame_id = key
                        self.dynamic_detection_observations_pub.publish(msg)
                    except IndexError:
                        pass
    
    
    def send_state(self):
        
        main_livin = (time.time()-ml.md.last_time_livin) < 1.0
        spinner_livin = (time.time()-self.last_time_livin) < 1.0
        
        dict_to_send = {
            "real": str(self.is_real).lower(),
            "main_livin": str(main_livin).lower(),
            "spinner_livin": str(spinner_livin).lower(),
            "fps": -1,
            "seq": -1,
        }
        
        if sl.scene is not None:
            dict_to_send['scene_objects'] = sl.scene.O
            # dict_to_send['scene_object_positions'] = sl.scene.object_positions_real

        # if ml.md.goal_pose and ml.md.goal_joints:
        #     structures_str = [structure.object_stack for structure in ml.md.structures]
        #     textStatus += f"eef: {str(round(ml.md.eef_pose.position.x,2))} {str(round(ml.md.eef_pose.position.y,2))} {str(round(ml.md.eef_pose.position.z,2))}\ng p: {str(round(ml.md.goal_pose.position.x,2))} {str(round(ml.md.goal_pose.position.y,2))} {str(round(ml.md.goal_pose.position.z,2))}\ng q:{str(round(ml.md.goal_pose.orientation.x,2))} {str(round(ml.md.goal_pose.orientation.y,2))} {str(round(ml.md.goal_pose.orientation.z,2))} {str(round(ml.md.goal_pose.orientation.w,2))}\nAttached: {ml.md.attached}\nbuild_mode {ml.md.build_mode}\nobject_touch and focus_id {ml.md.object_focus_id} {ml.md.object_focus_id}\nStructures: {str(structures_str)}\n"
        if gl.gd.present():
            dict_to_send["fps"] = round(gl.gd.hand_frames[-1].fps)
            dict_to_send["seq"] = gl.gd.hand_frames[-1].seq
            
            dict_to_send["gesture_type_selected"] = gl.sd.previous_gesture_observed_data_action
            dict_to_send["gs_state_action"] = GestureSentence.process_gesture_queue(gl.gd.gestures_queue)
            dict_to_send["gs_state_objects"] = gl.gd.target_objects
            dict_to_send["gs_state_ap"] = gl.gd.ap
        
        if gl.gd.l.static.relevant():
            static_n = len(gl.gd.static_info().filenames)
            dict_to_send['l_static_names'] = gl.gd.Gs_static
            dict_to_send['l_static_probs'] = [gl.gd.l.static[-1][n].probability for n in range(static_n)]
            dict_to_send['l_static_activated'] = [str(gl.gd.l.static[-1][n].activated).lower() for n in range(static_n)]

        if gl.gd.r.static.relevant():
            static_n = len(gl.gd.static_info().filenames)
            dict_to_send['r_static_names'] = gl.gd.Gs_static
            dict_to_send['r_static_probs'] = [gl.gd.r.static[-1][n].probability for n in range(static_n)]
            dict_to_send['r_static_activated'] = [str(gl.gd.r.static[-1][n].activated).lower() for n in range(static_n)]
        
        if gl.gd.l.dynamic and gl.gd.l.dynamic.relevant():
            dynamic_n = len(gl.gd.dynamic_info().filenames)
            dict_to_send['l_dynamic_names'] = gl.gd.Gs_dynamic
            dict_to_send['l_dynamic_probs'] = list(gl.gd.l.dynamic[-1].probabilities_norm)
            dict_to_send['l_dynamic_activated'] = [str(gl.gd.l.dynamic[-1][n].activated).lower() for n in range(dynamic_n)]
        
        if gl.gd.r.dynamic and gl.gd.r.dynamic.relevant():
            dynamic_n = len(gl.gd.dynamic_info().filenames)
            dict_to_send['r_dynamic_names'] = gl.gd.Gs_dynamic
            dict_to_send['r_dynamic_probs'] = list(gl.gd.r.dynamic[-1].probabilities_norm)
            dict_to_send['r_dynamic_activated'] = [str(gl.gd.r.dynamic[-1][n].activated).lower() for n in range(dynamic_n)]
            # self.get_logger().info(f"r dynamic enabled {dict_to_send['r_dynamic_probs']}.. {dict_to_send['r_dynamic_activated']}, {dict_to_send['r_dynamic_names']}")

        if gl.gd.l.static and gl.gd.l.static.relevant() is not None:  
            try:
                dict_to_send['l_static_relevant_biggest_id'] = gl.gd.l.static.relevant().biggest_probability_id
            except AttributeError:
                dict_to_send['l_static_relevant_biggest_id'] = -1

        if gl.gd.r.static and gl.gd.r.static.relevant() is not None:  
            try:
                dict_to_send['r_static_relevant_biggest_id'] = gl.gd.r.static.relevant().biggest_probability_id
            except AttributeError:
                dict_to_send['r_static_relevant_biggest_id'] = -1

        if gl.gd.l.dynamic and gl.gd.l.dynamic.relevant() is not None:
            try:
                dict_to_send['l_dynamic_relevant_biggest_id'] = gl.gd.l.dynamic.relevant().biggest_probability_id
            except AttributeError:
                dict_to_send['l_dynamic_relevant_biggest_id'] = -1

        if gl.gd.r.dynamic and gl.gd.r.dynamic.relevant() is not None:
            try:
                dict_to_send['r_dynamic_relevant_biggest_id'] = gl.gd.r.dynamic.relevant().biggest_probability_id
            except AttributeError:
                dict_to_send['r_dynamic_relevant_biggest_id'] = -1
        
        
        compound_gestures = gl.gd.c[-1]
        dict_to_send['compound_activated'] = ['false'] * len(gl.gd.c.info.names)
        dict_to_send['compound_names'] = list(gl.gd.c.info.names)
        
        if compound_gestures is not None:
            dict_to_send['compound_activated'] = [str(a).lower() for a in compound_gestures.activates]
        print("dict_to_send['compound_activated']", dict_to_send['compound_activated'], "dict_to_send['compound_names']", dict_to_send['compound_names'])
        
        
        data_as_str = str(dict_to_send)
        data_as_str = data_as_str.replace("'", '"')
        
        self.all_states_pub.publish(String(data=data_as_str))

    def detection_thread(self, freq=1., args={}):
        if not ROS: raise Exception("ROS cannot be imported!")

        rclpy.init()

        settings.gesture_detection_on = True
        settings.launch_gesture_detection = True

        roscm = ROSComm()

        configGestures = ParseYAML.load_gesture_config_file(settings.paths.custom_settings_yaml)
        hand_mode_set = configGestures['using_hand_mode_set']
        hand_mode = dict(configGestures['hand_mode_sets'][hand_mode_set])

        rate = roscm.create_rate(freq)

        while rclpy.ok():
            if gl.gd.hand_frames:
                send_g_data(roscm, hand_mode, args)

            rate.sleep()

    def spin_until_future_complete_(self, future):
        if rossem is not None:
            while rclpy.spin_until_future_complete(self, future, timeout_sec=0.01) is not None:
                rossem.release()
                time.sleep(0.01)
                rossem.acquire()
        else:
            raise Exception("[ERROR] NotImplementedError!")


class MirracleSetupInterface(ROSComm):
    ''' November 2023 '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert CrowtologyClient is not None, "CrowtologyClient not Found!"
        
        self.crowracle = CrowtologyClient(node=self)
        self.onto = self.crowracle.onto
        self.LANG = 'en'
        # self.get_logger().info(self.onto)

        #create listeners (synchronized)
        self.nlTopic = "/nlp/command"
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        
        self.add_dummy_cube()
        self.add_dummy_cube()
        print(self.get_objects_from_onto())
        print("Ready")
        # if not NOT_PROFILING:
        #     StatTimer.init()

    @staticmethod
    def mocked_update_scene():
        s = None
        s = cbgo.srcmodules.Scenes.Scene(init='object', random=False)
        sl.scene = s
        return s


    NAME2TYPE = {
        'cube': 'Object',
        
    }

    def update_scene(self):
        
        s = None
        s = cbgo.srcmodules.Scenes.Scene(init='', objects=[], random=False)
        s.objects = []
        
        objects = self.get_objects_from_onto()
        ''' object list; item is dictionary containing uri, id, color, color_nlp_name_CZ, EN, nlp_name_CZ, nlp_name_EN; absolute_location'''

        '''
        import rdflib
        uri = rdflib.term.URIRef('http://imitrob.ciirc.cvut.cz/ontologies/crow#test_CUBE_498551_od_498551')
        '''
        # [{'uri': rdflib.term.URIRef('http://imitrob.ciirc.cvut.cz/ontologies/crow#test_CUBE_498551_od_498551'), 'id': 'od_498551', 'color': rdflib.term.URIRef('http://imitrob.ciirc.cvut.cz/ontologies/crow#COLOR_GREEN'), 'color_nlp_name_CZ': 'zelená', 'color_nlp_name_EN': 'green', 'nlp_name_CZ': 'kostka', 'nlp_name_EN': 'cube', 'absolute_location': [-0.34157065, 0.15214929, -0.24279054]}]
        # [{'uri': rdflib.term.URIRef('http://imitrob.ciirc.cvut.cz/ontologies/crow#test_CUBE_498551_od_498551'), 'id': 'od_498551', 'color': rdflib.term.URIRef('http://imitrob.ciirc.cvut.cz/ontologies/crow#COLOR_GREEN'), 'color_nlp_name_CZ': 'zelená', 'color_nlp_name_EN': 'green', 'nlp_name_CZ': 'kostka', 'nlp_name_EN': 'cube', 'absolute_location': [-0.34157065, 0.15214929, -0.24279054]}]

        # Colors:
        # [COLOR_GREEN. 

        for object in objects:
            ''' o is dictionary containing properties '''
            uri = object['uri']
            id = object['uri'].split("#")[-1]
            color = object['color']
            color_nlp_name_CZ = object['color_nlp_name_CZ']
            color_nlp_name_EN = object['color_nlp_name_EN']
            nlp_name_CZ = object['nlp_name_CZ']
            nlp_name_EN = object['nlp_name_EN']
            absolute_location = object['absolute_location']
            print(f"MY ID: {id}")
            
            o = cbgo.srcmodules.Objects.Object(name=id, position_real=np.array(absolute_location), random=False)
            # o.quaternion = np.array(object['pose'][1])
            # o.color_uri = color
            o.color = color_nlp_name_EN
            # o.color_name_CZ = color_nlp_name_CZ
            
            # o.nlp_name_CZ = nlp_name_CZ
            # o.nlp_name_EN = nlp_name_EN
            # o.crow_id = id
            # o.crow_uri = uri
            
            s.objects.append(o)

        sl.scene = s        
        # self.get_logger().info(f"scene: {sl.scene}")
        return s

    def match_object_in_onto(self, obj):
        onto = self.get_objects_from_onto()
        obj = json.loads(obj)
        for o in onto:
           url = str(o["uri"])
           if url is not None and url == obj[0]["target"][0]:
              return o
        return None
        
        
    def add_dummy_cube(self):
        o_list = self.get_objects_from_onto()
        if len(o_list) == 0:
           print("Adding a dummy cube into ontology")
           self.crowracle.add_test_object("CUBE")
        
    def get_objects_from_onto(self):
        o_list = self.crowracle.getTangibleObjectsProps()
        #print("Onto objects:")
        #print(o_list)
        return o_list
        
    def get_action_from_cmd(self, cmd):
        action = json.loads(cmd)[0]["action_type"].lower()
        if action in ACTION_TRANSL.keys():
           return ACTION_TRANSL[action]
        else:
           print("No czech translation for action " + action)
           return action
        
        
    def publish_trajectory(self, trajectory):
        #TODO choose proper msg type
        action = json.dumps(trajectory)
        msg = StampedString()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.data = action
        print(f'Publishing {msg.data}')
        self.trajectory_publisher.publish(msg)
        

    def keep_alive(self):
        self.pclient.nlp_alive = time.time()







def init(robot_interface='', setup='standalone'):
    global roscm, rossem
    rclpy.init(args=None)
    rossem = threading.Semaphore()

    if setup == 'standalone':
        roscm = ROSComm(robot_interface=robot_interface)
    elif setup == 'mirracle':
        roscm = MirracleSetupInterface(robot_interface=robot_interface)
    else: raise Exception()