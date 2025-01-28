
from hri_msgs.msg import HRICommand as HRICommandMSG # Download https://github.com/ichores-research/modality_merging to workspace
import rclpy
from gesture_msgs.msg import DetectionSolution, DetectionObservations
from gesture_msgs.srv import SaveHandRecord, GetModelConfig
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import time
import collections
import numpy as np
from naive_merger.HriCommand import HriCommand


class GestureSentenceGetter():

    def __init__(self, rosnode=None):
        super(GestureSentenceGetter, self).__init__()
        if rosnode is not None:
            self._rosnode = rosnode
        else:
            self._rosnode = self
        self.rosnode.create_subscription(HRICommandMSG, "/teleop_gesture_toolbox/hricommand_original", self.gestures_receive_callback, qos_profile=QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT), callback_group=self.rosnode.callback_group) # receive gestures
        

        self.hricommand_queue = collections.deque(maxlen=5)

        self.get_static_model_config = self.rosnode.create_client(GetModelConfig, '/teleop_gesture_toolbox/static_detection_info')
        # while not self.get_static_model_config.wait_for_service(timeout_sec=1.0):
        #     print('service not available, waiting again...')
        self.get_dynamic_model_config = self.rosnode.create_client(GetModelConfig, '/teleop_gesture_toolbox/dynamic_detection_info')
        # while not self.get_dynamic_model_config.wait_for_service(timeout_sec=1.0):
        #     print('service not available, waiting again...')

        self._Gs_static = None
        self._Gs_dynamic = None


    @property
    def rosnode(self):
        return self._rosnode

    def wait_gestures(self):
        while rclpy.ok():
            if len(self.hricommand_queue) > 0:
                break
            time.sleep(1.0)

        static_gesture_p = np.array(self.hricommand_queue[-1].pv_dict["gesture"].p[:len(self.Gs_static)])
        dynamic_gesture_p = np.array(self.hricommand_queue[-1].pv_dict["gesture"].p[len(self.Gs_static):])

        return self.Gs_static[np.argmax(static_gesture_p)], self.Gs_dynamic[np.argmax(dynamic_gesture_p)]


    def gestures_receive_callback(self, msg):
        self.hricommand_queue.append(HriCommand.from_ros(msg))


    def call_static_model_config_service(self):
        self.future = self.get_static_model_config.call_async(GetModelConfig.Request())
        # rclpy.spin_until_future_complete(self, self.future)
        while True:
            time.sleep(0.2)
            try:
                self.future.result().gestures
                break
            except:
                pass
        return list(self.future.result().gestures)
    
    def call_dynamic_model_config_service(self):
        self.future = self.get_dynamic_model_config.call_async(GetModelConfig.Request())
        # rclpy.spin_until_future_complete(self, self.future)
        while True:
            time.sleep(0.2)
            try:
                self.future.result().gestures
                break
            except:
                pass
        return list(self.future.result().gestures)
    
    @property
    def Gs_static(self):
        ''' Get list of static gestures once '''
        if self._Gs_static is None:
            self._Gs_static = self.call_static_model_config_service()
        return self._Gs_static

    @property
    def Gs_dynamic(self):
        ''' Get list of dynamic gestures once '''
        if self._Gs_dynamic is None:
            self._Gs_dynamic = self.call_dynamic_model_config_service()
        return self._Gs_dynamic

    @property
    def Gs(self):
        Gs = self.Gs_static.copy()
        Gs.extend(self.Gs_dynamic)
        return Gs