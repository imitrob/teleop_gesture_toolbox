

import rclpy
from gesture_msgs.msg import DetectionSolution, DetectionObservations
from gesture_msgs.srv import SaveHandRecord, GetModelConfig

class GestureListService():
    ''' Calls Gesture Detector service to get gesture names.
    '''
    def __init__(self):
        super(GestureListService, self).__init__()

        self.get_static_model_config = self.create_client(GetModelConfig, '/teleop_gesture_toolbox/static_detection_info')
        while not self.get_static_model_config.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...', flush=True)
        self.get_dynamic_model_config = self.create_client(GetModelConfig, '/teleop_gesture_toolbox/dynamic_detection_info')
        while not self.get_dynamic_model_config.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...', flush=True)

        self._Gs_static = None
        self._Gs_dynamic = None


    def call_static_model_config_service(self):
        self.future = self.get_static_model_config.call_async(GetModelConfig.Request())
        rclpy.spin_until_future_complete(self, self.future)
        return list(self.future.result().gestures)
    
    def call_dynamic_model_config_service(self):
        self.future = self.get_dynamic_model_config.call_async(GetModelConfig.Request())
        rclpy.spin_until_future_complete(self, self.future)
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