

import collections
import gesture_msgs.msg as rosm
from gesture_detector.hand_processing.frame_lib import Frame

class HandListener():
    def __init__(self):
        super(HandListener, self).__init__()
        self.hand_frames = collections.deque(maxlen=5)
        self.create_subscription(rosm.Frame, '/hand_frame', self.hand_frame_callback, 10)
        
    def hand_frame_callback(self, data):
        ''' Hand data received by ROS msg is saved '''
        f = Frame()
        f.import_from_ros(data)
        self.hand_frames.append(f)
