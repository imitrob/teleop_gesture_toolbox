

import collections
import gesture_msgs.msg as rosm
from gesture_detector.hand_processing.frame_lib import Frame

class HandListener():
    def __init__(self):
        super(HandListener, self).__init__()
        self.hand_frames = collections.deque(maxlen=5)
        self.create_subscription(rosm.Frame, '/teleop_gesture_toolbox/hand_frame', self.hand_frame_callback, 10)
        print("Note: `You need ros2 run gesture_detector leap` and `sudo leapd` running", flush=True)

    def hand_frame_callback(self, data):
        ''' Hand data received by ROS msg is saved '''
        f = Frame()
        f.import_from_ros(data)
        self.hand_frames.append(f)
