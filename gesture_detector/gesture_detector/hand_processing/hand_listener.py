

import collections
import gesture_msgs.msg as rosm
from gesture_detector.hand_processing.frame_lib import Frame

from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy

class HandListener():
    def __init__(self):
        super(HandListener, self).__init__()
        self.hand_frames = collections.deque(maxlen=10)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(rosm.Frame, '/teleop_gesture_toolbox/hand_frame', self.hand_frame_callback, qos)
        print("Note: `You need ros2 run gesture_detector leap` and `sudo leapd` running", flush=True)

    def hand_frame_callback(self, data):
        ''' Hand data received by ROS msg is saved '''
        f = Frame()
        f.import_from_ros(data)
        self.hand_frames.append(f)
