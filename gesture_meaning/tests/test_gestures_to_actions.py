

from gesture_meaning.gesture_meaning_service import GestureToMeaningNode, OneToOne_Sample
from gesture_msgs.srv import GestureToMeaning
import rclpy

def test_gestures_to_actions():
    rclpy.init()
    rosnode = GestureToMeaningNode()

    req = GestureToMeaning.Request()
    req.gestures.gesture_names = OneToOne_Sample.G
    gesture_probs = [0.0] * len(OneToOne_Sample.G)
    gesture_probs[0] = 1.0
    req.gestures.gesture_probs = gesture_probs

    response = rosnode.G2I_service_callback(req, GestureToMeaning.Response())

    assert response.intent.target_action == 'move_up'

if __name__ == "__main__":
    test_gestures_to_actions()