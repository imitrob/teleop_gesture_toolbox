

from gesture_meaning.get_gesture_meaning import GestureToMeaningNode, OneToOne_Sample
from teleop_msgs.srv import G2I


def test_gestures_to_actions():
    rosnode = GestureToMeaningNode()

    req = G2I.request()
    req.gesture_names = OneToOne_Sample.G
    gesture_probs = [0.0] * len(OneToOne_Sample.G)
    gesture_probs[0] = 1.0
    req.gesture_probs = gesture_probs

    response = rosnode.G2I_service_callback(req, G2I.response())

    assert response.intent.target_action == 'move_up'

if __name__ == "__main__":
    test_gestures_to_actions()