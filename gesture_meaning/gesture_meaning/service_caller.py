


from typing import Iterable
from gesture_msgs.srv import GestureToMeaning
import rclpy
from rclpy.node import Node



class GetGestureMeaning():
    def __init__(self):
        super(GetGestureMeaning, self).__init__()
        self.cli = self.create_client(GestureToMeaning, '/teleop_gesture_toolbox/get_meaning')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
        self.req = GestureToMeaning.Request()

    def __call__(self, gesture_names: Iterable[str], gesture_probs: Iterable[float]):
        self.req.gestures.gesture_names = gesture_names
        self.req.gestures.gesture_probs = gesture_probs

        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

class RosNode(Node):
    def __init__(self):
        super().__init__('gesture_meaning_caller')

class GetGestureMeaningNode(GetGestureMeaning, RosNode):
    def __init__(self):
        super(GetGestureMeaningNode, self).__init__()


# def map(s, max_gesture_probs, target_objects):
#     if len(target_objects) == 0: # no object added, no pointing was made
#         ''' Object not given -> use eef position '''
#         focus_point = s.r.eef_position_real
#     else:
#         object_name_1 = target_objects[0]
#         if s.get_object_by_name(object_name_1) is None:
#             print(f"Object target is not in the scene!, object name: {object_name_1} objects: {s.O}")
#             GestureSentence.clearing()
#             return None
#         focus_point = s.get_object_by_name(object_name_1).position_real

#     if focus_point is None: return

#     # target object to focus point {target_objects} -> {focus_point}

#     sr = s.to_ros(SceneRos())
#     sr.focus_point = np.array(focus_point, dtype=float)
#     print(f"[INFO] Aggregated gestures: {list(gestures_queue_proc)}")

#     time.sleep(0.01)

#     g2i_tester = G2IRosNode(init_node=False, inference_type='1to1', load_model='M3v10_D6.pkl', ignore_unfeasible=True)

#     gestures = gl.gd.gestures_queue_to_ros(gestures_queue_proc, GesturesRos())
#     response = g2i_tester.G2I_service_callback( \
#         G2I.Request(gestures=gestures, scene=sr),
#         G2I.Response()
#         )

#     response_intent = response.intent

#     self.gesture_sentence_publisher_mapped.publish(GestureSentence.export_mapped_to_HRICommand(s, ret))

if __name__ == "__main__":
    rclpy.init()
    client = GetGestureMeaningNode()

    gesture_names = ['swipe_up', 'swipe_left', 'swipe_down', 'swipe_right', 'pinch','grab', 'point', 'two', 'three', 'four', 'five', 'thumbsup']
    gesture_probs = [0.0] * 12
    gesture_probs[0] = 1.0

    response = client(gesture_names, gesture_probs)

    print(response)
    print(response.intent)






