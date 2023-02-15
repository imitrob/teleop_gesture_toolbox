import sys
import numpy as np
from os_and_utils.utils import get_cbgo_path
sys.path.append(get_cbgo_path())

from context_based_gesture_operation.agent_nodes.g2i import G2IRosNode

# Btree imports
import rclpy
from rclpy.node import Node
from context_based_gesture_operation.srv import G2I
from context_based_gesture_operation.msg import Intent
from context_based_gesture_operation.msg import Scene as SceneRos
from context_based_gesture_operation.msg import Gestures as GesturesRos
from srcmodules.Scenes import Scene
from srcmodules.Gestures import Gestures
from srcmodules.Actions import Actions
from srcmodules.Objects import Object
from srcmodules.SceneFieldFeatures import SceneFieldFeatures


class GGPipeline():
    ''' Performing fake detections by pressing GUI button
    '''
    def __init__(self, g2i_model='M3v8_D4_1.pkl', scene_def_id=8):
        self.gestures_queue_fake = []
        self.gestures_queue_real = []

        self.focused_objects_fake = []
        self.focused_objects_real = []

        self.auxiliary_parameters_fake = []
        self.auxiliary_parameters_real = []

        self.s = None

        self.g2i = G2IRosNode(load_model=g2i_model)#M3v10_D5.pkl')
        self.scene_def_id = scene_def_id

    def __call__(self, real=True):
        '''
        Parameters:
            real (Bool): real/fake detections - Fake takes GUI gesture source

        1. - 4. Data gather - G, O, A
        '''
        assert (self.gestures_queue_real != [] or self.gestures_queue_fake != []), "gestures_queue not filled!"
        #assert (self.focused_objects_real != [] or self.focused_objects_fake != []), "focused_objects not filled!"
        #assert (self.auxiliary_parameters_real is not [] or self.auxiliary_parameters_fake is not []), "auxiliary_parameters not filled!"
        assert (self.s is not None), "scene not filled!"

        ''' 1. Static & Dynamic gestures are added to gl.gd.gestures_queue
            - a. Using GUI (Fake)
            - b. Using Leap (Real)
        '''
        if real:
            gestures_queue = self.gestures_queue_real
        else:
            gestures_queue = self.gestures_queue_fake
        self.gestures_queue_real = []
        self.gestures_queue_fake = []

        ''' 2. Object focused are added
            - a. Using GUI (Fake) - ComboBox
            - b. Using Deictic gesture (Real)
        '''
        if real:
            focused_objects = self.focused_objects_real
        else:
            focused_objects = self.focused_objects_fake

        ''' 3. Auxiliary parameters
            - a. Using GUI (Fake) -
            - b. Using Metric gestures (Real)
        '''
        if real:
            auxiliary_parameters = self.auxiliary_parameters_real
        else:
            auxiliary_parameters = self.auxiliary_parameters_fake

        ''' 4. Scene and context data
            - s scene object
        '''
        s = self.s

        ''' 5. Gestures to Human Intent (G2I)
        '''
        # Filter gestures into only one, more general gesture
        chosen_gesture = gestures_queue[-1]
        # What focus point to apply to the G2I detection ?
        if focused_objects != []: # If any object - use last focused object
            target_object = s.get_object_by_name(focused_object[-1]).position_real
        else: # Use eef position
            target_object = s.r.eef_position_real
        # Apply g2i approach
        target_intent_action, target_object = self.g2i.predict_with_scene_gesture_and_target_object(s, gesture, target_object, scene_def_id=self.scene_def_id)

        ''' 6. Apply behavior tree to get to the goal
            - Each target_intent_action should have its own b-tree ?
        '''
        b_tree = BTree(...)

        while True:
            next_target_action = b_tree.pick_next_action(...)

            getattr(ml.RealRobotActionLib,action)(object_names)
            if next_target_action is None: break

        return "Done"
