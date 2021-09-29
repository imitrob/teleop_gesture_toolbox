import tf
from geometry_msgs.msg import PoseStamped, Quaternion, Point

import sys
from os.path import expanduser, isfile
import numpy as np
HOME = expanduser("~")
import time
from copy import deepcopy
from math import asin, atan2
from itertools import permutations, combinations

sys.path.append("/usr/lib/Leap")
sys.path.append(HOME+"/LeapSDK/lib/x64")
sys.path.append(HOME+"/LeapSDK/lib")

import Leap, sys, thread, time
from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture
import settings


class SampleListener(Leap.Listener):
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    state_names = ['STATE_INVALID', 'STATE_START', 'STATE_UPDATE', 'STATE_END']

    def on_init(self, controller):
        print("Initialized")

    def on_connect(self, controller):
        print("Connected")
        # Enable gestures
        controller.enable_gesture(Leap.Gesture.TYPE_CIRCLE)
        controller.enable_gesture(Leap.Gesture.TYPE_KEY_TAP)
        controller.enable_gesture(Leap.Gesture.TYPE_SCREEN_TAP)
        controller.enable_gesture(Leap.Gesture.TYPE_SWIPE)
        while not settings.mo:
            time.sleep(2)
            print("[Leap listener] No Move object published")

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print("Disconnected")

    def on_exit(self, controller):
        print("Exited")

    def on_frame(self, controller):
        frame = controller.frame()
        settings.frames.append(frame)
        ## Update forward kinematics in settings
        settings.timestamps.append(frame.timestamp)
        fps = frame.current_frames_per_second
        timestamp = frame.timestamp

        # Access: frame.id, frame.timestamp, len(frame.hands), len(frame.fingers), len(frame.tools), len(frame.gestures()
        fa = settings.FrameAdv()
        for hand in frame.hands:
            if hand.is_left:
                fah = fa.l
            elif hand.is_right:
                fah = fa.r
            fah.visible = True
            fah.grab = hand.grab_strength
            fah.pinch = hand.pinch_strength
            fah.conf = hand.confidence
            ## Pose of palm
            q = tf.transformations.quaternion_from_euler(hand.palm_normal.roll, hand.direction.pitch, hand.direction.yaw)
            pose = PoseStamped()
            pose.pose.orientation = Quaternion(*q)
            pose.pose.position = Point(hand.palm_position[0],hand.palm_position[1],hand.palm_position[2])
            pose.header.stamp.secs = frame.timestamp//1000000
            pose.header.stamp.nsecs = 1000*(frame.timestamp%1000000)

            fah.pPose = deepcopy(pose)
            fah.pRaw = deepcopy([hand.palm_position[0],hand.palm_position[1],hand.palm_position[2], hand.palm_normal.roll, hand.direction.pitch, hand.direction.yaw])
            fah.pNormDir = deepcopy([hand.palm_normal[0], hand.palm_normal[1], hand.palm_normal[2], hand.direction[0], hand.direction[1], hand.direction[2]])

            ## Stand of each finger
            position_of_fingers = []
            for i in range(0,5):
                bone_1 = hand.fingers[i].bone(0)
                if i == 0:
                    bone_1 = hand.fingers[i].bone(1)
                bone_4 = hand.fingers[i].bone(3)
                q1 = tf.transformations.quaternion_from_euler(0.0, asin(-bone_1.direction[1]), atan2(bone_1.direction[0], bone_1.direction[2])) # roll, pitch, yaw
                q2 = tf.transformations.quaternion_from_euler(0.0, asin(-bone_4.direction[1]), atan2(bone_4.direction[0], bone_4.direction[2])) # roll, pitch, yaw
                oc = np.dot(q1, q2)
                fah.OC[i] = oc

                position_of_fingers.append([bone_4.next_joint[0], bone_4.next_joint[1], bone_4.next_joint[2]])

            fah.TCH12 = np.sum(np.power(np.subtract(position_of_fingers[1], position_of_fingers[0]),2))/1000
            fah.TCH23 = np.sum(np.power(np.subtract(position_of_fingers[2], position_of_fingers[1]),2))/1000
            fah.TCH34 = np.sum(np.power(np.subtract(position_of_fingers[3], position_of_fingers[2]),2))/1000
            fah.TCH45 = np.sum(np.power(np.subtract(position_of_fingers[4], position_of_fingers[3]),2))/1000
            fah.TCH13 = np.sum(np.power(np.subtract(position_of_fingers[2], position_of_fingers[0]),2))/1000
            fah.TCH14 = np.sum(np.power(np.subtract(position_of_fingers[3], position_of_fingers[0]),2))/1000
            fah.TCH15 = np.sum(np.power(np.subtract(position_of_fingers[4], position_of_fingers[0]),2))/1000

            fah.vel = [hand.palm_velocity[0]/1000, hand.palm_velocity[1]/1000, hand.palm_velocity[2]/1000]
            # TODO: change 0.1 to var from settings
            if hand.palm_velocity[0]/1000 < 0.1 and hand.palm_velocity[1]/1000 < 0.1 and hand.palm_velocity[2]/1000 < 0.1:
                fah.time_last_stop = deepcopy(frame.timestamp)
            else:
                # TODO: do also for left
                fah.time_last_stop = settings.frames_adv[-1].r.time_last_stop
            ## Finding since frame
            targetTimestamp = timestamp - settings.gd.r.SINCE_FRAME_TIME * 1000000
            sinceFrameIndex = 0
            for i in range(0,len(settings.timestamps)):
                if abs(settings.timestamps[-i*1] - targetTimestamp) < 20000: # 20ms
                    sinceFrameIndex = -i*1
                    break
            if sinceFrameIndex != 0:
                rot_axis = [0.] * 3
                rot_axis_euler = [0.] * 3
                prev_frame = settings.frames[-sinceFrameIndex]
                for hand_ in prev_frame.hands:
                    if hand_.id == hand.id:
                         rot_axis = [hand.palm_normal.roll-hand_.palm_normal.roll, hand.direction.pitch-hand_.direction.pitch, hand.direction.yaw-hand_.direction.yaw]
                         rot_axis_euler = [hand.palm_normal.roll, hand.direction.pitch, hand.direction.yaw]

                #print("rr", rot_axis)
                q = tf.transformations.quaternion_from_euler(rot_axis[0], rot_axis[1], rot_axis[2])
                fah.rot = Quaternion(q[0], q[1], q[2], q[3])
                fah.rotRaw = [rot_axis[0], rot_axis[1], rot_axis[2]]
                fah.rotRawEuler = rot_axis_euler

            ## Data for training
            # bone directions and angles
            hand_direction = np.array(hand.direction)
            hand_angles = np.array([0., asin(-hand.direction[1]), atan2(hand.direction[0], hand.direction[2])])
            v = np.array(hand.palm_position.to_float_array()) - np.array(hand.wrist_position.to_float_array())

            wrist_direction = v / np.sqrt(np.sum(v**2))
            wrist_angles = np.array([0., asin(-wrist_direction[1]), atan2(wrist_direction[0], wrist_direction[2])])
            bone_direction, bone_angles = np.zeros([5,4,3]), np.zeros([5,4,3])
            for i in range(0,5):
                for j in range(0,4):
                    bone_direction[i][j] = np.array(hand.fingers[i].bone(j).direction.to_float_array())
                    bone_angles[i][j] = np.array((0., asin(-hand.fingers[i].bone(j).direction[1]), atan2(hand.fingers[i].bone(j).direction[0], hand.fingers[i].bone(j).direction[2])))

            # bone angles differences
            fah.wrist_hand_angles_diff = wrist_angles - hand_angles
            fah.fingers_angles_diff = np.zeros([5,4,3])
            for i in range(0,5):
                for j in range(0,4):
                    if j == 0:
                        d1 = hand_angles
                    else:
                        d1 = bone_angles[i][j-1]
                    d2 = bone_angles[i][j]
                    fah.fingers_angles_diff[i][j] = d1 - d2

            # distance between finger positions
            palm_position = np.array(hand.palm_position.to_float_array())
            fah.pos_diff_comb = []
            combs = position_of_fingers; combs.extend([palm_position])
            for comb in combinations(combs,2):
                fah.pos_diff_comb.append(np.array(comb[0]) - np.array(comb[1]))

            # palm and pointing finger
            fah.index_position = hand.fingers[1].bone(3).next_joint.to_float_array()

        ## Gesture Detection
        GestureDetection.all()

        for gesture in frame.gestures():
            if gesture.type == Leap.Gesture.TYPE_CIRCLE and 'circ' in settings.GESTURE_NAMES:
                circle = CircleGesture(gesture)

                # Determine clock direction using the angle between the pointable and the circle normal
                if circle.pointable.direction.angle_to(circle.normal) <= Leap.PI/2:
                    settings.gd.r.gests[settings.gd.r.GESTS["circ"]].clockwise = True
                else:
                    settings.gd.r.gests[settings.gd.r.GESTS["circ"]].clockwise = False

                # Calculate the angle swept since the last frame
                swept_angle = 0
                if circle.state != Leap.Gesture.STATE_START:
                    previous_update = CircleGesture(controller.frame(1).gesture(circle.id))
                    swept_angle =  (circle.progress - previous_update.progress) * 2 * Leap.PI
                    settings.gd.r.gests[settings.gd.r.GESTS["circ"]].progress = circle.progress
                    settings.gd.r.gests[settings.gd.r.GESTS["circ"]].angle = swept_angle * Leap.RAD_TO_DEG
                    settings.gd.r.gests[settings.gd.r.GESTS["circ"]].radius = circle.radius

                if circle.state == Leap.Gesture.STATE_STOP:
                    settings.gd.r.gests[settings.gd.r.GESTS["circ"]].time_visible = 1
                    settings.gd.r.gests[settings.gd.r.GESTS["circ"]].toggle = True
                    settings.gd.r.gests[settings.gd.r.GESTS["circ"]].progress = circle.progress
                    settings.gd.r.gests[settings.gd.r.GESTS["circ"]].angle = swept_angle * Leap.RAD_TO_DEG
                    settings.gd.r.gests[settings.gd.r.GESTS["circ"]].radius = circle.radius

            if gesture.type == Leap.Gesture.TYPE_SWIPE and 'swipe' in settings.GESTURE_NAMES:
                swipe = SwipeGesture(gesture)
                if gesture.state != Leap.Gesture.STATE_START:
                    settings.gd.r.gests[settings.gd.r.GESTS["swipe"]].in_progress = True
                if gesture.state == Leap.Gesture.STATE_STOP:
                    settings.gd.r.gests[settings.gd.r.GESTS["swipe"]].in_progress = False
                    settings.gd.r.gests[settings.gd.r.GESTS["swipe"]].time_visible = 1
                    settings.gd.r.gests[settings.gd.r.GESTS["swipe"]].toggle = True
                    settings.gd.r.gests[settings.gd.r.GESTS["swipe"]].direction = [swipe.direction[0], swipe.direction[1], swipe.direction[2]]
                    settings.gd.r.gests[settings.gd.r.GESTS["swipe"]].speed = swipe.speed

            if gesture.type == Leap.Gesture.TYPE_KEY_TAP and 'pin' in settings.GESTURE_NAMES:
                keytap = KeyTapGesture(gesture)
                if gesture.state != Leap.Gesture.STATE_START:
                    settings.gd.r.gests[settings.gd.r.GESTS["pin"]].in_progress = True
                if gesture.state == Leap.Gesture.STATE_STOP:
                    settings.gd.r.gests[settings.gd.r.GESTS["pin"]].in_progress = False
                    settings.gd.r.gests[settings.gd.r.GESTS["pin"]].direction = [keytap.direction[0], keytap.direction[1], keytap.direction[2]]
                    settings.gd.r.gests[settings.gd.r.GESTS["pin"]].time_visible = 1
                    settings.gd.r.gests[settings.gd.r.GESTS["pin"]].toggle = True

            if gesture.type == Leap.Gesture.TYPE_SCREEN_TAP and 'touch' in settings.GESTURE_NAMES:
                screentap = ScreenTapGesture(gesture)
                if gesture.state != Leap.Gesture.STATE_START:
                    settings.gd.r.gests[settings.gd.r.GESTS["touch"]].in_progress = True
                if gesture.state == Leap.Gesture.STATE_STOP:
                    settings.gd.r.gests[settings.gd.r.GESTS["touch"]].in_progress = False
                    settings.gd.r.gests[settings.gd.r.GESTS["touch"]].direction = [screentap.direction[0], screentap.direction[1], screentap.direction[2]]
                    settings.gd.r.gests[settings.gd.r.GESTS["touch"]].time_visible = 1
                    settings.gd.r.gests[settings.gd.r.GESTS["touch"]].toggle = True

        settings.frames_adv.append(fa)

    def state_string(self, state):
        if state == Leap.Gesture.STATE_START:
            return "STATE_START"

        if state == Leap.Gesture.STATE_UPDATE:
            return "STATE_UPDATE"

        if state == Leap.Gesture.STATE_STOP:
            return "STATE_STOP"

        if state == Leap.Gesture.STATE_INVALID:
            return "STATE_INVALID"


class GestureDetection():
    @staticmethod
    def all():
        if settings.frames_adv and settings.mo:
            GestureDetection.processTch()
            GestureDetection.processOc()

            if 'grab' in settings.GESTURE_NAMES: GestureDetection.processPose_grab()
            if 'pinch' in settings.GESTURE_NAMES: GestureDetection.processPose_pinch()
            if 'point' in settings.GESTURE_NAMES: GestureDetection.processPose_point()
            if 'respectful' in settings.GESTURE_NAMES: GestureDetection.processPose_respectful()
            if 'spock' in settings.GESTURE_NAMES: GestureDetection.processPose_spock()
            if 'rock' in settings.GESTURE_NAMES: GestureDetection.processPose_rock()
            if 'victory' in settings.GESTURE_NAMES: GestureDetection.processPose_victory()
            if 'italian' in settings.GESTURE_NAMES: GestureDetection.processPose_italian()

            if 'move_in_axis' in settings.GESTURE_NAMES: GestureDetection.processGest_move_in_axis()
            if 'rotation_in_axis' in settings.GESTURE_NAMES: GestureDetection.processGest_rotation_in_axis()

            if 'move_in_axis' in settings.GESTURE_NAMES: GestureDetection.processComb_goToConfig()

    @staticmethod
    def processTch():
        fa = settings.frames_adv[-1]
        if fa.r.visible:
            if fa.r.conf > settings.gd.r.MIN_CONFIDENCE:
                settings.gd.r.conf = True
            else:
                settings.gd.r.conf = False

            if fa.r.TCH12 > settings.gd.r.TCH_TURN_ON_DIST[0] and settings.gd.r.conf:
                settings.gd.r.tch12 = False
            elif fa.r.TCH12 < settings.gd.r.TCH_TURN_OFF_DIST[0]:
                settings.gd.r.tch12 = True
            if fa.r.TCH23 > settings.gd.r.TCH_TURN_ON_DIST[1] and settings.gd.r.conf:
                settings.gd.r.tch23 = False
            elif fa.r.TCH23 < settings.gd.r.TCH_TURN_OFF_DIST[1]:
                settings.gd.r.tch23 = True
            if fa.r.TCH34 > settings.gd.r.TCH_TURN_ON_DIST[2] and settings.gd.r.conf:
                settings.gd.r.tch34 = False
            elif fa.r.TCH34 < settings.gd.r.TCH_TURN_OFF_DIST[2]:
                settings.gd.r.tch34 = True
            if fa.r.TCH45 > settings.gd.r.TCH_TURN_ON_DIST[3] and settings.gd.r.conf:
                settings.gd.r.tch45 = False
            elif fa.r.TCH45 < settings.gd.r.TCH_TURN_OFF_DIST[3]:
                settings.gd.r.tch45 = True

            if fa.r.TCH13 > settings.gd.r.TCH_TURN_ON_DIST[4] and settings.gd.r.conf:
                settings.gd.r.tch13 = False
            elif fa.r.TCH13 < settings.gd.r.TCH_TURN_OFF_DIST[4]:
                settings.gd.r.tch13 = True
            if fa.r.TCH14 > settings.gd.r.TCH_TURN_ON_DIST[5] and settings.gd.r.conf:
                settings.gd.r.tch14 = False
            elif fa.r.TCH14 < settings.gd.r.TCH_TURN_OFF_DIST[5]:
                settings.gd.r.tch14 = True
            if fa.r.TCH15 > settings.gd.r.TCH_TURN_ON_DIST[6] and settings.gd.r.conf:
                settings.gd.r.tch15 = False
            elif fa.r.TCH15 < settings.gd.r.TCH_TURN_OFF_DIST[6]:
                settings.gd.r.tch15 = True

    @staticmethod
    def processOc():
        fa = settings.frames_adv[-1]
        if fa.r.visible:
            gd = settings.gd.r
            if fa.r.conf > gd.MIN_CONFIDENCE:
                gd.conf = True
            else:
                gd.conf = False

            for i in range(0,5):
                if fa.r.OC[i] > gd.OC_TURN_ON_THRE[i] and gd.conf:
                    gd.oc[i] = True
                elif fa.r.OC[i] < gd.OC_TURN_OFF_THRE[i]:
                    gd.oc[i] = False

    @staticmethod
    def processPose_grab():
        fa = settings.frames_adv[-1]
        if fa.l.visible:
            gd = settings.gd.l
            g = gd.poses[gd.POSES["grab"]]
            g.prob = fa.l.grab
            # gesture toggle processing
            if fa.l.grab > g.TURN_ON_THRE:
                g.toggle = True
            elif fa.l.grab < g.TURN_OFF_THRE:
                g.toggle = False
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["grab"]]
            g.prob = fa.r.grab
            # gesture toggle processing
            if fa.r.grab > g.TURN_ON_THRE and gd.conf:
                g.toggle = True
            elif fa.r.grab < g.TURN_OFF_THRE:
                g.toggle = False

    @staticmethod
    def processPose_pinch():
        fa = settings.frames_adv[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["pinch"]]
            g.prob = fa.r.pinch
            if fa.r.pinch > g.TURN_ON_THRE and gd.conf:
                g.toggle = True
                g.time_visible += 0.01
            elif fa.r.pinch < g.TURN_OFF_THRE:
                g.toggle = False
                g.time_visible = 0.0

    @staticmethod
    def processPose_point():
        ''' tch, oc functions need to be called before to get fingers O/C
        '''
        fa = settings.frames_adv[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["point"]]
            if gd.oc[1] is True and gd.oc[2] is False and gd.oc[3] is False and gd.oc[4] is False:
                g.toggle = True
                g.time_visible += 0.01
            elif gd.oc[1] is False or gd.oc[3] is True or gd.oc[4] is True:
                g.toggle = False
                g.time_visible = 0.0

    @staticmethod
    def processPose_respectful():
        ''' tch, oc functions need to be called before to get fingers O/C
        '''
        fa = settings.frames_adv[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["respectful"]]
            if gd.oc[0] is False and gd.oc[1] is True and gd.oc[2] is True and gd.oc[3] is True and gd.oc[4] is False:
                g.toggle = True
                g.time_visible = 1
            elif gd.oc[0] is True or gd.oc[1] is False or gd.oc[2] is False or gd.oc[3] is False or gd.oc[4] is True:
                g.toggle = False

    @staticmethod
    def processPose_spock():
        ''' tch, oc functions need to be called before to get fingers O/C
        '''
        fa = settings.frames_adv[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["spock"]]
            if gd.oc[1] is True and gd.oc[2] is True and gd.oc[3] is True and gd.oc[4] is True and gd.tch23 is True and gd.tch34 is False and gd.tch45 is True:
                g.toggle = True
                g.time_visible = 1
            elif gd.oc[1] is False or gd.oc[2] is False or gd.oc[3] is False or gd.oc[4] is False or gd.tch23 is False or gd.tch34 is True or gd.tch45 is False:
                g.toggle = False

    @staticmethod
    def processPose_rock():
        ''' tch, oc functions need to be called before to get fingers O/C
        '''
        fa = settings.frames_adv[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["rock"]]
            if gd.oc[1] is True and gd.oc[4] is True and gd.oc[2] is False and gd.oc[3] is False:
                g.toggle = True
                g.time_visible = 1
            elif gd.oc[1] is False or gd.oc[2] is True or gd.oc[3] is True or gd.oc[4] is False:
                g.toggle = False

    @staticmethod
    def processPose_victory():
        ''' tch, oc functions need to be called before to get fingers O/C
        '''
        fa = settings.frames_adv[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["victory"]]
            if gd.oc[1] is True and gd.oc[2] is True and gd.oc[3] is False and gd.oc[4] is False and gd.oc[0] is False:
                g.toggle = True
                g.time_visible = 1
            elif gd.oc[1] is False or gd.oc[2] is False or gd.oc[3] is True or gd.oc[4] is True or gd.oc[0] is True:
                g.toggle = False

    @staticmethod
    def processPose_italian():
        ''' tch, oc functions need to be called before to get fingers O/C
        '''
        fa = settings.frames_adv[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["italian"]]
            if gd.tch12 is True and gd.tch23 is True and gd.tch34 is True and gd.tch45 is True:
                g.toggle = True
                g.time_visible = 1
            elif gd.tch12 is False or gd.tch23 is False or gd.tch34 is False or gd.tch45 is False:
                g.toggle = False

    @staticmethod
    def processComb_goToConfig():
        ''' tch, oc functions need to be called before to get fingers O/C
        '''
        fa = settings.frames_adv[-1]
        g = settings.gd.r.gests[settings.gd.r.GESTS["move_in_axis"]]
        g_time = settings.gd.r.poses[settings.gd.r.POSES["point"]].time_visible
        if g_time > 2:
            if g.toggle[0] and g.move[0]:
                settings.WindowState = 1
            if g.toggle[0] and not g.move[0]:
                settings.WindowState = 0


    @staticmethod
    def processGest_move_in_axis():
        '''
        '''
        fa = settings.frames_adv[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.gests[gd.GESTS["move_in_axis"]]
            g_tmp = deepcopy(g.toggle)
            if abs(fa.r.vel[0]) > g.MIN_THRE and fa.r.vel[1] < g.MAX_THRE and fa.r.vel[2] < g.MAX_THRE:
                g.toggle[0] = True
                g.time_visible = 1
                g.move[0] = True if fa.r.vel[0] > g.MIN_THRE else False
                if g_tmp[0] == False:
                    settings.mo.gestureGoalPoseUpdate(0, g.move[0])
            else:
                g.toggle[0] = False
            if abs(fa.r.vel[1]) > g.MIN_THRE and fa.r.vel[0] < g.MAX_THRE and fa.r.vel[2] < g.MAX_THRE:
                g.time_visible = 1
                g.toggle[1] = True
                g.move[1] = True if fa.r.vel[1] > g.MIN_THRE else False
                if g_tmp[1] == False:
                    settings.mo.gestureGoalPoseUpdate(1, g.move[1])
            else:
                g.toggle[1] = False
            if abs(fa.r.vel[2]) > g.MIN_THRE and fa.r.vel[0] < g.MAX_THRE and fa.r.vel[1] < g.MAX_THRE:
                g.time_visible = 1
                g.toggle[2] = True
                g.move[2] = True if fa.r.vel[2] > g.MIN_THRE else False
                if g_tmp[2] == False:
                    settings.mo.gestureGoalPoseUpdate(2, g.move[2])
            else:
                g.toggle[2] = False


    @staticmethod
    def processGest_rotation_in_axis():
        '''
        '''
        fa = settings.frames_adv[-1]
        if fa.r.visible:
            euler = fa.r.pRaw[3:6]
            gd = settings.gd.r
            g = gd.gests[gd.GESTS["rotation_in_axis"]]
            g_tmp = deepcopy(g.toggle)
            if (euler[0] > g.MAX_THRE[0] or euler[0] < g.MIN_THRE[0]):
                g.time_visible = 1
                g.toggle[0] = True
                g.move[0] = True if euler[0] > g.MAX_THRE[0] else False
                if g_tmp[0] == False:
                    settings.mo.gestureGoalPoseRotUpdate(0, g.move[0])
            else:
                g.toggle[0] = False
            if (euler[1] > g.MAX_THRE[1] or euler[1] < g.MIN_THRE[1]):
                g.toggle[1] = True
                g.time_visible = 1
                g.move[1] = True if euler[1] > g.MAX_THRE[1] else False
                if g_tmp[1] == False:
                    settings.mo.gestureGoalPoseRotUpdate(1, g.move[1])
            else:
                g.toggle[1] = False
            if (euler[1] > g.MAX_THRE[2] or euler[1] < g.MIN_THRE[2]):
                g.toggle[2] = True
                g.time_visible = 1
                g.move[2] = True if euler[2] > g.MAX_THRE[2] else False
                if g_tmp[2] == False:
                    settings.mo.gestureGoalPoseRotUpdate(2, g.move[2])
            else:
                g.toggle[2] = False


def main():
    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)
    while True: pass
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)


if __name__ == "__main__":
    settings.init()
    main()
    print("Leap Listener Exit")
