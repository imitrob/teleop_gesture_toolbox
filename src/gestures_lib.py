#!/usr/bin/env python3.8

## TODO: Move gestures detection lib from leapmotionlistener.py

import os
import gdown

import settings


class GesturesDetectionClass():
    def __init__(self):
        pass

    @staticmethod
    def download_networks_gdrive():
        # get one dir above
        NETWORKS_PATH = '/'.join((settings.NETWORK_PATH).split('/')[:-2])
        gdown.download_folder(settings.NETWORKS_DRIVE_URL, output=NETWORKS_PATH)


    def change_current_network(self, network=None):
        ''' Switches learned file
        '''
        pass

    @staticmethod
    def get_networks():
        ''' Looks at the settings.NETWORK_PATH folder and returns every file with extension *.pkl
        '''
        networks = []
        for file in os.listdir(settings.NETWORK_PATH):
            if file.endswith(".pkl"):
                networks.append(file)
        return networks

    @staticmethod
    def receive_hand_data_callback(msg):
        ''' Puts msg as new record into
        '''
        Frame()

        framemsg = Framemsg()
        framemsg.fps = f.fps
        framemsg.hands = f.hands
        framemsg.header.secs = f.secs
        framemsg.header.nsecs = f.nsecs
        framemsg.header.seq = f.seq

        framemsg.leapgestures.circle_toggle = f.leapgestures.circle.toggle
        framemsg.leapgestures.circle_in_progress = f.leapgestures.circle.in_progress
        framemsg.leapgestures.circle_clockwise = f.leapgestures.circle.clockwise
        framemsg.leapgestures.circle_progress = f.leapgestures.circle.progress
        framemsg.leapgestures.circle_angle = f.leapgestures.circle.angle
        framemsg.leapgestures.circle_radius = f.leapgestures.circle.radius
        framemsg.leapgestures.circle_state = f.leapgestures.circle.state
        framemsg.leapgestures.swipe_toggle = f.leapgestures.swipe.toggle
        framemsg.leapgestures.swipe_in_progress = f.leapgestures.swipe.in_progress
        framemsg.leapgestures.swipe_direction = f.leapgestures.swipe.direction
        framemsg.leapgestures.swipe_speed = f.leapgestures.swipe.speed
        framemsg.leapgestures.swipe_state = f.leapgestures.swipe.state
        framemsg.leapgestures.keytap_toggle = f.leapgestures.keytap.toggle
        framemsg.leapgestures.keytap_in_progress = f.leapgestures.keytap.in_progress
        framemsg.leapgestures.keytap_direction = f.leapgestures.keytap.direction
        framemsg.leapgestures.keytap_state = f.leapgestures.keytap.state
        framemsg.leapgestures.screentap_toggle = f.leapgestures.screentap.toggle
        framemsg.leapgestures.screentap_in_progress = f.leapgestures.screentap.in_progress
        framemsg.leapgestures.screentap_direction = f.leapgestures.screentap.direction
        framemsg.leapgestures.screentap_state = f.leapgestures.screentap.state

        for (handmsg, hand) in [(framemsg.l, f.l), (framemsg.r, f.r)]:
            handmsg.id = hand.id
            handmsg.is_left = hand.is_left
            handmsg.is_right = hand.is_right
            handmsg.is_valid = hand.is_valid
            handmsg.grab_strength = hand.grab_strength
            handmsg.pinch_strength = hand.pinch_strength
            handmsg.confidence = hand.confidence
            handmsg.palm_normal = hand.palm_normal()
            handmsg.direction = hand.direction()
            handmsg.palm_position = hand.palm_position()

            handmsg.finger_bones = []
            for finger in hand.fingers:
                for bone in finger.bones:
                    bonemsg = Bonemsg()

                    basis = bone.basis[0](), bone.basis[1](), bone.basis[2]()
                    bonemsg.basis = [item for sublist in basis for item in sublist]
                    bonemsg.direction = bone.direction()
                    bonemsg.next_joint = bone.next_joint()
                    bonemsg.prev_joint = bone.prev_joint()
                    bonemsg.center = bone.center()
                    bonemsg.is_valid = bone.is_valid
                    bonemsg.length = bone.length
                    bonemsg.width = bone.width

                    handmsg.finger_bones.append(bonemsg)

            handmsg.palm_velocity = hand.palm_velocity()
            basis = hand.basis[0](), hand.basis[1](), hand.basis[2]()
            handmsg.basis = [item for sublist in basis for item in sublist]
            handmsg.palm_width = hand.palm_width
            handmsg.sphere_center = hand.sphere_center()
            handmsg.sphere_radius = hand.sphere_radius
            handmsg.stabilized_palm_position = hand.stabilized_palm_position()
            handmsg.time_visible = hand.time_visible
            handmsg.wrist_position = hand.wrist_position()

    def ros_publish(self, f):
        # f hand_classes.Frame() -> framemsg mirracle_gestures.msg/Frame
        framemsg = Framemsg()
        framemsg.fps = f.fps
        framemsg.hands = f.hands
        framemsg.header.secs = f.secs
        framemsg.header.nsecs = f.nsecs
        framemsg.header.seq = f.seq

        framemsg.leapgestures.circle_toggle = f.leapgestures.circle.toggle
        framemsg.leapgestures.circle_in_progress = f.leapgestures.circle.in_progress
        framemsg.leapgestures.circle_clockwise = f.leapgestures.circle.clockwise
        framemsg.leapgestures.circle_progress = f.leapgestures.circle.progress
        framemsg.leapgestures.circle_angle = f.leapgestures.circle.angle
        framemsg.leapgestures.circle_radius = f.leapgestures.circle.radius
        framemsg.leapgestures.circle_state = f.leapgestures.circle.state
        framemsg.leapgestures.swipe_toggle = f.leapgestures.swipe.toggle
        framemsg.leapgestures.swipe_in_progress = f.leapgestures.swipe.in_progress
        framemsg.leapgestures.swipe_direction = f.leapgestures.swipe.direction
        framemsg.leapgestures.swipe_speed = f.leapgestures.swipe.speed
        framemsg.leapgestures.swipe_state = f.leapgestures.swipe.state
        framemsg.leapgestures.keytap_toggle = f.leapgestures.keytap.toggle
        framemsg.leapgestures.keytap_in_progress = f.leapgestures.keytap.in_progress
        framemsg.leapgestures.keytap_direction = f.leapgestures.keytap.direction
        framemsg.leapgestures.keytap_state = f.leapgestures.keytap.state
        framemsg.leapgestures.screentap_toggle = f.leapgestures.screentap.toggle
        framemsg.leapgestures.screentap_in_progress = f.leapgestures.screentap.in_progress
        framemsg.leapgestures.screentap_direction = f.leapgestures.screentap.direction
        framemsg.leapgestures.screentap_state = f.leapgestures.screentap.state

        for (handmsg, hand) in [(framemsg.l, f.l), (framemsg.r, f.r)]:
            handmsg.id = hand.id
            handmsg.is_left = hand.is_left
            handmsg.is_right = hand.is_right
            handmsg.is_valid = hand.is_valid
            handmsg.grab_strength = hand.grab_strength
            handmsg.pinch_strength = hand.pinch_strength
            handmsg.confidence = hand.confidence
            handmsg.palm_normal = hand.palm_normal()
            handmsg.direction = hand.direction()
            handmsg.palm_position = hand.palm_position()

            handmsg.finger_bones = []
            for finger in hand.fingers:
                for bone in finger.bones:
                    bonemsg = Bonemsg()

                    basis = bone.basis[0](), bone.basis[1](), bone.basis[2]()
                    bonemsg.basis = [item for sublist in basis for item in sublist]
                    bonemsg.direction = bone.direction()
                    bonemsg.next_joint = bone.next_joint()
                    bonemsg.prev_joint = bone.prev_joint()
                    bonemsg.center = bone.center()
                    bonemsg.is_valid = bone.is_valid
                    bonemsg.length = bone.length
                    bonemsg.width = bone.width

                    handmsg.finger_bones.append(bonemsg)

            handmsg.palm_velocity = hand.palm_velocity()
            basis = hand.basis[0](), hand.basis[1](), hand.basis[2]()
            handmsg.basis = [item for sublist in basis for item in sublist]
            handmsg.palm_width = hand.palm_width
            handmsg.sphere_center = hand.sphere_center()
            handmsg.sphere_radius = hand.sphere_radius
            handmsg.stabilized_palm_position = hand.stabilized_palm_position()
            handmsg.time_visible = hand.time_visible
            handmsg.wrist_position = hand.wrist_position()

        self.frame_publisher.publish(framemsg)


class Network():
    def __init__(self, file):
        self.name = file


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
