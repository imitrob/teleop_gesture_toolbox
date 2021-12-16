#!/usr/bin/env python3.8

## TODO: Move gestures detection lib from leapmotionlistener.py

import os
import gdown

import settings


class GestureDetection():
    def __init__(self):
        self.l = GestureDataHand()
        self.r = GestureDataHand()

    @staticmethod
    def download_networks_gdrive():
        # get one dir above
        NETWORKS_PATH = '/'.join((settings.paths.network_path).split('/')[:-2])
        gdown.download_folder(settings.configGestures['networks_drive_url'], output=NETWORKS_PATH)

    @staticmethod
    def get_networks():
        ''' Looks at the settings.paths.network_path folder and returns every file with extension *.pkl
        '''
        networks = []
        for file in os.listdir(settings.paths.network_path):
            if file.endswith(".pkl"):
                networks.append(file)
        return networks


class GestureDataHand():
    '''
    '''
    def __init__(self):

        configGestures = ParseYAML.load_gesture_config_file(paths.custom_settings_yaml)
        gestures = ParseYAML.load_gestures_file(paths.custom_settings_yaml, ret='obj')

        self.conf = False
        self.MIN_CONFIDENCE = configGestures['MIN_CONFIDENCE']
        self.SINCE_FRAME_TIME = configGestures['SINCE_FRAME_TIME']

        self.tch12, self.tch23, self.tch34, self.tch45 = [False] * 4
        self.tch13, self.tch14, self.tch15 = [False] * 3
        self.TCH_TURN_ON_DIST = configGestures['TCH_TURN_ON_DIST']
        self.TCH_TURN_OFF_DIST = configGestures['TCH_TURN_OFF_DIST']

        self.oc = [False] * 5
        self.OC_TURN_ON_THRE =  configGestures['OC_TURN_ON_THRE']
        self.OC_TURN_OFF_THRE = configGestures['OC_TURN_OFF_THRE']


        self.poses = []
        self.gests = []
        for gesture in Gs:
            using_set = configGestures['using_set']
            try:
                if gesture in gestures[using_set].keys():
                    if gestures[using_set][gesture]['dynamic'] == 'false' or gestures[using_set][gesture]['dynamic'] == False:
                        self.poses.append(PoseData(name=gesture, data=gestures[using_set][gesture]))
                    else:
                        self.gests.append(GestureData(name=gesture, data=gestures[using_set][gesture]))
                else:
                    raise Exception("[ERROR* Settings] Information about gesture with name: ", gesture, " could not be found in gesture_recording.yaml file. Make an entry there.")
            except KeyError:
                print("gesture key", gesture, " and keys ", gestures[using].keys())


        # Create links for gestures
        self.POSES = {}
        for n, i in enumerate(self.poses):
            self.POSES[i.NAME] = n
        self.GESTS = {}
        for n, i in enumerate(self.gests):
            self.GESTS[i.NAME] = n

        self.final_chosen_pose = 0
        self.final_chosen_gesture = 0

        self.pymcout = None

class PoseData():
    def __init__(self, name, data):
        data = ParseYAML.parseStaticGesture(data)

        self.NAME = name
        self.prob = 0.0
        self.toggle = False
        self.TURN_ON_THRE = data['turnon']
        self.TURN_OFF_THRE = data['turnoff']
        self.time_visible = 0.0
        self.filename = data['filename']

class GestureData():
    def __init__(self, name, data):
        data = ParseYAML.parseDynamicGesture(data)

        self.NAME = name
        if data['var_len'] > 1:
            self.prob = [0.0] * data['var_len']
            self.toggle = [False] * data['var_len']
        else:
            self.prob = 0.0
            self.toggle = False
        self.time_visible = 0.0
        self.in_progress = False
        self.direction = [0.0,0.0,0.0]
        self.speed = 0.0
        self.filename = data['filename']

        # for circle movement
        self.clockwise = False
        self.angle = 0.0
        self.progress = 0.0
        self.radius = 0.0
        # for move_in_axis thresholds
        self.MIN_THRE = data['minthre']
        self.MAX_THRE = data['maxthre']
        ## move in x,y,z, Positive/Negative
        self.move = [False, False, False]


class Network():
    def __init__(self, file):
        self.name = file


class GestureDetection():
    @staticmethod
    def all():
        if settings.frames_adv and settings.mo:
            GestureDetection.processTch()
            GestureDetection.processOc()

            if 'grab' in settings.Gs: GestureDetection.processPose_grab()
            if 'pinch' in settings.Gs: GestureDetection.processPose_pinch()
            if 'point' in settings.Gs: GestureDetection.processPose_point()
            if 'respectful' in settings.Gs: GestureDetection.processPose_respectful()
            if 'spock' in settings.Gs: GestureDetection.processPose_spock()
            if 'rock' in settings.Gs: GestureDetection.processPose_rock()
            if 'victory' in settings.Gs: GestureDetection.processPose_victory()
            if 'italian' in settings.Gs: GestureDetection.processPose_italian()

            if 'move_in_axis' in settings.Gs: GestureDetection.processGest_move_in_axis()
            if 'rotation_in_axis' in settings.Gs: GestureDetection.processGest_rotation_in_axis()

            if 'move_in_axis' in settings.Gs: GestureDetection.processComb_goToConfig()

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
