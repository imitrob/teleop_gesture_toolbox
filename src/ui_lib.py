#!/usr/bin/python2.7
""" Launches Interface Application
    - Loads config from include/custom_settings/application.yaml
"""

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys, random
import settings
import numpy as np
import tf
from copy import deepcopy
import time
from os.path import expanduser, isfile, isdir
from sklearn.metrics import confusion_matrix
from scipy import sparse
from threading import Thread
import matplotlib.pyplot as plt
import csv
import pickle
import ctypes
from threading import Timer
from visualizer_lib import VisualizerLib
import yaml
import io

# ros msg classes
from geometry_msgs.msg import PoseStamped, Quaternion, Pose, Point

class Example(QMainWindow):

    def __init__(self):
        super(Example, self).__init__()

        with open(settings.CUSTOM_SETTINGS_YAML+"gesture_recording.yaml", 'r') as stream:
            gestures_data_loaded = yaml.safe_load(stream)
        with open(settings.CUSTOM_SETTINGS_YAML+"application.yaml", 'r') as stream:
            app_data_loaded = yaml.safe_load(stream)
        global LEFT_MARGIN, RIGHT_MARGIN, BOTTOM_MARGIN, ICON_SIZE, START_PANEL_Y, BarMargin
        LEFT_MARGIN = app_data_loaded['LEFT_MARGIN']
        RIGHT_MARGIN = app_data_loaded['RIGHT_MARGIN']
        BOTTOM_MARGIN = app_data_loaded['BOTTOM_MARGIN']
        ICON_SIZE = app_data_loaded['ICON_SIZE']
        START_PANEL_Y = app_data_loaded['START_PANEL_Y']
        BarMargin = app_data_loaded['BarMargin']

        self.setMinimumSize(QSize(500, 400)) # Minimum window size
        self.lbl1 = QLabel('Poses & Gestures', self)
        self.lbl1.setGeometry(20, 36, 150, 50)
        self.lbl2 = QLabel('Observations', self)
        self.lbl2.setGeometry(self.size().width()-140, 36, 100, 50)

        ## View Configuration App
        settings.WindowState = 0
        self.ViewState = True
        ## Cursor Picking (on Configuration page)
        self.pickedSolution = np.zeros(settings.NumConfigBars)
        self.pickedTime = np.zeros(settings.NumConfigBars)

        self.lblConfNames = app_data_loaded['ConfigurationPage']['ItemNames']
        self.lblConfValues = ['0.', '0.', '0.', '0.', '0.', '0.', '0.', '0.']
        self.lblConfNamesObj = []
        self.lblConfValuesObj = []
        for i in range(0, settings.NumConfigBars[0]*settings.NumConfigBars[1]):
            self.lblConfNamesObj.append(QLabel(self.lblConfNames[i], self))
            self.lblConfValuesObj.append(QLabel(self.lblConfValues[i], self))

        ## Label Observation Poses
        self.lblObservationPosesNames = ['conf.', 'oc1', 'oc2', 'oc3', 'oc4', 'oc5', 'tch12', 'tch23', 'tch34', 'tch45', 'tch13', 'tch14', 'tch15', 'x_vel', 'y_vel', 'z_vel', 'x_rot', 'y_rot', 'z_rot']
        self.lblObservationPosesValues = ['0.0' , '0.0', '0.0', '0.0', '0.0', '0.0', '0.0',   '0.0',   '0.0',   '0.0',   '0.0',   '0.0',   '0.0',   '0.0',   '0.0',   '0.0',   '0.0',  '0.0', '0.0']
        self.lblObservationPosesNamesObj = []
        self.lblObservationPosesValuesObj = []
        for i in range(0, len(self.lblObservationPosesNames)):
            self.lblObservationPosesValuesObj.append(QLabel(self.lblObservationPosesValues[i], self))
            self.lblObservationPosesNamesObj.append(QLabel(self.lblObservationPosesNames[i], self))
        for i in self.lblObservationPosesValuesObj:
            i.setVisible(False)
        ## Label Gestrues Poses
        self.lblGesturesPosesNames = settings.GESTURE_NAMES
        self.lblGesturesPosesNamesObj = []
        for i in range(0, len(self.lblGesturesPosesNames)):
            self.lblGesturesPosesNamesObj.append(QLabel(self.lblGesturesPosesNames[i], self))

        self.lblCreateConfusionMatrixInfo = QLabel("", self)
        self.btnConf = QPushButton('Confusion matrix', self)
        self.btnConf.setToolTip('Create <b>Confusion</b> matrix')
        self.btnConf.resize(self.btnConf.sizeHint())
        self.btnConf.clicked.connect(self.button_confustion_mat)

        self.comboPlayNLive = QComboBox(self)
        self.comboPlayNLive.addItem("Live hand")
        self.comboPlayNLive.addItem("Play path")
        self.comboPlayNLive.activated[str].connect(self.onComboPlayNLiveChanged)
        self.comboPlayNLive.setGeometry(LEFT_MARGIN+130, START_PANEL_Y-10,ICON_SIZE*2,ICON_SIZE/2)

        self.comboPickPlayTraj = QComboBox(self)
        for path in settings.sp:
            self.comboPickPlayTraj.addItem(path.NAME)
        self.comboPickPlayTraj.activated[str].connect(self.onComboPickPlayTrajChanged)
        self.comboPickPlayTraj.setGeometry(LEFT_MARGIN+130+ICON_SIZE*2, START_PANEL_Y-10,ICON_SIZE*2,ICON_SIZE/2)

        self.comboLiveMode = QComboBox(self)
        self.comboLiveMode.addItem("Default")
        self.comboLiveMode.addItem("Interactive")
        self.comboLiveMode.addItem("Gesture based")
        self.comboLiveMode.activated[str].connect(self.onComboLiveModeChanged)
        self.comboLiveMode.setGeometry(LEFT_MARGIN+130+ICON_SIZE*2, START_PANEL_Y-10,ICON_SIZE*2,ICON_SIZE/2)
        ## Control of the movement exectution
        self.btnPlayMove = QPushButton('Forward', self)
        self.btnPlayMove.clicked.connect(self.button_play_move)
        self.btnPlayMove.setGeometry(LEFT_MARGIN+130+ICON_SIZE*4, START_PANEL_Y-10,ICON_SIZE,ICON_SIZE/2)
        self.btnPlayMove2 = QPushButton('Backward', self)
        self.btnPlayMove2.clicked.connect(self.button_play_move2)
        self.btnPlayMove2.setGeometry(LEFT_MARGIN+130+ICON_SIZE*5, START_PANEL_Y-10,ICON_SIZE,ICON_SIZE/2)
        self.btnPlayMove3 = QPushButton('Stop', self)
        self.btnPlayMove3.clicked.connect(self.button_play_move3)
        self.btnPlayMove3.setGeometry(LEFT_MARGIN+130+ICON_SIZE*6, START_PANEL_Y-10,ICON_SIZE,ICON_SIZE/2)

        self.btnSave = QPushButton("Save", self)
        self.btnSave.clicked.connect(self.button_save)
        self.recording = False # Bool if recording is happening
        self.REC_TIME = gestures_data_loaded['Recording']['Length'] # sec
        self.dir_queue = []

        self.lblStatus = QLabel('Status bar', self)
        self.lblStatus.setGeometry(LEFT_MARGIN+130, START_PANEL_Y, 200, 100)

        self.comboInteractiveSceneChanges = QComboBox(self)
        self.comboInteractiveSceneChanges.addItem("Scene 1 Drawer")
        self.comboInteractiveSceneChanges.addItem("Scene 2 Pick/Place")
        self.comboInteractiveSceneChanges.addItem("Scene 3 Push button")
        self.comboInteractiveSceneChanges.addItem("Scene 4 - 2 Pick/Place")
        self.comboInteractiveSceneChanges.activated[str].connect(self.onInteractiveSceneChanged)
        self.comboInteractiveSceneChanges.setGeometry(LEFT_MARGIN+130+ICON_SIZE*4, START_PANEL_Y-10,ICON_SIZE*2,ICON_SIZE/2)


        # Move Page
        lbls = ['Pos. X:', 'Pos. Y:', 'Pos. Z:', 'Ori. X:', 'Ori. Y:', 'Ori. Z:', 'Ori. W:']
        self.movePageGoPoseLabels = []
        self.movePageGoPoseEdits = []
        for i in range(0,7):
            self.movePageGoPoseLabels.append(QLabel(self))
            self.movePageGoPoseLabels[-1].setText(lbls[i])
            self.movePageGoPoseLabels[-1].move(LEFT_MARGIN+20, START_PANEL_Y+i*32)
            self.movePageGoPoseEdits.append(QLineEdit(self))
            self.movePageGoPoseEdits[-1].move(LEFT_MARGIN+80, START_PANEL_Y+i*32)
            self.movePageGoPoseEdits[-1].resize(200, 32)
        self.movePageGoPoseButton = QPushButton("Go To Pose", self)
        self.movePageGoPoseButton.clicked.connect(self.go_to_pose_button)
        self.movePageGoPoseButton.move(LEFT_MARGIN+80, START_PANEL_Y+7*32)

        # Scene Page


        self.timer = QBasicTimer()
        self.timer.start(100, self)
        self.step = 0

        self.play_status = 0

        menubar = self.menuBar()
        viewMenu = menubar.addMenu('View')
        pageMenu = menubar.addMenu('Page')
        configMenu = menubar.addMenu('Config')
        sceneMenu = menubar.addMenu('Scene')
        testingMenu = menubar.addMenu('Testing')

        ## Menu items -> View options
        viewOptionsAction = QAction('View gestures', self, checkable=True)
        viewOptionsAction.setStatusTip('View gestures')
        viewOptionsAction.setChecked(True)
        viewOptionsAction.triggered.connect(self.toggleMenu)
        ## Menu items -> Go to page
        viewGoToInfoAction = QAction('Info page', self)
        viewGoToInfoAction.setStatusTip('Info page')
        viewGoToInfoAction.triggered.connect(self.goToInfo)
        viewGoToControlAction = QAction('Control page', self)
        viewGoToControlAction.setStatusTip('Control page')
        viewGoToControlAction.triggered.connect(self.goToConfig)
        viewGoToMoveAction = QAction('Move page', self)
        viewGoToMoveAction.setStatusTip('Move page')
        viewGoToMoveAction.triggered.connect(self.goToMove)

        # The environment
        impMenu = QMenu('Area choose', self)
        switchEnvAboveAct = QAction('Above', self)
        switchEnvAboveAct.triggered.connect(self.switchEnvAbove)
        switchEnvWallAct = QAction('Wall', self)
        switchEnvWallAct.triggered.connect(self.switchEnvWall)
        switchEnvTableAct = QAction('Table', self)
        switchEnvTableAct.triggered.connect(self.switchEnvTable)
        impMenu.addAction(switchEnvAboveAct)
        impMenu.addAction(switchEnvWallAct)
        impMenu.addAction(switchEnvTableAct)
        impMenu2 = QMenu('Orientation', self)
        impAct4 = QAction('Fixed', self, checkable=True)
        impAct4.triggered.connect(self.impAct4)
        impMenu2.addAction(impAct4)
        print_path_trace_action = QAction('Print path trace', self, checkable=True)
        print_path_trace_action.triggered.connect(self.print_path_trace)

        SCENES = settings.getSceneNames()
        for index, SCENE in enumerate(SCENES):
            action = QAction('Scene '+str(index)+' '+SCENE, self)
            action.triggered.connect(
                lambda checked, index=index: self.goScene(index))
            sceneMenu.addAction(action)

        initTestAction = QAction('Initialization test', self)
        initTestAction.triggered.connect(settings.mo.testInit)
        tableTestAction = QAction('Table test', self)
        tableTestAction.triggered.connect(settings.mo.testMovements)
        inputTestAction = QAction('Test by input', self)
        inputTestAction.triggered.connect(settings.mo.testMovementsInput)

        ## Add actions to the menu
        viewMenu.addAction(viewOptionsAction)
        pageMenu.addAction(viewGoToInfoAction)
        pageMenu.addAction(viewGoToControlAction)
        pageMenu.addAction(viewGoToMoveAction)
        configMenu.addMenu(impMenu)
        configMenu.addMenu(impMenu2)
        configMenu.addAction(print_path_trace_action)
        testingMenu.addAction(tableTestAction)
        testingMenu.addAction(initTestAction)
        testingMenu.addAction(inputTestAction)

        thread = Thread(target = self.play_method)
        thread.start()

        self.setGeometry(app_data_loaded['WindowXY'][0],app_data_loaded['WindowXY'][1], app_data_loaded['WindowSize'][0], app_data_loaded['WindowSize'][1])
        self.setWindowTitle('Interface')
        self.show()

        ## Initialize Visible Objects Array
        self.AllVisibleObjects = []
        self.AllVisibleObjects.append(ObjectQt('lblStatus',self.lblStatus,0))
        for n,obj in enumerate(self.lblObservationPosesNamesObj):
            self.AllVisibleObjects.append(ObjectQt('lblObservationPosesNamesObj'+str(n),obj,0))
        for n,obj in enumerate(self.lblGesturesPosesNamesObj):
            self.AllVisibleObjects.append(ObjectQt('lblGesturesPosesNamesObj'+str(n),obj,0))
        self.AllVisibleObjects.append(ObjectQt('lbl1',self.lbl1,0))
        self.AllVisibleObjects.append(ObjectQt('lbl2',self.lbl2,0))
        self.AllVisibleObjects.append(ObjectQt('comboPlayNLive',self.comboPlayNLive,0))
        self.AllVisibleObjects.append(ObjectQt('btnConf',self.btnConf,0))
        self.AllVisibleObjects.append(ObjectQt('btnSave',self.btnSave,0))

        self.AllVisibleObjects.append(ObjectQt('comboPickPlayTraj',self.comboPickPlayTraj,0,view_group=['ViewState', 'play']))
        self.AllVisibleObjects.append(ObjectQt('btnPlayMove' ,self.btnPlayMove, 0,view_group=['ViewState', 'play']))
        self.AllVisibleObjects.append(ObjectQt('btnPlayMove2',self.btnPlayMove2,0,view_group=['ViewState', 'play']))
        self.AllVisibleObjects.append(ObjectQt('btnPlayMove3',self.btnPlayMove3,0,view_group=['ViewState', 'play']))

        self.AllVisibleObjects.append(ObjectQt('comboLiveMode',self.comboLiveMode,0,view_group=['ViewState', 'live']))
        self.AllVisibleObjects.append(ObjectQt('comboInteractiveSceneChanges',self.comboInteractiveSceneChanges,0,view_group=['ViewState', 'live']))

        for n,obj in enumerate(self.lblConfNamesObj):
            self.AllVisibleObjects.append(ObjectQt('lblConfNamesObj'+str(n),obj,1))
        for n,obj in enumerate(self.lblConfValuesObj):
            self.AllVisibleObjects.append(ObjectQt('lblConfValuesObj'+str(n),obj,1))
        for n,obj in enumerate(self.movePageGoPoseLabels):
            self.AllVisibleObjects.append(ObjectQt('movePageGoPoseLabels'+str(n),obj,2))
        for n,obj in enumerate(self.movePageGoPoseEdits):
            self.AllVisibleObjects.append(ObjectQt('movePageGoPoseEdits'+str(n),obj,2))
        self.AllVisibleObjects.append(ObjectQt('movePageGoPoseButton',self.movePageGoPoseButton,2))

        print("[Interface] Done")

    def go_to_pose_button(self):
        vals = []
        for obj in self.movePageGoPoseEdits:
            val = None
            try:
                val = float(obj.text())
            except:
                print("[ERROR*] Value Error!")
                val = 0.0
            vals.append(val)
        pose = Pose()
        pose.position = Point(*vals[0:3])
        pose.orientation = Quaternion(*vals[3:7])
        settings.goal_pose = pose

    def keyPressEvent(self, event):
        '''
        '''
        KEYS = [self.mapQtKey(key) for key in settings.GESTURE_KEYS]
        if event.key() in KEYS:
            self.recording = True
            for n, key in enumerate(KEYS):
                if event.key() == key:
                    self.dir_queue.append(self.lblGesturesPosesNames[n])
                    self.caller = RepeatableTimer(self.REC_TIME, self.save_data, ())
                    self.caller.start()
        if event.key() == Qt.Key_X:
            self.caller = RepeatableTimer(self.REC_TIME, self.vis_path, ())
            self.caller.start()
        event.accept()

    def button_confustion_mat(self, e):
        thread = Thread(target = self.button_confustion_mat_)
        thread.start()

    def vis_path(self):
        ''' TODO:
        '''
        viz = VisualizerLib()
        viz.visualize_new_fig(title="Path", dim=3)
        #viz.visualize_3d(settings.goal_pose_array, storeObj=settings, color='b', label="leap", units='m')
        data = [settings.extv(pose.position) for pose in list(settings.eef_pose_array)]
        viz.visualize_3d(data=data, storeObj=settings.figdata, color='r', label="robot", units='m')
        data = [settings.extv(settings.mo.transformLeapToScene(settings.frames_adv[i].r.pPose.pose).position) for i in range(0, settings.BUFFER_LEN)]
        viz.visualize_3d(data=data, storeObj=settings.figdata, color='b', label="leap", units='m')
        viz.show()


    def button_play_move(self, e):
        self.play_status = 1
    def button_play_move2(self, e):
        self.play_status = -1
    def button_play_move3(self, e):
        self.play_status = 0
    def button_save(self, e):
        #self.recording = True
        #self.caller = RepeatableTimer(1, save_data, ())
        pass

    def save_data(self):
        print("saving data")
        self.recording = False
        n_sample = ""
        dir = self.dir_queue.pop(0)
        if not isdir(settings.LEARN_PATH+dir):
            from os import mkdir
            mkdir(settings.LEARN_PATH+dir)
        for i in range(0,200):
            if not isfile(settings.LEARN_PATH+dir+"/"+str(i)+".pkl"):
                n_sample = str(i)
                break

        with open(settings.LEARN_PATH+dir+"/"+str(n_sample)+'.pkl', 'wb') as output:
            pickle.dump(settings.frames_adv, output, pickle.HIGHEST_PROTOCOL)

        print("Gesture movement ", dir," saved")



    def play_method(self):
        while True:
            time.sleep(0.1)
            settings.HoldValue += self.play_status
            if self.play_status == 1 and settings.HoldValue > 100:
                settings.HoldValue = 99
                self.play_status = 0
            if self.play_status == -1 and settings.HoldValue < 0:
                settings.HoldValue = 1
                self.play_status = 0


    def button_confustion_mat_(self):
        NUM_SAMPLES = 5
        DELAY_BETW_SAMPLES = 0.5
        y_true = []
        y_pred = []

        poses_list = [p.NAME for p in settings.gd.r.poses]
        for n, i in enumerate(poses_list):
            for j in range(0,NUM_SAMPLES):
                self.lblCreateConfusionMatrixInfo.setText(i+" "+str(j))
                time.sleep(DELAY_BETW_SAMPLES)
                for m,g in enumerate(poses_list):
                    if settings.gd.r.poses[m].toggle:
                        y_true.append(n)
                        y_pred.append(m)

        self.lblCreateConfusionMatrixInfo.setText("Done (Saved as confusionmatrix.csv)")
        cm = confusion_matrix(y_true, y_pred)
        cm_ = np.vstack((poses_list,cm))
        poses_list.insert(0,"Confusion matrix")
        cm__ = np.hstack((np.array([poses_list]).transpose(),cm_))
        with open('confusionmatrix.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(cm__)



    def toggleMenu(self, state):
        if state:
            self.ViewState = True
        else:
            self.ViewState = False
    def goToInfo(self):
        settings.WindowState = 0
    def goToConfig(self):
        settings.WindowState = 1
    def goToMove(self):
        settings.WindowState = 2

    def gestures_goal_init_procedure(self):
        settings.md.gestures_goal_pose.position = settings.md.ENV['start']
        settings.md.gestures_goal_pose.orientation = settings.md.ENV['ori']

    # Switch the environment functions
    def switchEnvAbove(self):
        settings.md.ENV = settings.md.ENV_DAT['above']
        self.gestures_goal_init_procedure()
    def switchEnvWall(self):
        settings.md.ENV = settings.md.ENV_DAT['wall']
        self.gestures_goal_init_procedure()
    def switchEnvTable(self):
        settings.md.ENV = settings.md.ENV_DAT['table']
        self.gestures_goal_init_procedure()

    # Fixed orientation function
    def impAct4(self, state):
        if state:
            settings.FIXED_ORI_TOGGLE = True
        else:
            settings.FIXED_ORI_TOGGLE = False
    def print_path_trace(self, state):
        if state:
            settings.print_path_trace = True
        else:
            settings.print_path_trace = False

    def goScene(self, index):
        SCENES = settings.getSceneNames()
        settings.mo.make_scene(scene=SCENES[index])

    def onComboPlayNLiveChanged(self, text):
        if text=="Live hand":
            settings.md.Mode = 'live'
        elif text=="Play path":
            settings.md.Mode = 'play'
    def onComboPickPlayTrajChanged(self, text):
        settings.mo.changePlayPath(text)
    def onComboLiveModeChanged(self, text):
        settings.mo.changeLiveMode(text)
    def onInteractiveSceneChanged(self, text):
        if text == "Scene 1 Drawer":
            settings.mo.make_scene(scene='drawer')
            settings.md.ENV = settings.md.ENV_DAT['wall']
        elif text == "Scene 2 Pick/Place":
            settings.mo.make_scene(scene='pickplace')
            settings.md.ENV = settings.md.ENV_DAT['table']
        elif text == "Scene 3 Push button":
            settings.mo.make_scene(scene='pushbutton')
            settings.md.ENV = settings.md.ENV_DAT['table']
        elif text == "Scene 4 - 2 Pick/Place":
            settings.mo.make_scene(scene='pickplace2')
            settings.md.ENV = settings.md.ENV_DAT['table']
        else: raise Exception("Item not on a list!")
        self.gestures_goal_init_procedure()

    def paintEvent(self, e):
        ## Value updates
        self.w = settings.w = self.size().width()
        self.h = settings.h = self.size().height()

        ## Set all objects on page visible (the rest set invisible)
        for obj in self.AllVisibleObjects:
            # Every object that belongs to that group are conditioned by that group
            if obj.page == settings.WindowState and (self.ViewState if 'ViewState' in obj.view_group else True) and ((settings.md.Mode=='live') if 'live' in obj.view_group else True) and ((settings.md.Mode=='play') if 'play' in obj.view_group else True):
                #print("DEBUG", obj.NAME, " ", (obj.page == settings.WindowState), (self.ViewState if 'ViewState' in obj.view_group else True), " ", ((settings.md.Mode=='live') if 'live' in obj.view_group else True), " " ,((settings.md.Mode=='play') if 'play' in obj.view_group else True))
                obj.qt.setVisible(True)
            else:
                obj.qt.setVisible(False)

        ## Point given window
        if settings.WindowState == 0:
            self.mainPage(e)
        if settings.WindowState == 1:
            self.configPage(e)
        if settings.WindowState == 2:
            self.movePage(e)


        QMainWindow.paintEvent(self, e)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 3))

        if len(settings.frames_adv) > 10:
            for n in range(1,10):
                if settings.frames_adv[-n-1].r.visible and settings.frames_adv[-n].r.visible:
                    p1 = settings.mo.transformLeapToUIsimple(settings.frames_adv[-n].r.pPose.pose)
                    p2 = settings.mo.transformLeapToUIsimple(settings.frames_adv[-n-1].r.pPose.pose)
                    painter.setPen(QPen(Qt.red, p1.position.z))
                    painter.drawLine(p1.position.x, p1.position.y, p2.position.x, p2.position.y)
            last_pose = settings.frames_adv[-1].r.pPose
            pose_c = settings.mo.transformLeapToUIsimple(last_pose.pose)
            x_c,y_c = pose_c.position.x, pose_c.position.y

            rad = settings.gd.r.poses[settings.gd.r.POSES["pointing"]].time_visible*80
            if rad > 100:
                rad = 100

            painter.setPen(QPen(Qt.yellow, 4))
            painter.drawEllipse(x_c-rad/2,y_c-rad/2, rad, rad)

        #painter.setPen(QPen(Qt.black, 1))
        #painter.drawLine(self.w/2,self.h-20, self.w-RIGHT_MARGIN-ICON_SIZE, self.h-20)
        #painter.drawLine(self.w/2,self.h-20, self.w/2, START_PANEL_Y+ICON_SIZE)
        #self.lblStartAxis.setGeometry(self.w/2+5, self.h-70, 100, 50)
        #self.lblStartAxis.setText(str(settings.md.ENV['start']))

        ## Paint the bones structure
        painter.setPen(QPen(Qt.black, 1))
        if settings.frames:
            hand = settings.frames[-1].hands[0]
            palm = hand.palm_position
            wrist = hand.wrist_position
            elbow = hand.arm.elbow_position
            pose_palm = Pose()
            pose_palm.position = Point(palm[0], palm[1], palm[2])
            pose_wrist = Pose()
            pose_wrist.position = Point(wrist[0], wrist[1], wrist[2])
            pose_elbow = Pose()
            pose_elbow.position = Point(elbow[0], elbow[1], elbow[2])
            pose_palm_ = settings.mo.transformLeapToUIsimple(pose_palm)
            pose_wrist_ = settings.mo.transformLeapToUIsimple(pose_wrist)
            x, y = pose_palm_.position.x, pose_palm_.position.y
            x_, y_ = pose_wrist_.position.x, pose_wrist_.position.y
            painter.drawLine(x, y, x_, y_)
            pose_elbow_ = settings.mo.transformLeapToUIsimple(pose_elbow)
            x, y = pose_elbow_.position.x, pose_elbow_.position.y
            painter.drawLine(x, y, x_, y_)
            for finger in hand.fingers:
                for b in range(0, 4):
                    bone = finger.bone(b)
                    pose_bone_prev = Pose()
                    pose_bone_prev.position = Point(bone.prev_joint[0], bone.prev_joint[1], bone.prev_joint[2])
                    pose_bone_next = Pose()
                    pose_bone_next.position = Point(bone.next_joint[0], bone.next_joint[1], bone.next_joint[2])
                    pose_bone_prev_ = settings.mo.transformLeapToUIsimple(pose_bone_prev)
                    pose_bone_next_ = settings.mo.transformLeapToUIsimple(pose_bone_next)
                    x, y = pose_bone_prev_.position.x, pose_bone_prev_.position.y
                    x_, y_ = pose_bone_next_.position.x, pose_bone_next_.position.y
                    painter.drawLine(x, y, x_, y_)


    def movePage(self, e):
        pass

    def mainPage(self, e):
        w = self.w
        h = self.h
        qp = QPainter()
        qp.begin(self)
        textStatus = ""
        if settings.goal_pose and settings.goal_joints:
            textStatus += "eef: "+str(round(settings.eef_pose.position.x,2))+" "+str(round(settings.eef_pose.position.y,2))+" "+str(round(settings.eef_pose.position.z,2))
            textStatus += '\ng p:'+str(round(settings.goal_pose.position.x,2))+" "+str(round(settings.goal_pose.position.y,2))+" "+str(round(settings.goal_pose.position.z,2))
            textStatus += '\ng q:'+str(round(settings.goal_pose.orientation.x,2))+" "+str(round(settings.goal_pose.orientation.y,2))+" "+str(round(settings.goal_pose.orientation.z,2))+" "+str(round(settings.goal_pose.orientation.w,2))


        self.btnConf.setGeometry(LEFT_MARGIN+130, h-10-ICON_SIZE, ICON_SIZE*2,ICON_SIZE/2)
        self.btnSave.setGeometry(LEFT_MARGIN+130, h-10-ICON_SIZE*2, ICON_SIZE*2,ICON_SIZE/2)
        if self.recording:
            qp.setBrush(QBrush(Qt.red, Qt.SolidPattern))
            qp.drawEllipse(LEFT_MARGIN+130+ICON_SIZE*2, h-10-ICON_SIZE, ICON_SIZE/2,ICON_SIZE/2)
            qp.setBrush(QBrush(Qt.black, Qt.NoBrush))
        self.lblCreateConfusionMatrixInfo.setGeometry(LEFT_MARGIN+130, h-ICON_SIZE,ICON_SIZE*2,ICON_SIZE)
        self.lblStatus.setText(textStatus)

        if self.ViewState:
            self.lbl2.move(self.size().width()-RIGHT_MARGIN-40, 36)
            # up late
            if settings.frames_adv:
                # hand fingers
                for i in range(1,6):
                    if settings.gd.r.oc[i-1]:
                        qp.drawPixmap(w-RIGHT_MARGIN-100, START_PANEL_Y-20, ICON_SIZE, ICON_SIZE, QPixmap(settings.GRAPHICS_PATH+"hand"+str(i)+"open.png"))
                    else:
                        qp.drawPixmap(w-RIGHT_MARGIN-100, START_PANEL_Y-20, ICON_SIZE, ICON_SIZE, QPixmap(settings.GRAPHICS_PATH+"hand"+str(i)+"closed.png"))
                # arrows
                g = settings.gd.r.gests[settings.gd.r.GESTS["move_in_axis"]]
                X = w-RIGHT_MARGIN-100-ICON_SIZE
                Y = START_PANEL_Y-20
                if g.toggle[0] and g.move[0]:
                    qp.drawPixmap(X, Y, ICON_SIZE, ICON_SIZE, QPixmap(settings.GRAPHICS_PATH+"arrow_right.png"))
                if g.toggle[0] and not g.move[0]:
                    qp.drawPixmap(X, Y, ICON_SIZE, ICON_SIZE, QPixmap(settings.GRAPHICS_PATH+"arrow_left.png"))
                if g.toggle[1] and g.move[1]:
                    qp.drawPixmap(X, Y, ICON_SIZE, ICON_SIZE, QPixmap(settings.GRAPHICS_PATH+"arrow_up.png"))
                if g.toggle[1] and not g.move[1]:
                    qp.drawPixmap(X, Y, ICON_SIZE, ICON_SIZE, QPixmap(settings.GRAPHICS_PATH+"arrow_down.png"))
                if g.toggle[2] and g.move[2]:
                    qp.drawPixmap(X, Y, ICON_SIZE, ICON_SIZE, QPixmap(settings.GRAPHICS_PATH+"arrow_front.png"))
                if g.toggle[2] and not g.move[2]:
                    qp.drawPixmap(X, Y, ICON_SIZE, ICON_SIZE, QPixmap(settings.GRAPHICS_PATH+"arrow_back.png"))

            # left lane
            POSE_FILE_IMAGES = [p.filename for p in settings.gd.r.poses]
            GEST_FILE_IMAGES = [p.filename for p in settings.gd.r.gests][0:4]
            for n, i in enumerate(POSE_FILE_IMAGES):
                qp.drawPixmap(LEFT_MARGIN, START_PANEL_Y+n*ICON_SIZE, ICON_SIZE, ICON_SIZE, QPixmap(settings.GRAPHICS_PATH+i))
                if settings.gd.r.poses[n].toggle:
                    qp.drawRect(LEFT_MARGIN,START_PANEL_Y+(n)*ICON_SIZE, ICON_SIZE, ICON_SIZE)
                qp.drawLine(LEFT_MARGIN+ICON_SIZE+2, START_PANEL_Y+(n+1)*ICON_SIZE, LEFT_MARGIN+ICON_SIZE+2, START_PANEL_Y+(n+1)*ICON_SIZE-settings.gd.r.poses[n].prob*ICON_SIZE)
            for n, i in enumerate(GEST_FILE_IMAGES):
                qp.drawPixmap(LEFT_MARGIN, START_PANEL_Y+(n+len(POSE_FILE_IMAGES))*ICON_SIZE, ICON_SIZE, ICON_SIZE, QPixmap(settings.GRAPHICS_PATH+i))
                if settings.gd.r.gests[n].toggle:
                    qp.drawRect(LEFT_MARGIN,START_PANEL_Y+(n+len(POSE_FILE_IMAGES))*ICON_SIZE, ICON_SIZE, ICON_SIZE)
                qp.drawLine(LEFT_MARGIN+ICON_SIZE+2, START_PANEL_Y+(n+1+len(POSE_FILE_IMAGES))*ICON_SIZE, LEFT_MARGIN+ICON_SIZE+2, START_PANEL_Y+(n+1+len(POSE_FILE_IMAGES))*ICON_SIZE-settings.gd.r.gests[n].prob*ICON_SIZE)
            for n, i in enumerate(self.lblGesturesPosesNamesObj):
                i.setVisible(True)
                i.move(LEFT_MARGIN+ICON_SIZE+5, START_PANEL_Y+n*ICON_SIZE)
            if settings.pymcout is not None:
                n_ = settings.pymcout
                qp.drawPixmap(LEFT_MARGIN+ICON_SIZE+10,START_PANEL_Y+(n_)*ICON_SIZE, ICON_SIZE, ICON_SIZE, QPixmap(settings.GRAPHICS_PATH+"arrow_left.png"))
            # circ options
            g = settings.gd.r.gests[settings.gd.r.GESTS["circ"]]
            if g.toggle:
                X = LEFT_MARGIN+130
                Y = START_PANEL_Y+len(POSE_FILE_IMAGES)*ICON_SIZE
                ARRL = 10  # Arrow length
                radius_2mm = g.radius/2
                qp.drawEllipse(X,Y, radius_2mm, radius_2mm)
                rh = radius_2mm/2
                if g.clockwise == True:
                    qp.drawLine(X, Y+rh, X-ARRL, Y+ARRL+rh)
                    qp.drawLine(X, Y+rh, X+ARRL, Y+ARRL+rh)
                else:
                    qp.drawLine(X, Y+rh, X-ARRL, Y-ARRL+rh)
                    qp.drawLine(X, Y+rh, X+ARRL, Y-ARRL+rh)

            if settings.md.Mode == 'play':
                if settings.gd.l.poses[settings.gd.l.POSES['grab']].toggle:
                    qp.drawPixmap(w/2, 50, 20, 20, QPixmap(settings.GRAPHICS_PATH+"hold.png"))
                    if settings.HoldPrevState == False:
                        settings.HoldAnchor = settings.HoldValue - settings.frames_adv[-1].l.pPose.pose.position.x/len(settings.sp[settings.md.PickedPath].poses)
                    #settings.HoldValue = settings.frames_adv[-1].l.pPose.pose.position.x/2 + 100
                    settings.HoldValue = settings.HoldAnchor + settings.frames_adv[-1].l.pPose.pose.position.x/len(settings.sp[settings.md.PickedPath].poses)
                    settings.HoldValue = settings.HoldAnchor + settings.frames_adv[-1].l.pPose.pose.position.x/2
                    if settings.HoldValue > 100: settings.HoldValue = 100
                    if settings.HoldValue < 0: settings.HoldValue = 0
                settings.HoldPrevState = settings.gd.l.poses[settings.gd.l.POSES['grab']].toggle
                diff_pose_progress = 100/len(settings.sp[settings.md.PickedPath].poses)
                for i in range(0, len(settings.sp[settings.md.PickedPath].poses)):
                    qp.fillRect(LEFT_MARGIN+diff_pose_progress*i*((w-40.0)/100.0), 30, 2, 20, Qt.black)
                qp.fillRect(LEFT_MARGIN+diff_pose_progress*settings.currentPose*((w-40.0)/100.0)+2, 35, diff_pose_progress*((w-40.0)/100.0), 10, Qt.red)
                qp.drawRect(LEFT_MARGIN, 30, w-40, 20)
                qp.fillRect(LEFT_MARGIN+settings.HoldValue*((w-40.0)/100.0), 30, 10, 20, Qt.black)

            # right lane
            for n, i in enumerate(self.lblObservationPosesNamesObj):
                i.setVisible(True)
                i.move(w-RIGHT_MARGIN, START_PANEL_Y+n*ICON_SIZE/2)
            if settings.frames and settings.frames_adv:
                ValuesArr = []
                ToggleArr = []
                ValuesArr.append(round(settings.frames_adv[-1].r.conf,2))
                oc = [round(i,2) for i in settings.frames_adv[-1].r.OC]
                ValuesArr.extend(oc)
                ValuesArr.append(round(settings.frames_adv[-1].r.TCH12, 2))
                ValuesArr.append(round(settings.frames_adv[-1].r.TCH23, 2))
                ValuesArr.append(round(settings.frames_adv[-1].r.TCH34, 2))
                ValuesArr.append(round(settings.frames_adv[-1].r.TCH45, 2))
                ValuesArr.append(round(settings.frames_adv[-1].r.TCH13, 2))
                ValuesArr.append(round(settings.frames_adv[-1].r.TCH14, 2))
                ValuesArr.append(round(settings.frames_adv[-1].r.TCH15, 2))
                vel = [round(i,2) for i in settings.frames_adv[-1].r.vel]
                ValuesArr.extend(vel)
                rr = settings.frames_adv[-1].r.rotRaw
                ValuesArr.append(round(rr[0], 2))
                ValuesArr.append(round(rr[1], 2))
                ValuesArr.append(round(rr[2], 2))

                ToggleArr.append(settings.gd.r.conf)
                ToggleArr.extend(settings.gd.r.oc)
                ToggleArr.append(settings.gd.r.tch12)
                ToggleArr.append(settings.gd.r.tch23)
                ToggleArr.append(settings.gd.r.tch34)
                ToggleArr.append(settings.gd.r.tch45)
                ToggleArr.append(settings.gd.r.tch13)
                ToggleArr.append(settings.gd.r.tch14)
                ToggleArr.append(settings.gd.r.tch15)
                ToggleArr.extend(settings.gd.r.gests[settings.gd.r.GESTS["move_in_axis"]].toggle)
                ToggleArr.extend(settings.gd.r.gests[settings.gd.r.GESTS["rotation_in_axis"]].toggle)

                for n, i in enumerate(self.lblObservationPosesValuesObj):
                    i.move(w-RIGHT_MARGIN, START_PANEL_Y+(n+0.5)*ICON_SIZE/2)
                    i.setText(str(ValuesArr[n]))
                    qp.drawLine(w-RIGHT_MARGIN-5+ValuesArr[n]*ICON_SIZE, START_PANEL_Y+(n+1)*ICON_SIZE/2, w-RIGHT_MARGIN-5, START_PANEL_Y+(n+1)*ICON_SIZE/2)
                    if ToggleArr[n]:
                        qp.drawRect(w-RIGHT_MARGIN-5, START_PANEL_Y+(n)*ICON_SIZE/2+5, ICON_SIZE, ICON_SIZE/2-10)
            # orientation
            if self.cursorEnabled():
                roll, pitch, yaw = tf.transformations.euler_from_quaternion([settings.frames_adv[-1].r.pPose.pose.orientation.x, settings.frames_adv[-1].r.pPose.pose.orientation.y, settings.frames_adv[-1].r.pPose.pose.orientation.z, settings.frames_adv[-1].r.pPose.pose.orientation.w])
                x = np.cos(yaw)*np.cos(pitch)
                y = np.sin(yaw)*np.cos(pitch)
                z = np.sin(pitch)
                last_pose = settings.frames_adv[-1].r.pPose

                last_pose_ = settings.mo.transformLeapToUIsimple(last_pose.pose)
                x_c,y_c = last_pose_.position.x, last_pose_.position.y
                qp.setPen(QPen(Qt.blue, 4))
                qp.drawLine(x_c, y_c, x_c+y*2*ICON_SIZE, y_c-z*2*ICON_SIZE)


        qp.end()

    def configPage(self, e):
        w = self.size().width()
        h = self.size().height()
        # computation part
        qp = QPainter()
        qp.begin(self)
        BarW = (w-LEFT_MARGIN-RIGHT_MARGIN)/settings.NumConfigBars[1]
        BarH = (h-START_PANEL_Y-BOTTOM_MARGIN)/settings.NumConfigBars[0]
        X_START = [int(LEFT_MARGIN+i*BarW+BarMargin) for i in range(0,settings.NumConfigBars[1])]
        Y_START = [int(START_PANEL_Y+BarMargin+i*BarH) for i in range(0, settings.NumConfigBars[0])]
        X_LEN = int(BarW-BarMargin)
        Y_LEN = int(BarH-BarMargin)
        X_END = np.add(X_START,int(BarW-BarMargin))
        Y_END = np.add(Y_START,int(BarH-BarMargin))
        X_BOUND = tuple(zip(X_START, X_END))
        Y_BOUND = tuple(zip(Y_START, Y_END))
        # picking part
        if not self.cursorEnabled():
            return
        last_pose = settings.frames_adv[-1].r.pPose
        last_pose_ = settings.mo.transformLeapToUIsimple(last_pose.pose)
        x,y = last_pose_.position.x, last_pose_.position.y
        x_ = (np.min(X_BOUND, 1) < x) & (x < np.max(X_BOUND, 1))
        y_ = (np.min(Y_BOUND, 1) < y) & (y < np.max(Y_BOUND, 1))
        prevPicked = deepcopy(self.pickedSolution)
        for n, i in enumerate(x_):
            for m, j in enumerate(y_):
                valueInt, valueStr = self.readConfigPageValues(m, n)
                self.lblConfNamesObj[n+m*settings.NumConfigBars[1]].move(X_START[n], Y_START[m]-25)
                self.lblConfNamesObj[n+m*settings.NumConfigBars[1]].setVisible(True)
                self.lblConfValuesObj[n+m*settings.NumConfigBars[1]].move(X_START[n]+BarW-40-BarMargin, Y_START[m]-25)
                self.lblConfValuesObj[n+m*settings.NumConfigBars[1]].setVisible(True)
                self.lblConfValuesObj[n+m*settings.NumConfigBars[1]].setText(valueStr)
                qp.drawRect(X_START[n], Y_START[m], X_LEN, Y_LEN)
                picked = (i and j)
                if (picked == True) and (picked == self.pickedSolution[m, n]):
                    self.pickedTime[m, n] += 0.1
                    if self.pickedTime[m, n] > 2.0:
                        self.saveConfigPageValues(m,n,abs(Y_START[m] - y))
                        self.pickedTime[m, n] = 0
                else:
                    self.pickedTime[m, n] = 0
                self.pickedSolution[m, n] = picked
                qp.fillRect(X_START[n], Y_START[m], X_LEN, valueInt, QColor('black'))


        qp.end()

    def saveConfigPageValues(self, m, n, value):
        # Items on the list
        # ['Gripper open', 'Applied force', 'Work reach', 'Shift start x', 'Speed', 'Scene change', 'Mode', 'Path']
        settings.VariableValues[m, n] = value
        if m == 0:
            if   n == 0:
                settings.md.gripper = value
            elif n == 1:
                settings.md.applied_force = value
            elif n == 2:
                settings.md.SCALE = value/50
            elif n == 3:
                settings.md.ENV['start'].x = value/100
        elif m == 1:
            if   n == 0:
                settings.md.speed = value
            elif n == 1:
                scenes = settings.getSceneNames()
                v = int(len(scenes)/125. * value)
                if v >= len(scenes):
                    v = len(scenes)-1
                settings.mo.make_scene(scene=scenes[v])
            elif n == 2:
                modes = settings.getModes()
                v = int(len(modes)/125. * value)
                if v >= len(modes):
                    v = len(modes)-1
                settings.md.Mode = modes[v]
            elif n == 3:
                paths = settings.getPathNames()
                v = int(len(paths)/125. * value)
                if v >= len(paths):
                    v = len(paths)-1
                settings.mo.changePlayPath(path_=paths[v])

    def readConfigPageValues(self, m, n):
        string = None
        if m == 0:
            if   n == 0:
                value = settings.md.gripper
            elif n == 1:
                value = settings.md.applied_force
            elif n == 2:
                value = settings.md.SCALE*50
            elif n == 3:
                value = settings.md.ENV['start'].x*100
        elif m == 1:
            if   n == 0:
                value = settings.md.speed
            elif n == 1:
                if settings.scene:
                    scenes = settings.getSceneNames()
                    value = scenes.index(settings.scene.NAME)
                    string = settings.scene.NAME
                else: value = 0
            elif n == 2:
                modes = settings.getModes()
                value = modes.index(settings.md.Mode)
                string = settings.md.Mode
            elif n == 3:
                value = settings.md.PickedPath
                paths = settings.getPathNames()
                string = paths[value]

        settings.VariableValues[m, n] = float(value)
        if not string:
            string = str(value)
        return value, string


    def cursorEnabled(self):
        ''' Checks if enough samples are made
        '''
        if len(settings.frames_adv) >= 10 and settings.frames_adv[-1].r:
            return True
        return False

    def timerEvent(self, event):
        if not settings.frames_adv:
            return
        fa = settings.frames_adv[-1]
        for i in settings.gd.r.gests[0:4]: # circ, swipe, pin, touch
            if i.time_visible > 0:
                i.time_visible -= 0.1
            else:
                i.toggle = False
        if fa.r.visible == False:
            for i in settings.gd.r.gests:
                i.toggle = False if isinstance(i.toggle, bool) else [False] * len(i.toggle)
            for i in settings.gd.r.poses:
                i.toggle = False
        if fa.l.visible == False:
            for i in settings.gd.l.gests:
                i.toggle = False if isinstance(i.toggle, bool) else [False] * len(i.toggle)
            for i in settings.gd.l.poses:
                i.toggle = False

        self.step = self.step + 1
        self.update()

    def mapQtKey(self, key):
        mapDict = {
            '0': Qt.Key_0 ,
            '1': Qt.Key_1 ,
            '2': Qt.Key_2 ,
            '3': Qt.Key_3 ,
            '4': Qt.Key_4 ,
            '5': Qt.Key_5 ,
            '6': Qt.Key_6 ,
            '7': Qt.Key_7 ,
            '8': Qt.Key_8 ,
            '9': Qt.Key_9 ,
            'a': Qt.Key_A ,
            'b': Qt.Key_B ,
            'c': Qt.Key_C ,
            'd': Qt.Key_D ,
            'e': Qt.Key_E ,
            'f': Qt.Key_F ,
            'g': Qt.Key_G ,
            'h': Qt.Key_H ,
            'i': Qt.Key_I ,
            'j': Qt.Key_J ,
            'k': Qt.Key_K ,
            'l': Qt.Key_L ,
            'm': Qt.Key_M ,
            'n': Qt.Key_N ,
            'o': Qt.Key_O ,
            'p': Qt.Key_P ,
            'q': Qt.Key_Q ,
            'r': Qt.Key_R ,
            's': Qt.Key_S ,
            't': Qt.Key_T ,
            'u': Qt.Key_U ,
            'v': Qt.Key_V ,
            'w': Qt.Key_W ,
            'x': Qt.Key_X ,
            'y': Qt.Key_Y ,
            'z': Qt.Key_Z
            }
        return mapDict[key]

class ObjectQt():
    def __init__(self, NAME=None, qt=None, page=None, view_group=['ViewState']):
        ''' Informations about app objects
        Parameters:
            NAME (Str): Name of object
            qr (Qt Object): Interaction variable
            page (Int): On what page this object is
            view_group (Str[]): In which groups this object belongs
        '''
        self.NAME = NAME
        self.qt = qt
        self.page = page
        self.view_group = view_group

class RepeatableTimer(object):
    def __init__(self, interval, function, args=[], kwargs={}):
        self._interval = interval
        self._function = function
        self._args = args
        self._kwargs = kwargs
    def start(self):
        t = Timer(self._interval, self._function, *self._args, **self._kwargs)
        t.start()

def main():
    while not settings.mo:
        time.sleep(2)
        print("[Interface] moveit not connected")
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    settings.init()
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    print("UI Exit")
