#!/usr/bin/env python3.8
""" Launches Interface Application
    - Loads config from include/custom_settings/application.yaml
"""

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys, random, time, csv, yaml
import settings
if __name__ == '__main__': settings.init()
from os_and_utils.nnwrapper import NNWrapper
import os_and_utils.move_lib as ml
if __name__ == '__main__': ml.init()
import gestures_lib as gl
if __name__ == '__main__': gl.init()
import os_and_utils.scenes as sl
if __name__ == '__main__': sl.init()

import numpy as np
from threading import Thread, Timer
from copy import deepcopy
from os.path import expanduser, isfile, isdir
from os_and_utils.transformations import Transformations as tfm
from os_and_utils.utils import point_by_ratio
try:
    from sklearn.metrics import confusion_matrix
except ModuleNotFoundError:
    print("[WARNING*] Sklearn library not installed -> confusion_matrix won't be plotted!")

import rospy
# ros msg classes
from geometry_msgs.msg import Quaternion, Pose, Point
from mirracle_gestures.srv import ChangeNetwork, SaveHandRecord

class Example(QMainWindow):

    def __init__(self):
        super(Example, self).__init__()

        with open(settings.paths.custom_settings_yaml+"recording.yaml", 'r') as stream:
            recording_data_loaded = yaml.safe_load(stream)
        with open(settings.paths.custom_settings_yaml+"application.yaml", 'r') as stream:
            app_data_loaded = yaml.safe_load(stream)
        global LEFT_MARGIN, RIGHT_MARGIN, BOTTOM_MARGIN, ICON_SIZE, START_PANEL_Y, START_PANEL_Y_GESTURES, BarMargin
        LEFT_MARGIN = app_data_loaded['LEFT_MARGIN']
        RIGHT_MARGIN = app_data_loaded['RIGHT_MARGIN']
        BOTTOM_MARGIN = app_data_loaded['BOTTOM_MARGIN']
        ICON_SIZE = app_data_loaded['ICON_SIZE']
        START_PANEL_Y = app_data_loaded['START_PANEL_Y']
        START_PANEL_Y_GESTURES = START_PANEL_Y + 80
        BarMargin = app_data_loaded['BarMargin']

        LeftPanelMaxIterms = app_data_loaded['LeftPanelMaxIterms']
        RightPanelMaxIterms = app_data_loaded['RightPanelMaxIterms']

        self.setMinimumSize(QSize(500, 400)) # Minimum window size
        self.lbl1 = QLabel('Left Hand', self)
        self.lbl1.setGeometry(20, 36, 150, 50)
        self.lbl2 = QLabel('Right Hand', self)
        self.lbl2.setGeometry(self.size().width()-140, 36, 100, 50)
        self.lbl3 = QLabel('Gestures', self)
        self.lbl3.setGeometry(20, START_PANEL_Y_GESTURES-30, 150, 50)
        self.lbl4 = QLabel('Gestures', self)
        self.lbl4.setGeometry(self.size().width()-140, START_PANEL_Y_GESTURES-30, 100, 50)

        ## View Configuration App
        settings.WindowState = 0
        self.GesturesViewState = False
        self.MoveViewState = True
        self.OneTimeTurnOnGesturesViewStateOnLeapMotionSignIn = True

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

        ## Right panel initialization (Observations)
        self.lblRightPanelNamesObj = []
        self.lblRightPanelValuesObj = []
        for i in range(0, RightPanelMaxIterms):
            self.lblRightPanelValuesObj.append(QLabel("", self))
            self.lblRightPanelNamesObj.append(QLabel("", self))
        for i in self.lblRightPanelValuesObj:
            i.setVisible(False)

        self.comboPlayNLive = QComboBox(self)
        self.comboPlayNLive.addItem("Play path")
        self.comboPlayNLive.activated[str].connect(self.onComboPlayNLiveChanged)
        self.comboPlayNLive.setGeometry(LEFT_MARGIN+130, START_PANEL_Y-10,ICON_SIZE*2,int(ICON_SIZE/2))

        self.comboPickPlayTraj = QComboBox(self)
        for path in sl.paths:
            self.comboPickPlayTraj.addItem(path.name)
        self.comboPickPlayTraj.activated[str].connect(self.onComboPickPlayTrajChanged)
        self.comboPickPlayTraj.setGeometry(LEFT_MARGIN+130+ICON_SIZE*2, START_PANEL_Y-10,ICON_SIZE*2,int(ICON_SIZE/2))

        self.comboLiveMode = QComboBox(self)
        self.comboLiveMode.addItem("Default")
        self.comboLiveMode.addItem("Interactive")
        self.comboLiveMode.addItem("Gesture based")
        self.comboLiveMode.activated[str].connect(self.onComboLiveModeChanged)
        self.comboLiveMode.setGeometry(LEFT_MARGIN+130+ICON_SIZE*2, START_PANEL_Y-10,ICON_SIZE*2,int(ICON_SIZE/2))
        ## Control of the movement exectution
        self.btnPlayMove = QPushButton('Forward', self)
        self.btnPlayMove.clicked.connect(self.button_play_move)
        self.btnPlayMove.setGeometry(LEFT_MARGIN+130+ICON_SIZE*4, START_PANEL_Y-10,ICON_SIZE,int(ICON_SIZE/2))
        self.btnPlayMove2 = QPushButton('Backward', self)
        self.btnPlayMove2.clicked.connect(self.button_play_move2)
        self.btnPlayMove2.setGeometry(LEFT_MARGIN+130+ICON_SIZE*5, START_PANEL_Y-10,ICON_SIZE,int(ICON_SIZE/2))
        self.btnPlayMove3 = QPushButton('Stop', self)
        self.btnPlayMove3.clicked.connect(self.button_play_move3)
        self.btnPlayMove3.setGeometry(LEFT_MARGIN+130+ICON_SIZE*6, START_PANEL_Y-10,ICON_SIZE,int(ICON_SIZE/2))

        self.recording = False # Bool if recording is happening
        self.REC_TIME = recording_data_loaded['Length'] # [s]
        self.dir_queue = []

        self.lblStatus = QLabel('Status bar', self)
        self.lblStatus.setGeometry(LEFT_MARGIN+130, START_PANEL_Y+32, 200, 100)

        self.comboInteractiveSceneChanges = QComboBox(self)
        self.comboInteractiveSceneChanges.addItem("Scene 1 Drawer")
        self.comboInteractiveSceneChanges.addItem("Scene 2 Pick/Place")
        self.comboInteractiveSceneChanges.addItem("Scene 3 Push button")
        self.comboInteractiveSceneChanges.addItem("Scene 4 - 2 Pick/Place")
        self.comboInteractiveSceneChanges.activated[str].connect(self.onInteractiveSceneChanged)
        self.comboInteractiveSceneChanges.setGeometry(LEFT_MARGIN+130+ICON_SIZE*4, START_PANEL_Y-10,ICON_SIZE*2,int(ICON_SIZE/2))

        # Bottom
        self.btnRecordActivate = QPushButton('Keyboard recording', self)
        self.btnRecordActivate.clicked.connect(self.record_with_keys)
        self.btnRecordActivate.setGeometry(LEFT_MARGIN+130, START_PANEL_Y+30,ICON_SIZE*2,int(ICON_SIZE/2))

        # Move Page
        lbls = ['Pos. X:', 'Pos. Y:', 'Pos. Z:', 'Ori. X:', 'Ori. Y:', 'Ori. Z:', 'Ori. W:', 'Gripper:']
        lblsVals = ['0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '1.0', '0.0']
        self.movePageGoPoseLabels = []
        self.movePageGoPoseEdits = []
        for i in range(0,7):
            self.movePageGoPoseLabels.append(QLabel(self))
            self.movePageGoPoseLabels[-1].setText(lbls[i])
            self.movePageGoPoseLabels[-1].move(LEFT_MARGIN+20, START_PANEL_Y+i*32)
            self.movePageGoPoseEdits.append(QLineEdit(self))
            self.movePageGoPoseEdits[-1].move(LEFT_MARGIN+80, START_PANEL_Y+i*32)
            self.movePageGoPoseEdits[-1].resize(200, 32)
            self.movePageGoPoseEdits[-1].setText(lblsVals[i])
        self.movePageGoPoseButton = QPushButton("Go To Pose", self)
        self.movePageGoPoseButton.clicked.connect(self.go_to_pose_button)
        self.movePageGoPoseButton.move(LEFT_MARGIN+80, START_PANEL_Y+7*32)
        ''' Gripper control '''
        i = 7
        self.movePageGoPoseLabels.append(QLabel(self))
        self.movePageGoPoseLabels[-1].setText(lbls[i])
        self.movePageGoPoseLabels[-1].move(LEFT_MARGIN+20, START_PANEL_Y+(i+1)*32)
        self.movePageGoPoseEdits.append(QLineEdit(self))
        self.movePageGoPoseEdits[-1].move(LEFT_MARGIN+80, START_PANEL_Y+(i+1)*32)
        self.movePageGoPoseEdits[-1].resize(200, 32)
        self.movePageGoPoseEdits[-1].setText(lblsVals[i])
        self.movePageGoGripperButton = QPushButton("Actuate gripper", self)
        self.movePageGoGripperButton.clicked.connect(self.actuate_gripper_button)
        self.movePageGoGripperButton.move(LEFT_MARGIN+80, START_PANEL_Y+9*32)

        self.movePageOpenGripperButton = QPushButton("Open gripper", self)
        self.movePageOpenGripperButton.clicked.connect(self.open_gripper_button)
        self.movePageOpenGripperButton.move(LEFT_MARGIN+80, START_PANEL_Y+10*32)
        self.movePageCloseGripperButton = QPushButton("Close gripper", self)
        self.movePageCloseGripperButton.clicked.connect(self.close_gripper_button)
        self.movePageCloseGripperButton.move(LEFT_MARGIN+80+100, START_PANEL_Y+10*32)



        self.movePageUseEnvAboveButton = QPushButton('Above env.', self)
        self.movePageUseEnvAboveButton.clicked.connect(self.movePageUseEnvAboveButtonFun)
        self.movePageUseEnvAboveButton.setGeometry(LEFT_MARGIN+300, START_PANEL_Y+100,ICON_SIZE*2,int(ICON_SIZE/2))
        self.movePageUseEnvWallButton = QPushButton('Wall env.', self)
        self.movePageUseEnvWallButton.clicked.connect(self.movePageUseEnvWallButtonFun)
        self.movePageUseEnvWallButton.setGeometry(LEFT_MARGIN+300, START_PANEL_Y+140,ICON_SIZE*2,int(ICON_SIZE/2))
        self.movePageUseEnvTableButton = QPushButton('Table env.', self)
        self.movePageUseEnvTableButton.clicked.connect(self.movePageUseEnvTableButtonFun)
        self.movePageUseEnvTableButton.setGeometry(LEFT_MARGIN+300, START_PANEL_Y+180,ICON_SIZE*2,int(ICON_SIZE/2))
        self.movePagePoseNowButton = QPushButton('Set current pose', self)
        self.movePagePoseNowButton.clicked.connect(self.movePagePoseNowButtonFun)
        self.movePagePoseNowButton.setGeometry(LEFT_MARGIN+300, START_PANEL_Y+20,ICON_SIZE*2,int(ICON_SIZE/2))

        self.timer = QBasicTimer()
        self.timer.start(100, self)
        self.step = 0

        self.play_status = 0

        menubar = self.menuBar()
        viewMenu = menubar.addMenu('View')
        pageMenu = menubar.addMenu('Page')
        configMenu = menubar.addMenu('Robot config.')
        sceneMenu = menubar.addMenu('Scene')
        testingMenu = menubar.addMenu('Testing')
        leapmotionMenu = menubar.addMenu('Gestures')

        ## Menu items -> View options
        viewOptionsAction = QAction('View gestures data', self, checkable=True)
        viewOptionsAction.setStatusTip('View gestures data')
        viewOptionsAction.setChecked(False)
        viewOptionsAction.triggered.connect(self.toggleViewGesturesMenu)
        viewMoveOptionsAction = QAction('View move data', self, checkable=True)
        viewMoveOptionsAction.setStatusTip('View move data')
        viewMoveOptionsAction.setChecked(True)
        viewMoveOptionsAction.triggered.connect(self.toggleViewMoveMenu)
        ## Menu items -> Go to page
        viewGoToInfoAction = QAction('Info page', self)
        viewGoToInfoAction.setStatusTip('Info page')
        viewGoToInfoAction.triggered.connect(self.goToInfo)
        viewGoToControlAction = QAction('Control page', self)
        viewGoToControlAction.setStatusTip('Control page (beta)')
        viewGoToControlAction.triggered.connect(self.goToConfig)
        viewGoToMoveAction = QAction('Move page', self)
        viewGoToMoveAction.setStatusTip('Move page')
        viewGoToMoveAction.triggered.connect(self.goToMove)

        # The environment
        impMenu = QMenu('Environment', self)
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
        fixedOriAct = QAction('Fixed (default as chosen env.)', self, checkable=True)
        fixedOriAct.triggered.connect(self.fixedOriAct)
        impMenu2.addAction(fixedOriAct)
        print_path_trace_action = QAction('Print path trace', self, checkable=True)
        print_path_trace_action.triggered.connect(self.print_path_trace)
        print_path_trace_action.checked = False

        record_with_keys_action = QAction('Record train data with keyboard keys', self, checkable=True)
        record_with_keys_action.triggered.connect(self.record_with_keys)
        download_networks_action = QAction('Download networks from gdrive', self)
        download_networks_action.triggered.connect(self.download_networks_gdrive)
        self.network_menu = QMenu('Pick detection network', self)
        self.networks = gl.gd.get_networks()
        self.network_actions = []
        for index,network in enumerate(self.networks):
            action = QAction(network, self)
            action.triggered.connect(
                lambda checked, network=network: self.changeNetwork(network))
            self.network_actions.append(action)
            self.network_menu.addAction(action)
        self.lblInfo = QLabel("", self)
        self.confusion_mat_action = QAction('Test gestures', self)
        self.confusion_mat_action.setToolTip('Plus Generate <b>Confusion</b> matrix')
        self.confusion_mat_action.triggered.connect(self.confustion_mat)


        SCENES = sl.scenes.names()
        for index, SCENE in enumerate(SCENES):
            action = QAction('Scene '+str(index)+' '+SCENE, self)
            action.triggered.connect(
                lambda checked, index=index: self.goScene(index))
            sceneMenu.addAction(action)

        initTestAction = QAction('Initialization test', self)
        initTestAction.triggered.connect(self.thread_testInit)
        tableTestAction = QAction('Table test', self)
        tableTestAction.triggered.connect(self.thread_testMovements)
        inputTestAction = QAction('Test by input', self)
        inputTestAction.triggered.connect(self.thread_testMovementsInput)
        inputPlotJointsAction = QAction('Plot joints path now', self)
        inputPlotJointsAction.triggered.connect(self.thread_inputPlotJointsAction)
        inputPlotPosesAction = QAction('Plot poses path now', self)
        inputPlotPosesAction.triggered.connect(self.thread_inputPlotPosesAction)

        ## Add actions to the menu
        viewMenu.addAction(viewOptionsAction)
        viewMenu.addAction(viewMoveOptionsAction)
        pageMenu.addAction(viewGoToInfoAction)
        pageMenu.addAction(viewGoToControlAction)
        pageMenu.addAction(viewGoToMoveAction)
        configMenu.addMenu(impMenu)
        configMenu.addMenu(impMenu2)
        configMenu.addAction(print_path_trace_action)
        testingMenu.addAction(tableTestAction)
        testingMenu.addAction(initTestAction)
        testingMenu.addAction(inputTestAction)
        testingMenu.addAction(inputPlotJointsAction)
        testingMenu.addAction(inputPlotPosesAction)
        testingMenu.addAction(self.confusion_mat_action)
        leapmotionMenu.addAction(record_with_keys_action)
        leapmotionMenu.addMenu(self.network_menu)
        leapmotionMenu.addAction(download_networks_action)

        thread = Thread(target = self.play_method)
        thread.start()

        self.setGeometry(app_data_loaded['WindowXY'][0],app_data_loaded['WindowXY'][1], app_data_loaded['WindowSize'][0], app_data_loaded['WindowSize'][1])
        self.setWindowTitle('Interface')
        self.show()

        ''' Initialize Visible Objects Array
            - AllVisibleObjects array specifies each object a specific view_group
            - Object visibility management across application
        '''
        self.AllVisibleObjects = []
        self.AllVisibleObjects.append(ObjectQt('lblStatus',self.lblStatus,0,view_group=['MoveViewState']))
        for n,obj in enumerate(self.lblRightPanelNamesObj):
            self.AllVisibleObjects.append(ObjectQt('lblRightPanelNamesObj'+str(n),obj,0,view_group=['GesturesViewState']))
        self.AllVisibleObjects.append(ObjectQt('lbl1',self.lbl1,0,view_group=['GesturesViewState']))
        self.AllVisibleObjects.append(ObjectQt('lbl2',self.lbl2,0,view_group=['GesturesViewState']))
        self.AllVisibleObjects.append(ObjectQt('lbl3',self.lbl3,0,view_group=['GesturesViewState']))
        self.AllVisibleObjects.append(ObjectQt('lbl4',self.lbl4,0,view_group=['GesturesViewState']))
        self.AllVisibleObjects.append(ObjectQt('lblInfo',self.lbl2,0,view_group=['GesturesViewState']))
        self.AllVisibleObjects.append(ObjectQt('comboPlayNLive',self.comboPlayNLive,0,view_group=['MoveViewState']))

        self.AllVisibleObjects.append(ObjectQt('comboPickPlayTraj',self.comboPickPlayTraj,0,view_group=['MoveViewState', 'play']))
        self.AllVisibleObjects.append(ObjectQt('btnPlayMove' ,self.btnPlayMove, 0,view_group=['MoveViewState', 'play']))
        self.AllVisibleObjects.append(ObjectQt('btnPlayMove2',self.btnPlayMove2,0,view_group=['MoveViewState', 'play']))
        self.AllVisibleObjects.append(ObjectQt('btnPlayMove3',self.btnPlayMove3,0,view_group=['MoveViewState', 'play']))

        self.AllVisibleObjects.append(ObjectQt('comboLiveMode',self.comboLiveMode,0,view_group=['MoveViewState', 'live']))
        self.AllVisibleObjects.append(ObjectQt('comboInteractiveSceneChanges',self.comboInteractiveSceneChanges,0,view_group=['MoveViewState', 'live']))

        self.AllVisibleObjects.append(ObjectQt('btnRecordActivate',self.btnRecordActivate,0,view_group=['MoveViewState']))

        for n,obj in enumerate(self.lblConfNamesObj):
            self.AllVisibleObjects.append(ObjectQt('lblConfNamesObj'+str(n),obj,1,view_group=['MoveViewState']))
        for n,obj in enumerate(self.lblConfValuesObj):
            self.AllVisibleObjects.append(ObjectQt('lblConfValuesObj'+str(n),obj,1,view_group=['MoveViewState']))
        for n,obj in enumerate(self.movePageGoPoseLabels):
            self.AllVisibleObjects.append(ObjectQt('movePageGoPoseLabels'+str(n),obj,2,view_group=['MoveViewState']))
        for n,obj in enumerate(self.movePageGoPoseEdits):
            self.AllVisibleObjects.append(ObjectQt('movePageGoPoseEdits'+str(n),obj,2,view_group=['MoveViewState']))
        self.AllVisibleObjects.append(ObjectQt('movePageGoPoseButton',self.movePageGoPoseButton,2,view_group=['MoveViewState']))
        self.AllVisibleObjects.append(ObjectQt('movePageGoGripperButton',self.movePageGoGripperButton,2,view_group=['MoveViewState']))
        self.AllVisibleObjects.append(ObjectQt('movePageOpenGripperButton',self.movePageOpenGripperButton,2,view_group=['MoveViewState']))
        self.AllVisibleObjects.append(ObjectQt('movePageCloseGripperButton',self.movePageCloseGripperButton,2,view_group=['MoveViewState']))

        self.AllVisibleObjects.append(ObjectQt('movePageUseEnvAboveButton',self.movePageUseEnvAboveButton,2,view_group=['MoveViewState']))
        self.AllVisibleObjects.append(ObjectQt('movePageUseEnvWallButton',self.movePageUseEnvWallButton,2,view_group=['MoveViewState']))
        self.AllVisibleObjects.append(ObjectQt('movePageUseEnvTableButton',self.movePageUseEnvTableButton,2,view_group=['MoveViewState']))
        self.AllVisibleObjects.append(ObjectQt('movePagePoseNowButton',self.movePagePoseNowButton,2,view_group=['MoveViewState']))

        self.setMouseTracking(True)
        self.mousex, self.mousey = 0.,0.
        self.updateLeftRightPanel(rightPanelNames=['r conf.', 'l conf.'])
        print("[Interface] Done")

    def updateLeftRightPanel(self, leftPanelNames=None, rightPanelNames=None):
        ''' Update names on left or right panel
        '''
        if rightPanelNames:
            for i in range(len(rightPanelNames)):
                obj = self.lblRightPanelNamesObj[i]
                obj.setText(rightPanelNames[i])
        if leftPanelNames:
            for i in range(len(leftPanelNames)):
                obj = self.lblLeftPanelNamesObj[i]
                obj.setText(leftPanelNames[i])

    def getRightPanelValues(self):
        ''' Get values for Right panel values
        '''
        values = []
        values.append(round(ml.md.frames[-1].r.confidence,2))
        values.append(round(ml.md.frames[-1].l.confidence,2))
        return values

    def getRightPanelActivates(self):
        activates = []
        activates.append(ml.md.frames[-1].r.confidence > settings.yaml_config_gestures['min_confidence'])
        activates.append(ml.md.frames[-1].l.confidence > settings.yaml_config_gestures['min_confidence'])
        return activates

    def mouseMoveEvent(self, event):
        self.mousex, self.mousey = event.x(), event.y()

    def go_to_pose_button(self):
        ''' Takes the text inputs given by user and change robot goal_pose
        '''
        vals = []
        for obj in self.movePageGoPoseEdits[0:7]:
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
        ml.md.goal_pose = pose
        ml.md.m.go_to_pose(pose)

    def actuate_gripper_button(self):
        ''' Takes text input float and control gripper position
        '''
        val = None
        try:
            val = float(self.movePageGoPoseEdits[7].text())
        except:
            print("[ERROR*] Value Error!")
            val = 0.0
        ml.md.m.set_gripper(val, effort=0.04, eef_rot=-1, action="", object="")

    def open_gripper_button(self):
        ml.md.m.set_gripper(1.0, action="release")

    def close_gripper_button(self):
        ml.md.m.set_gripper(0.0)

    def keyPressEvent(self, event):
        ''' Callbacky for every keyboard button press
        '''
        if settings.record_with_keys:
            KEYS = [self.mapQtKey(key) for key in gl.gd.Gs_keys]
            if event.key() in KEYS:
                self.recording = True
                for n, key in enumerate(KEYS):
                    if event.key() == key:
                        self.dir_queue.append(gl.gd.Gs[n])
                        self.caller = RepeatableTimer(self.REC_TIME, self.save_data, ())
                        self.caller.start()
        else:
            print("[Interface] Key have been read, but recording is not activated!")
        event.accept()


    def changeNetwork(self, network, type='static'):
        ''' ROS service send request about network change
        '''

        rospy.wait_for_service(f'/mirracle_gestures/change_{type}_network')
        try:
            change_network = rospy.ServiceProxy(f'/mirracle_gestures/change_{type}_network', ChangeNetwork)
            response = change_network(data=network)
            Gs = [g.lower() for g in response.Gs]
            settings.args = response.args
            print("[UI] Gestures & Network changed, new set of gestures: "+str(", ".join(Gs)))
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        settings.paths.gesture_network_file = network

        gl.gd.gesture_change_srv(local_data=response)

    def download_networks_gdrive(self):
        ''' Downloads all networks from google drive
            1. Files download
            2. Network info update
        '''
        gl.gd.download_networks_gdrive()
        # Update Networks Menu
        self.network_menu.clear()
        self.networks = gl.gd.get_networks()
        self.network_actions = []
        for network in self.networks:
            action = QAction(network, self)
            action.triggered.connect(
                lambda checked, network=network: self.changeNetwork(network))
            self.network_menu.addAction(action)
            self.network_actions.append(action)

        time.sleep(1)

    def confustion_mat(self, e):
        thread = Thread(target = self.confustion_mat_)
        thread.start()
    def button_play_move(self, e):
        self.play_status = 1
    def button_play_move2(self, e):
        self.play_status = -1
    def button_play_move3(self, e):
        self.play_status = 0

    def save_data(self):
        ''' Saving record data in this thread will be outdated, ROS service will be created
        '''
        rospy.wait_for_service('save_hand_record')
        try:
            save_hand_record = rospy.ServiceProxy('save_hand_record', SaveHandRecord)
            resp1 = save_hand_record(directory=settings.paths.learn_path+self.dir_queue.pop(0), save_method='numpy', recording_length=1.0)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        self.recording = False

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


    def confustion_mat_(self):
        self.lblInfo.setText("Show gesture disp. here")
        time.sleep(5)
        NUM_SAMPLES = 5
        DELAY_BETW_SAMPLES = 0.5
        y_true = []
        y_pred = []

        gl.gd.r.static.names
        static_gestures_list = gl.gd.r.static.names()
        for n, i in enumerate(static_gestures_list):
            for j in range(0,NUM_SAMPLES):
                self.lblInfo.setText(i+" "+str(j))
                time.sleep(DELAY_BETW_SAMPLES)
                for m,g in enumerate(static_gestures_list):
                    if gl.gd.r.poses[m].toggle:
                        y_true.append(n)
                        y_pred.append(m)

        self.lblInfo.setText("Done (Saved as confusionmatrix.csv)")
        cm = confusion_matrix(y_true, y_pred)
        cm_ = np.vstack((static_gestures_list,cm))
        static_gestures_list.insert(0,"Confusion matrix")
        cm__ = np.hstack((np.array([static_gestures_list]).transpose(),cm_))
        with open('confusionmatrix.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(cm__)



    def toggleViewGesturesMenu(self, state):
        self.GesturesViewState = state
    def toggleViewMoveMenu(self, state):
        self.MoveViewState = state
    def goToInfo(self):
        settings.WindowState = 0
    def goToConfig(self):
        settings.WindowState = 1
    def goToMove(self):
        settings.WindowState = 2

    def gestures_goal_init_procedure(self):
        ml.md.gestures_goal_pose.position = ml.md.ENV['start']
        ml.md.gestures_goal_pose.orientation = ml.md.ENV['ori']

    # Switch the environment functions
    def switchEnvAbove(self):
        ml.md.ENV = ml.md.ENV_DAT['above']
        self.gestures_goal_init_procedure()
    def switchEnvWall(self):
        ml.md.ENV = ml.md.ENV_DAT['wall']
        self.gestures_goal_init_procedure()
    def switchEnvTable(self):
        ml.md.ENV = ml.md.ENV_DAT['table']
        self.gestures_goal_init_procedure()

    def movePageUseEnvAboveButtonFun(self):
        self.movePageGoPoseEdits[3].setText(str(ml.md.ENV_DAT['above']['ori'].x))
        self.movePageGoPoseEdits[4].setText(str(ml.md.ENV_DAT['above']['ori'].y))
        self.movePageGoPoseEdits[5].setText(str(ml.md.ENV_DAT['above']['ori'].z))
        self.movePageGoPoseEdits[6].setText(str(ml.md.ENV_DAT['above']['ori'].w))
    def movePageUseEnvWallButtonFun(self):
        self.movePageGoPoseEdits[3].setText(str(ml.md.ENV_DAT['wall']['ori'].x))
        self.movePageGoPoseEdits[4].setText(str(ml.md.ENV_DAT['wall']['ori'].y))
        self.movePageGoPoseEdits[5].setText(str(ml.md.ENV_DAT['wall']['ori'].z))
        self.movePageGoPoseEdits[6].setText(str(ml.md.ENV_DAT['wall']['ori'].w))
    def movePageUseEnvTableButtonFun(self):
        self.movePageGoPoseEdits[3].setText(str(ml.md.ENV_DAT['table']['ori'].x))
        self.movePageGoPoseEdits[4].setText(str(ml.md.ENV_DAT['table']['ori'].y))
        self.movePageGoPoseEdits[5].setText(str(ml.md.ENV_DAT['table']['ori'].z))
        self.movePageGoPoseEdits[6].setText(str(ml.md.ENV_DAT['table']['ori'].w))
    def movePagePoseNowButtonFun(self):
        self.movePageGoPoseEdits[0].setText(str(ml.md.goal_pose.position.x))
        self.movePageGoPoseEdits[1].setText(str(ml.md.goal_pose.position.y))
        self.movePageGoPoseEdits[2].setText(str(ml.md.goal_pose.position.z))

        self.movePageGoPoseEdits[3].setText(str(ml.md.goal_pose.orientation.x))
        self.movePageGoPoseEdits[4].setText(str(ml.md.goal_pose.orientation.y))
        self.movePageGoPoseEdits[5].setText(str(ml.md.goal_pose.orientation.z))
        self.movePageGoPoseEdits[6].setText(str(ml.md.goal_pose.orientation.w))

    # Fixed orientation function
    def fixedOriAct(self, state):
        settings.ORIENTATION_MODE = 'fixed' if state else 'free'
    def print_path_trace(self, state):
        settings.print_path_trace = state
    def record_with_keys(self, state):
        if not state: state = True
        settings.record_with_keys = state

    def goScene(self, index):
        scenes = sl.scenes.names()
        sl.scenes.make_scene(ml.md.m, scenes[index])

    def onComboPlayNLiveChanged(self, text):
        if text=="Live hand":
            ml.md.mode = 'live'
        elif text=="Play path":
            ml.md.mode = 'play'
    def onComboPickPlayTrajChanged(self, text):
        ml.md.changePlayPath(text)
    def onComboLiveModeChanged(self, text):
        ml.md.changeLiveMode(text)
    def onInteractiveSceneChanged(self, text):
        if text == "Scene 1 Drawer":
            sl.scenes.make_scene('drawer')
            ml.md.ENV = ml.md.ENV_DAT['wall']
        elif text == "Scene 2 Pick/Place":
            sl.scenes.make_scene('pickplace')
            ml.md.ENV = ml.md.ENV_DAT['table']
        elif text == "Scene 3 Push button":
            sl.scenes.make_scene('pushbutton')
            ml.md.ENV = ml.md.ENV_DAT['table']
        elif text == "Scene 4 - 2 Pick/Place":
            sl.scenes.make_scene('pickplace2')
            ml.md.ENV = ml.md.ENV_DAT['table']
        else: raise Exception("Item not on a list!")
        self.gestures_goal_init_procedure()

    def paintEvent(self, e):
        ## Window Resolution update
        self.w = settings.w = self.size().width()
        self.h = settings.h = self.size().height()

        ## Set all objects on page visible (the rest set invisible)
        for obj in self.AllVisibleObjects:
            # Every object that belongs to that group are conditioned by that group
            if obj.page == settings.WindowState and (self.GesturesViewState if 'GesturesViewState' in obj.view_group else True) and  (self.MoveViewState if 'MoveViewState' in obj.view_group else True) and ((ml.md.mode=='live') if 'live' in obj.view_group else True) and ((ml.md.mode=='play') if 'play' in obj.view_group else True):
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

        for h in ['l', 'r']:
            if getattr(ml.md, h+'_present')():
                pts = len(ml.md.frames)
                if pts > 10: pts = 10
                for n in range(1,pts):
                    #if ml.md.frames[-n-1].l.visible and ml.md.frames[-n].l.visible:
                    if getattr(ml.md.frames[-n-1], h).visible and getattr(ml.md.frames[-n], h).visible:
                        #p1 = tfm.transformLeapToUIsimple(ml.md.frames[-n].l.palm_pose())
                        p1 = tfm.transformLeapToUIsimple(getattr(ml.md.frames[-n], h).palm_pose())
                        #p2 = tfm.transformLeapToUIsimple(ml.md.frames[-n-1].l.palm_pose())
                        p2 = tfm.transformLeapToUIsimple(getattr(ml.md.frames[-n-1], h).palm_pose())
                        painter.setPen(QPen(Qt.red, p1.position.z))
                        painter.drawLine(p1.position.x, p1.position.y, p2.position.x, p2.position.y)
        if gl.gd.r.static[-1] and self.cursor_enabled() and 'point' in gl.gd.r.static.info.names:
            pose_c = tfm.transformLeapToUIsimple(ml.md.frames[-1].r.palm_pose())
            x_c,y_c = pose_c.position.x, pose_c.position.y

            rad = gl.gd.r.static[-1].point.time_visible*80
            if rad > 100:
                rad = 100

            painter.setPen(QPen(Qt.yellow, 4))
            painter.drawEllipse(x_c-rad/2,y_c-rad/2, rad, rad)

        #painter.setPen(QPen(Qt.black, 1))
        #painter.drawLine(self.w/2,self.h-20, self.w-RIGHT_MARGIN-ICON_SIZE, self.h-20)
        #painter.drawLine(self.w/2,self.h-20, self.w/2, START_PANEL_Y+ICON_SIZE)
        #self.lblStartAxis.setGeometry(self.w/2+5, self.h-70, 100, 50)
        #self.lblStartAxis.setText(str(ml.md.ENV['start']))



        ''' Draw the bone structure '''
        for h in ['l', 'r']:
            painter.setPen(QPen(Qt.black, 1))
            #if ml.md.r_present():
            if getattr(ml.md, h+'_present')():
                hand = getattr(ml.md.frames[-1], h)
                palm = hand.palm_position()
                wrist = hand.wrist_position()

                elbow = hand.elbow_position()
                pose_palm = Pose()
                pose_palm.position = Point(palm[0], palm[1], palm[2])
                pose_wrist = Pose()
                pose_wrist.position = Point(wrist[0], wrist[1], wrist[2])
                pose_elbow = Pose()
                pose_elbow.position = Point(elbow[0], elbow[1], elbow[2])
                pose_palm_ = tfm.transformLeapToUIsimple(pose_palm)
                pose_wrist_ = tfm.transformLeapToUIsimple(pose_wrist)
                x, y = pose_palm_.position.x, pose_palm_.position.y
                x_, y_ = pose_wrist_.position.x, pose_wrist_.position.y
                painter.drawLine(x, y, x_, y_)
                pose_elbow_ = tfm.transformLeapToUIsimple(pose_elbow)
                x, y = pose_elbow_.position.x, pose_elbow_.position.y
                painter.drawLine(x, y, x_, y_)

                if h == 'l':
                    ''' Set builder mode '''
                    nBuild_modes = len(ml.md.build_modes)
                    id_build_mode = ml.md.build_modes.index(ml.md.build_mode)
                    painter.setPen(QPen(Qt.blue, 4))
                    for n, i in enumerate(ml.md.build_modes):
                        x_bm, y_bm = point_by_ratio((x, y),(x_, y_), 0.5+0.5*(n/nBuild_modes))

                        if n == id_build_mode:
                            painter.setBrush(QBrush(Qt.blue, Qt.SolidPattern))
                        else:
                            painter.setBrush(QBrush(Qt.blue, Qt.NoBrush))
                        painter.drawEllipse(x+x_bm-5, y+y_bm-5, 10, 10)

                    painter.setPen(QPen(Qt.black, 1))

                for finger in hand.fingers:
                    for b in range(0, 4):
                        bone = finger.bones[b]
                        pose_bone_prev = Pose()
                        pose_bone_prev.position = Point(bone.prev_joint[0], bone.prev_joint[1], bone.prev_joint[2])
                        pose_bone_next = Pose()
                        pose_bone_next.position = Point(bone.next_joint[0], bone.next_joint[1], bone.next_joint[2])
                        pose_bone_prev_ = tfm.transformLeapToUIsimple(pose_bone_prev)
                        pose_bone_next_ = tfm.transformLeapToUIsimple(pose_bone_next)
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
        if ml.md.goal_pose and ml.md.goal_joints:
            structures_str = [structure.object_stack for structure in ml.md.structures]
            textStatus += f"eef: {str(round(ml.md.eef_pose.position.x,2))} {str(round(ml.md.eef_pose.position.y,2))} {str(round(ml.md.eef_pose.position.z,2))}\ng p: {str(round(ml.md.goal_pose.position.x,2))} {str(round(ml.md.goal_pose.position.y,2))} {str(round(ml.md.goal_pose.position.z,2))}\ng q:{str(round(ml.md.goal_pose.orientation.x,2))} {str(round(ml.md.goal_pose.orientation.y,2))} {str(round(ml.md.goal_pose.orientation.z,2))} {str(round(ml.md.goal_pose.orientation.w,2))}\nAttached: {ml.md.attached}\nbuild_mode {ml.md.build_mode}\nobject_touch and focus_id {ml.md.object_focus_id} {ml.md.object_focus_id}\nStructures: {str(structures_str)}"

        if self.recording:
            qp.setBrush(QBrush(Qt.red, Qt.SolidPattern))
            qp.drawEllipse(LEFT_MARGIN+130+ICON_SIZE*2, h-10-ICON_SIZE, ICON_SIZE/2,ICON_SIZE/2)
            qp.setBrush(QBrush(Qt.black, Qt.NoBrush))
        self.lblInfo.setGeometry(LEFT_MARGIN+130, h-ICON_SIZE,ICON_SIZE*5,ICON_SIZE)
        self.lblStatus.setText(textStatus)

        self.lbl3.setText(settings.get_hand_mode()['l'])
        self.lbl4.setText(settings.get_hand_mode()['r'])

        self.lbl2.move(self.size().width()-RIGHT_MARGIN-40, 36)
        self.lbl4.move(self.size().width()-RIGHT_MARGIN-40, START_PANEL_Y_GESTURES-30)

        if self.GesturesViewState:
            # up late
            if ml.md.frames:
                # hand fingers
                n = 0
                if ml.md.frames[-1].l.confidence > settings.yaml_config_gestures['min_confidence']:
                    qp.drawRect(LEFT_MARGIN,START_PANEL_Y+(n)*ICON_SIZE, ICON_SIZE, ICON_SIZE)
                qp.drawLine(LEFT_MARGIN+ICON_SIZE+2, START_PANEL_Y+(n+1)*ICON_SIZE, LEFT_MARGIN+ICON_SIZE+2, int(START_PANEL_Y+(n+1)*ICON_SIZE-round(ml.md.frames[-1].l.confidence,2)*ICON_SIZE))
                for i in range(0,5):
                    if ml.md.frames[-1].l.oc_activates[i]:
                        qp.drawPixmap(LEFT_MARGIN, START_PANEL_Y, ICON_SIZE, ICON_SIZE, QPixmap(settings.paths.graphics_path+"hand"+str(i+1)+"open_left.png"))
                    else:
                        qp.drawPixmap(LEFT_MARGIN, START_PANEL_Y, ICON_SIZE, ICON_SIZE, QPixmap(settings.paths.graphics_path+"hand"+str(i+1)+"closed_left.png"))

                if ml.md.frames[-1].r.confidence > settings.yaml_config_gestures['min_confidence']:
                    qp.drawRect(w-RIGHT_MARGIN,START_PANEL_Y+(n)*ICON_SIZE, ICON_SIZE, ICON_SIZE)
                qp.drawLine(w-RIGHT_MARGIN+ICON_SIZE+2, START_PANEL_Y+(n+1)*ICON_SIZE, w-RIGHT_MARGIN+ICON_SIZE+2, int(START_PANEL_Y+(n+1)*ICON_SIZE-round(ml.md.frames[-1].r.confidence,2)*ICON_SIZE))
                for i in range(0,5):
                    if ml.md.frames[-1].r.oc_activates[i]:
                        qp.drawPixmap(w-RIGHT_MARGIN, START_PANEL_Y, ICON_SIZE, ICON_SIZE, QPixmap(settings.paths.graphics_path+"hand"+str(i+1)+"open.png"))
                    else:
                        qp.drawPixmap(w-RIGHT_MARGIN, START_PANEL_Y, ICON_SIZE, ICON_SIZE, QPixmap(settings.paths.graphics_path+"hand"+str(i+1)+"closed.png"))

                ''' Direction of hand '''
                for h, X in [('l', LEFT_MARGIN+ICON_SIZE), ('r', w-RIGHT_MARGIN-ICON_SIZE)]:
                    point_direction = getattr(ml.md.frames[-1], h).point_direction()
                    if point_direction[0] < 0.0:
                        qp.drawPixmap(X, START_PANEL_Y, ICON_SIZE, ICON_SIZE, QPixmap(settings.paths.graphics_path+"arrow_right.png"))
                    if point_direction[0] > 0.0:
                        qp.drawPixmap(X, START_PANEL_Y, ICON_SIZE, ICON_SIZE, QPixmap(settings.paths.graphics_path+"arrow_left.png"))
                    if point_direction[1] < 0.0:
                        qp.drawPixmap(X, START_PANEL_Y, ICON_SIZE, ICON_SIZE, QPixmap(settings.paths.graphics_path+"arrow_up.png"))
                    if point_direction[1] > 0.0:
                        qp.drawPixmap(X, START_PANEL_Y, ICON_SIZE, ICON_SIZE, QPixmap(settings.paths.graphics_path+"arrow_down.png"))

            ''' Left side lane - Gestures '''
            static_gs_file_images = gl.gd.static_info().filenames
            static_gs_names = gl.gd.static_info().names
            dynamic_gs_file_images = gl.gd.dynamic_info().filenames
            dynamic_gs_names = gl.gd.dynamic_info().names
            if gl.gd.l.static.relevant():
                for n, i in enumerate(static_gs_file_images):
                    image_filename = settings.paths.graphics_path+i
                    image_filename = f"{image_filename[:-4]}_left{image_filename[-4:]}"
                    qp.drawPixmap(LEFT_MARGIN, START_PANEL_Y_GESTURES+n*ICON_SIZE, ICON_SIZE, ICON_SIZE, QPixmap(image_filename))

                    if gl.gd.l.static[-1][n].activated:
                        qp.drawRect(LEFT_MARGIN,START_PANEL_Y_GESTURES+(n)*ICON_SIZE, ICON_SIZE, ICON_SIZE)
                    qp.drawLine(LEFT_MARGIN+ICON_SIZE+2, START_PANEL_Y_GESTURES+(n+1)*ICON_SIZE, LEFT_MARGIN+ICON_SIZE+2, int(START_PANEL_Y_GESTURES+(n+1)*ICON_SIZE-gl.gd.l.static[-1][n].probability*ICON_SIZE))
                    qp.drawText(LEFT_MARGIN+ICON_SIZE+5, START_PANEL_Y_GESTURES+n*ICON_SIZE+10, static_gs_names[n])
            if gl.gd.r.static.relevant():
                for n, i in enumerate(static_gs_file_images):
                    image_filename = settings.paths.graphics_path+i
                    qp.drawPixmap(w-RIGHT_MARGIN, START_PANEL_Y_GESTURES+n*ICON_SIZE, ICON_SIZE, ICON_SIZE, QPixmap(image_filename))

                    if gl.gd.r.static[-1][n].activated:
                        qp.drawRect(w-RIGHT_MARGIN,START_PANEL_Y_GESTURES+(n)*ICON_SIZE, ICON_SIZE, ICON_SIZE)
                    qp.drawLine(w-RIGHT_MARGIN+ICON_SIZE+2, START_PANEL_Y_GESTURES+(n+1)*ICON_SIZE, w-RIGHT_MARGIN+ICON_SIZE+2, int(START_PANEL_Y_GESTURES+(n+1)*ICON_SIZE-gl.gd.r.static[-1][n].probability*ICON_SIZE))
                    qp.drawText(w-RIGHT_MARGIN+ICON_SIZE+5, START_PANEL_Y_GESTURES+n*ICON_SIZE+10, static_gs_names[n])


            if gl.gd.r.dynamic and gl.gd.r.dynamic.relevant():
                probabilities = gl.gd.r.dynamic[-1].probabilities_norm
                for n, i in enumerate(dynamic_gs_file_images):
                    qp.drawPixmap(LEFT_MARGIN, START_PANEL_Y_GESTURES+(n+len(static_gs_file_images))*ICON_SIZE, ICON_SIZE, ICON_SIZE, QPixmap(settings.paths.graphics_path+i))

                    if gl.gd.r.dynamic[-1][n].activated:
                        qp.drawRect(LEFT_MARGIN,START_PANEL_Y_GESTURES+(n+len(static_gs_file_images))*ICON_SIZE, ICON_SIZE, ICON_SIZE)
                    qp.drawLine(LEFT_MARGIN+ICON_SIZE+2, START_PANEL_Y_GESTURES+(n+1+len(static_gs_file_images))*ICON_SIZE, LEFT_MARGIN+ICON_SIZE+2, START_PANEL_Y_GESTURES+(n+1+len(static_gs_file_images))*ICON_SIZE-probabilities[n]*ICON_SIZE)
                    qp.drawText(LEFT_MARGIN+ICON_SIZE+5, START_PANEL_Y_GESTURES+(n+len(static_gs_file_images))*ICON_SIZE+10, dynamic_gs_names[n])

            if gl.gd.l.static and gl.gd.l.static.relevant():
                n_ = gl.gd.l.static.relevant().biggest_probability_id
                qp.drawPixmap(LEFT_MARGIN+ICON_SIZE+10,START_PANEL_Y_GESTURES+(n_)*ICON_SIZE, ICON_SIZE, ICON_SIZE, QPixmap(settings.paths.graphics_path+"arrow_left.png"))
            if gl.gd.r.dynamic and gl.gd.r.dynamic.relevant():
                n_ = gl.gd.r.dynamic.relevant().biggest_probability_id
                qp.drawPixmap(LEFT_MARGIN+ICON_SIZE+10,START_PANEL_Y_GESTURES+(n_+len(static_gs_file_images))*ICON_SIZE, ICON_SIZE, ICON_SIZE, QPixmap(settings.paths.graphics_path+"arrow_left.png"))


            # circ options
            ### DEPRECATED
            if 'circ' in gl.gd.r.dynamic.info.names:
                g = gl.gd.r.dynamic.circ
                if g.activate:
                    X = LEFT_MARGIN+130
                    Y = START_PANEL_Y+len(static_gs_file_images)*ICON_SIZE
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

        if self.MoveViewState:
            if ml.md.mode == 'play':
                if gl.gd.relevant(time_now=rospy.Time.now(), hand='l', type='static') and hasattr(gl.gd.l.static[-1], 'grab'):
                    if gl.gd.l.static[-1].grab.activated:
                        qp.drawPixmap(w/2, 50, 20, 20, QPixmap(settings.paths.graphics_path+"hold.png"))
                        if settings.HoldPrevState == False:
                            settings.HoldAnchor = settings.HoldValue - ml.md.frames[-1].l.palm_pose().position.x/len(sl.paths[ml.md.picked_path].poses)
                        settings.HoldValue = settings.HoldAnchor + ml.md.frames[-1].l.palm_pose().position.x/len(sl.paths[ml.md.picked_path].poses)
                        settings.HoldValue = settings.HoldAnchor + ml.md.frames[-1].l.palm_pose().position.x/2
                        if settings.HoldValue > 100: settings.HoldValue = 100
                        if settings.HoldValue < 0: settings.HoldValue = 0

                    settings.HoldPrevState = gl.gd.l.static.grab.toggle
                diff_pose_progress = 100/len(sl.paths[ml.md.picked_path].poses)
                for i in range(0, len(sl.paths[ml.md.picked_path].poses)):
                    qp.fillRect(LEFT_MARGIN+diff_pose_progress*i*((w-40.0)/100.0), 30, 2, 20, Qt.black)
                qp.fillRect(LEFT_MARGIN+diff_pose_progress*settings.currentPose*((w-40.0)/100.0)+2, 35, diff_pose_progress*((w-40.0)/100.0), 10, Qt.red)
                qp.drawRect(LEFT_MARGIN, 30, w-40, 20)
                qp.fillRect(LEFT_MARGIN+settings.HoldValue*((w-40.0)/100.0), 30, 10, 20, Qt.black)

        if self.GesturesViewState:
            '''
            # right lane
            for n, i in enumerate(self.lblRightPanelNamesObj):
                i.setVisible(True)
                i.move(w-RIGHT_MARGIN, int(START_PANEL_Y+n*ICON_SIZE/2))
            if ml.md.present():
                values, activates = self.getRightPanelValues(), self.getRightPanelActivates()
                for n in range(len(values)):
                    obj = self.lblRightPanelValuesObj[n]
                    obj.move(w-RIGHT_MARGIN, START_PANEL_Y+(n+0.5)*ICON_SIZE/2)
                    obj.setText(str(values[n]))
                    qp.drawLine(w-RIGHT_MARGIN-5+values[n]*ICON_SIZE, START_PANEL_Y+(n+1)*ICON_SIZE/2, w-RIGHT_MARGIN-5, START_PANEL_Y+(n+1)*ICON_SIZE/2)
                    if activates[n]:
                        qp.drawRect(w-RIGHT_MARGIN-5, START_PANEL_Y+(n)*ICON_SIZE/2+5, ICON_SIZE, ICON_SIZE/2-10)
            '''
            # orientation
            '''
            if self.cursor_enabled():
                roll, pitch, yaw = ml.md.frames[-1].r.palm_euler()
                x = np.cos(yaw)*np.cos(pitch)
                y = np.sin(yaw)*np.cos(pitch)
                z = np.sin(pitch)

                last_pose_ = tfm.transformLeapToUIsimple(ml.md.frames[-1].r.palm_pose())
                x_c,y_c = last_pose_.position.x, last_pose_.position.y
                qp.setPen(QPen(Qt.blue, 4))
                qp.drawLine(x_c, y_c, x_c+y*2*ICON_SIZE, y_c-z*2*ICON_SIZE)
            '''

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
        if self.cursor_enabled():
            last_pose_ = tfm.transformLeapToUIsimple(ml.md.frames[-1].r.palm_pose())
            x,y = last_pose_.position.x, last_pose_.position.y
        else:
            x,y = self.mousex, self.mousey
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
        # ['Gripper open', 'Applied force', 'Work reach', 'Shift start x', 'Speed', 'Scene change', 'mode', 'Path']
        settings.VariableValues[m, n] = value
        if m == 0:
            if   n == 0:
                ml.md.gripper = value
            elif n == 1:
                ml.md.applied_force = value
            elif n == 2:
                ml.md.SCALE = value/50
            elif n == 3:
                ml.md.ENV['start'].x = value/100
        elif m == 1:
            if   n == 0:
                ml.md.speed = value
            elif n == 1:
                scenes = settings.getSceneNames()
                v = int(len(scenes)/125. * value)
                if v >= len(scenes):
                    v = len(scenes)-1
                sl.scene.make_scene(scene=scenes[v])
            elif n == 2:
                modes = ml.md.modes()
                v = int(len(modes)/125. * value)
                if v >= len(modes):
                    v = len(modes)-1
                ml.md.mode = modes[v]
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
                value = ml.md.gripper
            elif n == 1:
                value = ml.md.applied_force
            elif n == 2:
                value = ml.md.scale*50
            elif n == 3:
                value = ml.md.ENV['start'].x*100
        elif m == 1:
            if   n == 0:
                value = ml.md.speed
            elif n == 1:
                if sl.scene:
                    scenes = sl.scenes.names()
                    value = scenes.index(sl.scene.name)
                    string = sl.scene.name
                else: value = 0
            elif n == 2:
                modes = ml.md.modes()
                value = modes.index(ml.md.mode)
                string = ml.md.mode
            elif n == 3:
                value = ml.md.picked_path
                paths = sl.paths.names()
                string = paths[value]

        settings.VariableValues[m, n] = float(value)
        if not string:
            string = str(value)
        return value, string


    def cursor_enabled(self):
        ''' Checks if enough samples are made
        '''
        if ml.md.r_present() and len(ml.md.frames) >= 10:
            return True
        return False

    def timerEvent(self, event):
        if ml.md.frames and self.OneTimeTurnOnGesturesViewStateOnLeapMotionSignIn:
            self.OneTimeTurnOnGesturesViewStateOnLeapMotionSignIn = False
            self.GesturesViewState = True
            self.comboPlayNLive.addItem("Live hand")

        ''' DEPRECATED
        if ml.md.frames and gl.gd.r.dynamic.info.names:
            fa = ml.md.frames[-1]
            for i in gl.gd.r.dynamic[-1][0:4]: # circ, swipe, pin, touch
                if i.time_visible > 0:
                    i.time_visible -= 0.1
                else:
                    i.toggle = False
            if fa.r.visible == False:
                for i in gl.gd.r.dynamic[-1]:
                    i.toggle = False if isinstance(i.toggle, bool) else [False] * len(i.toggle)
                for i in gl.gd.r.static[-1]:
                    i.toggle = False
            if fa.l.visible == False:
                for i in gl.gd.l.dynamic[-1]:
                    i.toggle = False if isinstance(i.toggle, bool) else [False] * len(i.toggle)
                for i in gl.gd.l.static[-1]:
                    i.toggle = False

            self.step = self.step + 1
        '''
        self.update()


    def mapQtKey(self, key):
        key = str(key)
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

    def thread_testInit(self):
        thread = Thread(target = ml.md.testInit)
        thread.start()
    def thread_testMovements(self):
        thread = Thread(target = ml.md.testMovements)
        thread.start()
    def thread_testMovementsInput(self):
        thread = Thread(target = ml.md.testMovementsInput)
        thread.start()
    def thread_inputPlotJointsAction(self):
        thread = Thread(target = ml.md.inputPlotJointsAction)
        thread.start()
    def thread_inputPlotPosesAction(self):
        thread = Thread(target = ml.md.inputPlotPosesAction)
        thread.start()


class ObjectQt():
    def __init__(self, NAME=None, qt=None, page=None, view_group=['GesturesViewState']):
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
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

def ui_thread_launch():

    if rospy.get_param("/mirracle_config/launch_ui", 'false') == "true":
        thread_ui = Thread(target = main)
        thread_ui.daemon=True
        thread_ui.start()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
