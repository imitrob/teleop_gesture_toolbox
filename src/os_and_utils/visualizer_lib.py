#!/usr/bin/env python
''' Visualization library:
    - Visualize up to 6 window figures (Full HD monitor)
    - Visualize n trajectories on one figure
    - 2D or 3D
    - In 3D it finds the best azimuth for view

Usage:
    1. Create obj
    viz = VisualizerLib()
    2. Create new fig
    viz.visualize_new_fig(title, dim=(2|3))
    3. Add trajectory data 2D or 3D based on setup figure holder
    viz.visualize_2d(data)
    viz.visualize_3d(data)
    4. You can visualize n trajectories at once
    viz.visualize_ntraj(data)
    5. Done with plotting
    viz.show()
'''
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time as t
import argparse

# Ensure package independency to ROS
try:
    from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Vector3Stamped, QuaternionStamped, Vector3
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    import rospy
    ROS_COMPATIBILITY = True

    trajectories = []
except ImportError:
    ROS_COMPATIBILITY = False

class VisualizerLib():
    def __init__(self):
        self.fig = None
        self.ax = None
        plt.ion()

    def show(self):
        plt.ioff()
        plt.show()

    def visualize_new_fig(self, title=None, dim=3):
        ''' Creates new figure instance window
            1. Creates new: fig, ax
            2. Moves new figure, position for 6 figures on 1080p monitor

        Args:
            title (str): sets title to figure
            dim (int): choose figure dimension (2|3)
        Returns:
            fig, ax (matplotlib objects): Used to store
        '''
        assert (dim == 3) or (dim == 2), "wrong dimension: "+str(dim)
        fig = plt.figure(figsize=(6,5))
        if dim == 3:
            ax = fig.add_subplot(111, projection='3d')
        if dim == 2:
            ax = fig.add_subplot(111)
        if type(title) is type("a"):
            fig.canvas.set_window_title(title)

        ## Move figure
        nfigs = len(list(map(plt.figure, plt.get_fignums())))-1 # get number of opened figures after creating new one
        BOXXMOVE = [0, 600, 1200, 0, 600, 1200, 0, 600, 1200, 0, 600, 1200]
        BOXYMOVE = [0, 0, 0, 600, 600, 600, 0, 0, 0, 600, 600, 600]
        self.move_figure(fig, BOXXMOVE[nfigs], BOXYMOVE[nfigs])

        self.fig = fig
        self.ax = ax

    def visualize_2d(self, data, color='', label="", transform='front', xlabel='X axis', ylabel='Y axis', scatter_pts=False, axvline=False):
        ''' Add trajectory to current figure for 2D Plot.

        Args:
            data (2D array): [[X1,Y1], [X2,Y2], [X3,Y3], ..] or possible to use zip([X1,X2,X3],[Y1,Y2,Y3])
            color (str): choose series color ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            label (str): series label (legend)
            transform (str): When data is 3D, it transforms to 2D
                look from 'top', 'front', 'left'
            xlabel, ylabel (str): Axis label legend
        '''
        data = np.array(data)
        try:
            data[:,0:2]
        except IndexError:
            print("[Visualizer lib] No data on input!")
            return
        if transform == 'top':
            data = data[:,1:3]
        if transform == 'front' or transform == '':
            data = data[:,0:2]
        if transform == 'left':
            data = np.delete(data, 1, 0)
        assert len(data[0]) == 2, "Data not valid, points are not [x,y] type"
        plt.ion() # turn on interactive model

        xt, yt = [], []
        for n, point in enumerate(data):
            xt.append(point[0])
            yt.append(point[1])
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(b=True)

        #plt.axis('equal')
        COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        if color=="":
            color=COLORS[len(self.ax.lines)%7]

        self.ax.plot(xt,yt,c=color, label=(label))#+" "+str(len(dparsed))) )
        self.ax.scatter(xt[0], yt[0], marker='o', color='black', zorder=2)
        self.ax.scatter(xt[-1], yt[-1], marker='x', color='black', zorder=2)
        if axvline:
            self.ax.axvline(x=xt[-1], color=color)
        if scatter_pts:
            self.ax.scatter(xt, yt, marker='|', color='black', zorder=2, alpha=0.3)

        if label != "":
            plt.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.0))

        #plt.annotate("Num points:", xy=(-0.15, 1.0), xycoords='axes fraction')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def visualize_3d(self, data, color='', label="", xlabel='X axis', ylabel='Y axis', zlabel='Z axis'):
        ''' Add trajectory to current figure for 3D Plot.

        Args:
            data (3D array): [[X1,Y1,Z1], [X2,Y2,Z2], [X3,Y3,Z3], ..] or possible to use zip([X1,X2,X3],[Y1,Y2,Y3],[Z1,Z2,Z3])
            color (str): choose series color ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            label (str): series label (legend)
            xlabel, ylabel, zlabel (str): Axis label legend
        '''
        if data is None:
            print("[Visualizer lib] No data on input!")
            return
        if ROS_COMPATIBILITY:
            data = self.dataROStoList(data)
        # data must be in (n x 3)
        assert len(data[0]) == 3, "Data not valid, points are not [x,y,z] type"

        plt.ion() # turn on interactive mode
        fig = self.fig
        ax = self.ax
        # Differentiate between points and scene type
        if not (type(data) == type([1]) or type(data) == type(np.array([1]))):
            data = convert_scene_to_points(data)

        xt, yt, zt = [], [], []
        for n, point in enumerate(data):
            xt.append(point[0])
            yt.append(point[1])
            zt.append(point[2])
        ax.elev = 20
        ax.azim = self.find_best_azimuth(data)
        ax.alpha = 0
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        try:
            plt.axis('equal')
        except:
            plt.axis('auto')

        COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        if color=="":
            color=COLORS[len(ax.lines)%7]

        self.ax.plot3D(xt,yt,zt,c=color, label=(label))#+" "+str(len(data))) )
        self.ax.scatter(xt[0], yt[0], zt[0], marker='o', color='black', zorder=2)
        self.ax.scatter(xt[-1], yt[-1], zt[-1], marker='x', color='black', zorder=2)
        plt.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.0))
        #plt.annotate("Num points:", xy=(-0.15, 1.0), xycoords='axes fraction')
        fig.canvas.draw()
        fig.canvas.flush_events()

    def find_best_azimuth(self, data):
        ''' Computes best view angle to 3D trajectory

        Args:
            data (3D array): [[X1,Y1,Z1], [X2,Y2,Z2], [X3,Y3,Z3], ..] or possible to use zip([X1,X2,X3],[Y1,Y2,Y3],[Z1,Z2,Z3])
        Returns:
            angle_q (float): azimuth in [deg]
        '''
        data = np.array(data)
        x1 = data[0, 0]
        x2 = data[len(data)-1, 0]
        y1 = data[0, 1]
        y2 = data[len(data)-1, 1]
        xd = abs(x2 - x1)
        yd = abs(y2 - y1)
        angle_q = np.rad2deg(np.arctan2(xd,yd))
        return angle_q

    def visualize_ntraj(self, data, n=None, xlabel='X axis', ylabel='Y axis', zlabel='Z axis', dim=3, transform="top"):
        ''' Plot multiple trajecories. visualize_2d/3d() called n times.

        Args:
            data (4D/3D array): array of 3D/2D trajectories
            n (int): number of trajectories (len(data) if n is None)
            xlabel, ylabel, zlabel (str): Axis label legend (Passed to visualize_2d/3d functions)
        '''
        assert (dim == 3) or (dim == 2), "wrong dimension: "+str(dim)
        if n==None:
            n = len(data)
        for i in range(0,n):
            traj = []
            for j in data[i]:
                traj.append(j)
            if dim == 3:
                self.visualize_3d(traj, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
            if dim == 2:
                self.visualize_2d(traj, xlabel=xlabel, ylabel=ylabel, transform=transform)

    def visualize_refresh(self, s=60):
        ''' Can be used to rotate 3D trajectory with 10Hz.
        '''
        i = 0
        while i<s*10:
            if i == 900:
                i = 0
            i += 1
            t.sleep(0.1)
            self.ax.elev = 90 - i/2.0
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def move_figure(self, f, x, y):
        ''' Move figure's upper left corner to pixel (x, y)
        '''

        backend = matplotlib.get_backend()
        if backend == 'TkAgg':
            f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
        elif backend == 'WXAgg':
            f.canvas.manager.window.SetPosition((x, y))
        else:
            # This works for QT and GTK
            # You can also use window.setGeometry
            f.canvas.manager.window.move(x, y)


    def dataROStoList(self, data):
        ''' array of Points,Poses,PoseStamped to List
        '''
        new_data = []
        if isinstance(data[0], Pose):
            for pose in data:
                new_data.append([pose.position.x, pose.position.y, pose.position.z])
        elif isinstance(data[0], Point):
            for point in data:
                new_data.append([point.x, point.y, point.z])
        elif isinstance(data[0], PoseStamped):
            for pose_stamped in data:
                new_data.append([pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z])
        else:
            return data
        return new_data

    def rosecho(self, topic):
        rospy.init_node("asd", anonymous=True)
        rospy.Subscriber(topic, FollowJointTrajectoryGoal, save_trajectory)

        print("Press to plot harvested trajectory data")
        input()
        global trajectories
        if not trajectories: print("No trajectories were published")
        self.visualize_new_fig('Trajectory positions', dim=2)
        # trajectories shape [ trajectory x 4 (position,velocity,acceleration,time_from_start) x 7 x points]
        for trajectory in trajectories:
            data = []
            for i, j in zip(trajectory[3],[t[1] for t in trajectory[0]]):
                data.append([i,j])
            self.visualize_2d(data, axvline=True)
        self.visualize_new_fig('Trajectory velocities', dim=2)
        for trajectory in trajectories:
            data = []
            for i, j in zip(trajectory[3],[t[1] for t in trajectory[1]]):
                data.append([i,j])
            self.visualize_2d(data, axvline=True)
        self.visualize_new_fig('Trajectory acceleration', dim=2)
        for trajectory in trajectories:
            data = []
            for i, j in zip(trajectory[3],[t[1] for t in trajectory[2]]):
                data.append([i,j])
            self.visualize_2d(data, axvline=True)
        self.show()

def save_trajectory(msg):
    positions = [point.positions for point in msg.trajectory.points]
    velocities = [point.velocities for point in msg.trajectory.points]
    accelerations = [point.accelerations for point in msg.trajectory.points]
    times_from_start = [point.time_from_start.to_sec() for point in msg.trajectory.points]
    stamp = msg.trajectory.header.stamp.to_sec()
    #for n,time_from_start in enumerate(times_from_start):
    #    times_from_start[n] += stamp
    tfs0 = times_from_start[0]
    for n,time_from_start in enumerate(times_from_start):
        times_from_start[n] -= tfs0

    global trajectories
    trajectories.append([positions, velocities, accelerations, times_from_start])
    print("Trajectory saved")


def main(args):
    ''' Test of printing multiple plots at once.
    '''
    viz = VisualizerLib()

    if args.rosecho == 'true':
        if not ROS_COMPATIBILITY:
            raise Exception("Argument rosecho is true, while ROS libraries cannot be imported!")
        viz.rosecho(args.topic)

        return

    viz.visualize_new_fig("plot1")
    viz.visualize_2d([[0,0,0],[0,1,1],[0,2,2],[0,3,3]])
    viz.visualize_2d([[1,0,0],[1,1,1],[1,2,2],[1,3,3]])

    viz.visualize_new_fig("plot2")
    data = np.array([[[0,0,0],[0,1,1],[0,2,2],[0,3,3]], [[1,0,0],[1,1,1],[1,2,2],[1,3,3]]])
    viz.visualize_ntraj(data)

    viz.visualize_new_fig(dim=2)
    viz.visualize_2d([[0,0,0],[0,1,1],[0,2,2],[0,3,3]])

    viz.visualize_new_fig(dim=2)
    viz.visualize_2d([[0,0,0],[0,1,1],[0,2,2],[0,3,3]])

    viz.visualize_new_fig(dim=2)
    viz.visualize_2d([[0,0,0],[0,1,1],[0,2,2],[0,3,3]])

    viz.visualize_new_fig(dim=2)
    viz.visualize_2d([[0,0,0],[0,1,1],[0,2,2],[0,3,3]])

    viz.show()
    return


    def plotPosesCallViz(self, dataPosePlot, dataPoseGoalsPlot, load_data=True, md=None):
        ''' Visualize data + show. Loading series from:
            - eef poses: settings.dataPosePlot
            - goal poses: settings.dataPoseGoalsPlot

        Parameters:
            load_data (bool): Loads the data from:
                - eef poses: md.goal_pose_array
                - goal poses: md.goal_pose_array
        '''

        if load_data:
            dataPosePlot = [pt.position for pt in list(md.goal_pose_array)]
            dataPoseGoalsPlot = [pt.position for pt in list(md.goal_pose_array)]

        if not settings.dataPosePlot:
            print("[ERROR*] No data when plotting poses were found, probably call with param: load_data=True")
            return

        # Plot positions
        self.visualize_new_fig(title="Trajectory executed - vis. poses of panda eef:", dim=3)
        self.visualize_3d(data=dataPosePlot, color='b', label="Real trajectory poses")
        self.visualize_3d(data=dataPoseGoalsPlot, color='r', label="Goal poses")


    def plotJointsCallViz(self, load_data=False, plotToppraPlan=False, plotVelocities=True, plotAccelerations=False, plotEfforts=False, md=None):
        ''' NEED TO BE UPDATED!

        Visualize data + show. Loading series from:
                - Sended trajectory values: settings.sendedPlot, settings.sendedPlotVel
                - The joint states values: settings.realPlot, settings.realPlotVel
                - Section of toppra execution, start/end pts: [settings.point_before_toppra, settings.point_after_toppra]
                - Section of trajectory replacement, start/end pts: [settings.point_after_toppra, settings.point_after_replace]
            Note: Every plot visualization takes ~200ms

        Parameters:
            load_data (bool): Loads joint_states positions and velocities to get up-to-date trajectories
            plotVelocities (bool): Plots velocities
            plotToppraPlan (bool): Plots toppra RobotTrajectory plan
            plotAccelerations (bool): Plots accelerations
            plotEfforts (bool): Plots efforts
        '''
        # Load/Update Data
        if load_data or settings.simulator == 'coppelia':
            dataJointPlot = [pt.position[settings.NJ] for pt in list(md.joint_states)]
            timeJointPlot = [pt.header.stamp.to_sec() for pt in list(md.joint_states)]
            settings.realPlot = zip(timeJointPlot, dataJointPlot)
            timeJointPlotVel = [pt.header.stamp.to_sec() for pt in list(md.joint_states)]
            dataJointPlotVel = [pt.velocity[settings.NJ] for pt in list(md.joint_states)]
            settings.realPlotVel = zip(timeJointPlotVel, dataJointPlotVel)

        # Plot positions
        settings.viz.visualize_new_fig(title="Trajectory number "+str(settings.loopn)+" executed - vis. position of panda_joint"+str(settings.NJ+1), dim=2)
        if settings.robot == 'panda' and (settings.simulator == 'gazebo' or settings.simulator == 'real'):
            settings.viz.visualize_2d(data=settings.sendedPlot, color='r', label="Replaced (sended) trajectory position", scatter_pts=True)
            settings.viz.visualize_2d(data=[settings.point_before_toppra, settings.point_after_toppra], color='y', label="Toppra executing")
            settings.viz.visualize_2d(data=[settings.point_after_toppra, settings.point_after_replace], color='k', label="Replace executing")
        else:
            pass
        settings.viz.visualize_2d(data=settings.realPlot, color='b', label="Real (joint states) trajectory position", xlabel='time (global rospy) [s]', ylabel='joint positons [rad]')

        # Plot velocities
        if plotVelocities:
            settings.viz.visualize_new_fig(title="Trajectory number "+str(settings.loopn)+" executed - vis. velocity of panda_joint"+str(settings.NJ+1), dim=2)

            if settings.robot == 'panda' and (settings.simulator == 'gazebo' or settings.simulator == 'real'):
                settings.viz.visualize_2d(data=settings.sendedPlotVel, color='r', label="Replaced (sended) trajectory velocity", scatter_pts=True)
            else:
                pass
            settings.viz.visualize_2d(data=settings.realPlotVel, color='b', label="Real (states) velocity", xlabel='time (global rospy) [s]', ylabel='joint velocities [rad/s]')

        # Plot accelerations
        if plotAccelerations:
            dataPlot = [pt.accelerations[settings.NJ] for pt in md._goal.trajectory.points]
            timePlot = [pt.time_from_start.to_sec()+md._goal.trajectory.header.stamp.to_sec() for pt in md._goal.trajectory.points]
            timeJointPlot = [pt.header.stamp.to_sec() for pt in list(md.joint_states)]
            dataJointPlot = [pt.effort[settings.NJ] for pt in list(md.joint_states)]
            settings.figdata = visualizer_lib.visualize_new_fig(title="Loop"+str(settings.loopn)+" ACC", dim=2)
            settings.viz.visualize_2d(data=zip(timePlot, dataPlot), color='r', label="sended trajectory accelerations", transform='front')
            settings.viz.visualize_2d(data=zip(timeJointPlot, dataJointPlot), color='b', label="real efforts")

        # Plot efforts
        if plotEfforts:
            #dataPlot = [pt.effort[settings.NJ] for pt in md._goal.trajectory.points]
            timePlot = [pt.time_from_start.to_sec()+md._goal.trajectory.header.stamp.to_sec() for pt in md._goal.trajectory.points]
            timeJointPlot = [pt.header.stamp.to_sec() for pt in list(md.joint_states)]
            dataJointPlot = [pt.effort[settings.NJ] for pt in list(md.joint_states)]
            settings.viz.visualize_new_fig(title="Path", dim=2)
            #settings.viz.visualize_2d(data=zip(timePlot, dataPlot), color='r', label="sended trajectory effort")
            settings.viz.visualizer_lib.visualize_2d(data=zip(timeJointPlot, dataJointPlot), color='b', label="real effort")

        if plotToppraPlan:
            self.plot_plan(plan=settings.toppraPlan)



if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='')

    parser.add_argument('--rosecho', default='false', type=str, help='(default=%(default)s)', choices=['true','false'])
    parser.add_argument('--topic', default='/followjointtrajectorygoalforplot', type=str)
    args=parser.parse_args()

    main(args)
