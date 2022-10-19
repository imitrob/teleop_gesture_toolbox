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
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time as t
import argparse, random

from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Circle, Ellipse
from itertools import product
from mpl_toolkits.mplot3d import art3d

#import seaborn as sns
#sns.set_theme(style="darkgrid")

try:
    from os_and_utils import settings; settings.init()
except ModuleNotFoundError:
    pass

# Ensure package independency to ROS
try:
    from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Vector3Stamped, QuaternionStamped, Vector3
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    import rclpy
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

    def visualize_new_fig(self, title=None, dim=3, move_figure=True):
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
        if move_figure:
            self.move_figure(fig, BOXXMOVE[nfigs], BOXYMOVE[nfigs])

        self.fig = fig
        self.ax = ax

    def savefig(self, dir):
        plt.savefig(dir)

    def visualize_2d(self, data, color='', label="", transform='front', xlabel='X axis', ylabel='Y axis', scatter_pts=False, axvline=False, start_stop_mark=True):
        ''' Add trajectory to current figure for 2D Plot.

        Args:
            data (2D array): [[X1,Y1], [X2,Y2], [X3,Y3], ..] or possible to use list(zip([X1,X2,X3],[Y1,Y2,Y3]))
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
        if start_stop_mark:
            self.ax.scatter(xt[0], yt[0], marker='o', color='black', zorder=2)
            self.ax.scatter(xt[-1], yt[-1], marker='x', color='black', zorder=2)
        if axvline:
            self.ax.axvline(x=xt[-1], color=color)
        if scatter_pts:
            self.ax.scatter(xt, yt, marker='|', color='black', zorder=2, alpha=0.3)

        if label != "":
            self.ax.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.0))

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
        rclpy.init(args=None)
        raise Exception("TODO")

        rclpy.Subscriber(topic, FollowJointTrajectoryGoal, save_trajectory)

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

    @staticmethod
    def cuboid_data(center, size):
        """
           Create a data array for cuboid plotting.


           ============= ================================================
           Argument      Description
           ============= ================================================
           center        center of the cuboid, triple
           size          size of the cuboid, triple, (x_length,y_width,z_height)
           :type size: tuple, numpy.array, list
           :param size: size of the cuboid, triple, (x_length,y_width,z_height)
           :type center: tuple, numpy.array, list
           :param center: center of the cuboid, triple, (x,y,z)


          """
        # suppose axis direction: x: to left; y: to inside; z: to upper
        # get the (left, outside, bottom) point
        o = [a - b / 2 for a, b in zip(center, size)]
        # get the length, width, and height
        l, w, h = size
        x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
             [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
             [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
             [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
        y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
             [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
             [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
             [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
        z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
             [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
             [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
             [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
        return np.array(x), np.array(y), np.array(z)

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
                - eef poses: md.eef_pose_array
                - goal poses: md.goal_pose_array
        '''

        if load_data:
            dataPosePlot = [pt.position for pt in list(md.eef_pose_array)]
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
        settings.viz.visualize_2d(data=settings.realPlot, color='b', label="Real (joint states) trajectory position", xlabel='time (global ros) [s]', ylabel='joint positons [rad]')

        # Plot velocities
        if plotVelocities:
            settings.viz.visualize_new_fig(title="Trajectory number "+str(settings.loopn)+" executed - vis. velocity of panda_joint"+str(settings.NJ+1), dim=2)

            if settings.robot == 'panda' and (settings.simulator == 'gazebo' or settings.simulator == 'real'):
                settings.viz.visualize_2d(data=settings.sendedPlotVel, color='r', label="Replaced (sended) trajectory velocity", scatter_pts=True)
            else:
                pass
            settings.viz.visualize_2d(data=settings.realPlotVel, color='b', label="Real (states) velocity", xlabel='time (global ros) [s]', ylabel='joint velocities [rad/s]')

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

class ScenePlot:
    '''
    promp_paths = approach.construct_promp_trajectories2(X, Y, start='mean')
    promp_paths_0_0 = approach.construct_promp_trajectories2(X, Y, start='0')
    promp_paths_0_1 = approach.construct_promp_trajectories2(X, Y, start='')
    promp_paths_test1 = approach.construct_promp_trajectories2(X, Y, start='test')

    promp_paths_grab_mean = [promp_paths[0], promp_paths_0_0[0], promp_paths_0_1[0]]
    my_plot(X[Y==0], promp_paths_grab_mean)

    promp_paths_kick_mean = [promp_paths[1], promp_paths_0_0[1], promp_paths_0_1[1]]
    my_plot(X[Y==1], promp_paths_kick_mean)

    promp_paths_nothing_mean = [promp_paths[2], promp_paths_0_0[2], promp_paths_0_1[2]]
    my_plot(X[Y==2], promp_paths_nothing_mean)
    '''
    @staticmethod
    def get_variabilities_dimensions(point_variability, path_normal):

        def get_radius_elipse_with_normal(a,b, vx,vy):
            ''' Radius from direction vector, input is normal vector
            Parameters:
                a,b (Float): elipse dimensions
                normal (Float): normal vector
            Returns:
                r (Float): radius
            '''
            theta = np.arctan2(vy,vx)
            theta_dir = theta + np.pi/2

            e = np.sqrt(1-(b**2/a**2))
            r = a * np.sqrt(1-e**2 * (np.sin(theta_dir)**2))
            return r
        x = get_radius_elipse_with_normal(point_variability[0], point_variability[1], path_normal[0], path_normal[1])
        y = get_radius_elipse_with_normal(point_variability[0], point_variability[2], path_normal[0], path_normal[2])

        return x,y

    @staticmethod
    def add_boundbox(data):
        # TODO:

        # Create cubic bounding box to simulate equal aspect ratio
        X = np.array(boundbox[0]); Y = np.array(boundbox[1]); Z = np.array(boundbox[2])
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax.plot([xb], [yb], [zb], 'w')

    @staticmethod
    def my_plot(data, promp_paths, waypoints=None, leap=True, boundbox=[[-1.0,1.0],[-1.0,1.0],[0.0,2.0]], filename='', size=(6,5), legend=[], series_marking='d', promp_paths_variances=[]):
        '''
        Parameters:
            promp_paths_variances (Float[n x 6]): n path points, 6 = (x,y,z,var_x,var_y,var_z)

        '''
        if legend == []: legend = [f'Series {n}' for n in range(len(promp_paths))]

        plt.rcParams["figure.figsize"] = size
        ax = plt.axes(projection='3d')
        for path in data:
            ax.plot3D(path[:,0], path[:,1], path[:,2], 'blue', alpha=0.2)
            ax.scatter(path[:,0][0], path[:,1][0], path[:,2][0], marker='o', color='black', zorder=2)
            ax.scatter(path[:,0][-1], path[:,1][-1], path[:,2][-1], marker='x', color='black', zorder=2)
        colors = ['blue','black', 'yellow', 'red', 'cyan', 'green', 'blue','black', 'yellow', 'red', 'cyan', 'green']
        annotations = [('left','top'), ('left','top'), ('right','bottom'), ('right','bottom'),('left','bottom')]

        ''' Plot 3D paths '''
        for n in range(len(promp_paths)):
            path = promp_paths[n]
            if waypoints is not None: waypoints_ = waypoints[n]
            else: waypoints_ = {}
            ax.plot3D(path[:,0], path[:,1], path[:,2], colors[n], label=f"{legend[n]}", alpha=1.0, linewidth=3)
            ax.scatter(path[:,0][0], path[:,1][0], path[:,2][0], marker='o', color='black', zorder=2, s=30)
            ax.scatter(path[:,0][-1], path[:,1][-1], path[:,2][-1], marker='x', color='black', zorder=2, s=30)
            npoints = 5
            p = int(len(path[:,0])/npoints)
            if series_marking == '%':
                for m in range(npoints):
                    ax.text(path[:,0][m*p], path[:,1][m*p], path[:,2][m*p], str(100*m*p/len(path[:,0]))+"%", color='black',
                                fontsize="small", weight='light',
                                horizontalalignment=annotations[m][0],
                                verticalalignment=annotations[m][1])
            elif series_marking == 'd':
                ax.scatter(path[:,0][1:-1], path[:,1][1:-1], path[:,2][1:-1], marker='d', color='black', zorder=2)
            for m, waypoint_key in enumerate(list(waypoints_.keys())):
                waypoint = waypoints_[waypoint_key]
                s = f"wp {m} "
                if waypoint.gripper is not None: s += f'(gripper {waypoint.gripper})'
                if waypoint.eef_rot is not None: s += f'(eef_rot {waypoint.eef_rot})'
                ax.text(waypoint.p[0], waypoint.p[1], waypoint.p[2], s)
        ''' Plot 3D paths with variances '''
        for n in range(len(promp_paths_variances)):
            path = promp_paths_variances[n]

            ax.plot3D(path[:,0], path[:,1], path[:,2], colors[n], label=f"{legend[n]}", alpha=1.0)
            ax.scatter(path[:,0][0], path[:,1][0], path[:,2][0], marker='o', color='black', zorder=2)
            ax.scatter(path[:,0][-1], path[:,1][-1], path[:,2][-1], marker='x', color='black', zorder=2)
            #npoints = 10
            #p = len(path[:,0])//npoints
            for m in range(len(path[:,0])):#npoints):
                i = m#*p
                if 0 > i-1: continue
                if len(path[:,0]) <= i+1: continue
                # subtract indices (i + 1, i)
                v2 = np.array([path[:,0][i+1]-path[:,0][i], path[:,1][i+1]-path[:,1][i], path[:,2][i+1]-path[:,2][i]])
                # subtract indices (i, i - 1)
                v1 = np.array([path[:,0][i]-path[:,0][i-1], path[:,1][i]-path[:,1][i-1], path[:,2][i]-path[:,2][i-1]])

                direction_path_vector = v1 + v2
                variances = (path[:,3][i], path[:,4][i], path[:,5][i])

                ellipse_dim = ScenePlot.get_variabilities_dimensions(variances, direction_path_vector)
                #patch = Circle((0,0), variances[0], facecolor = 'b', alpha = .2)
                patch = Ellipse((0,0), ellipse_dim[0], height=0.01, facecolor = 'b', alpha = .2)
                ax.add_patch(patch)

                normal_vector = direction_path_vector
                pathpatch_2d_to_3d(patch, z = 0, normal = normal_vector)
                pathpatch_translate(patch, (path[:,0][i], path[:,1][i], path[:,2][i]))

        ax.legend()
        # Leap Motion
        if leap:
            X,Y,Z = VisualizerLib.cuboid_data([0.475, 0.0, 0.0], (0.004, 0.010, 0.001))
            ax.plot_surface(X, Y, Z, color='grey', rstride=1, cstride=1, alpha=0.5)
            ax.text(0.475, 0.0, 0.0, 'Leap Motion')

        # compatibiliy issues, need to import it here
        sl = None
        try:
            from os_and_utils import scenes as sl; sl.init()
        except:
            pass
        if sl.scene:
            for n in range(len(sl.scene.object_poses)):
                pos = sl.scene.object_poses[n].position
                size = sl.scene.object_sizes[n]
                X,Y,Z = VisualizerLib.cuboid_data([pos.x, pos.y, pos.z], (size.x, size.y, size.z))
                ax.plot_surface(X, Y, Z, color='yellow', rstride=1, cstride=1, alpha=0.8)

        if boundbox:
            # Create cubic bounding box to simulate equal aspect ratio
            X = np.array(boundbox[0]); Y = np.array(boundbox[1]); Z = np.array(boundbox[2])
            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
            Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
            Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
            Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
            # Comment or uncomment following both lines to test the fake bounding box:
            for xb, yb, zb in zip(Xb, Yb, Zb):
               ax.plot([xb], [yb], [zb], 'w')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.text(0.05, 0.0,0.95,'• - start\n× - target', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
        if filename:
            plt.savefig(f'{os.path.expanduser("~")}/Documents/{filename}.png', format='png')

        save_fig_and_data(plt, np.array((data, promp_paths, waypoints), dtype=object), filename)
        plt.show()
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    import fastdtw, os
    sin1 = np.sin(np.linspace(0, 10, 100))
    sin2 = np.sin(np.linspace(2, 12, 100))
    data = np.array([sin1, sin2])
    dist, ind = fastdtw.fastdtw(sin1, sin2)
    dtw_plot(data,ind)
    '''
    @staticmethod
    def dtw_plot(data, ind, name='dtw_plot'):
        '''
        Parameters:
            data (Float[2, n]): Two paths: Representative & Test
            ind (Float[m, 2]): Indices: Representative & Test, m>=n
            name (Str): For plot & data save
        '''
        data1 = data[0]
        data2 = data[1]
        plt.xlabel('Time [-]')
        plt.ylabel('Values [-]')
        plt.plot(data.T, linewidth=2)
        plt.subplots_adjust(left=0.15)
        plt.grid(True)
        indice_lines = []
        for ind1, ind2 in ind:
            indice_lines.append([[ind1, ind2], [data1[ind1], data2[ind2]]])
            plt.plot([ind1, ind2], [data1[ind1], data2[ind2]], linewidth=1, color='grey')

        save_fig_and_data(plt, np.array((indice_lines, data.T), dtype=object), name)

    '''
    import fastdtw
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    data_repre = np.array([np.linspace(0,1,100), np.sin(np.linspace(0,1,100)), np.cos(np.linspace(0,1,100))])
    data_repre.shape
    data_repre_var = np.ones((3,100)) * 0.1

    data_test = np.array([np.linspace(0,1,100)+1, np.sin(np.linspace(0,1,100)), np.cos(np.linspace(0,1,100))])

    data = np.array([np.vstack((data_repre, data_repre_var)), data_test], dtype=object)
    data = np.array([data_repre, data_test], dtype=object)

    dist, ind = fastdtw.fastdtw(data[0][0:3].T, data[1].T)

    data[0].shape
    data = [data[0].T, data[1].T]

    dtw_3dplot(data, ind)

    import scipy.stats
    scipy.stats.norm(0, 1)
    scipy.stats.norm(0, 1).pdf(0)
    scipy.stats.norm(0, 1).cdf(0)

    scipy.stats.norm(100, 12).pdf(101)

    data, ind = np.load(f'/home/{os.getlogin()}/Pictures/dtw_3dplot_2022-04-27 14:38:53.141357.npy', allow_pickle=True)
    dtw_3dplot(data, ind, name='dtw_3dplot2')
    '''
    @staticmethod
    def dtw_3dplot(data, ind, name='dtw_3dplot'):
        '''
        Parameters:
            data (Float[2, n, 3|6]): Two paths: Representative & Test
            ind (Float[m, 2]): Indices: Representative & Test, m>=n
            name (Str): For plot & data save
        '''
        plt.subplots_adjust(left=0.15)
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')

        data1 = data[0].T # n x 3|6 -> 3|6 x n
        data2 = data[1].T # n x 3|6 -> 3 x n

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.grid(True)
        indice_lines = []
        for ind1, ind2 in ind:
            indice_lines.append([[data1[0][ind1], data2[0][ind2]], [data1[1][ind1], data2[1][ind2]], [data1[2][ind1], data2[2][ind2]]])
            ax.plot3D([data1[0][ind1], data2[0][ind2]], [data1[1][ind1], data2[1][ind2]], [data1[2][ind1], data2[2][ind2]], linewidth=1, color='grey', label='_nolegend_')
        # rewrite -> to be on top
        ax.plot3D(data1[0],data1[1],data1[2], linewidth=4)
        ax.plot3D(data2[0],data2[1],data2[2], linewidth=4)
        ax.legend(['Representative', 'Test path'])

        if len(data1) > 3:
            for _ in range(100):
                random_path = []
                for i in range(len(data1[0])):
                    random_path.append([random.uniform(data1[0][i]-data1[3][i], data1[0][i]+data1[3][i]),
                                        random.uniform(data1[1][i]-data1[4][i], data1[1][i]+data1[4][i]),
                                        random.uniform(data1[2][i]-data1[5][i], data1[2][i]+data1[4][i])])
                random_path = np.array(random_path)
                ax.plot3D(random_path[:,0],random_path[:,1],random_path[:,2], linewidth=1, color='b', alpha=0.2)

        save_fig_and_data(plt, np.array((data, ind), dtype=object), name)

def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                  [-d[2],     0,  d[0]],
                  [d[1], -d[0],    0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0,0,0), index)

    normal /= np.linalg.norm(normal) #Make sure the vector is normalised

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color

    verts = path.vertices #Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector
    M = rotation_matrix(d) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])

def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta

def save_fig_and_data(plt, data, name, d=''):
    if os.path.isfile(f"/home/{os.getlogin()}/Pictures/conf.svg"): d=f"_{datetime.datetime.now()}"
    plt.savefig(f"/home/{os.getlogin()}/Pictures/{name}{d}.svg", format='svg')

    np.save(f'/home/{os.getlogin()}/Pictures/{name}{d}.npy', data)
    #dd = np.load(f'/home/{os.getlogin()}/Pictures/{name}.npy', allow_pickle=True)


if False:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib.patches import Circle
    from itertools import product

    ax = plt.axes(projection = '3d') #Create axes

    p = Circle((0,0), .2) #Add a circle in the yz plane
    ax.add_patch(p)
    pathpatch_2d_to_3d(p, z = 0.5, normal = 'x')
    pathpatch_translate(p, (0, 0.5, 0))

    p = Circle((0,0), .2, facecolor = 'r') #Add a circle in the xz plane
    ax.add_patch(p)
    pathpatch_2d_to_3d(p, z = 0.5, normal = 'y')
    pathpatch_translate(p, (0.5, 1, 0))

    p = Circle((0,0), .2, facecolor = 'g') #Add a circle in the xy plane
    ax.add_patch(p)
    pathpatch_2d_to_3d(p, z = 0, normal = 'z')
    pathpatch_translate(p, (0.5, 0.5, 0))

    for normal in product((-1, 1), repeat = 3):
        p = Circle((0,0), .2, facecolor = 'y', alpha = .2)
        ax.add_patch(p)
        pathpatch_2d_to_3d(p, z = 0, normal = normal)
        pathpatch_translate(p, 0.5)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='')

    parser.add_argument('--rosecho', default='false', type=str, help='(default=%(default)s)', choices=['true','false'])
    parser.add_argument('--topic', default='/followjointtrajectorygoalforplot', type=str)
    args=parser.parse_args()

    main(args)
