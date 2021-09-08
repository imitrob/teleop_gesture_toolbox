#!/usr/bin/env python2.7
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

    def visualize_2d(self, data, color='', label="", transform='front', xlabel='X axis', ylabel='Y axis', scatter_pts=False):
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

        plt.axis('equal')
        COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        if color=="":
            color=COLORS[len(self.ax.lines)%7]

        self.ax.plot(xt,yt,c=color, label=(label))#+" "+str(len(dparsed))) )
        self.ax.scatter(xt[0], yt[0], marker='o', color='black', zorder=2)
        self.ax.scatter(xt[-1], yt[-1], marker='x', color='black', zorder=2)
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
        ax.azim = find_best_azimuth(data)
        ax.alpha = 0
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        plt.axis('equal')

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
        print("vis done")

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

def main():
    ''' Test of printing multiple plots at once.
    '''
    viz = VisualizerLib()

    viz.visualize_new_fig("plot1")
    viz.visualize_2d([[0,0,0],[0,1,1],[0,2,2],[0,3,3]])
    viz.visualize_2d([[1,0,0],[1,1,1],[1,2,2],[1,3,3]])

    viz.visualize_new_fig("plot2")
    data = np.array([[[0,0,0],[0,1,1],[0,2,2],[0,3,3]], [[1,0,0],[1,1,1],[1,2,2],[1,3,3]]])
    viz.visualize_ntraj(data)

    viz.visualize_new_fig()
    viz.visualize_2d([[0,0,0],[0,1,1],[0,2,2],[0,3,3]])

    viz.visualize_new_fig()
    viz.visualize_2d([[0,0,0],[0,1,1],[0,2,2],[0,3,3]])

    viz.visualize_new_fig()
    viz.visualize_2d([[0,0,0],[0,1,1],[0,2,2],[0,3,3]])

    viz.visualize_new_fig()
    viz.visualize_2d([[0,0,0],[0,1,1],[0,2,2],[0,3,3]])

    viz.show()
    return

if __name__ == "__main__":
    main()
