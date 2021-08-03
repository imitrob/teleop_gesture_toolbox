
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time as t
from std_msgs.msg import ColorRGBA
''' Usage:
    fig, ax = visualize_new_fig() -> creates new figure
    visualize_2d() -> adds new trajectory
    visualize_2d_ntraj -> adds n trajectories
    visualize_2d_joints_ntraj -> adds n trajectories from joints
    Before closing app:
    plt.ioff() # turn off interactive mode
    plt.show()
'''

def visualize_new_fig(title=None, dim=3):
    ''' 1. Creates new: fig, ax
        2. Moves new figure, position for 6 figures on 1080p monitor
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
    BOXXMOVE = [0, 600, 1200, 0, 600, 1200]
    BOXYMOVE = [0, 0, 0, 600, 600, 600]
    move_figure(fig, BOXXMOVE[nfigs], BOXYMOVE[nfigs])

    return fig, ax


def visualize_2d(data, storeObj, color='', label="", transform='front', units='m'):
    ''' Visualization in 2D
        Options: color
                 label - legend
                 transform - When data is 3D, it transforms to 2D
                           - look from 'top', 'front', 'left'
    '''
    data = np.array(data)
    if transform == 'top':
        data = data[:,1:3]
    if transform == 'front':
        data = data[:,0:2]
    if transform == 'left':
        data = np.delete(data, 1, 0)
    assert len(data[0]) == 2, "Data not valid, points are not [x,y] type"
    plt.ion() # turn on interactive model

    xt, yt = [], []
    for n, point in enumerate(data):
        xt.append(point[0])
        yt.append(point[1])
    storeObj.ax.set_xlabel('X axis ['+units+']')
    storeObj.ax.set_ylabel('Y axis ['+units+']')
    storeObj.ax.grid(b=True)

    plt.axis('equal')
    COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if color=="":
        color=COLORS[len(storeObj.ax.lines)%7]

    storeObj.ax.plot(xt,yt,c=color, label=(label))#+" "+str(len(dparsed))) )
    storeObj.ax.scatter(xt[0], yt[0], marker='o', color='black', zorder=2)
    storeObj.ax.scatter(xt[-1], yt[-1], marker='x', color='black', zorder=2)
    if label != "":
        plt.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.0))

    #plt.annotate("Num points:", xy=(-0.15, 1.0), xycoords='axes fraction')
    storeObj.fig.canvas.draw()
    storeObj.fig.canvas.flush_events()

def visualize_3d(data, storeObj, color='', label="", units='m'):
    ''' Add trajectory to current figure for 3D Plot.
    '''
    # data must be in (n x 3)
    assert len(data[0]) == 3, "Data not valid, points are not [x,y,z] type"

    plt.ion() # turn on interactive mode
    fig = storeObj.fig
    ax = storeObj.ax
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
    ax.set_xlabel('X axis ['+units+']')
    ax.set_ylabel('Y axis ['+units+']')
    ax.set_zlabel('Z axis ['+units+']')
    plt.axis('equal')

    COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if color=="":
        color=COLORS[len(ax.lines)%7]

    ax.plot3D(xt,yt,zt,c=color, label=(label))#+" "+str(len(data))) )
    storeObj.ax.scatter(xt[0], yt[0], zt[0], marker='o', color='black', zorder=2)
    storeObj.ax.scatter(xt[-1], yt[-1], zt[-1], marker='x', color='black', zorder=2)
    plt.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.0))
    #plt.annotate("Num points:", xy=(-0.15, 1.0), xycoords='axes fraction')
    fig.canvas.draw()
    fig.canvas.flush_events()

def convert_scene_to_points(data):
    dparsed = []
    # convert to array
    try:
        points = data.joint_trajectory.points
        assert points != [], "No points in given plan!"
        for point in points:
            T = moveit_lib.iiwa_forward_kinematics(point.positions)
            position = [T[0,3], T[1,3], T[2,3]]
            dparsed.append(position)
            #point.time_from_start
    except NameError:
        print(data.joint_trajectory.points[0].positions,"\n","Name Error, Data is in unsupported type: ", type(data))
    return dparsed

def find_best_azimuth(data):
    data = np.array(data)
    x1 = data[0, 0]
    x2 = data[len(data)-1, 0]
    y1 = data[0, 1]
    y2 = data[len(data)-1, 1]
    xd = abs(x2 - x1)
    yd = abs(y2 - y1)
    angle_q = np.rad2deg(np.arctan2(xd,yd))
    return angle_q

def visualize_ntraj(Q, storeObj, n=None, units='m', dim=3, transform="top"):
    ''' Plot multiple trajecories. visualize_2d() called n times.
    '''
    assert (dim == 3) or (dim == 2), "wrong dimension: "+str(dim)
    if n==None:
        n = len(Q)
    for i in range(0,n):
        traj = []
        for j in Q[i]:
            traj.append(j)
        if dim == 3:
            visualize_3d(traj, storeObj, units=units)
        if dim == 2:
            visualize_2d(traj, storeObj, units=units, transform=transform)

def visualize_2d_joints_ntraj(Q, storeObj, n=None, color='b', label=""):
    ''' Plot multiple trajecories for JOINTS. visualize_2d() called n times with iiwa_forward_kinematics().
    '''
    from moveit_lib import iiwa_forward_kinematics
    if n==None:
        n = len(Q)
    for i in range(0,n):
        traj = []
        for j in Q[i]:
            traj.append(iiwa_forward_kinematics(j))
        visualize_2d(traj, storeObj, color=color, label=(str(i)+label))

def visualize_refresh(storeObj, s=60):
    i = 0
    while i<s*10:
        if i == 900:
            i = 0
        i += 1
        t.sleep(0.1)
        storeObj.ax.elev = 90 - i/2.0
        storeObj.fig.canvas.draw()
        storeObj.fig.canvas.flush_events()

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

def draw_path(self, mg=None, size=0.002, color=ColorRGBA(1, 0, 0, 1), ns='path'):
    publisher = rospy.Publisher('/ee_path', Marker, queue_size=1)
    m = Marker()
    m.header.frame_id = 'world'
    m.header.stamp = rospy.Time.now()
    m.type = m.SPHERE_LIST
    m.pose.orientation.w = 1
    m.scale.x = size
    m.scale.y = size
    m.scale.z = size
    m.color = color
    # m.color.a = 0.9
    # m.color.r = 1.0
    m.action = m.ADD
    m.ns = ns
    m.pose.position.x = 0.0  # robot_info.get_eef2tip_transform(group_name).transform.translation.x
    m.pose.position.y = 0.0  # robot_info.get_eef2tip_transform(group_name).transform.translation.y
    m.pose.position.z = -0.0  # robot_info.get_eef2tip_transform(group_name).transform.translation.z
    # while ta_client.state() != GoalStatus.SUCCEEDED:
    pose = PoseStamped(header=Header(frame_id='world', stamp=rospy.Time.now()))
    while True:
        # draw the line
        pose.header.stamp = rospy.Time.now()
        pose = mg.get_current_pose(mg.get_end_effector_link())  # type: PoseStamped
        # pose_dict = self.get_eef_pose()
        # joint_angles = ta_client.get_current_tp().positions
        # fk_result = fk.getFK(mg.get_end_effector_link(), mg.get_active_joints(), joint_angles)  # type: GetPositionFKResponse
        # p = Point(*pose_dict['position'])
        p = Point()
        p.x = pose.pose.position.x
        p.y = pose.pose.position.y
        p.z = pose.pose.position.z

        m.colors.append(ColorRGBA(m.color.r, m.color.g, m.color.b, m.color.a))
        m.action = Marker.ADD
        m.points.append(p)
        publisher.publish(m)
        yield True

def main():
    ''' Test of printing multiple plots at once.
    '''
    plt.ion()

    fig, ax = visualize_2d_new_fig("plota")
    storeObj = type('storeObj', (object,), {'fig' : fig, 'ax': ax})
    visualize_2d([[0,0,0],[0,1,1],[0,2,2],[0,3,3]], storeObj=storeObj)
    visualize_2d([[1,0,0],[1,1,1],[1,2,2],[1,3,3]], storeObj=storeObj)

    fig, ax = visualize_2d_new_fig("plot22222")
    storeObj = type('storeObj', (object,), {'fig' : fig, 'ax': ax})
    Q = np.array([[[0,0,0],[0,1,1],[0,2,2],[0,3,3]], [[1,0,0],[1,1,1],[1,2,2],[1,3,3]]])
    visualize_2d_ntraj(Q, storeObj)

    fig, ax = visualize_2d_new_fig()
    storeObj = type('storeObj', (object,), {'fig' : fig, 'ax': ax})
    visualize_2d([[0,0,0],[0,1,1],[0,2,2],[0,3,3]], storeObj=storeObj)

    fig, ax = visualize_2d_new_fig()
    storeObj = type('storeObj', (object,), {'fig' : fig, 'ax': ax})
    visualize_2d([[0,0,0],[0,1,1],[0,2,2],[0,3,3]], storeObj=storeObj)

    fig, ax = visualize_2d_new_fig()
    storeObj = type('storeObj', (object,), {'fig' : fig, 'ax': ax})
    visualize_2d([[0,0,0],[0,1,1],[0,2,2],[0,3,3]], storeObj=storeObj)

    fig, ax = visualize_2d_new_fig()
    storeObj = type('storeObj', (object,), {'fig' : fig, 'ax': ax})
    visualize_2d([[0,0,0],[0,1,1],[0,2,2],[0,3,3]], storeObj=storeObj)

    plt.ioff()
    plt.show()
    return




if __name__ == "__main__":
    main()
