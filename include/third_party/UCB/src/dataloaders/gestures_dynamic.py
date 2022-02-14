import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
from itertools import accumulate

#sys.path.append("/home/petr/my_ws/src/mirracle_gestures/src/")
sys.path.append(os.path.abspath(os.getcwd()+'/../../../../../src'))
from os_and_utils.nnwrapper import NNWrapper
from os_and_utils.loading import HandDataLoader, DatasetLoader
from os_and_utils.utils import GlobalPaths
import settings; settings.init(change_working_directory=False)
import gestures_lib as gl; gl.init()

########################################################################################################################

def get_n_testtrain(Ylen, ratio=0.3):
    n_test = int(Ylen * ratio)
    n_train = Ylen - n_test
    return n_train, n_test

def get(data_path,seed,fixed_order=False,pc_valid=0,args=None):


    # id: n update
    _map = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0
    }
    Gs = gl.gd.l.dynamic.info.names
    dataloader_args = {'interpolate':1, 'discards':1, 'normalize':1, 'normalize_dim':1, 'n':0}
    dataloader_args['n'] = args.dataset_n_time_samples

    ###########################################################################################

    data={}
    taskcla=[]
    #size=[1,57,1] # from: size = [1,28,28], TODO: discard last Dimension somehow to size=[1,87]

    if 'static' in dataloader_args:
        g_type = 'static'
        X, Y = DatasetLoader(dataloader_args).load_static(GlobalPaths(change_working_directory=False).learn_path, Gs=Gs)
    else:
        g_type = 'dynamic'
        X, Y = DatasetLoader(dataloader_args).load_dynamic(GlobalPaths(change_working_directory=False).learn_path, Gs=Gs)
    uniqY = list(set(Y))
    size = list(X[0:1].shape)

    #my_plot(X[Y==0], [])
    #my_plot(X[Y==1], [])
    #my_plot(X[Y==2], [])
    #my_plot(X[Y==3], [])
    #my_plot(X[Y==4], [])

    n_train, n_test = get_n_testtrain(len(Y))

    data_splited = {'train': {'X': None, 'Y':None}, 'test': {'X':None, 'Y':None}}

    data_splited1 = torch.utils.data.random_split(X, [n_train, n_test], generator=torch.Generator().manual_seed(seed))
    data_splited['train']['X'] = data_splited1[0]
    data_splited['test']['X'] = data_splited1[1]
    data_splited2 = torch.utils.data.random_split(Y, [n_train, n_test], generator=torch.Generator().manual_seed(seed))
    data_splited['train']['Y'] = data_splited2[0]
    data_splited['test']['Y'] = data_splited2[1]

    n_updates = len(list(set([_map[k] for k in list(_map.keys())])))

    def make_m(_map):
        prev = -1
        m = []
        for k in list(_map.keys()):
            curr = _map[k]
            if prev != curr:
                m.append(k)
                prev = curr
        return m
    m = make_m(_map)
    tmp = [_map[k] for k in list(_map.keys())]
    _, counts = np.unique(tmp, return_counts=True)
    counts = list(counts)

    for n,ncla in enumerate(counts):
        data[n]={}
        data[n]['ncla']=ncla
        data[n]['name']=f"{data[n]['ncla']}-{g_type}-n-{n}"


    for s in ['train','test']:
        #loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False,drop_last=True)
        for n in range(n_updates):
            data[n][s]={'x': [],'y': []}
        for image,label in zip(data_splited[s]['X'], data_splited[s]['Y']):
            ud = _map[label]
            data[ud][s]['x'].append(image)
            data[ud][s]['y'].append(label-m[ud])


    # "Unify" and save
    for n in list(range(n_updates)):
        for s in ['train','test']:
            data[n][s]['x']=torch.tensor(np.array(data[n][s]['x']),dtype=torch.float32).view(-1,size[0],size[1],size[2])
            data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)

    # Validation
    for t in data.keys():
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['test']['x'].clone()
        data[t]['valid']['y']=data[t]['test']['y'].clone()

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size

########################################################################################################################
'''
Dataset visualize
'''
"""
import matplotlib.pyplot as plt
from os_and_utils.visualizer_lib import VisualizerLib

def my_plot(data, promp_path_waypoints_tuple):

    plt.rcParams["figure.figsize"] = (20,20)
    ax = plt.axes(projection='3d')
    for path in data:
        ax.plot3D(path[:,0], path[:,1], path[:,2], 'blue', alpha=0.2)
        ax.scatter(path[:,0][0], path[:,1][0], path[:,2][0], marker='o', color='black', zorder=2)
        ax.scatter(path[:,0][-1], path[:,1][-1], path[:,2][-1], marker='x', color='black', zorder=2)
    colors = ['blue','black', 'yellow', 'red', 'cyan', 'green']
    for n,path_waypoints_tuple in enumerate(promp_path_waypoints_tuple):
        path, waypoints = path_waypoints_tuple
        ax.plot3D(path[:,0], path[:,1], path[:,2], colors[n], label=f"Series {str(n)}", alpha=1.0)
        ax.scatter(path[:,0][0], path[:,1][0], path[:,2][0], marker='o', color='black', zorder=2)
        ax.scatter(path[:,0][-1], path[:,1][-1], path[:,2][-1], marker='x', color='black', zorder=2)
        npoints = 5
        p = int(len(path[:,0])/npoints)
        for n in range(npoints):
            ax.text(path[:,0][n*p], path[:,1][n*p], path[:,2][n*p], str(100*n*p/len(path[:,0]))+"%")
        for n, waypoint_key in enumerate(list(waypoints.keys())):
            waypoint = waypoints[waypoint_key]
            s = f"wp {n} "
            if waypoint.gripper is not None: s += f'(gripper {waypoint.gripper})'
            if waypoint.eef_rot is not None: s += f'(eef_rot {waypoint.eef_rot})'
            ax.text(waypoint.p[0], waypoint.p[1], waypoint.p[2], s)
    ax.legend()
    # Leap Motion
    X,Y,Z = VisualizerLib.cuboid_data([0.475, 0.0, 0.0], (0.004, 0.010, 0.001))
    ax.plot_surface(X, Y, Z, color='grey', rstride=1, cstride=1, alpha=0.5)
    ax.text(0.475, 0.0, 0.0, 'Leap Motion')
    '''
    for n in range(len(sl.scene.object_poses)):
        pos = sl.scene.object_poses[n].position
        size = sl.scene.object_sizes[n]
        X,Y,Z = VisualizerLib.cuboid_data([pos.x, pos.y, pos.z], (size.x, size.y, size.z))
        ax.plot_surface(X, Y, Z, color='yellow', rstride=1, cstride=1, alpha=0.8)
    '''
    # Create cubic bounding box to simulate equal aspect ratio
    X = np.array([0.3,0.7]); Y = np.array([-0.2, 0.2]); Z = np.array([0.0, 0.5])
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
    #plt.savefig('/home/pierro/Documents/test_promp_nothing_4_differentstarts.png', format='png')
    plt.show()
"""
