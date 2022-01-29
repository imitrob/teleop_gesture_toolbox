import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations

#os.chdir(os.path.expanduser("~/promps_python"))
sys.path.insert(1, os.path.expanduser("~/promps_python"))

from promp.discrete_promp import DiscretePROMP
from promp.linear_sys_dyn import LinearSysDyn
from promp.promp_ctrl import PROMPCtrl
from numpy import diff

if False:
    pickle_in = open(os.path.expanduser('~/promps_python/data/data.pkl'),"rb")
    data = pickle.load(pickle_in, encoding='latin1')
    demos_list    = [data['steps']['states'][k][0][:,0] for k in range(100)]
    Ddemos_list   = [data['steps']['states'][k][0][:,1] for k in range(100)]

demos_list = []
Ddemos_list = []

def construct_promp_trajectories(Xpalm, Y):
    ''' Main function for generating trajectories

    '''
    assert isinstance(Y, np.ndarray), "Not right type"
    assert isinstance(Xpalm, np.ndarray), "Not right type"
    g = list(dict.fromkeys(Y))
    g
    counts = [list(Y).count(g_) for g_ in g]
    counts
    promp_paths = []
    for n in range(0,len(counts)):
        row = []
        for dim in range(0,3):
            demos_list = Xpalm[Y==n,:,dim]
            d_promp = DiscretePROMP(data=demos_list)
            d_promp.train()

            meanstart = np.mean([i[0] for i in demos_list])
            meangoal = np.mean([i[-1] for i in demos_list])
            d_promp.set_start(meanstart) #demos_list[0][0])
            d_promp.set_goal(meangoal) #demos_list[0][-1])
            pos_2, vel_2, acc_2 = d_promp.generate_trajectory(phase_speed=1.,   randomness=0.)

            row.append(pos_2.T[0])
        promp_paths.append(row)


    promp_paths = np.moveaxis(np.array(promp_paths), 1, 2)

    return promp_paths

    ## CHECK THE SERIES
    for n in range(0,1):
        for dim in range(0,1):
            demos_list = Xpalm[sum(counts[0:n]):sum(counts[0:n])+counts[n],0:100,dim]
            Ddemos_list = DXpalm[sum(counts[0:n]):sum(counts[0:n])+counts[n],0:100,dim]
            pos = demo_generate_traj()
            promp_paths[n, dim] = pos[0]
            plt.plot(pos, 'g')


def construct_promp_trajectories_waypoints(Xpalm, Y, waypoints={}):
    ''' Main function for generating trajectories
    Parameters:
        waypoints ({n x 3}): n waypoints, each waypoint has 3 dims
    '''
    assert isinstance(Y, np.ndarray), "Not right type"
    assert isinstance(Xpalm, np.ndarray), "Not right type"
    g = list(dict.fromkeys(Y))
    counts = [list(Y).count(g_) for g_ in g]
    promp_paths = []
    for n in range(0,len(counts)):
        row = []
        for dim in range(0,3):
            demos_list = Xpalm[Y==n,:,dim]
            d_promp = DiscretePROMP(data=demos_list)
            d_promp.train()

            for key in list(waypoints.keys()):
                d_promp.add_viapoint(key, waypoints[key][dim])

            pos_2, vel_2, acc_2 = d_promp.generate_trajectory(phase_speed=1.,   randomness=0.)

            row.append(pos_2.T[0])
        promp_paths.append(row)

    promp_paths = np.moveaxis(np.array(promp_paths), 1, 2)

    return np.array(promp_paths)

def construct_promp_trajectory_waypoints(Xpalm, waypoints={}):
    ''' Main function for generating trajectories
    Parameters:
        waypoints ({n x 3}): n waypoints, each waypoint has 3 dims
    '''
    startpos = [0.0, 0.0, 2.0]
    goalpos = [0.0, 0.0, 5.0]
    assert isinstance(Xpalm, np.ndarray), "Not right type"
    promp_paths = []
    for dim in range(0,3):
        demos_list = Xpalm[:,:,dim]
        d_promp = DiscretePROMP(data=demos_list)
        d_promp.train()

        for key in list(waypoints.keys()):
            if key == 0.0:
                d_promp.set_start(waypoints[key].p[dim])
            elif key == 1.0:
                print(f"assinging goal at {waypoints[key][dim]}")
                d_promp.set_goal(waypoints[key].p[dim])
            else:
                d_promp.add_viapoint(key, waypoints[key].p[dim])

        pos_2, vel_2, acc_2 = d_promp.generate_trajectory(phase_speed=1.,   randomness=0.)
        promp_paths.append(pos_2.T[0])

    promp_paths = np.moveaxis(np.array(promp_paths), 0, 1)

    return np.array(promp_paths)

def get_weights(Xpalm,DXpalm,Y):
    '''
        What to try:
        different batch size
        record more samples
        combinations between samples to get more weights
        differentiate with covariance matrix
    '''
    g = list(dict.fromkeys(Y))
    g
    counts = [list(Y).count(g_) for g_ in g]
    counts

    Xpalm.shape
    # samples x times x dim


    X = []
    Y_ = []
    for n in range(0,len(counts)): # gests cca 7?
        ind = list(range(0,counts[n],2))
        for m in range(0,len(ind)-1): # cca 41?
            w_dims = []
            for dim in range(0,3):
                demos_list = Xpalm[sum(counts[0:n])+ind[m]:sum(counts[0:n])+ind[m+1],0:100,dim]
                demos_list
                Ddemos_list = DXpalm[sum(counts[0:n])+ind[m]:sum(counts[0:n])+ind[m+1],0:100,dim]
                d_promp = DiscretePROMP(data=demos_list, num_bfs=35)
                d_promp.train()

                w_dims.extend(d_promp._sigma_W)
            X.append(w_dims)
            Y_.append(n)
    Y = Y_
    X = np.array(X)
    Y = np.array(Y)

    return X,Y



def plot_mean_and_sigma(mean, lower_bound, upper_bound, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), lower_bound, upper_bound, color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)


def demo_generate_traj():

    #add a via point
    # d_promp.add_viapoint(0.7, 5)
    # plt.scatter(0.7, 5, marker='*', s=100)

    #set the start and goal, the spatial scaling
    d_promp.set_start(demos_list[0][0])
    d_promp.set_goal(demos_list[0][-1])

    #add a via point
    # d_promp.add_viapoint(0.3, 2.25)
    # d_promp.add_viapoint(0.6, 2.25)
    # plt.scatter(0.7, 5, marker='*', s=100)

    for traj, traj_vel in zip(demos_list, Ddemos_list):
        plt.figure("ProMP-Pos")
        plt.plot(traj, 'k', alpha=0.2)
        plt.figure("ProMP-Vel")
        plt.plot(traj_vel, 'k', alpha=0.2)

    for _ in  range(1):

        pos_1, vel_1, acc_1 = d_promp.generate_trajectory(phase_speed=1.,  randomness=1e-1)
        pos_2, vel_2, acc_2 = d_promp.generate_trajectory(phase_speed=1.,   randomness=1e-1)
        pos_3, vel_3, acc_3 = d_promp.generate_trajectory(phase_speed=1., randomness=1e-1)
        #pos_2, vel_2, acc_2 = d_promp.generate_trajectory(phase_speed=1.,   randomness=0)

        plt.figure("ProMP-Pos")
        plt.plot(pos_1, 'r')
        plt.plot(pos_2, 'g')
        plt.plot(pos_3, 'b')


        plt.figure("ProMP-Vel")
        plt.plot(vel_1, 'r')
        plt.plot(vel_2, 'g')
        plt.plot(vel_3, 'b')


def create_demo_traj():
    """
    This funciton shows how to compute
    closed form control distribution from the trajectory distribution
    """

    #this is just used for demo purposes
    lsd = LinearSysDyn()

    state  = data['steps']['states'][0][0]
    action = data['steps']['actions'][0][0]

    promp_ctl = PROMPCtrl(promp_obj=d_promp)
    promp_ctl.update_system_matrices(A=lsd._A, B=lsd._B)

    ctrl_cmds_mean, ctrl_cmds_sigma = promp_ctl.compute_ctrl_traj(state_list=state)

    plt.figure("Ctrl cmds")

    for k in range(lsd._action_dim):

        mean        = ctrl_cmds_mean[:, k]
        lower_bound = mean - 3.*ctrl_cmds_sigma[:, k, k]
        upper_bound = mean + 3*ctrl_cmds_sigma[:, k, k]

        plot_mean_and_sigma(mean=mean, lower_bound=lower_bound, upper_bound=upper_bound, color_mean='g', color_shading='g')

    plt.plot(action, 'r')

def my_promp(data, out='w'):
    global d_promp
    d_promp = DiscretePROMP(data=data)
    d_promp.train()
    demo_generate_traj()
    create_demo_traj()
    plt.show()


def main():
    len(demos_list)
    demos_list[0].shape
    demos_list[0][0]
    d_promp = DiscretePROMP(data=demos_list)
    d_promp.train()
    d_promp._W.shape


    demo_generate_traj()
    create_demo_traj()
    plt.show()


if __name__ == '__main__':
    main()
