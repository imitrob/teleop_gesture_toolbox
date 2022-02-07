import sys
import settings
sys.path.append(settings.paths.promp_sebasutp_path)
from matplotlib import pyplot as plt

import robpy.full_promp as promp
import robpy.utils as utils
import numpy as np

#1) Take the first 10 striking movements from a file with recorded demonstrations
'''
with open('/home/pierro/promp/examples/strike_mov.npz','rb') as f:
    data = np.load(f, allow_pickle=True, encoding = 'bytes')
    time = data['time'][0:10]
    Q = data['Q'][0:10]

Q.shape
Q[0].shape

time.shape
len(time[0])
'''
# We want: Recordings x Timesamples x DoF

def construct_promp_trajectory_waypoints(Xpalm, waypoints={}):
    Q = Xpalm
    time = []
    for q in Q:
        time.append(np.linspace(0,1,len(q)))
    time = np.array(time, dtype=object)
    #2) Create a ProMP with basis functions: 3 RBFs with scale 0.25 and
    #   centers 0.25, 0.5 and 0.75. Use also polynomial basis functions of
    #   degree one (constant and linear term)
    full_basis = {
            'conf': [
                    {"type": "sqexp", "nparams": 4, "conf": {"dim": 3}},
                    {"type": "poly", "nparams": 0, "conf": {"order": 1}}
                ],
            'params': [np.log(0.25),0.25,0.5,0.75]
            }
    robot_promp = promp.FullProMP(basis=full_basis)

    #3) Train ProMP with NIW prior on the covariance matrix (as described in the paper)

    dof = len(Q[0][0])
    dim_basis_fun = 5
    inv_whis_mean = lambda v, Sigma: utils.make_block_diag(Sigma, dof)
    prior_Sigma_w = {'v':dim_basis_fun*dof, 'mean_cov_mle': inv_whis_mean}
    train_summary = robot_promp.train(time, q=Q, max_iter=10, prior_Sigma_w=prior_Sigma_w,
            print_inner_lb=True)


    #4) Plot some samples from the learned ProMP and conditioned ProMP

    n_samples = 1 # Number of samples to draw
    plot_dof = 3 # Degree of freedom to plot
    sample_time = [np.linspace(0,1,200) for i in range(n_samples)]

    #4.1) Make some samples from the unconditioned ProMP
    promp_samples = robot_promp.sample(sample_time)

    #4.2) Condition the ProMP to start at q_cond_init and draw samples from it
    def condition(t, T, q, qd=[], ignore_Sy=False, dt=0.1):
        q = np.array(q)
        qd = np.array(qd)
        if qd is np.array([]):
            robot_promp.condition(t=t, T=T, q=q, ignore_Sy=ignore_Sy)
        elif None in qd:

            #n = len(q)
            #Sigma_q = np.zeros([n,n])
            #for n,qd_ in enumerate(qd):
            #    if qd_ is None:
            #        # TODO: np.inf
            #        Sigma_q[n] = [0.,0.,0.]#[999.,999.,999.]
            #        qd[n] = 0.0
            robot_promp.condition(t=t,    T=T,         q=q,         ignore_Sy=ignore_Sy)
            robot_promp.condition(t=t+dt, T=int(T+dt), q=(q+dt*qd), ignore_Sy=ignore_Sy)
        else:
            robot_promp.condition(t=t,    T=T,         q=q,         ignore_Sy=ignore_Sy)
            robot_promp.condition(t=t+dt, T=int(T+dt), q=(q+dt*qd), ignore_Sy=ignore_Sy)


    #robot_promp.condition(t=0.5, T=1, q=[0.5, 0.15, 0.04], Sigma_q=[[0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]], ignore_Sy=False)
    #robot_promp.condition(t=0.6, T=1, q=[0.5, 0.3, 0.04], ignore_Sy=False)
    #condition(t=0.5, T=1, q=[0.5, 0.15, 0.04], qd=[1.0, 0.0, 0.0])

    for wp_t in list(waypoints.keys()):
        if waypoints[wp_t].v is not None: # condition also velocity
            condition(t=wp_t, T=1, q=waypoints[wp_t].p, qd=waypoints[wp_t].v)
        else:
            robot_promp.condition(t=wp_t, T=1, q=waypoints[wp_t].p, ignore_Sy=False)

    cond_samples = robot_promp.sample(sample_time)


    cond_sample = cond_samples[0]
    # cond sample shape: time x DoF
    return cond_sample




""" Toy example of a Probabilistic Movement Primitive

"""
if False:
    # TEMPORARY: -> WILL BE CHANGED
    import sys; sys.path.append("/home/pierro/promp")
    import robpy.full_promp as promp
    import robpy.utils as utils
    import numpy as np
    import argparse
    import os
    import matplotlib.pyplot as plt
    import json


    full_basis = {
            'conf': [
                    {"type": "sqexp", "nparams": 6, "conf": {"dim": 5}},
                    {"type": "poly", "nparams": 0, "conf": {"order": 0}}
                ],
            'params': [np.log(0.1),0.0,0.25,0.5,0.75,1.0]
            }
    dim_basis_fun = promp.dim_comb_basis(**full_basis)
    dof = 1
    w_dim = dof*dim_basis_fun
    test_mu_w = np.array([-10,20,-12,15,-13,-5])
    test_sig_w = 9*np.eye(w_dim)

    inv_whis_mean = lambda v, Sigma: np.diag(np.diag(Sigma))
    params = {
            'new_kernel': full_basis,
            #'prior_mu_w': {"m0": np.zeros(5*7), "k0": 1},
            #'prior_Sigma_w': {'v':dim_basis_fun*dof, 'mean_cov_mle': inv_whis_mean},
            'model_fname': "/tmp/promp.json",
            'diag_sy': True,
            'opt_basis_pars': False,
            'print_inner_lb': True,
            'no_Sw': False, #Approx E-Step with Dirac delta?
            'num_joints': dof,
            'max_iter': 30,
            'init_params': {'mu_w': np.zeros(w_dim),
                'Sigma_w': 1e8*np.eye(w_dim),
                'Sigma_y': np.eye(dof)}
            }

    def create_toy_data(n=100, T=30, missing_obs=[40,40]):
        p = promp.FullProMP(model={'mu_w': test_mu_w,
            'Sigma_w': test_sig_w,
            'Sigma_y': np.eye(1)}, num_joints=1, basis=full_basis)
        times = [np.linspace(0,1,T) for i in range(n)]
        if missing_obs is not None:
            for i in range(missing_obs[0]): times[i] = np.delete(times[i], range(1,int(T/2)))
            for i in range(missing_obs[1]): times[i+missing_obs[0]] = np.delete(times[i+missing_obs[0]], range(int(T/2),T-1))
            #times = [np.delete(times[i], range(1 + (i % (T/2)),T/2 - 1 + (i % (T/2)))) for i in range(n)]
        Phi = []
        X = p.sample(times, Phi=Phi, q=True)
        return times, Phi, X

    def trivial_train(times, Phi, X):
        W = []
        for i, phi in enumerate(Phi):
            y = X[i]
            phi_a = np.array(phi)
            w = np.dot(np.linalg.pinv(phi_a[:,0,:]),y[:,0])
            W.append(w)
        mu_w = np.mean(W,axis=0)
        Sigma_w = np.cov(W, rowvar=0)
        return mu_w, Sigma_w

    def promp_train(times, X):
        p = promp.FullProMP(num_joints=1, basis=full_basis)
        p.train(times=times, q=X, **params)
        return p.mu_w, p.Sigma_w

    times, Phi, X = create_toy_data()
    len(Phi[0])

    for i,t in enumerate(times):
        plt.plot(t, X[i])
    plt.show()

    mu_w_t, sig_w_t = trivial_train(times, Phi, X)
    mu_w_p, sig_w_p = promp_train(times, X)

    print(f"mu_w= {test_mu_w}")
    print(f"mu_w_trivial= {mu_w_t}")
    print(f"mu_w_em= {mu_w_p}")

    print(f"sig_w= {test_sig_w}")
    print(f"sig_w_trivial= {sig_w_t}")
    print(f"sig_w_em= {sig_w_p}")



    #### CONDITIONING
    """ Very simple conditioning example

    Most of the other examples work with a 7 DoF robot. In this example, we show
    simply a Polynomial with fixed parameters on a single degree of freedom, and
    show how conditioning work in this scenario.
    """

    import robpy.full_promp as promp
    import robpy.utils as utils
    import numpy as np
    import argparse
    import os
    import matplotlib.pyplot as plt
    import json

    print(promp.__file__)

    full_basis = {
            'conf': [
                    {"type": "poly", "nparams": 0, "conf": {"order": 3}}
                ],
            'params': []
            }
    dim_basis_fun = promp.dim_comb_basis(**full_basis)
    dof = 1
    w_dim = dof*dim_basis_fun

    p = promp.FullProMP(num_joints=1, basis=full_basis)
    p.mu_w = np.zeros(w_dim)
    p.Sigma_w = 1e2*np.eye(w_dim)
    p.Sigma_y = 1e-4*np.eye(dof)

    n_samples = 5 # Number of samples to draw
    sample_time = [np.linspace(0,1,200) for i in range(n_samples)]

    #1) Condition the ProMP to pass through q=0.5 at time 0, q=-1 at time 0.5 and q=1 at time 1.0

    #1.1) Procedure 1: Use the E-step doing all at the same time
    e_step = p.E_step(times=[[0.0,0.5,1.0]], Y = np.array([[[0.5],[-1.0],[1.0]]]))

    #1.2) Procedure 2: Condition the ProMP one by one obs
    p.condition(t=0.0, T=1.0, q=np.array([0.5]), ignore_Sy=False)
    p.condition(t=0.5, T=1.0, q=np.array([-1]), ignore_Sy=False)
    p.condition(t=1.0, T=1.0, q=np.array([1]), ignore_Sy=False)

    # Draw samples of recursive conditioning
    recursive_cond_samples = p.sample(sample_time)

    # Draw samples of E-step conditioning
    p.mu_w = e_step['w_means'][0]
    p.Sigma_w = e_step['w_covs'][0]
    e_step_cond_samples = p.sample(sample_time)

    for i in range(n_samples):
        plt.plot(sample_time[i], recursive_cond_samples[i][:,0], color='green')
        plt.plot(sample_time[i], e_step_cond_samples[i][:,0], color='blue')
    plt.title('Samples of the conditioned ProMPs')
    plt.show()
