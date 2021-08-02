import numpy as np
from matplotlib import pyplot as plt
import time

import sys
import os
from os.path import expanduser
HOME = expanduser("~")
sys.path.append(HOME+'/promp')
sys.path.insert(1, HOME+'/promp/examples/python_promp')
import robpy.full_promp as promp
import robpy.utils as utils
from visualizer_lib import visualize_2d, visualize_2d_joints_ntraj
from types import MethodType

def promp_(showcase, input_config=None, input_f='promp_input.npz', output_f='promp_output.npz'):
    #1) Take the first 10 striking movements from a file with recorded demonstrations
    numLeapRec = input_config['leap_records']
    with open(HOME+'/iiwa_ws/src/motion_primitives_vanc/include/data/'+input_f,'rb') as f:
        data = np.load(f, allow_pickle=True, encoding='latin1')
        time = data['time'][0:numLeapRec]
        Q = data['Q'][0:numLeapRec]

    # Input data may be in list form -> need array type
    Q = [np.array(q) for q in Q]

    full_basis = input_config['basis']
    robot_promp = promp.FullProMP(basis=full_basis)

    #2) Train ProMP with NIW prior on the covariance matrix (as described in the paper)
    dof = 7
    dim_basis_fun = input_config['dim_basis']
    inv_whis_mean = lambda v, Sigma: input_config['v_rate']*utils.make_block_diag(Sigma, dof) + input_config['Sigma_rate']*np.eye(dof*dim_basis_fun)
    prior_Sigma_w = {'v':dim_basis_fun*dof, 'mean_cov_mle': inv_whis_mean}
    train_summary = robot_promp.train(time, q=Q, max_iter=input_config['max_iter'], prior_Sigma_w=prior_Sigma_w,
            print_inner_lb=True)


    #3) Make some samples from the learned ProMP and conditioned ProMP
    n_samples = input_config['n_samples'] # Number of most probable samples
    sample_time = [np.linspace(0,1,200) for i in range(n_samples)]

    # Samples from the unconditioned ProMP
    promp_samples = robot_promp.sample(sample_time)

    # Condition the ProMP
    ## first joints in trajectory
    if input_config['q_cond_init'] is not None:
        q_cond_init = np.array(input_config['q_cond_init'])
    else:
        Q1 = Q[0]
        Q1_0 = Q1[0]
        q_cond_init = Q1_0
        print("Q1_0",Q1_0)

    robot_promp.condition(t=input_config['cond_time'], T=input_config['T'], q=q_cond_init, ignore_Sy=False)
    if input_config['q_cond_end'] is not None:
        q_cond_end = np.array(input_config['q_cond_end'])
        robot_promp.condition(t=1.9, T=input_config['T'], q=q_cond_end, ignore_Sy=False)
    cond_samples = robot_promp.sample(sample_time)

    #4) An example of conditioning in Task space
    import robpy.kinematics.forward as fwd
    # Compute the prior distribution in joint space at the desired time
    time_cartesian = 0.9
    mean_marg_w, cov_marg_w = robot_promp.marginal_w(np.array([0.0,time_cartesian,1.0]), q=True)
    prior_mu_q = mean_marg_w[1]
    prior_Sigma_q = cov_marg_w[1]

    # Compute the posterior distribution in joint space after conditioning in task space
    fwd_kin = fwd.BarrettKinematics()
    # Update robot to iiwa
    fwd_kin._link_matrices = MethodType(_link_matrices_iiwa, fwd_kin)
    prob_inv_kin = promp.ProbInvKinematics(fwd_kin)

    mu_cartesian = np.array(input_config['mu_cartesian']) #np.array([-0.62, -0.44, -0.34])
    Sigma_cartesian = input_config["Sigma_cartesian"]*np.eye(3) #0.02**2*np.eye(3)

    mu_q, Sigma_q = prob_inv_kin.inv_kin(mu_theta=prior_mu_q, sig_theta=prior_Sigma_q,
            mu_x = mu_cartesian, sig_x = Sigma_cartesian)

    # Finally, condition in joint space using the posterior joint space distribution
    robot_promp.condition(t=time_cartesian, T=input_config['T'], q=mu_q, Sigma_q=Sigma_q, ignore_Sy=False)
    task_cond_samples = robot_promp.sample(sample_time)


    #5) Save results to file
    np.savez(HOME+"/iiwa_ws/src/motion_primitives_vanc/src/"+output_f, Q=promp_samples[0], time=sample_time)
    result = {"promp_samples": promp_samples, "cond_samples": cond_samples, "task_cond_samples": task_cond_samples, 'time': sample_time}
    return result

def _link_matrices_iiwa(self, joints):
    ''' Update for ProMP robot parameters
    '''
    DH=np.array([[joints[0], 0.34, 0, -90],
                 [joints[1], 0.0, 0, 90],
                 [joints[2], 0.4, 0, 90],
                 [joints[3], 0.0, 0, -90],
                 [joints[4], 0.4, 0, -90],
                 [joints[5], 0.0, 0, 90],
                 [joints[6], 0.126, 0, 0]])
    hi = []
    hi.append(np.eye(4))
    for i in range(0, len(DH)):
        t = DH[i, 0]
        d = DH[i, 1]
        a = DH[i, 2]
        al= DH[i, 3]
        T = np.array([[math.cos(t), -math.sin(t)*math.cos(math.radians(al)), math.sin(t)*math.sin(math.radians(al)), a*math.cos(t)],
              [math.sin(t), math.cos(t)*math.cos(math.radians(al)), -math.cos(t)*math.sin(math.radians(al)), a*math.sin(t)],
              [0, math.sin(math.radians(al)), math.cos(math.radians(al)), d],
              [0, 0, 0, 1]])
        hi.append(T)
    hi.append(np.eye(4))
    return hi

if __name__ == "__main__":
    promp_()
