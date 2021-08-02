import math
import numba
import numpy as np
# import autograd.numpy as np
# from autograd import grad 
#@numba.jit
def to_rad(deg):
    return np.pi * deg / 180.0

#@numba.jit
def iiwa_forward_kinematics(joints, out='xyz'):
    ''' Direct Kinematics from iiwa structure. Using its dimensions and angles.
    '''
    #joints = [0,0,0,0,0,0,0]
    DH=np.array([[joints[0], 0.34, 0, -90],
                 [joints[1], 0.0, 0, 90],
                 [joints[2], 0.4, 0, 90],
                 [joints[3], 0.0, 0, -90],
                 [joints[4], 0.4, 0, -90],
                 [joints[5], 0.0, 0, 90],
                 [joints[6], 0.126, 0, 0]])
    Tr = np.eye(4)

    for i in range(0, len(DH)):
        t = DH[i, 0]
        d = DH[i, 1]
        a = DH[i, 2]
        al= DH[i, 3]
        # rad_al = math.radians(al)
        rad_al = to_rad(al)
        T = np.array([[np.cos(t), -np.sin(t)*np.cos(rad_al), np.sin(t)*np.sin(rad_al), a*np.cos(t)],
              [np.sin(t), np.cos(t)*np.cos(rad_al), -np.cos(t)*np.sin(rad_al), a*np.sin(t)],
              [0, np.sin(rad_al), np.cos(rad_al), d],
              [0, 0, 0, 1]])
        Tr = np.matmul(Tr, T)
    #pp = [Tr[0,3], Tr[1,3], T[2,3]]
    #pp
    if out=='xyz':
        return Tr[0:3,3]
        # return [Tr[0,3], Tr[1,3], Tr[2,3]]
    if out=='matrix':
        return Tr
    return None

​

#@numba.jit
def iiwa_jacobian(state):
    fun = iiwa_forward_kinematics
    eps = 0.001
    jacobian = np.zeros((3,7))
​
    inp = np.array(state)
    selector = np.array([0,1,2,3,4,5,6])

    for i in selector:
        jacobian[:,i] = (fun(inp + eps* (selector == i)) - fun(inp - eps* (selector == i))) / (2*eps)
    # print(jacobian)
    return jacobian

# def jacobian_test():
#     J = iiwa_jacobian(7*[0.5])
#     eef_vel = np.dot(J, np.array(7*[0.1]))
#
# import timeit
#
#
# print(timeit.timeit(jacobian_test, number=1000))
#
# jacobian_test()
# jacobian_test()
# jacobian_test()
#
# print('timeit')
#
# print(timeit.timeit(jacobian_test, number=1000))
#
# print(timeit.timeit(jacobian_test, number=1000))
​
# jacobian_test()
# grad_iiwa_fk = grad(iiwa_forward_kinematics)
# print(grad_iiwa_fk(7*[0.01]))
# print(iiwa_forward_kinematics(7*[0.01]))
