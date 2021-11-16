import numpy as np

class Raw_Kinematics():
    ''' Includes forward kinematics and jacobian computation for Panda and KUKA iiwa
        TODO: Check DH parameters on some example
    '''

    def __init__():
        pass

    @staticmethod
    def forward_kinematics(joints, robot='panda', out='xyz'):
        ''' Direct Kinematics from iiwa/panda structure. Using its dimensions and angles.
            TODO: Panda has link8 which is is not here of lenght ~0.106m
        '''
        if robot == 'iiwa':
            #             /theta   , d   , a,     alpha
            DH=np.array([[joints[0], 0.34, 0, -90],
                         [joints[1], 0.0, 0, 90],
                         [joints[2], 0.4, 0, 90],
                         [joints[3], 0.0, 0, -90],
                         [joints[4], 0.4, 0, -90],
                         [joints[5], 0.0, 0, 90],
                         [joints[6], 0.126, 0, 0]])
        elif robot == 'panda':
            #             /theta   , d   , a,     alpha
            DH=np.array([[joints[0], 0.333, 0,      0],
                         [joints[1], 0.0,   0,      -90],
                         [joints[2], 0.316, 0,      90],
                         [joints[3], 0.0,   0.0825, 90],
                         [joints[4], 0.384, -0.0825,-90],
                         [joints[5], 0.0,   0,      90],
                         [joints[6], 0.0, 0.088,  90]])
        else: raise Exception("Wrong robot name chosen!")

        Tr = np.eye(4)
        for i in range(0, len(DH)):
            t = DH[i, 0]
            d = DH[i, 1]
            a = DH[i, 2]
            al= DH[i, 3]
            T = np.array([[np.cos(t), -np.sin(t)*np.cos(np.radians(al)), np.sin(t)*np.sin(np.radians(al)), a*np.cos(t)],
                  [np.sin(t), np.cos(t)*np.cos(np.radians(al)), -np.cos(t)*np.sin(np.radians(al)), a*np.sin(t)],
                  [0, np.sin(np.radians(al)), np.cos(np.radians(al)), d],
                  [0, 0, 0, 1]])
            Tr = np.matmul(Tr, T)
        if out=='xyz':
            return [Tr[0,3], Tr[1,3], Tr[2,3]]
        if out=='matrix':
            return Tr
        return None

    @staticmethod
    def jacobian(state, robot='panda'):
        fun = Raw_Kinematics.forward_kinematics
        eps = 0.001
        jacobian = np.zeros((3,7))

        inp = np.array(state)
        selector = np.array([0,1,2,3,4,5,6])

        for i in selector:
            jacobian[:,i] = (np.array(fun(inp + eps* (selector == i), robot=robot)) - np.array(fun(inp - eps* (selector == i), robot=robot))) / (2*eps)
        # print(jacobian)
        return jacobian
