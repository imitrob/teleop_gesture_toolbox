'''
Look at tests/loading_test.py for example

Example to load MPs:
from os_and_utils.loading import HandDataLoader, DatasetLoader
self.X, self.Y, self.robot_promps = DatasetLoader(['interpolate']).load_mp(settings.paths.learn_path, self.Gs, approach)
Load dynamic:
dataloader_args = {'interpolate':1, 'discards':1, 'normalize':1, 'normalize_dim':1, 'n':0}
self.X, self.Y = DatasetLoader(dataloader_args).load_dynamic(settings.paths.learn_path, self.Gs)
'''
import numpy as np
import pickle
import sys,os
from copy import deepcopy
'''
#from leapmotion import frame_lib; frame_lib.__name__ = 'frame_lib'; frame_lib.__spec__.name = 'frame_lib'
if __name__ == '__main__':
    sys.path.append("..")
    sys.path.append("../leapmotion")
sys.path.append("leapmotion")
import frame_lib
'''
import leapmotion.frame_lib as frame_lib

from scipy.interpolate import interp1d
from sklearn.preprocessing import scale

from os_and_utils.transformations import Transformations as tfm

class HandDataLoader():
    def __init__(self, import_method='numpy', dataset_files=[]):
        if import_method not in ['numpy', 'pickle']:
            raise Exception(f"Invalid import_method: {import_method}, pick 'numpy or 'pickle'")
        self.import_method = import_method

        self.dataset_files = dataset_files

    def load_tmp(self, dir, type):
        dir = os.path.join(dir, "..")
        if not os.path.isfile(f"{dir}tmp_{type}{self.get_ext()}"):
            return False
        data = np.load(f"{dir}tmp_{type}{self.get_ext()}", allow_pickle=True).item()
        return data

    def load(self,dir,Gs,type):
        X,Y = self.load_directory(dir, Gs)
        dir_tmp = os.path.abspath(os.path.join(dir,f'../tmp_{type}'))
        np.save(dir_tmp, {'X': X, 'Y': Y, 'files':self.dataset_files})
        return X,Y

    def load_(self, dir, Gs,type):
        ''' Main function:
            1. Checks if tmp file exists
                - If not, it loads all directory, saves tmp, return X,Y
            2. If exists, checks the difference
                - If difference zero, return X,Y from tmp file
                - If some difference, loads again all directory, save tmp return X,Y
        '''
        data = self.load_tmp(dir,type)
        if not data:
            X,Y = self.load_directory(dir, Gs)
            dir_tmp = os.path.abspath(os.path.join(dir,f'../tmp_{type}'))
            np.save(dir_tmp, {'X': X, 'Y': Y, 'files':self.dataset_files})
            return X,Y

        data_files = data['files']
        real_files = self.get_files()
        diff = [item for item in real_files if item not in data_files]

        if diff == []:
            return data['X'], data['Y']
        else:
            X,Y = self.load_directory(dir, Gs)
            dir_tmp = os.path.abspath(os.path.join(dir,f'../tmp_{type}'))
            np.save(dir_tmp, {'X': X, 'Y': Y, 'files':self.dataset_files})
            return X, Y

    def get_files(self, dir, Gs):
        dataset_files = []
        for n, G in enumerate(Gs):
            i=0
            while os.path.isfile(f"{dir}/{G}/{str(i)}{self.get_ext()}"):
                i_file = f"{G}/{str(i)}{self.get_ext()}"
                with open(f"{dir}/{i_file}", 'rb') as input:
                    dataset_files.append(i_file)
                i+=1
        return dataset_files

    def load_directory(self, dir, Gs):
        ## Load data from file (%timeit ~7s)
        self.dataset_files = []
        X, Y = [], []
        for n, G in enumerate(Gs):
            i=0
            print(f"{dir}{G}/{str(i)}{self.get_ext()}")
            while os.path.isfile(f"{dir}/{G}/{str(i)}{self.get_ext()}"):
                i_file = f"{G}/{str(i)}{self.get_ext()}"
                print(f"i_file {i_file}")
                with open(f"{dir}/{i_file}", 'rb') as input:
                    self.dataset_files.append(i_file)

                    X.append(self.load_file(input))
                    Y.append(n)
                i+=1

        if X == []:
            print(('[WARN*] No data was imported! Check parameters path folder (learn_path) and gesture names (Gs)'))

        return X,Y

    def load_directory_update(self, dir, Gs):
        ## Load data from file (%timeit ~7s)
        X, Y = [], []
        for n, G in enumerate(Gs):
            i=0
            print(f"{dir}/{G}/{str(i)}{self.get_ext()}")
            while os.path.isfile(f"{dir}/{G}/{str(i)}{self.get_ext()}"):
                i_file = f"{G}/{str(i)}{self.get_ext()}"
                print(f"i_file {i_file}")
                with open(f"{dir}/{i_file}", 'rb') as input:
                    # Discard already learned part if it is in dataset_files array
                    if i_file in self.dataset_files:
                        i+=1
                        continue
                    self.dataset_files.append(i_file)

                    X.append(self.load_file(input))
                    Y.append(n)
                i+=1

        if X == []:
            print(('[WARN*] No data was imported! Check parameters path folder (learn_path) and gesture names (Gs)'))

        return X,Y

    def get_ext(self):
        if self.import_method=='numpy':
            return '.npy'
        elif self.import_method=='pickle':
            return '.pkl'

    def load_file(self, input):
        if self.import_method == 'numpy':
            return np.load(input, encoding='latin1', allow_pickle=True)
        elif self.import_method == 'pickle':
            return pickle.load(input, encoding="latin1")

#X,Y = HandDataLoader().load_directory('/home/<user>/<your dir>', ['asd'])


class DatasetLoader():
    def __init__(self, args={}):
        self.args = args

        self.hdl = HandDataLoader()

    def load_tmp(self, dir, type='static'):
        print(f"[Loading] Dataset tmp directory file is {dir}tmp_{type}{HandDataLoader().get_ext()}")
        if not os.path.isfile(f"{dir}tmp_{type}{HandDataLoader().get_ext()}"):
            return False
        data = np.load(f"{dir}tmp_{type}{HandDataLoader().get_ext()}", allow_pickle=True).item()
        return data

    def load_static(self, dir, Gs, out='', new=False):
        return self.load(dir=dir, Gs=Gs, type='static', out=out, new=new)

    def load_dynamic(self, dir, Gs, out='', new=False):
        return self.load(dir=dir, Gs=Gs, type='dynamic', out=out, new=new)

    def load_mp_tmp(self,dir,Gs):
        data = np.load(f"{dir}tmp_mp{HandDataLoader().get_ext()}", allow_pickle=True).item()
        return data['X'], data['Y']

    def load_mp(self,dir,Gs, approach, new=False):
        ## Load movements same as dynamic
        X, Y, diff = self.load(dir=dir, Gs=Gs, type='mp', out='diff', new=new)
        ## Check trained MPs
        robot_promps = []
        '''
        if diff:
        '''
        for i in range(len(list(set(Y)))):
            robot_promps.append(approach.construct_promp_trajectory(X[Y==i]))
        #print("saving")
        #np.save(f"{dir}tmp_trained_promps.pkl", robot_promps)
        #with open(f"{dir}tmp_trained_promps.pkl", 'wb') as file:
        #    pickle.dump(robot_promps, file)

        '''
        else:
            #robot_promps = np.load(f"{dir}tmp_trained_promps.pkl", allow_pickle=True).item()
            with open(f"{dir}tmp_trained_promps.pkl", 'rb') as file:
                robot_promps = pickle.load(file)
        '''
        return X,Y,robot_promps

    def load(self, dir, Gs, type = 'dynamic', out='', new=False):
        data = self.load_tmp(dir, type=type)
        files = self.hdl.get_files(dir,Gs)
        if not data or new:
            print("[Loading] No data, loading all directory")
            HandData, HandDataFlags = self.hdl.load_directory(dir, Gs)
            if type == 'dynamic':
                X,Y = self.get_dynamic(HandData,HandDataFlags)
            elif type == 'static':
                X,Y = self.get_static(HandData,HandDataFlags)
            elif type == 'mp':
                X,Y = self.get_mp(HandData,HandDataFlags)
            else: raise Exception("Wrong type")

            np.save(f"{dir}tmp_{type}", {'X': X, 'Y': Y, 'files': files})
            assert X.shape[0] == Y.shape[0], "Wrong Xs and Ys"
            if out=='diff':
                return X,Y,True
            return X,Y

        data_files = data['files']
        real_files = files
        diff = [item for item in real_files if item not in data_files]

        if diff == []:
            print("[Loading] No difference, tmp up to date")
            if out=='diff':
                return data['X'], data['Y'],False
            return data['X'], data['Y']
        else:
            print("[Loading] Difference, loading all directory again")
            HandData, HandDataFlags = self.hdl.load_directory(dir, Gs)
            if type == 'dynamic':
                X,Y = self.get_dynamic(HandData,HandDataFlags)
            elif type == 'static':
                X,Y = self.get_static(HandData,HandDataFlags)
            elif type == 'mp':
                X,Y = self.get_mp(HandData,HandDataFlags)
            else: raise Exception("Wrong type")

            np.save(f"{dir}tmp_{type}", {'X': X, 'Y': Y, 'files': files})
            assert X.shape[0] == Y.shape[0], "Wrong Xs and Ys"
            if out=='diff':
                return X,Y,True
            return X,Y


    def get_static(self, data, flags):
        X = data
        Y = flags

        X_, Y_ = [], []
        for n, X_n in enumerate(X):

            gesture_X = []
            for m, X_nt in enumerate(X_n):
                # Recording is done always only with one hand
                # right hand has priority
                if X_nt.r.visible:
                    gesture_X.append(np.array(X_nt.r.get_learning_data(definition=self.args['input_definition_version'])))
                elif X_nt.l.visible:
                    gesture_X.append(np.array(X_nt.l.get_learning_data(definition=self.args['input_definition_version'])))
            if not gesture_X: continue

            X_.append(gesture_X)
            Y_.append(Y[n])

        X = X_
        Y = Y_

        if 'interpolate' in self.args:
            X_interpolated = []
            for n,sample in enumerate(X):
                X_interpolated_sample = []
                for dim in range(0,len(sample[0])):
                    f2 = interp1d(np.linspace(0,1, num=len(np.array(sample)[:,dim])), np.array(sample)[:,dim], kind='cubic')
                    X_interpolated_sample.append(f2(np.linspace(0,1, num=101)))
                X_interpolated.append(np.array(X_interpolated_sample).T)
            X=np.array(X_interpolated)
            print("Data interpolated")


        X_ = []
        Y_ = []
        for n, X_n in enumerate(X):
            if 'middle' in self.args:
                X_.append(deepcopy(X_n[0]))
                Y_.append(deepcopy(Y[n]))
            elif 'average'in self.args: X_.append(deepcopy(avg_dataframe(X_n)))
            elif 'as_dimesion' in self.args: X_.append(deepcopy(X_n))
            elif 'take_every' in self.args:
                for i in range(0, len(X_n), int(self.args['take_every'])):
                    X_.append(deepcopy(X_n[i]))
                    Y_.append(deepcopy(Y[n]))
            else: # 'take_every' == 10
                for i in range(0, len(X_n), 10):
                    X_.append(deepcopy(X_n[i]))
                    Y_.append(deepcopy(Y[n]))

        X = np.array(X)
        Y = np.array(Y)
        print(f"shape 1 {np.array(X).shape} y {np.array(Y).shape}")
        X,Y = self.to_2D(X,Y)
        print(f"shape 2 {np.array(X).shape} y {np.array(Y).shape}")
        X,Y = self.postprocessing(X,Y)
        print(f"shape 3 {X.shape} y {Y.shape}")
        X,Y = self.discard(X,Y)
        print(f"shape 4 {X.shape} y {Y.shape}")

        return X, Y

    def discard(self, X, Y):
        discards = []
        for n,rec in enumerate(X):
            for obs in rec:
                if np.isnan(obs):
                    discards.append(n)
                    break
        X = np.delete(X,discards,axis=0)
        Y = np.delete(Y,discards)
        print(f"[Loading] Discarted {len(discards)} records")
        return X,Y

    #import numpy as np
    #A = np.array([0,1,2,3,4,5,6])
    #B = [1,3,5]
    #np.delete(A,B)

    def postprocessing(self, X, Y):

        try:
            X = scale(X)
        except:
            print("[Loading] Scale function failed")

        try:
            # Not in a loop, import can be here -> dependency
            import theano
            floatX = theano.config.floatX
            X = X.astype(floatX)
            Y = Y.astype(floatX)
        except:
            print("[Loading] Converting as X.astype(floatX) failed!")
        X = np.array(X)
        Y = np.array(Y)
        return X,Y

    def get_Xpalm(self, data):
        ## Pick: samples x time x palm_position
        Xpalm = []
        for sample in data:
            row = []
            for t in sample:

                l2 = np.linalg.norm(np.array(t.r.palm_position()) - np.array(t.r.index_position()))
                row.append([*t.r.palm_position(), l2])#,*t.r.index_position])
            Xpalm.append(row)
        return Xpalm

    def get_mp(self, data, flags):
        return self.get_dynamic(data, flags)

    def get_dynamic(self, data, flags):
        Y = flags
        ## Pick: samples x time x palm_position
        Xpalm = []
        for sample in data:
            row = []
            for t in sample:
                if t.r.visible:
                    #l2 = np.linalg.norm(np.array(t.r.palm_position()) - np.array(t.r.index_position()))
                    row.append([*t.r.palm_position()])#, np.linalg.norm(t.r.palm_velocity())])#, l2])#,*t.r.index_position])
            Xpalm.append(row)

        ''' Discard when trajectory points count < 5
        '''
        discards = []
        Xpalm = np.array(Xpalm, dtype=object)
        for n,p in enumerate(Xpalm):
            if len(p) < 5:
                discards.append(n)
        Xpalm = np.delete(Xpalm,discards,axis=0)
        Y = np.delete(Y,discards,axis=0)
        print(f"[Loading] Number {len(discards)} recordings discarded! discards {discards}")

        if 'normalize' in self.args:
            Xpalm = np.array(Xpalm)
            Xpalm_ = []
            for p in Xpalm:
                p_ = []
                p0 = deepcopy(p[len(p)//2])
                for n in range(0, len(p)):
                    p_.append(np.subtract(p[n], p0))
                Xpalm_.append(p_)

            Xpalm = Xpalm_

        ''' Interpolate palm_positions, to n time samples
        '''
        N = 100 if not 'n' in self.args else self.args['n']
        N_observations = 3 if not 'n_observations' in self.args else self.args['n_observations']
        Xpalm_interpolated = []
        invalid_ids = []
        for n,sample in enumerate(Xpalm):
            Xpalm_interpolated_sample = []
            try:
                for dim in range(0,N_observations):
                    f2 = interp1d(np.linspace(0,1, num=len(np.array(sample)[:,dim])), np.array(sample)[:,dim], kind='cubic')
                    Xpalm_interpolated_sample.append(f2(np.linspace(0,1, num=N)))
                Xpalm_interpolated.append(np.array(Xpalm_interpolated_sample).T)
            except IndexError:
                print("Sample with invalid data detected, IndexError")
                invalid_ids.append(n)
            except ValueError:
                print("Sample with invalid data detected, ValueError")
                invalid_ids.append(n)
        Xpalm=Xpalm_interpolated=np.array(Xpalm_interpolated)
        Y = np.delete(Y,invalid_ids,axis=0)

        '''
        dx = 1/100
        DXpalm_interpolated = []
        for sample in Xpalm_interpolated:
            DXpalm_interpolated_sample = []
            sampleT = sample.T
            for dim in range(0,3):
                DXpalm_interpolated_sample.append(np.diff(sampleT[dim]))
            DXpalm_interpolated.append(np.array(DXpalm_interpolated_sample).T)
        DXpalm_interpolated = np.array(DXpalm_interpolated)
        DXpalm = np.array(DXpalm_interpolated)
        '''

        # get direction vectors
        # Xpalm n x 32 x 3
        '''
        Xpalm_ = []
        for path in Xpalm:
            row = []
            for i in range(len(path)-1):
                point_ = path[i+1,0:3] - path[i,0:3]
                row.append([*point_])#, path[i,3]])
            row.append([0.,0.,0.])
            Xpalm_.append(row)
        Xpalm = np.array(Xpalm_)
        '''
        '''
        Xpalm_ = []
        for xpalm, xvel in zip(Xpalm, Xvel):
            xpalm_= []
            for obspalm, obsvel in zip(xpalm, xvel):
                xpalm_.append([*obspalm, *obsvel])
            Xpalm_.append(xpalm_)
        Xpalm = np.array(Xpalm_)
        '''

        #np.save('/home/petr/Documents/tmp.npy', Xpalm)
        #Xpalm = np.load('/home/petr/Documents/tmp.npy', allow_pickle=True)
        #Xpalm.shape
        '''
        if 'normalize_dim' in self.args:
            Xpalm_ = []
            Xpalm = np.swapaxes(Xpalm, 1, 2)
            for n,dim1 in enumerate(Xpalm):
                for m,dim2 in enumerate(dim1):
                    if (np.max(dim2) - np.min(dim2)) < 0.0000001:
                        Xpalm[n,m] = np.inf
                        continue
                    Xpalm[n,m] = (dim2 - np.min(dim1)) / (np.max(dim1) - np.min(dim1))
            Xpalm = np.swapaxes(Xpalm, 1, 2)
            Xpalm = np.array(Xpalm)
        '''
        ''' Reverse path: Start is end
            - How data are saved, it saves as reversed
        '''

        Xpalm = np.array(Xpalm)
        Xpalm_ = []
        for p in Xpalm:
            p_ = []
            for n in range(len(p)):
                p_.append(p[-n-1])
            Xpalm_.append(p_)

        Xpalm = np.array(Xpalm_)

        '''
        if 'scene_frame' in self.args:
        '''
        Xpath = tfm.transformLeapToBase_3D(Xpalm)

        ''' scale limit
        '''
        if 'scale_limit' in self.args:
            limit_distance = 0.2
            data_ = []
            for path in Xpath:
                path_ = []
                path = np.swapaxes(path, 0, 1)
                for dim in range(3):
                    _1d = path[dim]
                    if (_1d.max() - _1d.min()) > limit_distance:
                        path_.append(_1d/(_1d.max() - _1d.min())*limit_distance)
                    else:
                        path_.append(_1d)
                path_ = np.swapaxes(path_, 0, 1)
                data_.append(path_)
            Xpath = data_
            print("ADDED SCALE LIMIT")

        ''' Discard nan and inf
        '''
        discards = self.get_discard_indices(Xpalm,Y)
        #discards.extend(self.get_discard_indices(DXpalm,Y))

        Xpalm = np.delete(Xpalm,discards,axis=0)
        #DXpalm = np.delete(DXpalm,discards,axis=0)
        Y = np.delete(Y,discards,axis=0)
        print(f"[Loading] Number {len(discards)} recordings discarded! discards {discards}")

        return Xpalm, Y

    ''' e.g.: 2 gestures, 2 recordings for each gesture, 2 observations
    import numpy as np
    data = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    flags = [0,1]
    data, flags = to_2D(None,data,flags)
    data # array([[1, 2], [3, 4], [5, 6], [7, 8]])
    flags # array([0, 0, 1, 1])
    '''
    def to_2D(self, data, flags):
        '''
        Parameters:
            data (3D): [g, recording, observation]
            flags (1D): [g] (e.g. [0,1,2])
        Returns:
            data (2D): [recording, observation]
            flags (1D): [g*recordings] (e.g. [])
        '''
        data_, flags_ = [], []
        for n,gesture in enumerate(data):
            data_.extend(gesture)
            flags_.extend([flags[n]] * len(gesture))
        return np.array(data_), np.array(flags_)

    '''
    import numpy as np
    A = np.array([[[np.nan],[np.inf]],[[3],[4]]])
    np.delete(A,[1], axis=0)
    a = np.inf
    a == np.inf
    X = get_discard_indices(None, A, [0,0])
    X
    '''

    def get_discard_indices(self,X,Y):
        discards = []
        for n,rec in enumerate(X):
            if np.isinf(rec).any() or np.isnan(rec).any():
                discards.append(n)
        return discards

def avg_dataframe(data_n):
    data_avg = np.zeros([len(data_n[0])])
    for data_nt in data_n:
        data_avg += data_nt
    data_avg *= (1/len(data_n))
    return data_avg


if __name__ == '__main__':
    print("Updating tmp data files")

    import os_and_utils.settings as settings; settings.init()
    import gesture_classification.gestures_lib as gl; gl.init()

    dataloader_args = {'normalize':1, 'n':5, 'scene_frame':1, 'inverse':1}
    DatasetLoader(dataloader_args).load(settings.paths.learn_path, gl.gd.Gs_dynamic, new=True)
    DatasetLoader({'input_definition_version':1}).load_static(settings.paths.learn_path, gl.gd.Gs_static, new=True)
