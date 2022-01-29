
'''
Look at tests/loading_test.py for example
'''
import numpy as np
import pickle
import sys,os
from copy import deepcopy

sys.path.append('../leapmotion')
import frame_lib

from scipy.interpolate import interp1d
from sklearn.preprocessing import scale

class HandDataLoader():
    def __init__(self, import_method='numpy', dataset_files=[]):
        if import_method not in ['numpy', 'pickle']:
            raise Exception(f"Invalid import_method: {import_method}, pick 'numpy or 'pickle'")
        self.import_method = import_method

        self.dataset_files = dataset_files

    def load_tmp(self, dir):
        dir = os.path.join(dir, "..")
        if not os.path.isfile(f"{dir}/tmp{self.get_ext()}"):
            return False
        data = np.load(f"{dir}/tmp{self.get_ext()}", allow_pickle=True).item()
        return data

    def load(self,dir,Gs):
        X,Y = self.load_directory(dir, Gs)
        dir_tmp = os.path.abspath(os.path.join(dir,'../tmp'))
        np.save(dir_tmp, {'X': X, 'Y': Y, 'files':self.dataset_files})
        return X,Y

    def load_(self, dir, Gs):
        ''' Main function:
            1. Checks if tmp file exists
                - If not, it loads all directory, saves tmp, return X,Y
            2. If exists, checks the difference
                - If difference zero, return X,Y from tmp file
                - If some difference, loads again all directory, save tmp return X,Y
        '''
        data = self.load_tmp(dir)
        if not data:
            X,Y = self.load_directory(dir, Gs)
            dir_tmp = os.path.abspath(os.path.join(dir,'../tmp'))
            np.save(dir_tmp, {'X': X, 'Y': Y, 'files':self.dataset_files})
            return X,Y

        data_files = data['files']
        real_files = self.get_files()
        diff = [item for item in real_files if item not in data_files]

        if diff == []:
            return data['X'], data['Y']
        else:
            X,Y = self.load_directory(dir, Gs)
            dir_tmp = os.path.abspath(os.path.join(dir,'../tmp'))
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
            print(f"{dir}/{G}/{str(i)}{self.get_ext()}")
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

#X,Y = HandDataLoader().load_directory('/home/pierro/Downloads', ['asd'])


class DatasetLoader():
    def __init__(self, args={}):
        self.args = args

        self.hdl = HandDataLoader()

    def load_tmp(self, dir):
        print(f"[Loading] dir file {dir}/tmp{HandDataLoader().get_ext()}")
        if not os.path.isfile(f"{dir}/tmp{HandDataLoader().get_ext()}"):
            return False
        data = np.load(f"{dir}/tmp{HandDataLoader().get_ext()}", allow_pickle=True).item()
        return data

    def load_dynamic(self, dir, Gs):
        ''' Main function:
            - Get difference
            -
        '''
        data = self.load_tmp(dir)
        files = self.hdl.get_files(dir,Gs)
        if not data:
            print("[Loading] No data, loading all directory")
            HandData, HandDataFlags = self.hdl.load_directory(dir, Gs)
            X,Y = self.get_dynamic(HandData,HandDataFlags)

            np.save(f"{dir}tmp", {'X': X, 'Y': Y, 'files': files})
            assert X.shape[0] == Y.shape[0], "Wrong Xs and Ys"
            return X,Y

        data_files = data['files']
        real_files = files
        diff = [item for item in real_files if item not in data_files]

        if diff == []:
            print("[Loading] No difference, tmp up to date")
            return data['X'], data['Y']
        else:
            print("[Loading] Difference, loading all directory again")
            HandData, HandDataFlags = self.hdl.load_directory(dir, Gs)
            X,Y = self.get_dynamic(HandData,HandDataFlags)

            np.save(f"{dir}tmp", {'X': X, 'Y': Y, 'files': files})
            assert X.shape[0] == Y.shape[0], "Wrong Xs and Ys"
            return X,Y


    def get_static(self, data, flags):

        X = data
        Y = flags

        X_, Y_ = [], []
        if True: # 'observations configurations' in self.args:
            for n, X_n in enumerate(X):
                gesture_X = []
                for m, X_nt in enumerate(X_n):
                    gesture_X.append(np.array(X_nt.r.get_learning_data()))
                X_.append(gesture_X)
                Y_.append(Y[n])
            print("Defined dataset type: all_defined")


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

        X,Y = self.to_2D(X,Y)
        print(f"shape {np.array(X).shape} y {np.array(Y).shape}")
        X,Y = self.postprocessing(X,Y)
        print(f"shape {X.shape} y {Y.shape}")
        X,Y = self.discard(X,Y)
        print(f"shape {X.shape} y {Y.shape}")


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


    def get_dynamic(self, data, flags):
        Y = flags
        ## Pick: samples x time x palm_position
        Xpalm = []
        for sample in data:
            row = []
            for t in sample:
                if t.r.visible:
                    l2 = np.linalg.norm(np.array(t.r.palm_position()) - np.array(t.r.index_position()))
                    row.append([*t.r.palm_position()])#, l2])#,*t.r.index_position])
            Xpalm.append(row)

        if 'normalize' in self.args:
            Xpalm = np.array(Xpalm)
            Xpalm_ = []
            for p in Xpalm:
                p_ = []
                p0 = deepcopy(p[0])
                for n in range(0, len(p)):
                    p_.append(np.subtract(p[n], p0))
                Xpalm_.append(p_)

            Xpalm = Xpalm_

        ## Interpolate palm_positions, to 100 time samples
        if 'interpolate' in self.args:
            Xpalm_interpolated = []
            invalid_ids = []
            for n,sample in enumerate(Xpalm):
                Xpalm_interpolated_sample = []
                try:
                    for dim in range(0,3):
                        f2 = interp1d(np.linspace(0,1, num=len(np.array(sample)[:,dim])), np.array(sample)[:,dim], kind='cubic')
                        Xpalm_interpolated_sample.append(f2(np.linspace(0,1, num=101)))
                    Xpalm_interpolated.append(np.array(Xpalm_interpolated_sample).T)
                except IndexError:
                    print("Sample with invalid data detected")
                    invalid_ids.append(n)
            Xpalm=Xpalm_interpolated=np.array(Xpalm_interpolated)
            Y = np.delete(Y,invalid_ids,axis=0)

        ## Create derivation to palm_positions
        '''
        dx = 1/100
        DXpalm_interpolated = []
        for sample in Xpalm_interpolated:
            DXpalm_interpolated_sample = []
            sampleT = sample.T
            for dim in range(0,4):
                DXpalm_interpolated_sample.append(np.diff(sampleT[dim]))
            DXpalm_interpolated.append(np.array(DXpalm_interpolated_sample).T)
        DXpalm_interpolated = np.array(DXpalm_interpolated)
        DXpalm = np.array(DXpalm_interpolated)
        '''

        if 'normalize_dim' in self.args:
            Xpalm_ = []
            Xpalm = np.swapaxes(Xpalm, 1, 2)
            for n,dim1 in enumerate(Xpalm):
                for m,dim2 in enumerate(dim1):
                    if (np.max(dim2) - np.min(dim2)) < 0.0000001:
                        Xpalm[n,m] = np.inf
                        continue
                    Xpalm[n,m] = (dim2 - np.min(dim2)) / (np.max(dim2) - np.min(dim2))
            Xpalm = np.swapaxes(Xpalm, 1, 2)
            Xpalm = np.array(Xpalm)

        if 'discards' in self.args:
            discards = self.get_discard_indices(Xpalm,Y)
            #discards.extend(self.get_discard_indices(DXpalm,Y))

            Xpalm = np.delete(Xpalm,discards,axis=0)
            #DXpalm = np.delete(DXpalm,discards,axis=0)
            Y = np.delete(Y,discards,axis=0)

            print(f"[Loading] Number {len(discards)} recordings discarded! discards {discards}")


        return Xpalm, Y

    #import numpy as np
    #A = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    #to_2D(0,A)
    def to_2D(self, data, flags):
        data_, flags_ = [], []
        for n,gesture in enumerate(data):
            data_.extend(gesture)
            flags_.extend([flags[n]] * len(gesture))
        return data_, flags_

    #import numpy as np
    #A = np.array([[[np.nan],[np.inf]],[[3],[4]]])
    #np.delete(A,[1], axis=0)
    #a = np.inf
    #a == np.inf
    #X,Y = discard_d(None, A, [0,0])
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

#
