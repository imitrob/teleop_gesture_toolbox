import json
from typing import Iterable
import gesture_detector
from gesture_detector.utils.saving import JSONLoader
import numpy as np
import sys,os
from copy import deepcopy

from scipy.interpolate import interp1d

from gesture_detector.utils.transformations import Transformations as tfm

class HandDataLoader():
    ext = '.json'

    def load_directory(self, dir, Gs):
        """ %timeit ~7s

        Args:
            dir (str): Directory to load
            Gs (Iterable[str]): Gesture names: folders 

        Returns:
            X, Y: HandFrames of shape (demontrations, frames), Labels of shape (demonstrations)
        """
        X, Y = [], []
        for n, G in enumerate(Gs):
            i=0
            print(f"{dir}{G}/{str(i)}{self.ext}")
            while os.path.isfile(f"{dir}/{G}/{str(i)}{self.ext}"):
                i_file = f"{G}/{str(i)}{self.ext}"
                print(f"File {i_file}")
                with open(f"{dir}/{i_file}", 'rb') as input:
                    X.append(self.load_file(f"{dir}/{i_file}"))
                    Y.append(n)
                i+=1

        if X == []:
            print(('[WARN*] No data was imported! Check parameters path folder (learn_path) and gesture names (Gs)'))

        return X,Y
        
    def load_file(self, input_):
        return JSONLoader.load(input_)


class DatasetLoader():
    def __init__(self, args={}):
        self.args = args
        self.hdl = HandDataLoader()

    def load_static(self, dir: str, Gs: Iterable[str]):
        """Convenience function to get dataset of all static gestures
        """
        return self.load_dataset(dir=dir, Gs=Gs, type='static')

    def load_dynamic(self, dir: str, Gs: Iterable[str]):
        """Convenience function to get dataset of all dynamic gestures
        """
        return self.load_dataset(dir=dir, Gs=Gs, type='dynamic')


    def load_dataset(self, dir: str, Gs: Iterable[str], type: str = 'dynamic'):
        """Main function to load dataset

        Args:
            dir (str): Directory of gesture data
            Gs (Iterable[str]): Gesture names
            type (str, optional): Which type of gestures to load. Defaults to 'dynamic'.
            
        Raises:
            Exception: If type not recognized (not in 'static', 'dynamic')

        Returns:
            np.ndarray: Observations (X) of shape (samples, features)
            np.ndarray: Labels (Y) of shape (samples)
        """        
        HandData, HandDataFlags = self.hdl.load_directory(dir, Gs)
        
        if type == 'dynamic':
            X,Y = self.get_dynamic(HandData,HandDataFlags)
        elif type == 'static':
            X,Y = self.get_static(HandData,HandDataFlags)
        else: raise Exception("Wrong type")

        assert X.shape[0] == Y.shape[0], "Wrong Xs and Ys"
        return X,Y

    def get_static(self, data, flags):
        """Extracts features of hand data to learning observations
        """        
        X = data
        Y = flags
        
        ''' Extract features '''
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

        ''' Interpolate demonstration data to have same length '''
        X_interpolated = []
        for n,sample in enumerate(X):
            X_interpolated_sample = []
            for dim in range(0,len(sample[0])):
                f2 = interp1d(np.linspace(0,1, num=len(np.array(sample)[:,dim])), np.array(sample)[:,dim], kind='cubic')
                X_interpolated_sample.append(f2(np.linspace(0,1, num=101)))
            X_interpolated.append(np.array(X_interpolated_sample).T)
        X=np.array(X_interpolated)


        X_ = []
        Y_ = []
        ''' Frame dropping '''
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
        X = np.array(X_)
        Y = np.array(Y_)

        X,Y = self.postprocessing(X,Y)
        print(f"shape before discrading nans: X: {X.shape} y {Y.shape}")
        X,Y = self.discard(X,Y)
        print(f"shape after discrading nans: X: {X.shape} y {Y.shape}, addon Y[0].shape {Y[0].shape} Y[1].shape {Y[1].shape}")

        return X, Y

    def discard(self, X, Y):
        ''' Discard demonstrations where is any nan value '''
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

    def postprocessing(self, X, Y):
        
        import pytensor as pt
        floatX = pt.config.floatX
        X = X.astype(floatX)
        Y = Y.astype(floatX)
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
    path = "/home/petr/teleop_2_ws/build/gesture_detector/gesture_detector/gesture_data/"
    gesture_names = ['grab', 'pinch', 'point', 'two', 'three', 'four', 'five', 'thumbsup']
    new = True

    X, y = DatasetLoader({'input_definition_version':1, 'take_every': 4}).load_static(path, gesture_names, new=new)

    exit()
    with open(gesture_detector.saved_models_path+"network99.json", 'r') as f:
        data_loaded = json.load(f)
    Gs_static = data_loaded['Gs']
    
    with open(gesture_detector.saved_models_path+"DTW99.json", 'r') as f:
        data_loaded = json.load(f)
    Gs_dynamic = data_loaded['Gs']

    dataloader_args = {'normalize':1, 'n':5, 'scene_frame':1, 'inverse':1}
    DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, gesture_detector.gesture_data_path, Gs_dynamic, new=True)
    DatasetLoader({'input_definition_version':1}).load_static(gesture_detector.gesture_data_path, Gs_static, new=True)

    DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, Gs_dynamic, new=True)