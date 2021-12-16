import numpy as np
import pickle
import sys,os

sys.path.append('../leapmotion')
import frame_lib

class HandDataLoader():
    def __init__(self, import_method='numpy', dataset_files=[]):
        if import_method not in ['numpy', 'pickle']:
            raise Exception(f"Invalid import_method: {import_method}, pick 'numpy or 'pickle'")
        self.import_method = import_method

        self.dataset_files = dataset_files

    def load_directory(self, dir, Gs):
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
    def __init__(self):
        pass

    @staticmethod
    def get_dynamic(data):
        ## Pick: samples x time x palm_position
        Xpalm = []
        for sample in data:
            row = []
            for t in sample:
                if t.r.index_position == []:
                    t.r.index_position = [0.,0.,0.]

                l2 = np.linalg.norm(np.array(t.r.pRaw[0:3]) - np.array(t.r.index_position))
                row.append([*t.r.pRaw[0:3], l2])#,*t.r.index_position])
            Xpalm.append(row)

        if 'normalize' in args:
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
        Xpalm_interpolated = []
        invalid_ids = []
        for n,sample in enumerate(Xpalm):
            Xpalm_interpolated_sample = []
            try:
                for dim in range(0,4):
                    f2 = interp1d(np.linspace(0,1, num=len(np.array(sample)[:,dim])), np.array(sample)[:,dim], kind='cubic')
                    Xpalm_interpolated_sample.append(f2(np.linspace(0,1, num=101)))
                Xpalm_interpolated.append(np.array(Xpalm_interpolated_sample).T)
            except IndexError:
                print("Sample with invalid data detected")
                invalid_ids.append(n)
        Xpalm_interpolated=np.array(Xpalm_interpolated)
        Ydyn = np.delete(Ydyn, invalid_ids)

        ## Create derivation to palm_positions
        dx = 1/100
        DXpalm_interpolated = []
        for sample in Xpalm_interpolated:
            DXpalm_interpolated_sample = []
            sampleT = sample.T
            for dim in range(0,4):
                DXpalm_interpolated_sample.append(np.diff(sampleT[dim]))
            DXpalm_interpolated.append(np.array(DXpalm_interpolated_sample).T)
        DXpalm_interpolated = np.array(DXpalm_interpolated)

        Xpalm = np.array(Xpalm_interpolated)
        DXpalm = np.array(DXpalm_interpolated)

        Xpalm_ = []
        np.array(Xpalm).shape
        Xpalm = np.swapaxes(Xpalm, 1, 2)
        for n,dim1 in enumerate(Xpalm):
            for m,dim2 in enumerate(dim1):
                if (np.max(dim2) - np.min(dim2)) < 0.0000001:
                    Xpalm[n,m] = np.inf
                    continue
                Xpalm[n,m] = (dim2 - np.min(dim2)) / (np.max(dim2) - np.min(dim2))
        Xpalm = np.swapaxes(Xpalm, 1, 2)
        return Xpalm, DXpalm


#
