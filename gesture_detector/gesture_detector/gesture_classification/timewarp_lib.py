import time, random
import numpy as np
from fastdtw import fastdtw

import gesture_detector
from gesture_detector.utils.loading import HandDataLoader, DatasetLoader

#from os_and_utils.visualizer_lib import VisualizerLib, ScenePlot
from scipy.spatial.distance import euclidean
from copy import deepcopy


class fastdtw_():
    def __init__(self):
        print(f"[TimeWarp] Init done")

    def init(self, model, model_config):
        self.X = model['X']
        self.X_ProMP = model['X_ProMP']
        self.Y = model['Y']
        self.counts = model_config['counts']
        assert len(self.X) == len(self.Y)

        self.Gs = model_config['gestures']
        print(f"[TimeWarp] Gs: {self.Gs}, counts: {self.counts}, records: {sum(self.counts)}")

        self.method = model_config['method']
        print(f"[Sample thread] DTW method is: {self.method}")

    def sub_mid(self, x):
        x_ = []
        x0 = deepcopy(x[len(x)//2])
        for n in range(len(x)):
            x_.append(np.subtract(x[n], x0))
        return x_

    def eacheach(self):
        '''
        O(n^2), n=number of recordings

        samples x samples
        '''
        counts = self.counts
        results = np.zeros([sum(counts),])
        for j in range(sum(counts)):
            dist = np.zeros([sum(counts),])
            for i in range(sum(counts)):
                dist[i], _ = fastdtw(self.X[j], self.X[i])

            mean = np.zeros([len(counts),])
            for i in range(len(counts)):
                mean[i] = np.mean(dist[sum(counts[0:i]):sum(counts[0:i])+counts[i]])
            results[j] = np.argmin(mean)
        return results

    def sample_eacheach(self, x):
        '''
        sample_each(x)
        '''
        counts = self.counts
        dist = np.zeros([sum(counts),])
        for i in range(sum(counts)):
            dist[i], _ = fastdtw(self.X[i], x)

        mean = np.zeros([len(counts),])
        for i in range(len(counts)):
            mean[i] = np.mean(dist[sum(counts[0:i]):sum(counts[0:i])+counts[i]])
        return mean

    def random(self):
        counts = self.counts
        # samples x random
        results = np.zeros([sum(counts),])
        for j in range(sum(counts)):
            dist = np.zeros([len(counts),])
            for i in range(len(counts)):
                rand_ind = random.randint(0,counts[i]-1)
                dist[i], _ = fastdtw(self.X[j], self.X[sum(counts[0:i])], dist=euclidean)

            results[j] = np.argmin(dist)
        return results

    def sample_random(self, x):
        counts = self.counts
        dist = np.zeros([len(counts),])
        for i in range(len(counts)):
            rand_ind = random.randint(0,counts[i]-1)
            dist[i], _ = fastdtw(self.X[sum(counts[0:i])], x, dist=euclidean)

        return dist

    def euclidean(self):
        counts = self.counts
        results = np.zeros([sum(counts),])
        for j in range(sum(counts)):
            dist = np.zeros([len(counts),])
            for i in range(len(counts)):
                dist[i], _ = fastdtw(self.X[j], self.X_ProMP[i], dist=euclidean)
            results[j] = np.argmin(dist)
        return results

    def sample_euclidean(self, x):
        counts = self.counts

        dist = np.zeros([len(counts),])
        for i in range(len(counts)):
            dist[i], _ = fastdtw(x, self.X_ProMP[i], dist=euclidean)
        return dist

    def promp(self):
        counts = self.counts
        results = np.zeros([sum(counts),])
        for j in range(sum(counts)):
            dist = np.zeros([len(counts),])
            for i in range(len(counts)):
                x = self.sub_mid(self.X[j])
                dist[i], _ = fastdtw(x, self.X_ProMP[i], radius=1, dist=euclidean)
            results[j] = np.argmin(dist)
        return results

    def sample_promp(self, x):
        x = self.sub_mid(x)
        counts = self.counts

        dist = np.zeros([len(counts),])
        for i in range(0,len(counts)):
            dist[i], _ = fastdtw(x, self.X_ProMP[i], radius=1, dist=euclidean)
        return dist

    def crossvalidation(self):
        counts = self.counts
        # samples x random
        results = np.zeros([sum(counts),])
        j = 0
        for j in range(0,sum(counts)):
            dist = np.zeros([len(counts),])
            i = 0
            for i in range(0,len(counts)):
                self.X[j].shape
                self.X[index1:index2].shape

                rand_ind = random.randint(0,counts[i]-1)
                index1 = sum(counts[0:i])
                index2 = sum(counts[0:i])+rand_ind
                dist[i], _ = fastdtw(self.X[j], self.X[index1:index2], dist=euclidean)

            results[j] = np.argmin(dist)
        return results

    def sample_crossvalidation(self, x):
        counts = self.counts
        # samples x random
        dist = np.zeros([len(counts),])
        for i in range(0,len(counts)):
            rand_ind = random.randint(0,counts[i]-1)
            index1 = sum(counts[0:i])
            index2 = sum(counts[0:i])+rand_ind
            dist[i], _ = fastdtw(x, self.X[index1:index2], dist=euclidean)

        return dist

    def sample(self, x, y=None, print_out=False, format='inverse_array', checktype=True):
        if checktype:
            if not isinstance(x, list):
                x = x.data
                # # TEMP: # FIXME:
                if len(x) != 15: raise Exception("TODO: HERE !")
                x = np.array(x).reshape(5,3)

        t=time.time()
        if 'eacheach' == self.method:
            x_result = self.sample_eacheach(x)
        elif 'random' == self.method:
            x_result = self.sample_random(x)
        elif 'euclidean' == self.method:
            x_result = self.sample_euclidean(x)
        elif 'crossvalidation' == self.method:
            x_result = self.sample_crossvalidation(x)
        elif 'promp' == self.method:
            x_result = self.sample_promp(x)
        else: raise Exception("Wrong arg flag")
        if print_out: print(abs(t-time.time()))
        if print_out: print(f"Sample result: {np.argmin(x_result)}, real: {y}")

        if format == 'inverse_array':
            # dist to likelihoods
            return np.argmin(x_result), np.array([list(1/x_result)]).squeeze()
        
        return np.argmin(x_result), np.array(x_result).squeeze()

    def evaluate(self):
        ''' Generates confusion matrix and prints accuracy
        Parameters:
            method (Str):
            - 'eacheach':
            - 'random':
            - 'promp':
            - 'euclidean':
            - 'crossvalidation':
        '''
        t=time.time()
        if 'eacheach' == self.method:
            results = self.eacheach()
        elif 'random' == self.method:
            results = self.random()
        elif 'euclidean' == self.method:
            results = self.euclidean()
        elif 'crossvalidation' == self.method:
            results = self.crossvalidation()
        elif 'promp' == self.method:
            results = self.promp()
        else: raise Exception("Wrong arg flag")
        print(abs(t-time.time()))

        #confusion_matrix_pretty_print.my_conf_matrix_from_data(self.Y, results, self.Gs,
        #  annot=True, cmap = 'Greens', fmt='1.8f', fz=10, lw=25, cbar=False, figsize=[6,6], show_null_values=2, pred_val_axis='y', #name=self.method)

        print("Accuracy = {}%".format((self.Y == results).mean() * 100))


if __name__ == '__main__':
    # 91%
    dataloader_args = {'n':10} #, 'scene_frame':1, 'normalize':1}
    # <40% (?)
    dataloader_args = {'normalize':1, 'n':10, 'scene_frame':1, 'inverse':1,'scale_limit':1}
    #X, Y = DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, self.Gs, new=True)
    X, Y = DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, self.Gs)

    fdtw = fastdtw_(X, Y); fdtw.method = 'promp'
    ScenePlot.my_plot([], fdtw.X_ProMP, legend=['swipe down', 'swipe front right','swipe left','swipe up', 'no gesture'], boundbox=False, leap=False, series_marking='')


    def plot_true_false_paths(id = 0):
        true_path_ids, false_path_ids = np.zeros(np.sum(Y==id), dtype=bool), np.zeros(np.sum(Y==id), dtype=bool)
        for i in range(len(X[Y==id])):
            if np.argmax(fdtw.sample(X[Y==id][i], checktype=False)) == Y[Y==id][i]:
                true_path_ids[i] = True
            else:
                false_path_ids[i] = True


        ScenePlot.my_plot(X[Y==id][true_path_ids], [fdtw.X_ProMP[id]], boundbox=[[0,0.02],[0,0.02],[0,0.02]], leap=False)
        ScenePlot.my_plot(X[Y==id][false_path_ids], [fdtw.X_ProMP[id]], boundbox=[[0,0.02],[0,0.02],[0,0.02]], leap=False)
    plot_true_false_paths(0)
    plot_true_false_paths(1)



    id=0; ScenePlot.my_plot(X[Y==id][0:4], [fdtw.X_ProMP[id]], boundbox=[[0,0.02],[0,0.02],[0,0.02]], leap=False)
    id=1; ScenePlot.my_plot(X[Y==id][0:4], [fdtw.X_ProMP[id]], boundbox=[[0,0.1],[0,0.1],[0,0.1]], leap=False)
    id=2; ScenePlot.my_plot(X[Y==id][0:4], [fdtw.X_ProMP[id]], boundbox=[[0,0.1],[0,0.1],[0,0.1]], leap=False)
    id=3; ScenePlot.my_plot(X[Y==id][0:4], [fdtw.X_ProMP[id]], boundbox=[[0,0.1],[0,0.1],[0,0.1]], leap=False)
    id=4; ScenePlot.my_plot(X[Y==id], [fdtw.X_ProMP[id]], boundbox=[[0,0.1],[0,0.1],[0,0.1]], leap=False)

    ''' Poster picture -> distance between representative path and test path '''
    id = 1
    tp = 30
    dist, ind = fastdtw(X[Y==id][tp], fdtw.X_ProMP[id])
    ScenePlot.dtw_3dplot(np.array([fdtw.X_ProMP[id],X[Y==id][tp]]),ind)

    # All
    ScenePlot.my_plot([], fdtw.X_ProMP, legend=['swipe down', 'swipe front right','swipe left','swipe up', 'no gesture'], boundbox=False, leap=False)

    fdtw.method='eacheach'; fdtw.evaluate() # 95.7%, 3000sec.
    fdtw.method='random'; fdtw.evaluate() # 80%, 51sec.
    fdtw.method='euclidean'; fdtw.evaluate() # 95%, 53sec.

    def inspect(id = 0):


        dists = fdtw.sample_promp(X[0])
        out_id = np.argmin(dists)
        if out_id != fdtw.Y[id]:
            ScenePlot.my_plot([X[id]], [fdtw.X_ProMP[id]], boundbox=[[0,0.1],[0,0.1],[0,0.1]], leap=False)


    fdtw.method='promp'; fdtw.evaluate() # 95.2%, 53.5sec.
    fdtw.X.shape
    fdtw.Y.shape

    fdtw.X_ProMP[1]
    fdtw.X[70]
    id = 1
    fdtw.method='eacheach'
    fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True) # 15sec.
    fdtw.method = 'random'
    fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True) # 0.25sec.
    fdtw.method = 'euclidean'
    fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True) # 0.25sec.
    fdtw.method = 'promp'
    fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True) # 0.25sec.

    fdtw.X_ProMP[0][4][2] = -0.02
    fdtw.X_ProMP

    ScenePlot.my_plot([fdtw.X[id]], [(fdtw.X_ProMP[0], {}), (fdtw.X_ProMP[4], {})], boundbox=False, leap=False)

    fdtw.counts
    fdtw.X


    '''
    All with 30 trajectory waypoints
    '''
    dataloader_args = {'n': 30}
    X, Y = DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, fdtw.Gs, new=True)
    fdtw = fastdtw_(X, Y)

    fdtw.method='eacheach'; fdtw.evaluate()
    fdtw.method='random'; fdtw.evaluate()
    fdtw.method='euclidean'; fdtw.evaluate()
    fdtw.method='promp'; fdtw.evaluate()

    id = 40
    fdtw.method= 'eacheach'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'random'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'euclidean'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'promp'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)

    '''
    All with 20 trajectory waypoints
    '''
    dataloader_args = {'n': 20}
    X, Y = DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, fdtw.Gs, new=True)
    fdtw = fastdtw_(X, Y)

    fdtw.method='eacheach'; fdtw.evaluate()
    fdtw.method='random'; fdtw.evaluate()
    fdtw.method='euclidean'; fdtw.evaluate()
    fdtw.method='promp'; fdtw.evaluate()

    id = 40
    fdtw.method='eacheach'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'random'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'euclidean'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'promp'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)

    '''
    All with 10 trajectory waypoints
    '''
    dataloader_args = {'n': 10}
    X, Y = DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, fdtw.Gs, new=True)
    fdtw = fastdtw_(X, Y)

    fdtw.method='eacheach'; fdtw.evaluate()
    fdtw.method='random'; fdtw.evaluate()
    fdtw.method='euclidean'; fdtw.evaluate()
    fdtw.method='promp'; fdtw.evaluate()

    id = 40
    fdtw.method='eacheach'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'random'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'euclidean'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'promp'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)

    '''
    All with 5 trajectory waypoints
    '''
    dataloader_args = {'n': 5}
    X, Y = DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, fdtw.Gs)
    fdtw = fastdtw_(X, Y)

    X.shape

    fdtw.method='eacheach'; fdtw.evaluate()
    fdtw.method='random'; fdtw.evaluate()
    fdtw.method='euclidean'; fdtw.evaluate()
    fdtw.method='promp'; fdtw.evaluate()

    id = 40
    fdtw.method='eacheach'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'random'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'euclidean'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'promp'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)

    '''
    All with 4 trajectory waypoints
    '''
    dataloader_args = {'n': 4}
    X, Y = DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, fdtw.Gs, new=True)
    fdtw = fastdtw_(X, Y)

    X.shape

    fdtw.method='eacheach'; fdtw.evaluate()
    fdtw.method='random'; fdtw.evaluate()
    fdtw.method='euclidean'; fdtw.evaluate()
    fdtw.method='promp'; fdtw.evaluate()

    id = 40
    fdtw.method='eacheach'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'random'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'euclidean'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'promp'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)

    '''
    All with 3 trajectory waypoints
    '''
    dataloader_args = {'n': 3}
    X, Y = DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, fdtw.Gs, new=True)
    fdtw = fastdtw_(X, Y)

    X.shape

    fdtw.method='eacheach'; fdtw.evaluate()
    fdtw.method='random'; fdtw.evaluate()
    fdtw.method='euclidean'; fdtw.evaluate()
    fdtw.method='promp'; fdtw.evaluate()

    id = 40
    fdtw.method='eacheach'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'random'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'euclidean'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'promp'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)

    '''
    All with 2 trajectory waypoints
    '''
    dataloader_args = {'n': 2}
    X, Y = DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, fdtw.Gs, new=True)
    fdtw = fastdtw_(X, Y)

    X.shape

    fdtw.method='eacheach'; fdtw.evaluate()
    fdtw.method='random'; fdtw.evaluate()
    fdtw.method='euclidean'; fdtw.evaluate()
    fdtw.method='promp'; fdtw.evaluate()

    id = 40
    fdtw.method='eacheach'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'random'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'euclidean'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
    fdtw.method = 'promp'; fdtw.sample(x=fdtw.X[id], y=fdtw.Y[id], print_out=True)
