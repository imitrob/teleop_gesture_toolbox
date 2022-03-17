import time, random
import numpy as np
import sys; sys.path.append('..')
import settings; settings.init()
from fastdtw import fastdtw
from os_and_utils.loading import HandDataLoader, DatasetLoader
import gestures_lib as gl
from os_and_utils import confusion_matrix_pretty_print
from scipy.spatial.distance import euclidean
from copy import deepcopy
if __name__ == '__main__': gl.init()
try:
    gl.gd
except AttributeError:
    gl.init()

from promps import promp_paraschos as promp_approach

class fastdtw_():
    ''' Performs time warping to dataset
    - Uses X trajectory

    After time warp computation:
    - results (id's computed) are compared to Y (id's real)
    Parameters:
    X (ndarray): Trajectory positions of palm
    Y (1darray): Gesture (id)

    Usage:
    fdtw = fastdtw_(X, Y)
    fdtw.evaluate(method)
    '''
    def __init__(self, X=None, Y=None):
        if not np.array(X).any():
            dataloader_args = {'normalize':1, 'n':5, 'scene_frame':1, 'inverse':1}
            X, Y = DatasetLoader(dataloader_args).load_dynamic(settings.paths.learn_path, gl.gd.Gs_dynamic)

        self.X = X
        self.Y = Y
        assert len(self.X) == len(self.Y)
        g = list(dict.fromkeys(Y))
        self.counts = [list(Y).count(g_) for g_ in g]

        self.X_ProMP = promp_approach.construct_promp_trajectories(self.X, self.Y, condition=False)
        self.X_ProMP = np.array([[[ 0.0, 0.0,  0.05], # 'swipe_down'
            [ 0.0,  0.0,  0.025],
            [ 0.0,  0.0, 0.0],
            [ 0.0,  0.0, -0.025],
            [ 0.0,  0.0, -0.05]],
            [[-0.05, -0.0, -0.0], # 'swipe_front_right'
            [-0.025, -0.0, -0.0],
            [0.0, -0.0,  0.0],
            [0.025,  0.0,  0.0],
            [0.05,  0.0,  0.0]],
            [[ 0.05,  0.0,  0.0], # 'swipe_left'
            [ 0.025,  0.0,  0.0],
            [0.0, -0.0,  0.0],
            [-0.025, -0.0, -0.0],
            [-0.05, -0.0, -0.0]],
            [[-0.0,  0.0, -0.05], # 'swipe_up'
            [-0.0,  0.0, -0.025],
            [ 0.0, -0.0, 0.0],
            [ 0.0, -0.0,  0.025],
            [ 0.0, -0.0,  0.05]],
            [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00], # 'nothing_dyn'
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]])

        self.method = None

        print(f"[TimeWarp] Gs: {gl.gd.Gs_dynamic}, counts: {self.counts}, records: {sum(self.counts)}")
        print(f"[TimeWarp] Init done")

    def init(self, method):
        self.method = method
        print(f"[Sample thread] DTW method is: {method}")

    def norm(self, x):
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
                x = self.norm(self.X[j])
                dist[i], _ = fastdtw(x, self.X_ProMP[i], radius=10, dist=euclidean)
            results[j] = np.argmin(dist)
        return results

    def sample_promp(self, x):
        x = self.norm(x)
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

    def sample(self, x, y=None, print_out=False, format='inverse_array'):
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

        #print("x", x, "x_result", x_result)
        if format == 'inverse_array':
            # united format
            return [list(1/x_result)]
        return x_result

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

        confusion_matrix_pretty_print.plot_confusion_matrix_from_data(self.Y, results, gl.gd.Gs_dynamic,
          annot=True, cmap = 'Oranges', fmt='1.8f', fz=10, lw=25, cbar=False, figsize=[6,6], show_null_values=2, pred_val_axis='y', name=self.method)

        print("Accuracy = {}%".format((self.Y == results).mean() * 100))


if __name__ == '__main__':

    dataloader_args = {'n':5, 'scene_frame':1, 'normalize':1} # {'normalize':1, 'n':5, 'scene_frame':1, 'inverse':1}
    X, Y = DatasetLoader(dataloader_args).load_dynamic(settings.paths.learn_path, gl.gd.Gs_dynamic, new=True)
    X, Y = DatasetLoader(dataloader_args).load_dynamic(settings.paths.learn_path, gl.gd.Gs_dynamic)

    fdtw = fastdtw_(X, Y)
    import os_and_utils.visualizer_lib; reload(os_and_utils.visualizer_lib)
    from os_and_utils.visualizer_lib import VisualizerLib, ScenePlot

    id = 3
    x = np.array([fdtw.norm(x) for x in X])
    ScenePlot.my_plot([x[Y==id][0]], [(fdtw.X_ProMP[id], {})], boundbox=True, leap=False)

    ScenePlot.my_plot(x, [(fdtw.X_ProMP[id], {})], boundbox=True, leap=False)
    ScenePlot.my_plot([], [(fdtw.X_ProMP[0], {}), (fdtw.X_ProMP[1], {}), (fdtw.X_ProMP[2], {}), (fdtw.X_ProMP[3], {}), (fdtw.X_ProMP[4], {})], boundbox=False, leap=False)

    fdtw.method='eacheach'; fdtw.evaluate() # 95.7%, 3000sec.
    fdtw.method='random'; fdtw.evaluate() # 80%, 51sec.
    fdtw.method='euclidean'; fdtw.evaluate() # 95%, 53sec.
    fdtw.method='promp'; fdtw.evaluate() # 95.2%, 53.5sec.

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
    X, Y = DatasetLoader(dataloader_args).load_dynamic(settings.paths.learn_path, gl.gd.Gs_dynamic, new=True)
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
    X, Y = DatasetLoader(dataloader_args).load_dynamic(settings.paths.learn_path, gl.gd.Gs_dynamic, new=True)
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
    X, Y = DatasetLoader(dataloader_args).load_dynamic(settings.paths.learn_path, gl.gd.Gs_dynamic, new=True)
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
    X, Y = DatasetLoader(dataloader_args).load_dynamic(settings.paths.learn_path, gl.gd.Gs_dynamic)
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
    X, Y = DatasetLoader(dataloader_args).load_dynamic(settings.paths.learn_path, gl.gd.Gs_dynamic, new=True)
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
    X, Y = DatasetLoader(dataloader_args).load_dynamic(settings.paths.learn_path, gl.gd.Gs_dynamic, new=True)
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
    X, Y = DatasetLoader(dataloader_args).load_dynamic(settings.paths.learn_path, gl.gd.Gs_dynamic, new=True)
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






#
