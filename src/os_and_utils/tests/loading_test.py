import sys, os
import numpy as np

# append utils library relative to this file
sys.path.append('../..')
# create GlobalPaths() which changes directory path to package src folder
import settings; settings.init()

sys.path.append('leapmotion') # .../src/leapmotion
from os_and_utils.loading import HandDataLoader, DatasetLoader

def test_import_static(Gs=['grab']):
    ''' Prepares X, Y datasets from files format
        - Loads data from learn_path, gesture recordings are saved here
            - Every gesture has its own directory
            - Every gesture recording has its own file
            Pros: Easy manipulation with recording (deletion, move)
            Cons: Slower load time
    Parameters:
        learn_path (Str): Path to learn directory (gesture recordings)
        args (Str{}): Flags for import
            - 'normalize' - Palm trajectory will start at zero (x,y,z=0,0,0 at time t=0)
            - 'interpolate' - X data will be interpolated to have static length of 100 samples

        Gs (Str[]): Set of gesture names, given names corresponds to gesture folders (from which to load gestures)
        dataset_files (Str[]): Array of filenames of dataset, which will be discarted
    '''

    HandData, HandDataFlags = HandDataLoader().load_directory(GlobalPaths().learn_path, Gs)

    X, Y = DatasetLoader().get_static(HandData, HandDataFlags)

#test_import_static()

def test_import_dynamic(Gs=['grab']):

    HandData, HandDataFlags = HandDataLoader().load_directory(GlobalPaths().learn_path, Gs)

    Xpalm, DXpalm, Ypalm = DatasetLoader().get_dynamic(HandData, HandDataFlags)


#test_import_dynamic()

# New - oneliner

import gestures_lib as gl; gl.init()

X, Y = DatasetLoader(['normalize', 'interpolate']).load_dynamic(settings.paths.learn_path, gl.gd.l.dynamic.info.names)

from os_and_utils.transformations import Transformations as tfm

ID = 0
'''
X_scene_paths = []
for path in X[Y==ID]:
    new_path = []
    for point in path:
        new_path.append(tfm.transformLeapToBase(point, out='position'))
    X_scene_paths.append(new_path)
X_scene_paths = np.array(X_scene_paths)
'''
X_scene_paths = X[Y==ID]
np.mean(X_scene_paths[:,0,0])
np.mean(X_scene_paths[:,0,1])
np.mean(X_scene_paths[:,0,2])

np.mean(X_scene_paths[:,-1,0])
np.mean(X_scene_paths[:,-1,1])
np.mean(X_scene_paths[:,-1,2])
from os_and_utils.visualizer_lib import ScenePlot

gl.gd.l.dynamic.info.names[ID]
ScenePlot.my_plot(X_scene_paths, [])


'''
TESTING WITH MPs
'''
import promps.promp_paraschos as approach
import promps.promp_sebasutp as approach

gl.gd.l.mp.info.names
X, Y, robot_promps = DatasetLoader({'normalize':1, 'interpolate':1, 'n':5}).load_mp(settings.paths.learn_path, gl.gd.l.mp.info.names, approach, new=True)




np.save('/home/petr/Downloads/a',robot_promps, allow_pickle=True)

#
