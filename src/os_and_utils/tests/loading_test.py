import sys, os
import numpy as np

# append utils library relative to this file
sys.path.append('..')
# create GlobalPaths() which changes directory path to package src folder
from utils import GlobalPaths
GlobalPaths()

sys.path.append('leapmotion') # .../src/leapmotion
from loading import HandDataLoader, DatasetLoader

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

test_import_static()

def test_import_dynamic(Gs=['grab']):

    HandData, HandDataFlags = HandDataLoader().load_directory(GlobalPaths().learn_path, Gs)

    Xpalm, DXpalm, Ypalm = DatasetLoader().get_dynamic(HandData, HandDataFlags)


test_import_dynamic()

# New - oneliner

X, Y = DatasetLoader(['interpolate', 'discards']).load_dynamic(GlobalPaths().learn_path, Gs)












#
