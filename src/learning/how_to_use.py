'''
1. Download dataset & unzip
2. Use `import_data` function
Note:
    - If running standalone (without mirracle_gestures package), copy `settings.py` (from dataset) to your launch script directory
    - Default path is set to: /home/<user>/<workspace>/src/mirracle_gestures/include/data/learning/
'''
import import_data
import sys
import os

Gs_static = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
Gs_dynamic = ['pin', 'rotate', 'touch', 'swipe_left', 'swipe_right', 'swipe_up', 'swipe_down']
#
data = import_data.import_data(learn_path=None, Gs=Gs_dynamic)
# Cartesian palm coordinates (shape = [Recordings x Time x 4]), [X,Y,Z,(euclidean distance from palm to point finger tip)]
data['dynamic']['Xpalm']
# Cartesian palm velocities (shape = [Recordings x Time x 4])
data['dynamic']['DXpalm']
# Flags (gesture IDs) (shape = [Recordings])
data['dynamic']['Y']

# Static gestures data (shape = [Recordings x Observations])
data['static']['X']
# Static gestures flags (shape = [Recordings])
data['static']['Y']
