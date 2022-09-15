'''
1. Download dataset & unzip (https://drive.google.com/drive/u/0/folders/1lasIj7vPenx_ZkMyofvmro6-xtzYdyVm)
2. Use `import_data` function
Note:
    - If running standalone (without mirracle_gestures package), copy files from drive and run how_to_use.py
'''
import sys
sys.path.append("../src/")
sys.path.append("../src/learning/")
import import_data

Gs_static = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
Gs_dynamic = ['pin', 'rotate', 'touch', 'swipe_left', 'swipe_right', 'swipe_up', 'swipe_down']
#
data = import_data.import_data(learn_path=None, Gs=Gs_dynamic)
# Cartesian palm coordinates (shape = [Recordings x Time x 4]), [X,Y,Z,(euclidean distance from palm to point finger tip)]
print(data['dynamic']['Xpalm'].shape)

# Cartesian palm velocities (shape = [Recordings x Time x 4])
print(data['dynamic']['DXpalm'].shape)
# Flags (gesture IDs) (shape = [Recordings])
print(data['dynamic']['Y'].shape)

# Static gestures data (shape = [Recordings x Observations])
print(data['static']['X'].shape)
# Static gestures flags (shape = [Recordings])
print(data['static']['Y'].shape)
