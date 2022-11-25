import numpy as np
import pandas as pd
from visualizer_lib import VisualizerLib

gesture_columns_static = ['stamp[s]', 'seq', 'grab', 'pinch', 'point', 'two', 'three', 'four', 'five', 'thumbsup', 'grab', 'pinch', 'point', 'two', 'three', 'four', 'five', 'thumbsup']
gesture_columns_dynamic = ['stamp[s]', 'seq', 'swipe_down', 'swipe_front_right', 'swipe_left', 'swipe_up', 'round','swipe_down', 'swipe_front_right', 'swipe_left', 'swipe_up', 'round','swipe_down', 'swipe_front_right', 'swipe_left', 'swipe_up', 'round']
column_mp = ['id', 'id_primitive', 'tmp_action_stamp', 'vars', 'path', 'waypoints']

#record_n = 'record_2/'
record_n = input("Enter record folder name (e.g., record_8):")

try:
    gesture_data_static = np.load(f"{record_n}/gesture_data_left_static.npy")
    print("Shape of static data: ", gesture_data_static.shape)
    df = pd.DataFrame(columns=gesture_columns_static, data=gesture_data_static)
    print("Static data as table:")
    print(df)
except:
    print("No static data in this recording!")

try:
    gesture_data_dynamic = np.load(f"{record_n}/gesture_data_right_dynamic.npy")
    print("Shape of dynamic data: ", gesture_data_dynamic.shape)
    print("Dynamic data as table:")
    df = pd.DataFrame(columns=gesture_columns_dynamic, data=gesture_data_dynamic)
    print(df)
except:
    print("No dynamic data in this recording!")
try:
    gesture_data_mp = np.load(f"{record_n}/executed_MPs.npy", allow_pickle=True)
    df = pd.DataFrame(columns=column_mp, data=gesture_data_mp)
    print("Motion primitives data as table:")
    print(df)
except:
    print("No motion primitives data in this recording!")


def plot_static():
    viz = VisualizerLib()
    viz.visualize_new_fig('xx', dim=2, move_figure=False)
    for i in range(2,9):
        viz.visualize_2d(list(zip(gesture_data_static[:,1], gesture_data_static[:,i])), label=f"{gesture_columns_static[i]}", xlabel='time, leap seq [-] (~10 seq/sec)', ylabel='Gesture probability', start_stop_mark=False)
    viz.savefig(f"./{record_n}/plot_3")
    viz.show()
plot_static()

def plot_mp():
    viz = VisualizerLib()
    viz.visualize_new_fig("Executed MPs", dim=3, move_figure=False)
    # self.action_saves # id, id_primitive, tmp_action_stamp, vars, path
    for id, id_primitive, tmp_action_stamp, vars, path, waypoints in gesture_data_mp:
        viz.visualize_3d(path,label=f"id: {id}, id_primitive: {id_primitive}, stamp: {tmp_action_stamp}", xlabel='X', ylabel='Y', zlabel='Z')
    viz.show()
plot_mp()





#
