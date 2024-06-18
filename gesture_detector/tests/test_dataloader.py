
import json
import gesture_detector
from gesture_detector.utils.loading import DatasetLoader, HandDataLoader
from gesture_detector.utils.saving import FrameEncoder, JSONLoader
import numpy as np

from gesture_detector.hand_processing import frame_lib

def test_dataloader():
    with open(gesture_detector.saved_models_path+"network99.json", 'r') as f:
        data_loaded = json.load(f)
    Gs_static = data_loaded['Gs']
    
    with open(gesture_detector.saved_models_path+"DTW99.json", 'r') as f:
        data_loaded = json.load(f)
    Gs_dynamic = data_loaded['Gs']

    # dataloader_args = {'normalize':1, 'n':5, 'scene_frame':1, 'inverse':1}
    # DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, gesture_detector.gesture_data_path, Gs_dynamic, new=True)
    print(Gs_static)
    DatasetLoader({'input_definition_version':1}).load_static(gesture_detector.gesture_data_path, Gs_static, new=True)

    # DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, Gs_dynamic, new=True)


def encode_to_json():
    with open(gesture_detector.saved_models_path+"network99.json", 'r') as f:
        data_loaded = json.load(f)
    Gs_static = data_loaded['Gs']

    hdl = HandDataLoader()

    for gesture in Gs_static:
        frames, Y = hdl.load_directory(gesture_detector.gesture_data_path, [gesture])

        for n in range(len(frames)):
            JSONLoader.save(f"/home/petr/Downloads/{gesture}/{n}.json", frames[n])
            frame = JSONLoader.load(f"/home/petr/Downloads/{gesture}/{n}.json")
            print(frame)





if __name__ == '__main__':
    test_dataloader()
    # encode_to_json()