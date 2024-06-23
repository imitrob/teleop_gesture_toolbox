
import json
import gesture_detector
from gesture_detector.utils.loading import DatasetLoader, HandDataLoader
from gesture_detector.utils.saving import FrameEncoder, JSONLoader
import numpy as np

from gesture_detector.hand_processing import frame_lib

def test_dataloader():
    with open(gesture_detector.saved_models_path+"common_gestures.json", 'r') as f:
        data_loaded = json.load(f)
    Gs_static = data_loaded['gestures']
    
    with open(gesture_detector.saved_models_path+"directional_swipes.json", 'r') as f:
        data_loaded = json.load(f)
    Gs_dynamic = data_loaded['gestures']

    dataloader_args = {'normalize':1, 'n':5, 'scene_frame':1, 'inverse':1}
    DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, Gs_dynamic)
    
    DatasetLoader({'input_definition_version':1}).load_static(gesture_detector.gesture_data_path, Gs_static)

    DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, Gs_dynamic)


def custom_encode_to_json(gesture_data_path: str, model_name: str):
    """Encodes all dataset to json.
    """    
    with open(f"{gesture_detector.saved_models_path}/{model_name}.json", 'r') as f:
        data_loaded = json.load(f)
    Gs_static = data_loaded['gestures']

    hdl = HandDataLoader()

    for gesture in Gs_static:
        frames, Y = hdl.load_directory(gesture_detector.gesture_data_path, [gesture])

        for n in range(len(frames)):
            JSONLoader.save(f"{gesture_data_path}/{gesture}/{n}.json", frames[n])
            frame = JSONLoader.load(f"{gesture_data_path}/{gesture}/{n}.json")
            print(frame)





if __name__ == '__main__':
    test_dataloader()