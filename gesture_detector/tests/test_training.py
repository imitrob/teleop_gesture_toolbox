
import os
import pytest

import gesture_detector
from gesture_detector.gesture_classification.torch_lib import Experiments


@pytest.mark.skipif(not os.path.isdir(gesture_detector.gesture_data_path), reason="gesture_data dataset not present")
def test_gesture_training():
    """ Optimal config for 8 gestures, each >30 recordings (~2000 samples) per gestures
    """
    args = {
    'experiment': "train_custom",
    'gestures': ['grab','pinch','point','two','three','four','five','thumbsup'],
    'model_name': "common_gestures",
    'save': False,
    'n_hidden': 20,
    'split': 0.3, # 30% for testing
    'take_every': 4, # take every 4th sample from hand records
    'iter': 5000,
    'seed': 93457,
    'inference_type': "ADVI",
    'layers': 2,
    'gesture_type': "static",
    'engine': "Torch",
    'input_definition_version': 1,
    }
    experiments = Experiments()
    acc = experiments.train_custom(args=args)

    assert acc > 98, "Should reach at least 98% accuracy"

def test_load_and_evaluate():
    args = {
    'experiment': 'load_and_evaluate',
    'gestures': ['grab','pinch','point','two','three','four','five','thumbsup'],
    'model_name': "common_gestures",
    'save': False,
    'n_hidden': 20,
    'split': 0.3,
    'take_every': 4,
    'iter': 5000,
    'seed': 93457,
    'inference_type': "ADVI",
    'layers': 2,
    'gesture_type': "static",
    'engine': "Torch",
    'input_definition_version': 1,
    }
    experiments = Experiments()
    acc = experiments.load_and_evaluate(args=args)

    assert acc > 98, "Model has 99% accuracy"


if __name__ == '__main__':
    test_gesture_training()
    test_load_and_evaluate()
    
