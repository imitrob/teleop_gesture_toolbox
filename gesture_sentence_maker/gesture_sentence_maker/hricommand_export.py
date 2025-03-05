

import numpy as np
from hri_msgs.msg import HRICommand

PREF_OBJECT_INDEX = -1 # second last pointed object
PREF_STORAGE_INDEX = -2 # the last pointed object
OBJECT_INDEX = -1 # the last pointed object, if there is only single pointing

def extract_deictic_solution(solution):
    
    target_object_names = solution["object_names"]
    target_object_stamp = solution["target_object_stamp"]
    # distance to likelihoods
    target_object_probs = []
    for dist in solution["object_distances"]:
        p = str( 1 / (1 + float(dist)) )
        target_object_probs.append(p)
    assert len(target_object_names) == len(target_object_probs)
    return target_object_names, target_object_probs, target_object_stamp
        
def argmax(names, probs):
    assert len(names) == len(probs)

    if len(names) == 0:
        return ""
    
    return names[np.argmax(np.array(probs))]

def export_original_to_HRICommand(
        s, # Scene object 
        target_object_solutions, # Queue of each pointings
        gesture_probabilities = None, # List
        gesture_timestamps = None, # List
        gesture_names = None, # List 
        params = None, # Auxiliary parameters
    ):
    sentence_as_dict = {}

    if gesture_probabilities is not None:
        sentence_as_dict['target_gesture'] = str(argmax(gesture_names, gesture_probabilities))
        sentence_as_dict['target_gesture_timestamp'] = float(argmax(gesture_timestamps, gesture_probabilities))
        sentence_as_dict['gesture_names'] = gesture_names
        sentence_as_dict['gesture_probs'] = list(gesture_probabilities)
        sentence_as_dict['gesture_timestamp'] = list(gesture_timestamps)

        # fill in gesture parameters
        for name,value in params.items():
            sentence_as_dict[f"parameter_{name}"] = value

    if len(target_object_solutions) > 1:
        target_object_names,target_object_probs,tos = extract_deictic_solution(target_object_solutions[PREF_STORAGE_INDEX])
        target_storage_names,target_storage_probs,tss = extract_deictic_solution(target_object_solutions[PREF_OBJECT_INDEX])
        sentence_as_dict["object_names"] = list(target_object_names)
        sentence_as_dict["object_probs"] = list(target_object_probs)
        sentence_as_dict['target_object'] = str(argmax(target_object_names, target_object_probs))
        sentence_as_dict['target_object_timestamp'] = tos
        sentence_as_dict['object_classes'] = list(s.get_object_types(target_object_names))

        sentence_as_dict['storage_names'] = list(target_storage_names)
        sentence_as_dict['storage_probs'] = list(target_storage_probs)
        sentence_as_dict['target_storage'] = str(argmax(target_storage_names, target_storage_probs))
        sentence_as_dict['target_storage_timestamp'] = tss
        sentence_as_dict['storage_classes'] = list(s.get_object_types(target_storage_names))
    elif len(target_object_solutions) == 1:
        target_object_names,target_object_probs,tos = extract_deictic_solution(target_object_solutions[OBJECT_INDEX])
        sentence_as_dict["object_names"] = target_object_names
        sentence_as_dict["object_probs"] = list(target_object_probs)
        sentence_as_dict['target_object'] = str(argmax(target_object_names, target_object_probs))
        sentence_as_dict['target_object_timestamp'] = tos
        sentence_as_dict['object_classes'] = s.get_object_types(target_object_names)
    else:
        sentence_as_dict["object_names"] = []
        sentence_as_dict["object_probs"] = []
        sentence_as_dict["storage_names"] = []
        sentence_as_dict["storage_probs"] = []


    data_as_str = str(sentence_as_dict)
    data_as_str = data_as_str.replace("'", '"')

    return HRICommand(data=[str(data_as_str)])

def import_original_HRICommand_to_dict(hricommand):
    sentence_as_str = hricommand.data[0]
    return eval(sentence_as_str)
