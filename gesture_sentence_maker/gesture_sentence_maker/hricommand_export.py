

import json

import numpy as np
from hri_msgs.msg import HRICommand

PREF_OBJECT_INDEX = -1 # second last pointed object
PREF_STORAGE_INDEX = -2 # the last pointed object
OBJECT_INDEX = -1 # the last pointed object, if there is only single pointing

def extract_deictic_solution(solution):
    
    target_object_names = solution.object_names
    target_object_stamp = solution.target_object_stamp
    # distance to likelihoods
    target_object_probs = []
    for dist in solution.object_distances:
        p = str( 1 / (1 + float(dist)) )
        target_object_probs.append(p)
    assert len(target_object_names) == len(target_object_probs)
    return target_object_names, target_object_probs, target_object_stamp


def selected_name(solution):
    """Which object this solution means. Read off the solution, never re-derived
    from the likelihoods: the object was chosen by the evidence rule over a whole
    pointing run (deictic_evidence.select), and the argmax of one frame's
    distances is a different, weaker rule that can disagree with it."""
    return str(solution.target_object_name or "")


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
        sentence_as_dict['target_object'] = selected_name(target_object_solutions[PREF_STORAGE_INDEX])
        sentence_as_dict['target_object_timestamp'] = tos
        sentence_as_dict['object_classes'] = list(s.get_object_types(target_object_names))

        sentence_as_dict['storage_names'] = list(target_storage_names)
        sentence_as_dict['storage_probs'] = list(target_storage_probs)
        sentence_as_dict['target_storage'] = selected_name(target_object_solutions[PREF_OBJECT_INDEX])
        sentence_as_dict['target_storage_timestamp'] = tss
        sentence_as_dict['storage_classes'] = list(s.get_object_types(target_storage_names))
    elif len(target_object_solutions) == 1:
        target_object_names,target_object_probs,tos = extract_deictic_solution(target_object_solutions[OBJECT_INDEX])
        sentence_as_dict["object_names"] = target_object_names
        sentence_as_dict["object_probs"] = list(target_object_probs)
        sentence_as_dict['target_object'] = selected_name(target_object_solutions[OBJECT_INDEX])
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


def export_mapped_to_HRICommand(
        sentence_as_dict: dict,     # the original sentence, as published raw
        mapping,                    # gesture_meaning.one_to_one_mapping.OneToOneMapping
        gesture_names: list = None,
        gesture_probs: list = None,
        gesture_timestamps: list = None,
    ):
    """The same sentence with the gesture meaning added, for /modality/gestures.

    Action fields are added only when a linked combination of gestures was
    actually shown. Without them HriCommand.from_ros yields no action slot,
    which is the honest answer for a gesture nobody linked -- naming the argmax
    of an all-zero distribution would publish the first action of the
    vocabulary as a confident command.
    """
    d = dict(sentence_as_dict)
    shown = (mapping.best_combination(gesture_names, gesture_probs)
             if gesture_probs is not None else None)
    if shown is not None:
        gestures, action = shown
        actions, action_probs = mapping.map_probs(gesture_names, gesture_probs)
        d["action_names"] = list(actions)
        d["action_probs"] = [float(p) for p in action_probs]
        d["target_action"] = action
        # Stamp of the earliest gesture of the combination that was shown: the
        # moment the user began commanding it, which is what the merger
        # interleaves the voice words against.
        d["target_action_timestamp"] = _combination_stamp(
            gestures, gesture_names, gesture_timestamps,
            default=d.get("target_gesture_timestamp", -1.0))

    return HRICommand(data=[json.dumps(d)])


def _combination_stamp(gestures, gesture_names, gesture_timestamps, default):
    if not gesture_timestamps:
        return default
    stamp_of = {g.lower(): t for g, t in zip(gesture_names, gesture_timestamps)}
    stamps = [stamp_of[g] for g in gestures if g in stamp_of]
    return float(min(stamps)) if stamps else default
