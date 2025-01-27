

import numpy as np
from hri_msgs.msg import HRICommand


def get_object_probs(s, target_object_solutions):
    
    # get object names, this can be easily obtained from the function deictic this definitely was using that 
    # from that I can easily get object classes 
    # TODO: This sequence is repeated in this file

    target_object_names, target_object_probs = [], []
    target_storage_names, target_storage_probs = [], []
    if len(target_object_solutions) > 0:
        target_object_solution = target_object_solutions[0]

        target_object_names = target_object_solution["object_names"]
        # distance to likelihoods
        target_object_probs = []
        for dist in target_object_solution["object_distances"]:
            p = str( 1 / (1 + float(dist)) )
            target_object_probs.append(p)
    if len(target_object_solutions) > 1:
        target_storage_solution = target_object_solutions[1]

        target_storage_names = target_storage_solution["object_names"]
        # distance to likelihoods
        target_storage_probs = []
        for dist in target_storage_solution["object_distances"]:
            p = str( 1 / (1 + float(dist)) )
            target_storage_probs.append(p)

    ###############

    # Option 1: s.O all objects in the scene s
    # object_types = []
    # for object_name in s.O:
    #     object_types.append(s.get_object_by_name(object_name).type)
    
    # Option 2: All object saved as target_object_solutions
    object_types = []
    for object_name in target_object_names:
        o_ = s.get_object_by_name(object_name)
        if o_ is not None:
            object_types.append(o_.type)
        else:
            object_types.append('object')

    if len(target_object_names) != len(target_object_probs):
        return HRICommand(data=['{"invalid": "True"}'])
    # Collect the data
    if len(target_object_names) != len(target_object_probs): HRICommand(data=[str("")])
    
    return target_object_names, target_object_probs, object_types, target_storage_names, target_storage_probs

def argmax(names, probs):
    assert len(names) == len(probs)

    if len(names) == 0:
        return ""
    
    return str(names[np.argmax(np.array(probs))])




def export_original_to_HRICommand(s, target_object_solutions, max_probs, max_timestamps, Gs):

    target_object_names, target_object_probs, object_types, target_storage_names, target_storage_probs = get_object_probs(s, target_object_solutions)

    # Collect the data
    sentence_as_dict = {
        'target_object': argmax(target_object_names, target_object_probs),
        'target_storage': argmax(target_storage_names, target_storage_probs),
        'gesture_names': Gs, # Gesture names 
        'gesture_probs': list(max_probs), # Gesture probabilities 
        'gesture_timestamp': list(max_timestamps), # One timestamp
        'object_names': target_object_names, # This should be all object names detected on the scene
        'object_probs': list(target_object_probs), # This should be all object likelihoods 
        # 'object_timestamps': None, # TODO
        'object_classes': object_types, # Object type names as cbgo types 
        # Each object type should reference to object class
        # 'storage_names': [''], # TODO: some objects are storages, received by Ontology get function
        # 'storage_probs': [],
        # 'storage_timestamps': [],
        # 'parameter_values': [], 
        # 'parameter_timestamps': [],
        'storage_names': target_storage_names, # TODO: some objects are storages, received by Ontology get function
        'storage_probs': list(target_storage_probs),
    }
    
    data_as_str = str(sentence_as_dict)
    data_as_str = data_as_str.replace("'", '"')

    return HRICommand(data=[str(data_as_str)])

def import_original_HRICommand_to_dict(hricommand):
    sentence_as_str = hricommand.data[0]
    return eval(sentence_as_str)

def export_only_objects_to_HRICommand(s, target_object_solutions):
    target_object_names, target_object_probs, object_types, target_storage_names, target_storage_probs = get_object_probs(s, target_object_solutions)

    sentence_as_dict = {
        # 'target_action': str(intent.target_action),
        'target_object': argmax(target_object_names, target_object_probs),
        'target_storage': argmax(target_storage_names, target_storage_probs),
        # 'action_names': action_names, # Gesture names 
        # 'action_probs': list(action_probs), # Gesture probabilities 
        # 'action_timestamp': action_timestamp, # One timestamp
        # (Note: I can get timestamp for every activation)
        'object_names': target_object_names, # This should be all object names detected on the scene
        'object_probs': list(target_object_probs), # This should be all object likelihoods 
        # 'object_timestamps': None, # TODO
        'object_classes': list(object_types), # Object type names as cbgo types 
        # Each object type should reference to object class
        'storage_names': target_storage_names, # TODO: some objects are storages, received by Ontology get function
        'storage_probs': list(target_storage_probs),
        # 'storage_timestamps': [],
        # 'parameters': intent.auxiliary_parameters, 
        # 'parameter_values': [], 
        # 'parameter_timestamps': [],
    }
    data_as_str = str(sentence_as_dict)
    data_as_str = data_as_str.replace("'", '"')

    return HRICommand(data=[str(data_as_str)])        

def export_mapped_to_HRICommand(s, intent, target_object_solutions):
    action_names = intent.action_names
    action_probs = intent.action_probs
    action_timestamp = 0.
    
    target_object_names, target_object_probs, object_types, target_storage_names, target_storage_probs = get_object_probs(s, target_object_solutions)
    # I think I don't need to do this, it is filled before with real values
    # try:
    #     object_probs[object_names.index(intent.target_object)] = 1.0
    # except:
    #     # return HRICommand(data=['{"status": "invalid"}'])
    #     pass

    if len(action_names) != len(action_probs): return HRICommand(data=[str("")])
    
    for p in action_probs:
        if p > 1.0:
            raise Exception("DEBUG here, probability is over 1, something happnned here !,... ", action_probs)

    sentence_as_dict = {
        'target_action': str(intent.target_action),
        'target_object': argmax(target_object_names, target_object_probs),
        'target_storage': argmax(target_storage_names, target_storage_probs),
        'action_names': action_names, # Gesture names 
        'action_probs': list(action_probs), # Gesture probabilities 
        'action_timestamp': action_timestamp, # One timestamp
        # (Note: I can get timestamp for every activation)
        'object_names': target_object_names, # This should be all object names detected on the scene
        'object_probs': list(target_object_probs), # This should be all object likelihoods 
        # 'object_timestamps': None, # TODO
        'object_classes': list(object_types), # Object type names as cbgo types 
        # Each object type should reference to object class
        # 'storage_names': [''], # TODO: some objects are storages, received by Ontology get function
        # 'storage_probs': [],
        # 'storage_timestamps': [],
        'parameters': intent.auxiliary_parameters, 
        # 'parameter_values': [], 
        # 'parameter_timestamps': [],
        'storage_names': target_storage_names, # TODO: some objects are storages, received by Ontology get function
        'storage_probs': list(target_storage_probs),
    }
    data_as_str = str(sentence_as_dict)
    print(data_as_str)
    data_as_str = data_as_str.replace("'", '"')

    return HRICommand(data=[str(data_as_str)])
