

import numpy as np
from gesture_msgs.msg import HRICommand


def get_object_probs(s, target_object_infos):
    # get object names, this can be easily obtained from the function deictic this definitely was using that 
    # from that I can easily get object classes 
    # TODO: This sequence is repeated in this file
    if len(target_object_infos) > 1:
        target_object_infos = np.array(list(target_object_infos[-2]))
        target_storage_infos = np.array(list(target_object_infos[-1]))
    elif len(target_object_infos) > 0:
        target_object_infos = np.array(list(target_object_infos[-1]))
        target_storage_infos = []
    else:
        target_object_infos = []
        target_storage_infos = []

    target_object_names = [o[0] for o in target_object_infos]
    target_object_probs = [str(1/(1+1*float(o[1]))) for o in target_object_infos]

    target_storage_names = [o[0] for o in target_storage_infos]
    target_storage_probs = [str(1/(1+1*float(o[1]))) for o in target_storage_infos]
    ###############

    # Option 1: s.O all objects in the scene s
    # object_types = []
    # for object_name in s.O:
    #     object_types.append(s.get_object_by_name(object_name).type)
    
    # Option 2: All object saved as target_object_infos
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


def get_max_timestamps(gestures_queue):
    ''' for every gesture from set, get the timestamp,
        where the gesture had highest probability
    '''
    # number of gestures
    n_of_total_gestures = len(gestures_queue[-1][3])
    # arrays init zeros
    array_max_prob = [-1] * n_of_total_gestures
    array_max_timestamps = [-1] * n_of_total_gestures
    
    for detection in gestures_queue:
        gprobs = detection[3]    
        
        for n, (gprob, maxprob) in enumerate(zip(gprobs, array_max_prob)):
            if gprob > maxprob:
                array_max_prob[n] = gprob
                array_max_timestamps[n] = detection[0]
    
    assert -1 not in array_max_timestamps # all timestamps must be set
    # rc.roscm.get_logger().info(str(array_max_timestamps))
    return array_max_timestamps

def export_original_to_HRICommand(max_gesture_probs, target_object_infos, gestures_queue, Gs):

    s = get_scene()

    gesture_timestamp     = get_max_timestamps(gestures_queue)

    target_object_names, target_object_probs, object_types, target_storage_names, target_storage_probs = get_object_probs(s, target_object_infos)

    # Collect the data
    sentence_as_dict = {
        'gestures': Gs, # Gesture names 
        'gesture_probs': list(max_gesture_probs), # Gesture probabilities 
        'gesture_timestamp': list(gesture_timestamp), # One timestamp
        # (Note: I can get timestamp for every activation)
        'objects': target_object_names, # This should be all object names detected on the scene
        'object_probs': list(target_object_probs), # This should be all object likelihoods 
        # 'object_timestamps': None, # TODO
        'object_classes': object_types, # Object type names as cbgo types 
        # Each object type should reference to object class
        # 'storages': [''], # TODO: some objects are storages, received by Ontology get function
        # 'storage_probs': [],
        # 'storage_timestamps': [],
        # 'parameters': ap, # All detected auxiliary parameters
        # 'parameter_values': [], 
        # 'parameter_timestamps': [],
        'storages': target_storage_names, # TODO: some objects are storages, received by Ontology get function
        'storage_probs': list(target_storage_probs),
    }
    
    data_as_str = str(sentence_as_dict)
    data_as_str = data_as_str.replace("'", '"')

    return HRICommand(data=[str(data_as_str)])

def export_only_objects_to_HRICommand(s, target_object_infos):
    
    target_object_names, target_object_probs, object_types, target_storage_names, target_storage_probs = get_object_probs(s, target_object_infos)

    sentence_as_dict = {
        # 'target_action': str(intent.target_action),
        'target_object': argmax(target_object_names, target_object_probs),
        'target_storage': argmax(target_storage_names, target_storage_probs),
        # 'actions': action_names, # Gesture names 
        # 'action_probs': list(action_probs), # Gesture probabilities 
        # 'action_timestamp': action_timestamp, # One timestamp
        # (Note: I can get timestamp for every activation)
        'objects': target_object_names, # This should be all object names detected on the scene
        'object_probs': list(target_object_probs), # This should be all object likelihoods 
        # 'object_timestamps': None, # TODO
        'object_classes': list(object_types), # Object type names as cbgo types 
        # Each object type should reference to object class
        'storages': target_storage_names, # TODO: some objects are storages, received by Ontology get function
        'storage_probs': list(target_storage_probs),
        # 'storage_timestamps': [],
        # 'parameters': intent.auxiliary_parameters, 
        # 'parameter_values': [], 
        # 'parameter_timestamps': [],
    }
    data_as_str = str(sentence_as_dict)
    data_as_str = data_as_str.replace("'", '"')

    return HRICommand(data=[str(data_as_str)])        

def export_mapped_to_HRICommand(s, intent, target_object_infos):
    action_names = intent.action_names
    action_probs = intent.action_probs
    action_timestamp = 0.
    
    target_object_names, target_object_probs, object_types, target_storage_names, target_storage_probs = get_object_probs(s, target_object_infos)
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
        'actions': action_names, # Gesture names 
        'action_probs': list(action_probs), # Gesture probabilities 
        'action_timestamp': action_timestamp, # One timestamp
        # (Note: I can get timestamp for every activation)
        'objects': target_object_names, # This should be all object names detected on the scene
        'object_probs': list(target_object_probs), # This should be all object likelihoods 
        # 'object_timestamps': None, # TODO
        'object_classes': list(object_types), # Object type names as cbgo types 
        # Each object type should reference to object class
        # 'storages': [''], # TODO: some objects are storages, received by Ontology get function
        # 'storage_probs': [],
        # 'storage_timestamps': [],
        'parameters': intent.auxiliary_parameters, 
        # 'parameter_values': [], 
        # 'parameter_timestamps': [],
        'storages': target_storage_names, # TODO: some objects are storages, received by Ontology get function
        'storage_probs': list(target_storage_probs),
    }
    data_as_str = str(sentence_as_dict)
    print(data_as_str)
    data_as_str = data_as_str.replace("'", '"')

    return HRICommand(data=[str(data_as_str)])
