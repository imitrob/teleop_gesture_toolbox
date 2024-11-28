
from typing import List, Dict, Any

def find_pointed_objects_timewindowmax(
        deictic_solutions: Dict[str, Any],
        target_pointings_stamp: List[str], 
        window_size: float = 1.0,
        velocity_threshold: float = 0.05,
    ):
    """ The Combined Time Window and Velocity Threshold Method 
    Integrating temporal alignment and hand dynamics.
    - The time window accommodates slight misalignments between target_pointings_stamp and object likelihood peaks.
    - Filtering by hand velocity ensures that only periods where the hand is likely pointing are considered.
    - Reduces false positives from periods when the hand is moving.

    Args:
        deictic_solutions (Dict[str, Any]): 
        target_pointings_stamp (List[str]): 
        window_size (float, optional): Defaults to 1.0.
        velocity_threshold (float, optional): Defaults to 0.05.

    Returns:
        List[str]: Names for each pointing object selection.
    """

    pointed_objects = []
    for t_stamp in target_pointings_stamp:
        # Define the time window
        start_time = t_stamp - window_size / 2
        end_time = t_stamp + window_size / 2
        
        # Filter deictic_solutions within the time window
        window_solutions = [
            sol for sol in deictic_solutions
            if start_time <= sol["target_object_stamp"] <= end_time
        ]
        
        # Further filter based on hand velocity
        steady_hand_solutions = [
            sol for sol in window_solutions
            if sol["hand_velocity"] <= velocity_threshold
        ]

        # Accumulate likelihoods for each object
        object_likelihoods = {}
        for sol in steady_hand_solutions:
            for name, likelihood in zip(sol["object_names"], sol["object_likelihoods"]):
                if name not in object_likelihoods or likelihood > object_likelihoods[name]:
                    object_likelihoods[name] = likelihood
        
        # Select the object with the highest likelihood
        if object_likelihoods:
            pointed_object = max(object_likelihoods, key=object_likelihoods.get)
            pointed_objects.append(pointed_object)
        else:
            pointed_objects.append(None)  # No object found in this window
    return pointed_objects
