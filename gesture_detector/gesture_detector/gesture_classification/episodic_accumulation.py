
from collections import Counter
from gesture_detector.utils.utils import CustomDeque
import numpy as np

# How much less the oldest gesture of an episode counts than the newest. Redoing
# a gesture is how a user corrects themselves, so the later one has to win -- but
# only just, or a fresh unsure detection would beat a confident earlier one.
RECENCY_PENALTY = 0.1


class AccumulatedGestures(CustomDeque):
    def __init__(self):
        super(CustomDeque, self).__init__(maxlen=50)



    def gestures_queue_to_ros(self, rostemplate=None):
        ''' Either queue of activated gesture strings or it gesture probs vector
        Parameters:
            gestures_queue (Float[] or Str[])
        GesturesRos()
        '''
        rostemplate.probabilities.data = list(np.array(np.zeros(len(self.Gs)), dtype=float))

        return rostemplate

    def processing(self, ignored_gestures):
        """Note: Parameters are extracted from the last detected gesture
        Args:
            ignored_gestures (List[Str]): Those gestures will be ignored
        """        

        gestures_queue_processed = self.get_not_ignored_gestures(ignored_gestures)
        gestures_queue_processed = self.get_unique_gestures(gestures_queue_processed)
        
        if len(gestures_queue_processed) > 0:
            publish = True
        else:
            publish = False
            return publish, None, None, {}

        max_probs = self.max_extraction(gestures_queue_processed)
        max_timestamps = self.get_max_timestamps()

        return publish, max_probs, max_timestamps, gestures_queue_processed[-1]['params']
    
    def get_max_timestamps(self):
        ''' for every gesture from set, get the timestamp,
            where the gesture had highest probability
        '''
        # number of gestures
        n_of_total_gestures = len(self[-1]['probs'])
        # arrays init zeros
        array_max_prob = [-1] * n_of_total_gestures
        array_max_timestamps = [-1] * n_of_total_gestures
        
        for detection in self:
            gprobs = detection['probs']    
            
            for n, (gprob, maxprob) in enumerate(zip(gprobs, array_max_prob)):
                if gprob > maxprob:
                    array_max_prob[n] = gprob
                    array_max_timestamps[n] = detection['stamp']
        
        assert -1 not in array_max_timestamps # all timestamps must be set
        # rc.roscm.get_logger().info(str(array_max_timestamps))
        return array_max_timestamps

    def max_extraction(self, gestures_dict_processed):
        ''' Highest probability each gesture reached in the episode, with the
            older ones worth slightly less.

        Two gestures shown in one episode both saturate at 1.0, and the sentence
        then cannot say which of them the user meant. Weighting by age settles it
        on the last one performed. The weight is relative to the episode, so a
        single gesture, or several at one instant, are left untouched.
        '''
        probs = np.array([g['probs'] for g in gestures_dict_processed], dtype=float)
        stamps = np.array([g['stamp'] for g in gestures_dict_processed], dtype=float)
        span = stamps.max() - stamps.min()
        if span == 0:
            return np.max(probs, axis=0)
        weights = 1 - RECENCY_PENALTY * (stamps.max() - stamps) / span
        return np.max(probs * weights[:, None], axis=0)

    def get_not_ignored_gestures(self, ignored_gestures):
        gestures_queue_processed = []
        for gesture_queue_item in self:
            if gesture_queue_item['name'] not in ignored_gestures:
                gestures_queue_processed.append(gesture_queue_item)
        return gestures_queue_processed

    def get_unique_gestures(self, gestures_queue):
        gestures_dict = {}
        
        for gesture_trigger in gestures_queue:
            name = gesture_trigger['name']
            if name in gestures_dict.keys():
                # use the latest stamp
                gestures_dict[name]['stamp'] = max(gestures_dict[name]['stamp'], gesture_trigger['stamp'])
                # use the biggest gesture prob observed
                gestures_dict[name]['probs'] = list(np.max((gestures_dict[name]['probs'], gesture_trigger['probs']), axis=0))
                # stack triggered hand labels
                gestures_dict[name]['hand'] = gestures_dict[name]['hand'] + gesture_trigger['hand']
            else:
                gestures_dict[name] = gesture_trigger

        return list(gestures_dict.values())
