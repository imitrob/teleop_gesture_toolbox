"""Gesture-name to action-name mapping as a plain function call.

The OneToOneMapping node in gesture_meaning_service.py does the same thing
behind the /teleop_gesture_toolbox/get_meaning service, which is slow to call
per sentence. Mergers import this class instead and map gesture words inline.
"""


class OneToOneMapping:
    """One gesture is mapped into one action command.

    Action names follow the skill-command vocabulary of the reasoning merger
    (e.g. 'pick', not 'pick_up'). Gestures mapped to '' carry no action
    meaning; words that are not gesture names (e.g. grounded object names
    like 'cup1') are not touched.
    """
    mapping = {
        # 'gesture': -- to --> 'action'
        'swipe_up':           'move_up',
        'swipe_left':         'push',
        'swipe_down':         'put',
        'swipe_right':        'push',
        'swipe_front_right':  'push',
        'pinch':              'pick',
        'grab':               'pick',
        'point':              '',
        'two':                'unglue',
        'three':              'stack',
        'four':               'release',
        'five':               'stop',
        'thumbsup':           'pour',
        'no_moving':          '',
        'thumb':              '',
    }

    def gesture_to_action(self, gesture_name: str):
        """Return the action name for a gesture, '' for a gesture without
        action meaning, or None when the word is not a known gesture."""
        return self.mapping.get(gesture_name.lower())

    def map_stamped(self, stamped_words: list) -> list:
        """Convert a [[stamp, word], ...] list: gesture names are replaced by
        their action names, gestures without meaning are dropped, all other
        words (object names, ...) pass through unchanged."""
        mapped = []
        for stamp, word in stamped_words:
            action = self.gesture_to_action(word) if isinstance(word, str) else None
            if action is None:
                mapped.append([stamp, word])
            elif action != '':
                mapped.append([stamp, action])
        return mapped
