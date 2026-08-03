from gesture_meaning.gesture_icons import GESTURE_ICONS, gesture_icon


def test_icons_cover_the_detector_vocabulary():
    required = {
        "grab",
        "pinch",
        "five",
        "point",
        "thumbsup",
        "two",
        "three",
        "four",
        "swipe_up",
        "swipe_down",
        "swipe_left",
        "swipe_right",
        "swipe_forward",
        "swipe_backward",
        "swipe_front_right",
        "no_moving",
        "swipe_arc",
    }
    assert required <= GESTURE_ICONS.keys()
    assert GESTURE_ICONS["no_moving"] == "●"


def test_unknown_gestures_have_a_visible_fallback():
    assert gesture_icon("new_gesture")
