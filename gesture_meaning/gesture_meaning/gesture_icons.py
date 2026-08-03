"""Canonical, dependency-free illustrations for gesture names."""

GESTURE_ICONS = {
    "grab": "✊",
    "pinch": "🤏",
    "point": "☝️",
    "five": "🖐️",
    "two": "✌️",
    "three": "🤟",
    "four": "🖖",
    "thumbsup": "👍",
    "thumb": "👍",
    "no_moving": "●",
    "swipe_up": "⬆️",
    "swipe_down": "⬇️",
    "swipe_left": "⬅️",
    "swipe_right": "➡️",
    "swipe_forward": "↗️",
    "swipe_backward": "↙️",
    "swipe_front_right": "↘️",
    "swipe_arc": "🌈",
}

FALLBACK_GESTURE_ICON = "👋"


def gesture_icon(name: str) -> str:
    return GESTURE_ICONS.get(name, FALLBACK_GESTURE_ICON)
