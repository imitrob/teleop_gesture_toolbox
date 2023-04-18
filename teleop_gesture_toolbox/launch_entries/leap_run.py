#!/usr/bin/env python
''' Runs: python leap.py '''
import os
path_from_ws = f'src/teleop_gesture_toolbox/teleop_gesture_toolbox/hand_processing'
file = 'leap.py'

while True:
    try:
        from entry_template import run; run(path_from_ws, file)
    except:
        print("leap.py ended, running again")