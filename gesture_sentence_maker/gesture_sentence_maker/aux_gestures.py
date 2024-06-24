


def misc_gesture_handle():
    pass


def load_auxiliary_parameter(type='rotation', robot_feedback=True):
    s = get_scene()

    if type == 'rotation':
        direction_vector = np.cross(gl.gd.hand_frames[-1].palm_normal(), gl.gd.hand_frames[-1].palm_direction())
        xy = list(direction_vector[0:2])
        xy.reverse()
        rot = np.rad2deg(np.arctan2(*xy))
        if robot_feedback:
            GestureSentence.test_rot_ap(rot)
        else:
            time.sleep(0.5)

        ap = rot
        if robot_feedback: GestureSentence.test_rot_ap(0.0)
    elif type == 'distance':

        dist = gl.gd.hand_frames[-1].touch12
        if robot_feedback:
            GestureSentence.test_dist_ap(dist)
        else:
            time.sleep(0.5)

        ap = dist
        if robot_feedback: GestureSentence.test_dist_ap(0.0)
    return ap
