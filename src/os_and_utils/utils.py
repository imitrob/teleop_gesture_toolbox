
def ROS_ENABLED():
    return False
    try:
        from mirracle_gestures.msg import Frame as Framemsg
        return True
    except:
        return False
