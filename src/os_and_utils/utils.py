
def ROS_ENABLED():
    try:
        import rospy
        return True
    except:
        return False
