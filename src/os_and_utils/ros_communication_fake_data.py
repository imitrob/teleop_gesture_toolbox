'''
Works also as notebook
'''
import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from mirracle_gestures.msg import DetectionSolution, DetectionObservations, Hand, Frame
global fake_pub; fake_pub = None

def send_fake_recognition_data_thread(fake_data, send_period=0.1):
    '''
    Parameters:
        send_period (Float): Send new gesture data every 'send_period' [s]

    '''
    rospy.init_node("SendFakeRecognitionData", anonymous=True)

    global fake_pub, fake_hand_pub
    fake_pub = rospy.Publisher("/mirracle_gestures/static_detection_solutions", DetectionSolution, queue_size=5)
    fake_hand_pub = rospy.Publisher("/hand_frame", Frame, queue_size=5)

    print("[Fake Gestures Pub] Init Done")
    seq = 0
    rate = rospy.Rate(1/send_period)
    #while not rospy.is_shutdown():
    for seq in range(len(fake_data['probabilities'])):

        #seq+=1
        send_fake_hand_data(fake_data['compact_fake_hand_metrics'][seq])
        send_fake_recognition_data(fake_data['probabilities'][seq], fake_data['id'][seq], seq)

        rate.sleep()

    print("[Fake Gestures Pub] Exit")

def send_fake_recognition_data(probs, id, seq, hand='l', approach = 'PyMC3'):
    global fake_pub

    msg = DetectionSolution()
    msg.header.seq = seq
    msg.header.frame_id = hand
    msg.sensor_seq = seq # There is no real data from Leap, so it can be same
    msg.header.stamp = rospy.Time.now()
    msg.probabilities = Float64MultiArray(MultiArrayLayout([MultiArrayDimension()], 0), probs)
    msg.id = id
    msg.approach = approach

    fake_pub.publish(msg)

def send_fake_hand_data(compact_fake_hand_metrics, hand='r'):

    msg = Frame()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = hand
    h = getattr(msg, hand)
    h.visible = True
    if hand == 'r': h.is_right = True
    if hand == 'l': h.is_left = True
    h.grab_strength = compact_fake_hand_metrics[0]
    h.pinch_strength = compact_fake_hand_metrics[1]
    h.finger_bones[4+3].direction = compact_fake_hand_metrics[2:5]
    fake_hand_pub.publish(msg)

def make_fake_data(fake_gestures_data, fake_hand_data, Gs, time_samples_on_one_gesture=20):
    '''
    Parameters:
        fake_gestures_data (String[]): Array of gesture fake data shown
        Gs (String[]): All gestures names as list of strings
        time_samples_on_one_gesture (Int): How many times one gesture must be shown to activate it
    Returns:
        fake_data {'probabilities': shape=len(Gs) x n (Float), 'id': n (Int)}: Received solutions
    '''
    nGs = len(Gs) # Number of gestures

    fake_data = {
        'probabilities': [],
        'id': [],
        'compact_fake_hand_metrics': []
    }
    for i in range(len(fake_gestures_data)):
        for j in range(time_samples_on_one_gesture):
            id = Gs.index(fake_gestures_data[i]) # Name to ID

            probs = [0.0] * nGs
            probs[id] = 1.0
            fake_data['probabilities'].append(probs)
            fake_data['id'].append(id)
            fake_data['compact_fake_hand_metrics'].append(fake_hand_data[i])

    # Make last id, different than previous
    id = (fake_data['id'][-1]+1)%nGs
    probs = [0.0] * nGs
    probs[id] = 1.0
    fake_data['probabilities'].append(probs)

    fake_data['id'].append(id)
    fake_data['compact_fake_hand_metrics'].append([0.,0.,1.,0.,0.])

    return fake_data

if __name__ == '__main__':
    # Model fake data
    Gs = ['grab', 'pinch', 'point', 'two', 'three', 'four', 'five', 'thumbsup']

    fake_gestures_data = ['grab', 'point', 'grab', 'thumbsup', 'five', 'point', 'grab', 'thumbsup', 'five', 'five', 'point', 'grab', 'thumbsup', 'five', 'point', 'grab', 'thumbsup']
    fake_hand_data = [
    [0.,0.,1.,0.,0.],
    [0.,0.,1.,0.,0.],
    [1.,1.,1.,0.,0.],
    [0.,0.,1.,0.,0.],
    [0.,0.,1.,0.,0.],
    [0.,0.,1.,0.,0.],
    [0.,0.,1.,0.,0.],
    [0.,0.,1.,0.,0.],
    [0.,0.,-1.,0.,0.],
    [0.,0.,-1.,0.,0.],
    [0.,0.,1.,0.,0.],
    [1.,1.,1.,0.,0.],
    [1.,1.,1.,0.,0.],
    [1.,1.,1.,0.,0.],
    [1.,1.,1.,0.,0.],
    [0.,0.,1.,0.,0.],
    [0.,0.,1.,0.,0.]
    ]

    fake_data = make_fake_data(fake_gestures_data, fake_hand_data, Gs)

    send_fake_recognition_data_thread(fake_data)




#
