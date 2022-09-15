#include "hand_classes.h"

#include "Leap.h"
using namespace Leap;

/*
 * 1. Default declarations
 * 2. Declarations from Leap Classes
 *
 * Structure:
 * Frame -> Hand -> Finger -> Bone -> Vector
 * Frame -> LeapGestures
 */

/*
 * Defualt declarations
 */
 Vector_::Vector_(){
   this->x = 0.0;
   this->y = 0.0;
   this->z = 0.0;
 }

Bone_::Bone_(){
  for(int i = 0; i < 3; i++){
    this->basis[i] = *(new Vector_());
  }
  this->direction = *(new Vector_());
  this->next_joint = *(new Vector_());
  this->prev_joint = *(new Vector_());
  this->center = *(new Vector_());
  this->is_valid = false;
  this->length = 0.0;
  this->width = 0.0;
}

Finger_::Finger_(){
  for(int i = 0; i < 4; i++){
    this->bones[i] = *(new Bone_());
  }
}

Hand_::Hand_(){
  this->visible = false;
  this->id = 0;
  this->is_left = false;
  this->is_right = false;
  this->is_valid = false;
  this->grab_strength = 0.0;
  this->pinch_strength = 0.0;
  this->confidence = 0.0;
  this->palm_normal = *(new Vector_());
  this->direction = *(new Vector_());
  this->palm_position = *(new Vector_());
  for(int i = 0; i < 5; i++){
    this->fingers[i] = *(new Finger_());
  }

  this->palm_velocity = *(new Vector_());
  for(int i = 0; i < 3; i++){
    this->basis[i] = *(new Vector_());
  }
  this->palm_width = 0.0;
  this->sphere_center = *(new Vector_());
  this->sphere_radius = 0.0;
  this->stabilized_palm_position = *(new Vector_());
  this->time_visible = 0.0;
  this->wrist_position = *(new Vector_());
}

Frame_::Frame_(){
  this->seq = 0; // ID of frame
  this->secs = 0; // seconds of frame
  this->nsecs = 0; // nanoseconds
  this->fps = 0; // Frames per second
  this->hands = 0; // Number of hands
}

/*
 * Declaration from Leap Classes
 */
Vector_::Vector_(double x, double y, double z){
 this->x = x;
 this->y = y;
 this->z = z;
}

Vector_::Vector_(double* pvector){
 this->x = *(pvector + 0*sizeof(double));
 this->y = *(pvector + 1*sizeof(double));
 this->z = *(pvector + 2*sizeof(double));
}

Bone_::Bone_(Bone* pbone){
  Bone bone = *pbone;

  this->basis[0] = *(new Vector_((double*)bone.basis().xBasis.toFloatPointer()));
  this->basis[1] = *(new Vector_((double*)bone.basis().yBasis.toFloatPointer()));
  this->basis[2] = *(new Vector_((double*)bone.basis().zBasis.toFloatPointer()));

  this->direction = *(new Vector_((double*)bone.direction().toFloatPointer()));
  this->next_joint = *(new Vector_((double*)bone.nextJoint().toFloatPointer()));
  this->prev_joint = *(new Vector_((double*)bone.prevJoint().toFloatPointer()));
  this->center = *(new Vector_((double*)bone.center().toFloatPointer()));
  this->is_valid = bone.isValid();
  this->length = bone.length();
  this->width = bone.width();
}

Finger_::Finger_(Finger* pfinger){
  Finger finger = *pfinger;

  for (int b = 0; b < 4; ++b) {
    Bone::Type boneType = static_cast<Bone::Type>(b);
    Bone bone = finger.bone(boneType);
    this->bones[b] = *(new Bone_(&bone));
  }
}

Hand_::Hand_(Hand* phand){
  Hand hand = *phand;

  this->visible = true;
  this->id = hand.id();
  this->is_left = hand.isLeft();
  this->is_right = hand.isRight();
  this->is_valid = hand.isValid();
  this->grab_strength = hand.grabStrength();
  this->pinch_strength = hand.pinchStrength();
  this->confidence = hand.confidence();
  this->palm_normal = *(new Vector_((double*)hand.palmNormal().toFloatPointer()));
  this->direction = *(new Vector_((double*)hand.direction().toFloatPointer()));
  this->palm_position = *(new Vector_((double*)hand.palmPosition().toFloatPointer()));

  const FingerList fingers = hand.fingers();
  int i = 0;
  for (FingerList::const_iterator fl = fingers.begin(); fl != fingers.end(); ++fl) {
    Finger finger = *fl;
    this->fingers[i] = *(new Finger_(&finger));
    i++;
  }
  this->palm_velocity = *(new Vector_((double*)hand.palmVelocity().toFloatPointer()));
  this->basis[0] = *(new Vector_((double*)hand.basis().xBasis.toFloatPointer()));
  this->basis[1] = *(new Vector_((double*)hand.basis().yBasis.toFloatPointer()));
  this->basis[2] = *(new Vector_((double*)hand.basis().zBasis.toFloatPointer()));

  this->palm_width = hand.palmWidth();

  this->sphere_center = *(new Vector_((double*)hand.sphereCenter().toFloatPointer()));
  this->sphere_radius = hand.sphereRadius();
  this->stabilized_palm_position = *(new Vector_((double*)hand.stabilizedPalmPosition().toFloatPointer()));
  this->time_visible = hand.timeVisible();
  this->wrist_position = *(new Vector_((double*)hand.wristPosition().toFloatPointer()));

}

Frame_::Frame_(Frame* pframe, Frame* pprevframe){

  Frame frame = *pframe;
  Frame prevframe = *pprevframe;

  this->seq = frame.id();
  this->fps = frame.currentFramesPerSecond();
  this->secs = frame.timestamp()/1000000;
  this->nsecs = 1000*(frame.timestamp()%1000000);
  this->hands = frame.hands().count();

  HandList hands = frame.hands();
  for (HandList::const_iterator hl = hands.begin(); hl != hands.end(); ++hl) {
    Hand hand = *hl;
    if (hand.isLeft() == true){
      this->l = *(new Hand_(&hand));
    } else {
      this->r = *(new Hand_(&hand));
    }
  }
  LeapGestures_ leap_gestures;
  const GestureList gestures = frame.gestures();
  for (int g = 0; g < gestures.count(); ++g) {
    Gesture gesture = gestures[g];
    switch (gesture.type()) {
      case Gesture::TYPE_CIRCLE:
      {
        leap_gestures.circle_id = gesture.id();
        CircleGesture circle = gesture;
        leap_gestures.circle_state = circle.state();
        std::string clockwiseness;

        if (circle.pointable().direction().angleTo(circle.normal()) <= PI/2) {
          leap_gestures.circle_clockwise = true;
        } else {
          leap_gestures.circle_clockwise = false;
        }

        // Calculate angle swept since last frame
        float sweptAngle = 0;
        if (circle.state() == Gesture::STATE_STOP){
          leap_gestures.circle_in_progress = false;
          CircleGesture previousUpdate = CircleGesture(prevframe.gesture(circle.id()));
          sweptAngle = (circle.progress() - previousUpdate.progress()) * 2 * PI;
          leap_gestures.circle_progress = circle.progress();
          leap_gestures.circle_angle = sweptAngle * RAD_TO_DEG;
          leap_gestures.circle_radius = circle.radius();
          leap_gestures.circle_state = gesture.state();
        } else {
          leap_gestures.circle_in_progress = true;
          CircleGesture previousUpdate = CircleGesture(prevframe.gesture(circle.id()));
          sweptAngle = (circle.progress() - previousUpdate.progress()) * 2 * PI;
          leap_gestures.circle_progress = circle.progress();
          leap_gestures.circle_angle = sweptAngle * RAD_TO_DEG;
          leap_gestures.circle_radius = circle.radius();
          leap_gestures.circle_state = gesture.state();
        }
        break;
      }
      case Gesture::TYPE_SWIPE:
      {
        SwipeGesture swipe = gesture;

        leap_gestures.swipe_id = gesture.id();
        leap_gestures.swipe_state = gesture.state();
        for(int i = 0; i < 3; i++){
          leap_gestures.swipe_direction[i] = *(swipe.direction().toFloatPointer() + i*sizeof(float));
        }
        leap_gestures.swipe_speed = swipe.speed();
        if (swipe.state() == Gesture::STATE_STOP){
          leap_gestures.swipe_in_progress = false;
        } else {
          leap_gestures.swipe_in_progress = true;
        }
        break;
      }
      case Gesture::TYPE_KEY_TAP:
      {
        KeyTapGesture keytap = gesture;

        leap_gestures.keytap_id = gesture.id();
        leap_gestures.keytap_state = gesture.state();
        for(int i = 0; i < 3; i++){
          leap_gestures.keytap_position[i] = *(keytap.position().toFloatPointer() + i*sizeof(float));
        }
        for(int i = 0; i < 3; i++){
          leap_gestures.keytap_direction[i] = *(keytap.direction().toFloatPointer() + i*sizeof(float));
        }
        if (keytap.state() == Gesture::STATE_STOP){
          leap_gestures.keytap_in_progress = false;
        } else {
          leap_gestures.keytap_in_progress = true;
        }
        break;
      }
      case Gesture::TYPE_SCREEN_TAP:
      {
        ScreenTapGesture screentap = gesture;

        leap_gestures.screentap_id = gesture.id();
        leap_gestures.screentap_state = gesture.state();
        for(int i = 0; i < 3; i++){
          leap_gestures.screentap_position[i] = *(screentap.position().toFloatPointer() + i*sizeof(float));
        }
        for(int i = 0; i < 3; i++){
          leap_gestures.screentap_direction[i] = *(screentap.direction().toFloatPointer() + i*sizeof(float));
        }
        if (screentap.state() == Gesture::STATE_STOP){
          leap_gestures.screentap_in_progress = false;
        } else {
          leap_gestures.screentap_in_progress = true;
        }
        break;
      }
      default:
        break;
    }
  }
}

/*
 * Leap Motion gestures definitions
 */
LeapGestures_::LeapGestures_(){
  // circle
  this->circle_id = 0;
  this->circle_in_progress = false;
  this->circle_clockwise = false;
  this->circle_progress = 0;
  this->circle_angle = 0.0;
  this->circle_radius = 0.0;
  this->circle_state = 0;
  // swipe
  this->swipe_id = 0;
  this->swipe_in_progress = false;
  for(int i = 0; i < 3; i++){
    this->swipe_direction[i] = 0.0;
  }
  this->swipe_speed = 0.0;
  this->swipe_state = 0;
  // keytap
  this->keytap_id = 0;
  this->keytap_in_progress = false;
  for(int i = 0; i < 3; i++){
    this->keytap_direction[i] = 0.0;
  }
  for(int i = 0; i < 3; i++){
    this->keytap_position[i] = 0.0;
  }
  this->keytap_state = 0;
  // screentap
  this->screentap_id = 0;
  this->screentap_in_progress = false;
  for(int i = 0; i < 3; i++){
    this->screentap_direction[i] = 0.0;
  }
  for(int i = 0; i < 3; i++){
    this->screentap_position[i] = 0.0;
  }
  this->screentap_state = 0;
}


//
