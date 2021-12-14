
#include <iostream>
#include <cstring>
#include "Leap.h"

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "mirracle_gestures.msg/Frame.h"

#include "handclasses.h"

using namespace Leap;

class SampleListener : public Listener {
  public:
    virtual void onInit(const Controller&);
    virtual void onConnect(const Controller&);
    virtual void onDisconnect(const Controller&);
    virtual void onExit(const Controller&);
    virtual void onFrame(const Controller&);
    virtual void onFocusGained(const Controller&);
    virtual void onFocusLost(const Controller&);
    virtual void onDeviceChange(const Controller&);
    virtual void onServiceConnect(const Controller&);
    virtual void onServiceDisconnect(const Controller&);
    virtual void rosPublish(Frame_);
  private:
};

const std::string fingerNames[] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};
const std::string boneNames[] = {"Metacarpal", "Proximal", "Middle", "Distal"};
const std::string stateNames[] = {"STATE_INVALID", "STATE_START", "STATE_UPDATE", "STATE_END"};

void SampleListener::onInit(const Controller& controller) {
  this->is_recording = false;

  ros::Publisher this->publisher = n.advertise<std_msgs::String>("chatter", 1000);
  std::cout << "Initialized" << std::endl;
}

void SampleListener::onConnect(const Controller& controller) {
  std::cout << "Connected" << std::endl;
  controller.enableGesture(Gesture::TYPE_CIRCLE);
  controller.enableGesture(Gesture::TYPE_KEY_TAP);
  controller.enableGesture(Gesture::TYPE_SCREEN_TAP);
  controller.enableGesture(Gesture::TYPE_SWIPE);
}

void SampleListener::onDisconnect(const Controller& controller) {
  // Note: not dispatched when running in a debugger.
  std::cout << "Disconnected" << std::endl;
}

void SampleListener::onExit(const Controller& controller) {
  std::cout << "Exited" << std::endl;
}

void SampleListener::onFrame(const Controller& controller) {
  // Get the most recent frame and report some basic information
  const Frame frame = controller.frame();
  const Frame prevframe = controller.frame(1);
  // Convert (Leap Motion) Frame object frame to (Mirracle Object) Frame_
  // Reason -> Frame_ has simpler definitions and is open source (hand_classes.h)
  Frame_ frame_(frame, prevframe);

  //if (this->is_recording == true){
  //}

  //#ifdef ROS
  this::rosPublish(frame_);
  //#endif
}

void SampleListener::rosPublish(frame_){

Framemsg msg;
msg.header.seq = frame_.seq;
msg.header.secs = frame_.secs;
msg.header.nsecs = frane.nsecs;
msg.fps = frame_.fps;
msg.hands = frame_.hands;

Hand * hands[2] = {&(msg.l), &(msg.r)};
Hand * hands_[2] = {&(frame_.l), &(frame_.r)};
for(int i = 0; i < 2; i++){
  hand = *(hands[i]);
  hand_ = *(hands_[i]);

  hand.id = hand_.id;
  hand.is_left = hand_.is_left;
  hand.is_right = hand_.is_right;
  hand.is_valid = hand_.is_valid;
  hand.grab_strength = hand_.grab_strength;
  hand.pinch_strength = hand_.pinch_strength;
  hand.confidence = hand_.confidence;
  hand.palm_normal = hand_.palm_normal();
  hand.direction = hand_.direction();
  hand.palm_position = hand_.palm_position();

  int l = 0;
  for(int j = 0; j < 5; j++){
    finger = hand.fingers[j];
    for(int k = 0; k < 4; k++){
      bone = finger.bones[k];

      hand.finger_bones[l].basis = bone.basis();
      hand.finger_bones[l].direction = bone.direction();
      hand.finger_bones[l].next_joint = bone.next_joint();
      hand.finger_bones[l].prev_joint = bone.prev_joint();
      hand.finger_bones[l].center = bone.center();

      hand.finger_bones[l].is_valid = bone.is_valid;
      hand.finger_bones[l].length = bone.length;
      hand.finger_bones[l].width = bone.width;
      l++;
    }
  }

  hand.palm_velocity = hand_.palm_velocity();
  hand.basis = hand_.basis();
  hand.palm_width = hand_.palm_width;
  hand.sphere_center = hand_.sphere_center();
  hand.sphere_radius = hand_.sphere_radius;
  hand.stabilized_palm_position = hand_.stabilized_palm_position();
  hand.time_visible = hand_.time_visible;
  hand.wrist_position = hand_.wrist_position();
}

msg.leapgestures.circle_id = hand_.leapgestures.circle_id;
msg.leapgestures.circle_in_progress = hand_.leapgestures.circle_in_progress;
msg.leapgestures.circle_clockwise = hand_.leapgestures.circle_clockwise;
msg.leapgestures.circle_progress = hand_.leapgestures.circle_progress;
msg.leapgestures.circle_angle = hand_.leapgestures.circle_angle;
msg.leapgestures.circle_radius = hand_.leapgestures.circle_radius;
msg.leapgestures.circle_state = hand_.leapgestures.circle_state;

msg.leapgestures.swipe_id = hand_.leapgestures.swipe_id;
msg.leapgestures.swipe_in_progress = hand_.leapgestures.swipe_in_progress;
msg.leapgestures.swipe_direction = hand_.leapgestures.swipe_direction();
msg.leapgestures.swipe_speed = hand_.leapgestures.swipe_speed;
msg.leapgestures.swipe_state = hand_.leapgestures.swipe_state;

msg.leapgestures.keytap_id = hand_.leapgestures.keytap_id;
msg.leapgestures.keytap_in_progress = hand_.leapgestures.keytap_in_progress;
msg.leapgestures.keytap_direction = hand_.leapgestures.keytap_direction();
msg.leapgestures.keytap_position = hand_.leapgestures.keytap_position();
msg.leapgestures.keytap_speed = hand_.leapgestures.keytap_speed;
msg.leapgestures.keytap_state = hand_.leapgestures.keytap_state;

msg.leapgestures.screentap_id = hand_.leapgestures.screentap_id;
msg.leapgestures.screentap_in_progress = hand_.leapgestures.screentap_in_progress;
msg.leapgestures.screentap_direction = hand_.leapgestures.screentap_direction();
msg.leapgestures.screentap_state = hand_.leapgestures.screentap_state;
msg.leapgestures.screentap_position = hand_.leapgestures.screentap_position();

this->publisher.publish(msg);
}

void SampleListener::onFocusGained(const Controller& controller) {
  std::cout << "Focus Gained" << std::endl;
}

void SampleListener::onFocusLost(const Controller& controller) {
  std::cout << "Focus Lost" << std::endl;
}

void SampleListener::onDeviceChange(const Controller& controller) {
  std::cout << "Device Changed" << std::endl;
  const DeviceList devices = controller.devices();

  for (int i = 0; i < devices.count(); ++i) {
    std::cout << "id: " << devices[i].toString() << std::endl;
    std::cout << "  isStreaming: " << (devices[i].isStreaming() ? "true" : "false") << std::endl;
  }
}

void SampleListener::onServiceConnect(const Controller& controller) {
  std::cout << "Service Connected" << std::endl;
}

void SampleListener::onServiceDisconnect(const Controller& controller) {
  std::cout << "Service Disconnected" << std::endl;
}

int main(int argc, char** argv) {
  // ROS init
  ros::init(argc, argv, "leap_listener_depositor");
  ros::NodeHandle n;

  // Create a sample listener and controller
  SampleListener listener;
  Controller controller;

  // Have the sample listener receive events from the controller
  controller.addListener(listener);

  if (argc > 1 && strcmp(argv[1], "--bg") == 0)
    controller.setPolicy(Leap::Controller::POLICY_BACKGROUND_FRAMES);

  // Keep this process running until Enter is pressed
  std::cout << "Press Enter to quit..." << std::endl;
  std::cin.get();

  // Remove the sample listener when done
  controller.removeListener(listener);

  return 0;
}
