#include "Leap.h"
using namespace Leap;

class Vector_{
public:
  Vector_();
  Vector_(double x, double y, double z);
  Vector_(double* pvector);

  double x;
  double y;
  double z;
private:
};

class Bone_{
public:
  Bone_(Bone* pbone);
  Bone_();

  Vector_ basis[3];
  Vector_ direction;
  Vector_ next_joint;
  Vector_ prev_joint;
  Vector_ center;
  bool is_valid;
  double length;
  double width;
private:
};

class Finger_{
public:
  Finger_(Finger* pfinger);
  Finger_();

  Bone_ bones[4];
private:
};

class Hand_{
public:
  // Data processed for learning
  double wrist_angles[90] = {};
  double bone_angles[90] = {};
  double finger_distances[90] = {};

  bool visible;
  int64_t id;
  bool is_left;
  bool is_right;
  bool is_valid;
  double grab_strength;
  double pinch_strength;
  double confidence;
  Vector_ palm_normal;
  Vector_ direction;
  Vector_ palm_position;
  Finger_ fingers[5];
  Vector_ palm_velocity;
  Vector_ basis[3];
  double palm_width;
  Vector_ sphere_center;
  double sphere_radius;
  Vector_ stabilized_palm_position;
  double time_visible;
  Vector_ wrist_position;

  Hand_(Hand* phand);
  Hand_();
private:
};

class LeapGestures_{
public:
  LeapGestures_();

  // circle
  int circle_id;
  bool circle_in_progress;
  bool circle_clockwise;
  double circle_progress;
  double circle_angle;
  double circle_radius;
  double circle_state;
  // swipe
  int swipe_id;
  bool swipe_in_progress;
  double swipe_direction[3];
  double swipe_speed;
  int swipe_state;
  // keytap
  int keytap_id;
  bool keytap_in_progress;
  double keytap_direction[3];
  double keytap_position[3];
  int keytap_state;
  // screentap
  int screentap_id;
  bool screentap_in_progress;
  double screentap_direction[3];
  double screentap_position[3];
  int screentap_state;
private:
};


class Frame_{
public:
  unsigned int seq;
  unsigned int fps;
  unsigned int secs;
  unsigned int nsecs;
  unsigned int hands;
  // Hand data
  Hand_ l;
  Hand_ r;

  LeapGestures_ leapgestures;

  Frame_(Frame* pframe, Frame* pprevframe);
  Frame_();
private:
};
