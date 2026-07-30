
std_msgs/Header header

int32 target_object_id
string target_object_name

string[] object_names
float32[] object_distances

geometry_msgs/Point line_point_1
geometry_msgs/Point line_point_2

geometry_msgs/Point target_object_position # focus_point

float32 hand_velocity # velocity at capture time

# How many consecutive frames target_object has stayed the closest one, and how
# many are needed before it may name an object (see deictic_evidence.py).
# A viewer can show evidence/evidence_threshold as how strongly it highlights.
int32 evidence
int32 evidence_threshold