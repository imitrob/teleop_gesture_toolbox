# Coppelia Sim ROS Interface



## Available services:

##### How to setup:

```
from mirracle_gestures.srv import AddMesh, AddMeshResponse, RemoveMesh, RemoveMeshResponse, GripperControl, GripperControlResponse


rospy.wait_for_service('add_mesh')
try:
    add_mesh = rospy.ServiceProxy('add_mesh', AddMesh)
    resp1 = add_mesh(file, name, pose, frame_id)
except rospy.ServiceException as e:
    print("Service call failed: %s"%e)
```

### Add object

Creates object on scene from shape or mesh file.

Service name: add_object
Service msg type: AddObject from mirracle_gestures.srv
```
string file # Location of mesh file
string name # Name of created mesh
geometry_msgs/Pose pose # Pose of mesh
string shape # 'cube', 'sphere', 'cylinder', 'cone'
float32 size
string color # 'r', 'g', 'b', 'c', 'm', 'y', 'k'
float32 friction
string frame_id
---
bool success

```

### Remove object

Service name: remove_object
Service msg type: RemoveObject from mirracle_gestures.srv
```
string name # Name of created mesh
---
bool success
```

### Gripper control

Service name: gripper_control
Service msg type: GripperControl from mirracle_gestures.srv
```
float32 position
float32 effort
string action # choose 'grasp'/'release'/''
string object # grasping object
---
bool success
```
