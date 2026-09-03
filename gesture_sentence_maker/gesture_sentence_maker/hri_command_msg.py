"""Select the HRICommand ROS interface available in this workspace.

The Franka HRI stack owns the preferred ``hri_msgs/HRICommand`` type.  A
standalone gesture-toolbox installation has the same small wire shape in
``gesture_msgs`` so that its publishers, subscribers and dashboard continue to
work without Franka HRI being installed.

All toolbox code must import HRICommand from this module.  ROS topics still use
one concrete type at runtime; this module merely makes that choice before the
endpoints are created.
"""

try:
    from hri_msgs.msg import HRICommand
except ImportError:
    from gesture_msgs.msg import HRICommand


HRICommandMSG = HRICommand
HRI_COMMAND_PACKAGE = HRICommand.__module__.split(".", 1)[0]
HRI_COMMAND_TYPE = f"{HRI_COMMAND_PACKAGE}/msg/HRICommand"
HRI_COMMAND_ROSBRIDGE_TYPE = f"{HRI_COMMAND_PACKAGE}/HRICommand"

__all__ = [
    "HRICommand",
    "HRICommandMSG",
    "HRI_COMMAND_TYPE",
    "HRI_COMMAND_ROSBRIDGE_TYPE",
]
