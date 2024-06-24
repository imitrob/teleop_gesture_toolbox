



from geometry_msgs.msg import Pose, Point, Quaternion
from panda_ros import Panda


class PandaEnhanced(Panda):

    
    def go_on_top_of_object(self, nameobj, s):
        object = s.get_object_by_name(nameobj)
        if object is None:
            print(f"name of object {nameobj} not found")
            return
        q_final = self.get_quaternion_eef(object.quaternion, object.name)
        p = object.position_real

        pose = Pose(position=Point(x=p[0], y=p[1], z=p[2]+0.3), orientation=Quaternion(x=q_final[0],y=q_final[1],z=q_final[2],w=q_final[3]))
        
        self.go_to_pose(pose)

