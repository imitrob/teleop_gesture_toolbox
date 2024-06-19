
    @staticmethod
    def go_on_top_of_object_modular(nameobj, s):
        object = s.get_object_by_name(nameobj)
        if object is None:
            print(f"name of object {nameobj} not found")
            return
        q_final = RealRobotConvenience.get_quaternion_eef(object.quaternion, object.name)
        p = object.position_real

        # HERE SEND POSE TO TOPIC
        Pose(position=Point(x=p[0], y=p[1], z=p[2]+0.3), orientation=Quaternion(x=q_final[0],y=q_final[1],z=q_final[2],w=q_final[3]))
        
        # rc.roscm.

        # RealRobotConvenience.move_sleep()


    @staticmethod
    def correction_by_teleop():
        if rc.roscm.is_real:
            print(f"{cc.H}Teleop started{cc.E}")
            while not gl.gd.present():
                time.sleep(0.1)
            while gl.gd.present():
                t = time.time()
                while time.time()-t < 5.0:
                    md.live_handle_step(mod=1, scale=0.03, local_live_mode='no_eef_rotation')
                    time.sleep(0.01)
            print(f"{cc.H}Teleop ended{cc.E}")
        return True
