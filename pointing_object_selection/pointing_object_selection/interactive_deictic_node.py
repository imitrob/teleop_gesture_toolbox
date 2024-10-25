from pointing_object_selection.deictic_node import DeicticLibRos
from panda_ros import Panda

class DeicticLibPandaRos(Panda, DeicticLibRos):
    def step(self):
        deictic_solution = DeicticLibRos.step()
        self.go_on_top_of_object(deictic_solution['target_object_name'], self.scene)

