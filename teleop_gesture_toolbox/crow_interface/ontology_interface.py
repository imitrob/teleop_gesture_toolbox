
import rclpy
import os, json
import sys
import json
from pkg_resources import resource_filename
from rcl_interfaces.srv import GetParameters
from rclpy.exceptions import ParameterNotDeclaredException
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
import argparse
from rdflib.namespace import Namespace, RDF, RDFS, OWL, FOAF, XSD
import sys
import time
from crow_ontology.crowracle_client import CrowtologyClient
from crow_msgs.msg import StampedString
import traceback as tb

ONTO_IRI = "http://imitrob.ciirc.cvut.cz/ontologies/crow"
CROW = Namespace(f"{ONTO_IRI}#")

ACTION_TRANSL = {"seber":"lift", "pustit":"drop", "ukaž":"reach", "podej":"pass me"}

# NOT_PROFILING = True
# if not NOT_PROFILING:
#     from crow_control.utils.profiling import StatTimer

class OntologyInterface(Node): # Maybe it should be the root node? GesturesProcessor(Node):
    #CLIENT = None
    MAX_OBJ_REQUEST_TIME = 2.5

    def __init__(self, node_name="ontology_reader_and_adder", model_path=""):
        super().__init__(node_name)
        
        self.crowracle = CrowtologyClient(node=self)
        self.onto = self.crowracle.onto
        self.LANG = 'en'
        # self.get_logger().info(self.onto)

        #create listeners (synchronized)
        self.nlTopic = "/nlp/command"
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        
        self.add_dummy_cube()
        self.add_dummy_cube()
        print(self.get_objects_from_onto())
        print("Ready")
        # if not NOT_PROFILING:
        #     StatTimer.init()

            
        
    def match_object_in_onto(self, obj):
        onto = self.get_objects_from_onto()
        obj = json.loads(obj)
        for o in onto:
           url = str(o["uri"])
           if url is not None and url == obj[0]["target"][0]:
              return o
        return None
        
        
    def add_dummy_cube(self):
        o_list = self.get_objects_from_onto()
        if len(o_list) == 0:
           print("Adding a dummy cube into ontology")
           self.crowracle.add_test_object("CUBE")
        
    def get_objects_from_onto(self):
        o_list = self.crowracle.getTangibleObjectsProps()
        #print("Onto objects:")
        #print(o_list)
        return o_list
        
    def get_action_from_cmd(self, cmd):
        action = json.loads(cmd)[0]["action_type"].lower()
        if action in ACTION_TRANSL.keys():
           return ACTION_TRANSL[action]
        else:
           print("No czech translation for action " + action)
           return action
        
        
    def publish_trajectory(self, trajectory):
        #TODO choose proper msg type
        action = json.dumps(trajectory)
        msg = StampedString()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.data = action
        print(f'Publishing {msg.data}')
        self.trajectory_publisher.publish(msg)
        

    def keep_alive(self):
        self.pclient.nlp_alive = time.time()

def main():
    rclpy.init()
    try:
        oi = OntologyInterface()
        rclpy.spin(oi)
    except KeyboardInterrupt:
        print("User requested shutdown.")
    except BaseException as e:
        print(f"Some error had occured: {e}")
        tb.print_exc()
if __name__ == '__main__':
    main()
