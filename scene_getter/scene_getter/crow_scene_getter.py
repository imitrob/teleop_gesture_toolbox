
# Crow interface
import numpy as np
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rdflib.namespace import Namespace, RDF, RDFS, OWL, FOAF, XSD

from crow_ontology.crowracle_client import CrowtologyClient
from crow_msgs.msg import StampedString

from scene_getter.scene_lib.scene import Scene
from scene_getter.scene_lib.scene_object import SceneObject
from scene_getter import SceneGetter

ONTO_IRI = "http://imitrob.ciirc.cvut.cz/ontologies/crow"
CROW = Namespace(f"{ONTO_IRI}#")

class CrowSceneGetter(SceneGetter):
    def __init__(self):
        super().__init__(self)

        self.crowracle = CrowtologyClient(node=self)
        self.onto = self.crowracle.onto
        
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)

    def get_scene(self):
        s = None
        s = Scene()
        s.objects = []

        for object in self.crowracle.getTangibleObjectsProps():
            ''' o is dictionary containing properties '''
            uri = object['uri']
            id = object['uri'].split("#")[-1]
            color = object['color']
            color_nlp_name_CZ = object['color_nlp_name_CZ']
            color_nlp_name_EN = object['color_nlp_name_EN']
            nlp_name_CZ = object['nlp_name_CZ']
            nlp_name_EN = object['nlp_name_EN']
            absolute_location = object['absolute_location']
            # print(f"MY ID: {id}")
            
            o = SceneObject(name=id, position_real=np.array(absolute_location), random=False)
            # o.quaternion = np.array(object['pose'][1])
            # o.color_uri = color
            o.color = color_nlp_name_EN
            # o.color_name_CZ = color_nlp_name_CZ
            
            # o.nlp_name_CZ = nlp_name_CZ
            # o.nlp_name_EN = nlp_name_EN
            # o.crow_id = id
            # o.crow_uri = uri
            
            if id not in s.object_names:
                s.objects.append(o)

        return s       