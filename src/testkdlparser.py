#!/usr/bin/env python

import urdf_parser_py.urdf as pyurdf
urdfString = '/home/pierro/my_ws/src/relaxed_ik/src/RelaxedIK/urdfs/ur5.urdf'

pyurdf.URDF.__dict__
pyurdf.__dict__
pyurdf.treeFromFile(urdfString)
