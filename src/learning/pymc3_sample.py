#!/usr/bin/env python3
''' Run main_sample_thread.py with: gesture_detection_approach: pymc3
                                    in include/custom_settings/gesture_config.yaml
'''
import numpy as np
import theano

class PyMC3_Sample():
    def __init__(self):
        self.X_train = None
        self.approx = None
        self.neural_network = None
        self.sample_proba = None
        self._sample_proba = None

    def sample(self, data):
        return self.sample_proba([data],100).mean(0)

    def init(self, nn):
        self.X_train = nn.X_train
        self.approx = nn.approx
        self.neural_network = nn.neural_network
        x = theano.tensor.matrix("X")
        n = theano.tensor.iscalar("n")
        x.tag.test_value = np.empty_like(self.X_train[:10])
        n.tag.test_value = 100
        self._sample_proba = self.approx.sample_node(
            self.neural_network.out.distribution.p, size=n, more_replacements={self.neural_network["ann_input"]: x}
        )
        self.sample_proba = theano.function([x, n], self._sample_proba)





#
