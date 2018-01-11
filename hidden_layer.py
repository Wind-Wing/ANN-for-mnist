from layer import Layer
import numpy as np

class Hidden_layer(Layer):

    def __init__(self, pre_layer, output_shape):
        super(Hidden_layer, self).__init__(pre_layer, output_shape)
