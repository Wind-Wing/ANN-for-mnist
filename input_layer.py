from layer import Layer
import numpy as np

class Input_layer(Layer):
    # input_shape is the shape of image, in mnist is [28,28]
    def __init__(self, sample_shape):
        self.output_array = np.empty(sample_shape)
        self.epoch = 0

    # override parent forward_propagating()
    def forward_propagating(self, sample):
        # transform to vector before return
        return sample.reshape([sample.size, 1])

    # override parent backward_propagation()
    # represent the end of backward propagation
    # do nothing, just stop recusive calling
    def backward_propagating(self, bp_factor, learning_rate):
        pass

    
        
    
    

        