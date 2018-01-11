import numpy as np
import loss
import math

class Layer:
    def __init__(self, pre_layer, output_shape):
        self.pre_layer = pre_layer

        self.input_array = np.empty_like(self.pre_layer.output_array)
        self.output_array = np.empty(output_shape)

        # when calc, input will reshape to vector
        # recover to square after calc for tranport 
        # between neurons
        self.bias = np.random.randn(self.output_array.size, 1)
        self.weight = np.random.randn(self.input_array.size, self.output_array.size)
        self.weight = self.weight / math.sqrt(self.output_array.size)

    def forward_propagating(self, sample):
        self.input = self.pre_layer.forward_propagating(sample)
        xw = np.matmul(self.input.T, self.weight).T
        self.output = self.active_function(xw + self.bias)
        return self.output

    def active_function(self, x):
        
        # sigmoid
        return 1.0 / (1.0 + math.e ** -x) 
        # softplus
        #return np.log(1.0 + math.e ** x)

    def backward_propagating(self, bp_factor, learning_rate):
        # gradient descent on this layer
        d_weight = np.matmul(self.input, bp_factor.T )
        self.weight = self.weight - d_weight * learning_rate
        d_bias = np.matmul(bp_factor, np.ones([1,1]))
        self.bias = self.bias - d_bias * learning_rate

        # calc previous layer's bp_factor
        active_grad = loss.active_funciton_gradient(self.input)
        pre_bp_factor = np.matmul(self.weight, bp_factor) * active_grad
        self.pre_layer.backward_propagating(pre_bp_factor, learning_rate)
    
