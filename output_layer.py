from layer import Layer
import numpy as np
import loss

class Output_layer(Layer):
    def __init__(self, pre_layer, output_shape):
        super(Output_layer, self).__init__(pre_layer, output_shape)

    def inference(self, sample):
        # normalization
        output = self.forward_propagating(sample)
        sum_output = output.sum()
        return output / sum_output

    def infer_gradient(self, result):
        _sum = self.output.sum()
        _sub = np.empty(result.shape)
        for i in range(result.size):
            _sub[i] = _sum
        
        return (_sub - result) / _sum ** 2
    
        
    def optimize_loss(self, sample, label, learning_rate):
        # first forward propagation
        # must first fp, or self.output/input will not be calc
        infer_result = self.inference(sample)
        # then backward_propagation
        bp_factor0 = loss.loss_gradient(label, infer_result) * \
                     self.infer_gradient(self.output) * \
                     loss.active_funciton_gradient(self.output)
        self.backward_propagating(bp_factor0, learning_rate)




        
        
