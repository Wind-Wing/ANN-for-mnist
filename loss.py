import numpy as np
import math

def calc_loss(output_layer, data):
    sample = data[0]
    label = data[1]
    standard = np.zeros([10,1])
    standard[label] = 1.0

    infer = output_layer.inference(sample)
    
    variance = (abs(standard - infer) ** 2).sum()
    #standard_deviation = math.sqrt(variance)
    return variance / 2.0

# based on loss = 1/2 E (label_vector[i] - infer_vector[i]) ** 2
def loss_gradient(label, infer_result):
    stand_result = np.zeros(infer_result.shape)
    stand_result[label] = 1.0
    return infer_result - stand_result

# based on sigmoid
def active_funciton_gradient(infer_result):
    return infer_result * (1.0 - infer_result)
    # softplus
    #return 1.0 - 1.0 / math.e ** infer_result


    
