from input_layer import *
from hidden_layer import *
from output_layer import *
from read_dataset import *

train_data = read_data()
_images, _labels = zip(*train_data)

input_layer = Input_layer(_images[0].shape)
hidden_layer = Hidden_layer(input_layer, [7,7])
output_layer = Output_layer(hidden_layer, [10,1])

if __name__ == "__main__":
    #print(output_layer.inference(_images[0]))
    output_layer.optimize_loss(_images[0], _labels[0], 0.01)
