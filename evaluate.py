import numpy as np
from read_dataset import *
from loss import *

valid_data = read_data(is_training = False)

# this function can only deal with batch = 1, 
# for batch training needs change
def eval_once(output_layer, data):
    image = data[0]
    label = data[1]
    result = output_layer.inference(image)
    if int(np.argmax(result)) == label:
        return 1
    else: 
        return 0

def evaluate(output_layer, saver, data, epochs):
    total_num = len(data)
    accuracy_counter = 0
    total_loss = 0
    for i in data:
        accuracy_counter += eval_once(output_layer, i)
        total_loss += calc_loss(output_layer, i)

    print("--------------------------------------------")
    print(output_layer.inference(data[0][0]))
    
    print("epochs %d " % epochs)
    print("accuracy = %.4f " % (accuracy_counter / total_num))
    print("average_loss = %.4f" % (total_loss / total_num))
    print("--------------------------------------------")

    saver["accuracy_" + str(epochs)] = accuracy_counter / total_num
    saver["loss_" + str(epochs)] = total_loss / total_num
    return total_loss / total_num

if __name__ == "__main__":
    pass

