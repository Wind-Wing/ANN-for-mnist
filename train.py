import numpy as np
from model import *
from loss import *
import evaluate
import shelve

reload = True
save_dir = "./"
save_file = "save1"
num_epochs = 200
learning_rate = 0.05

saver = shelve.open(save_dir + save_file, 'c', writeback= True)

if reload == True :
    try:
        ann_model = saver["model"]
        epochs = saver["epochs"]
    except:
        ann_model = output_layer
        epochs = 1
else:
    ann_model = output_layer
    epochs = 1

for i in range(num_epochs):
    print("epoche # %d " % epochs)
    np.random.shuffle(train_data)
    train_counter = 0
    train_total = len(train_data)
    for j in train_data:
        _image = j[0]
        _label = j[1]
        train_counter += 1
        #if(train_counter % 1000 == 0):
        #    print("training : %d / %d " % (train_counter, train_total))
        ann_model.optimize_loss(_image, _label, learning_rate)
    avg_loss = evaluate.evaluate(ann_model, saver, evaluate.valid_data, epochs)
    epochs += 1
    learning_rate = 0.05 + avg_loss/2
    saver["epochs"] = epochs
    saver["model"] = ann_model


saver.close()
