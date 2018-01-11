import numpy as np
import struct
import random
 
train_imageFilename = 'train-images.idx3-ubyte'
train_labelFilename = "train-labels.idx1-ubyte"
valid_imageFilename = "t10k-images.idx3-ubyte"
valid_labelFilename = "t10k-labels.idx1-ubyte"

dataset_path = "./mnist/"
normalizing_normalizing = 255

def read_images(filepath):
    binfile = open(filepath , 'rb')
    buf = binfile.read()
    
    index = 0   
    # unpack_from : transform bit stream into select python variables
    # > means big end way
    # I means unsigned int
    fileHeadFormat = '>IIII'
    magicNumber, numImages , numRows , numColumns = struct.unpack_from(fileHeadFormat , buf , index)
    if magicNumber != 2051:
        raise Exception("MagicNumber not match, binary file damaged")
    index += struct.calcsize(fileHeadFormat)

    imageSize = str(numRows * numColumns)
    imageFormat = '>' + imageSize + 'B'

    #numImages = 1
    images = []
    for i in range(numImages):
        image = struct.unpack_from(imageFormat ,buf, index)
        index += struct.calcsize(imageFormat)
        
        image = np.array(image) / normalizing_normalizing
        images.append(image.reshape(numRows,numColumns))

    '''
    # test, show an image
    fig = plt.figure()
    plt.imshow(images[0] , cmap='binary')
    plt.show()
    '''
    return images    


def read_labels(filepath):
    binfile = open(filepath , 'rb')
    buf = binfile.read()
    
    index = 0
    # unpack_from : transform bit stream into select python variables
    # > means big end way
    # I means unsigned int
    fileHeadFormat = '>II'
    magicNumber, numLabels = struct.unpack_from(fileHeadFormat , buf , index)
    if magicNumber != 2049:
        raise Exception("MagicNumber not match, binary file damaged")
    index += struct.calcsize(fileHeadFormat)

    
    labelFormat = '>1B'
    labels = []
    for i in range(numLabels):
        label = struct.unpack_from(labelFormat ,buf, index)
        index += struct.calcsize(labelFormat)
        labels.append(label[0])
        
    return labels

def read_data(is_training = True):
    if is_training:
        imageFilename = train_imageFilename
        labelFilename = train_labelFilename
    else:
        imageFilename = valid_imageFilename
        labelFilename = valid_labelFilename

    images = read_images(dataset_path + imageFilename)
    labels = read_labels(dataset_path + labelFilename)
    
    data = list(zip(images, labels))
    return data

if __name__ == "__main__":
    print(read_images(dataset_path + train_imageFilename)[0])
