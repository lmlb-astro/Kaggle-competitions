import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten

class ConvModel():
    ## initialize the UNet
    def __init__(self):
        ## build UNet for a given number (T) of time steps
        self.model = self.build_model()
    
    ## The first block with two convolutional layers
    def block1(self, x, filters = 4, kernel_size = (3, 3), strides = (1, 1)):
        ## perform the first convolutional layer
        conv1 = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'SAME', activation = 'relu')(x)
        
        ## return the second convolutional layer
        return tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'SAME', activation = 'relu')(conv1)

    
    ## The downsample block (DB) of the model
    def DB(self, x, pool_size = (2, 1), filters = 4):
        ## max pool the data
        pooled_x = tf.keras.layers.MaxPool2D(pool_size = pool_size)(x)

        ## return the double convolutional layer (i.e. block1) on the pooled weights
        return self.block1(pooled_x, filters = filters)

    ## get the model
    def get_model(self):
        return self.model
    
    ## function to build the model
    def build_model(self):
        ## add the input layer
        inputs = tf.keras.Input(shape = (28, 28, 1))

        ## add the first convolutional block
        b1 = self.block1(inputs)

        ## add dropout
        drop = tf.keras.layers.Dropout(0.1)(b1)

        ## add the first downsample (DB) block
        db1 = self.DB(drop) 
        db2 = self.DB(db1)

        ## flatten the model
        flatten = Flatten(dtype='float32')
        fl = flatten(db2)
        print(fl.shape)

        ## add a few dense layers
        l1 = tf.keras.layers.Dense(256, activation="relu")(fl)
        l2 = tf.keras.layers.Dense(64, activation="relu")(l1)
        l3 = tf.keras.layers.Dense(32, activation="relu")(l2)

        outputs = tf.keras.layers.Dense(10, activation="softmax")(l3)

        return tf.keras.Model(inputs = inputs, outputs = outputs)








