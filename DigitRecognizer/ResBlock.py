import numpy as np
import tensorflow as tf

#### this file implements the following classes ####
# 1) A ResBlock(): a block of two convolutional layers using the Residual network design
####################################################

## Class for a residual convolutional block with 2 layers
class ResBlock(tf.keras.layers.Layer):
    
    def __init__(self, filters = 4, kernel_size = (3, 3), use_bias = True):
        ## initialize the parent class
        super().__init__()

        ## initialize the two convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = (1, 1), padding = 'SAME', use_bias = use_bias)
        self.conv2 = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = (1, 1), padding = 'SAME', use_bias = use_bias)

        ## add the batch normalization layers
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        ## add the ReLu layers
        self.relu1 = tf.keras.layers.ReLU()
        self.relu2 = tf.keras.layers.ReLU()
    

    
    ## call of the ResBlock()
    def call(self, inputs):
        ## first convolutional layer + normalization + activation
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        
        ## second convolutional layer +normalization
        x = self.conv2(x)
        x = self.batch_norm2(x)

        ## return the output of the residual block
        return self.relu2(inputs + x)

        
        