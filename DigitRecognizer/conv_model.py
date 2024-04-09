import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten

import TSConv as tsconv


## Class for the ConvBlock
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters = 4, conv_layer = 'Conv2D'):
        ## initialize the parent class
        super().__init__()

        ## Verify that the input for conv_layer is valid
        assert conv_layer == 'Conv2D' or conv_layer == 'TSConv', "Invalid argument, conv_layer only takes arguments Conv2D or TSConv"

        ## initialize the two convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME', activation = 'relu')
        self.conv2 = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME', activation = 'relu')
        ## initialize as time-shift convolutional layer instead when specified
        if(conv_layer == 'TSConv'):
            self.conv1 = tsconv.TSConv(filters = filters)
            self.conv2 = tsconv.TSConv(filters = filters)


    ## Call the convolutional block (ConvBlock) 
    def call(self, inputs):
        ## perform the first convolutional layer
        x = self.conv1(inputs)

        ## perform the second convolutional layer
        return self.conv2(x)


######################################################


## Class for the downsample (DB) block
class DB(tf.keras.layers.Layer):
    def __init__(self, filters = 4, conv_layer = 'Conv2D'):
        ## initialize the parent class
        super().__init__()

        ## initialize the max_pooling and the ConvBlock in the DB
        self.m_pool = tf.keras.layers.MaxPool2D(pool_size = (2, 1))
        self.block1 = ConvBlock(filters = filters, conv_layer = conv_layer)

    ## call the downsample block (DB)
    def call(self, inputs):
        x = self.m_pool(inputs)
        return self.block1(x)


######################################################


## class for the convolutional model
class ConvModel(tf.keras.Model):
    ## initialize the ConvModel
    def __init__(self, filters = 4, conv_layer = 'Conv2D'):
        ## initialize the parent class
        super(ConvModel, self).__init__()

        ## Verify that the input for conv_layer is valid
        assert conv_layer == 'Conv2D' or conv_layer == 'TSConv', "Invalid argument, conv_layer only takes arguments Conv2D or TSConv"
        self.input_layer = tf.keras.layers.InputLayer(input_shape = (28, 28, 1))

        ## The dropout layer with 10% dropout
        self.drop = tf.keras.layers.Dropout(0.1)
        
        ## Convolutional and downsample blocks
        self.block1 = ConvBlock(filters = 8, conv_layer = conv_layer)
        self.db1 = DB(filters = 8, conv_layer = conv_layer)

        self.block2 = ConvBlock(filters = 16, conv_layer = conv_layer)
        self.db2 = DB(filters = 16, conv_layer = conv_layer)

        self.block3 = ConvBlock(filters = 16, conv_layer = conv_layer)

        ## flatten
        self.flatten = Flatten(dtype='float32')

        ## dense layers
        #self.d1 = tf.keras.layers.Dense(256, activation="relu")
        self.d2 = tf.keras.layers.Dense(64, activation="relu")
        self.d3 = tf.keras.layers.Dense(32, activation="relu")
        
        self.dense_final = tf.keras.layers.Dense(10, activation="softmax")
        
    
    ## call the model
    def call(self, inputs):

        ## get the inputs
        inputs = self.input_layer(inputs)

        ## perform dropout
        x = self.drop(inputs)
        
        ## convolutional and downsample blocks
        x = self.block1(x)
        x = self.db1(x)
        x = self.block2(x)
        x = self.db2(x)
        x = self.block3(x)
        
        ## flatten
        x = self.flatten(x)
        print(x.shape)

        ## dense layers
        #x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        
        return self.dense_final(x)







