import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten


## Class for Block1
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters  = 4, kernel_size = (3, 3), strides = (1, 1)):
        ## initialize the parent class
        super().__init__()

        ## initialize the two convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size = kernel_size, strides = strides, padding = 'SAME', activation = 'relu')
        self.conv2 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size = kernel_size, strides = strides, padding = 'SAME', activation = 'relu')


    ## Call block1 
    def call(self, inputs):
        ## perform the first convolutional layer
        x = self.conv1(inputs)

        ## perform the first convolutional layer
        x = self.conv2(x)
        return x


######################################################


## Class for the downsample (DB) block
class DB(tf.keras.layers.Layer):
    def __init__(self, pool_size = (2, 1), filters = 4):
        ## initialize the parent class
        super().__init__()

        ## initialize the max_pooling and the ConvBlock
        self.m_pool = tf.keras.layers.MaxPool2D(pool_size = pool_size)
        self.block1 = ConvBlock(num_filters = filters)

    ## call the downsample block (DB)
    def call(self, inputs):
        x = self.m_pool(inputs)
        return self.block1(x)


######################################################


## class for the convolutional model
class ConvModel(tf.keras.Model):
    ## initialize the ConvModel
    def __init__(self, input_shape = (28, 28, 1), filters = 4, kernel_size = (3, 3), strides = (1, 1)):
        ## initialize the parent class
        super().__init__()
        
        ## Initiate the ConvModel
        ## first conv_block
        self.block1 = ConvBlock()

        ## downsample blocks
        self.db1 = DB()
        self.db2 = DB()

        ## flatten
        self.flatten = Flatten(dtype='float32')

        ## dense layers
        self.d1 = tf.keras.layers.Dense(256, activation="relu")
        self.d2 = tf.keras.layers.Dense(64, activation="relu")
        self.d3 = tf.keras.layers.Dense(32, activation="relu")
        
        self.dense_final = tf.keras.layers.Dense(10, activation="softmax")

    
    ## call the model
    def call(self, inputs):
        ## conv block
        x = self.block1(inputs)

        ## downsample blocks
        x = self.db1(x)
        x = self.db2(x)
        
        ## flatten
        x = self.flatten(x)

        ## dense layers
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        
        return self.dense_final(x)







