import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten

import TSConv as tsconv
import ResBlock as resbl


#### This file implements the following classes ####
# 1) ConvBlock(): A block with two convolutional layers connected through batch normalization
# 2) DB(): A downsample block which performs max pooling followed by a ConvBlock()
# 3) ConvModel(): A convolutional network for MNIST digit classification, consisting of a ConvBlock(), a DB(), a max_pool and a dense fully connected layer.
# 4) ResModel(): A residual convolutional network for MNIST digit classification, consisting of a ResBlock(), a max_pool and a dense fully connected layer.
# 5) ResModelLarge(): A largeresidual convolutional network for MNIST digit classification, consisting of a 3 x ResBlock() before each max pooling, a max_pool and a dense fully connected layer.
#####################################################


## Class for the ConvBlock with two convolutional layers
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters = 4, conv_layer = 'Conv2D', kernel_size = (3, 3), use_bias = True):
        ## initialize the parent class
        super().__init__()

        ## Verify that the input for conv_layer is valid
        assert conv_layer == 'Conv2D' or conv_layer == 'TSConv', "Invalid argument, conv_layer only takes arguments Conv2D or TSConv"

        ## initialize the two convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = (1, 1), padding = 'SAME', activation = 'relu', use_bias = use_bias)
        self.conv2 = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = (1, 1), padding = 'SAME', activation = 'relu', use_bias = use_bias)
        ## initialize as time-shift convolutional layer instead when specified
        if(conv_layer == 'TSConv'):
            self.conv1 = tsconv.TSConv(filters = filters)
            self.conv2 = tsconv.TSConv(filters = filters)

        ## add the batch normalization layer
        self.batch_norm1 = tf.keras.layers.BatchNormalization()


    ## Call the convolutional block (ConvBlock) 
    def call(self, inputs):
        ## perform the first convolutional layer
        x = self.conv1(inputs)

        ## perform the normalization
        x = self.batch_norm1(x)

        ## perform the second convolutional layer
        return self.conv2(x)


######################################################


## Class for the downsample (DB) block
class DB(tf.keras.layers.Layer):
    def __init__(self, filters = 4, conv_layer = 'Conv2D', kernel_size = (3, 3)):
        ## initialize the parent class
        super().__init__()

        ## initialize the max_pooling and the ConvBlock in the DB
        self.m_pool = tf.keras.layers.MaxPool2D(pool_size = (2, 2))
        self.block1 = ConvBlock(filters = filters, conv_layer = conv_layer, kernel_size = (3, 3))

    ## call the downsample block (DB)
    def call(self, inputs):
        x = self.m_pool(inputs)
        return self.block1(x)


######################################################


## class for the convolutional model
class ConvModel(tf.keras.Model):
    ## initialize the ConvModel
    def __init__(self, filters = [16, 32], conv_layer = 'Conv2D', drop_rate = 0.2, kernel_size = (3, 3)):
        ## initialize the parent class
        super(ConvModel, self).__init__()

        ## Verify that the input for conv_layer is valid
        assert conv_layer == 'Conv2D' or conv_layer == 'TSConv', "Invalid argument, conv_layer only takes arguments Conv2D or TSConv"
        #self.input_layer = tf.keras.layers.InputLayer(input_shape = (28, 28, 1))
        
        ## Convolutional and downsample blocks
        self.block1 = ConvBlock(filters = filters[0], conv_layer = conv_layer, kernel_size = kernel_size)
        self.db1 = DB(filters = filters[1], conv_layer = conv_layer, kernel_size = kernel_size)
        self.mp2d = tf.keras.layers.MaxPool2D()

        ## flatten
        self.flatten = Flatten(dtype='float32')

        ## dense layers
        self.d1 = tf.keras.layers.Dense(256, activation="relu")

        ## The dropout layer with custom rate
        self.drop1 = tf.keras.layers.Dropout(drop_rate)
        self.drop2 = tf.keras.layers.Dropout(drop_rate)
        
        self.dense_final = tf.keras.layers.Dense(10, activation="softmax")
        
    
    ## call the model
    def call(self, inputs):

        ## get the inputs
        #inputs = self.input_layer(inputs)
        
        ## convolutional and downsample blocks
        x = self.block1(inputs)
        x = self.db1(x)
        x = self.mp2d(x)
        x = self.drop1(x)
        
        ## flatten
        x = self.flatten(x)

        ## dense layers
        x = self.d1(x)

        ## perform dropout
        x = self.drop2(x)
        
        return self.dense_final(x)


## class for the residual model
class ResModel(tf.keras.Model):
    ## initialize the ConvModel
    def __init__(self, filters = [16, 32], drop_rate = 0.2, kernel_size = (3, 3)):
        ## initialize the parent class
        super(ResModel, self).__init__()

        ## add the input layer
        #self.input_layer = tf.keras.layers.InputLayer(input_shape = (28, 28, 1))
        
        ## The first convolutional, the residual and max pool blocks
        self.conv1 = tf.keras.layers.Conv2D(filters = filters[0], kernel_size = kernel_size, strides = (1, 1), padding = 'SAME')
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.resblock1 = resbl.ResBlock(filters = filters[0], kernel_size = kernel_size)
        self.mp2d1 = tf.keras.layers.MaxPool2D()
        self.resblock2 = resbl.ResBlock(filters = filters[1], kernel_size = kernel_size)
        self.mp2d2 = tf.keras.layers.MaxPool2D()

        ## flatten
        self.flatten = Flatten(dtype='float32')

        ## dense layers
        self.d1 = tf.keras.layers.Dense(256, activation="relu")

        ## The dropout layer with custom rate
        self.drop1 = tf.keras.layers.Dropout(drop_rate)
        self.drop2 = tf.keras.layers.Dropout(drop_rate)
        
        self.dense_final = tf.keras.layers.Dense(10, activation="softmax")
        
    
    ## call the model
    def call(self, inputs):

        ## get the inputs
        #inputs = self.input_layer(inputs)

        ## first convolutional layer and batch normalization
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        
        ## convolutional and downsample blocks
        x = self.resblock1(x)
        x = self.mp2d1(x)
        x = self.resblock2(x)
        x = self.mp2d2(x)
        x = self.drop1(x)
        
        ## flatten
        x = self.flatten(x)

        ## dense layers
        x = self.d1(x)

        ## perform dropout
        x = self.drop2(x)
        
        return self.dense_final(x)


## class for the residual large model
class ResModelLarge(tf.keras.Model):
    ## initialize the ConvModel
    def __init__(self, filters = [16, 32], drop_rate = 0.2, kernel_size = (3, 3)):
        ## initialize the parent class
        super().__init__()

        ## add the input layer
        #self.input_layer = tf.keras.layers.InputLayer(input_shape = (28, 28, 1))
        
        ## The first convolutional, the residual and max pool blocks
        self.conv1 = tf.keras.layers.Conv2D(filters = filters[0], kernel_size = kernel_size, strides = (1, 1), padding = 'SAME')
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.resblock1_1 = resbl.ResBlock(filters = filters[0], kernel_size = kernel_size)
        self.resblock1_2 = resbl.ResBlock(filters = filters[0], kernel_size = kernel_size)
        self.resblock1_3 = resbl.ResBlock(filters = filters[0], kernel_size = kernel_size)
        self.mp2d1 = tf.keras.layers.MaxPool2D()
        self.resblock2_1 = resbl.ResBlock(filters = filters[1], kernel_size = kernel_size)
        self.resblock2_2 = resbl.ResBlock(filters = filters[1], kernel_size = kernel_size)
        self.resblock2_3 = resbl.ResBlock(filters = filters[1], kernel_size = kernel_size)
        self.mp2d2 = tf.keras.layers.MaxPool2D()

        ## flatten
        self.flatten = Flatten(dtype='float32')

        ## dense layers
        self.d1 = tf.keras.layers.Dense(256, activation="relu")

        ## The dropout layer with custom rate
        self.drop1 = tf.keras.layers.Dropout(drop_rate)
        self.drop2 = tf.keras.layers.Dropout(drop_rate)
        
        self.dense_final = tf.keras.layers.Dense(10, activation="softmax")
        
    
    ## call the model
    def call(self, inputs):

        ## get the inputs
        #inputs = self.input_layer(inputs)

        ## first convolutional layer and batch normalization
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        
        ## convolutional and max pooling 
        x = self.resblock1_1(x) ## first group of layers
        x = self.resblock1_2(x)
        x = self.resblock1_3(x)
        x = self.mp2d1(x)
        
        x = self.resblock2_1(x) ## second group of layers
        x = self.resblock2_2(x)
        x = self.resblock2_3(x)
        x = self.mp2d2(x)
        x = self.drop1(x)
        
        ## flatten
        x = self.flatten(x)

        ## dense layers
        x = self.d1(x)

        ## perform dropout
        x = self.drop2(x)
        
        return self.dense_final(x)




