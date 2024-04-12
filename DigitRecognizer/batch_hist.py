import tensorflow as tf


## A class to store the loss and accuracy for each batch
class BatchHistory(tf.keras.callbacks.Callback):
    ## initialize the storage lists
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    ## append the losses and accuracies at the end of each batch
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))