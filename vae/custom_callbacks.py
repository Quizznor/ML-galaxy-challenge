from tensorflow import keras
import os
import numpy as np
import wandb
import tensorflow as tf
from tensorflow.keras import backend as k


class Save_Encoder(keras.callbacks.Callback):

    def __init__(self,
                 folder,
                 save_every,
                 name="encoder"):

        self.overall_idx = 0
        self.folder = folder
        self.save_every = save_every
        self.name=name

    def on_epoch_end(self, epoch, logs={}):
        """
        Call this function at the end of each batch. And save the model every n steps
        """

        self.model.encoder.save(self.folder+"/"+self.name+"_encoder"+"_epoch_{idx}.h5".format(idx = self.overall_idx))
        self.model.decoder.save(self.folder+"/"+self.name+"_decoder"+"_epoch_{idx}.h5".format(idx = self.overall_idx))
        
        self.overall_idx +=1