# Run the training in this file
import argparse
import numpy as np
import json
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import datetime
import wandb
from wandb.keras import WandbCallback
import os


from model import Sampling, get_encoder, get_decoder, VAE


print("Using tensorflow verion:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

physical_devices = tf.config.experimental.list_physical_devices('GPU')

parser = argparse.ArgumentParser()
parser.add_argument('--ConfigFile', nargs=1,
                    help='JSON config for Training',
                    type=argparse.FileType('r'))
arguments = parser.parse_args()
config_dict = json.load(arguments.ConfigFile[0])

#Login to wandb
#! wandb login config_dict["wandb_key"]
wandb.init(
    project = "galaxy",
    config = config_dict,
    )

run_name = wandb.run.name

BATCH_SIZE = config_dict["BATCH_SIZE"]

encoder = get_encoder(config_dict)
encoder.summary()
decoder = get_decoder(config_dict)
decoder.summary()

vae = VAE(encoder, decoder)

optimizer = keras.optimizers.Adam(
    learning_rate=config_dict["learning_rate"],
)

vae.compile(optimizer=optimizer)

#Load Data here


#set up the callbacks
start_time =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
dir_base = config_dict["save_model_folder"] + start_time + "_" + run_name

save_dir = os.path.join(dir_base, "models")
plot_dir = os.path.join(dir_base, "plots")

os.makedirs(save_dir, exist_ok=True) 
os.makedirs(plot_dir, exist_ok=True) 

print("All Functions and Classes defined! Start with the Training!")

history = vae.fit(X,
    verbose=2,
    epochs=config_dict["epochs"],
    callbacks=[
        WandbCallback(),
        ],
    )