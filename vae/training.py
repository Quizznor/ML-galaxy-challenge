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
import jax


from model import Sampling, get_encoder, get_decoder, VAE


print("Using tensorflow verion:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

physical_devices = tf.config.experimental.list_physical_devices('GPU')


rng = jax.random.PRNGKey(0)

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
latent_dim = config_dict["latent_dim"]

encoder = get_encoder(latent_dim)
decoder = get_decoder(latent_dim)

vae = VAE(encoder, decoder)

optimizer = keras.optimizers.Adam(
    learning_rate=config_dict["learning_rate"],
)

vae.compile(optimizer=optimizer)


print("Load Data")

with h5py.File(config_dict["training_data"], 'r') as F:
  labels = np.array(F['ans'])
  # reject some we don't care about, keep the rest
  (milkywaylikes_idx,) = np.where(labels > 3)
  images = np.array(F['images'][milkywaylikes_idx])
  # Milky way similar galaxies are class '3' ('-4' normalizes the labels again)
  labels = np.array(F['ans'][milkywaylikes_idx]) - 4

images = images.astype(np.float32) / 255.

# crop the image a bit to get 64x64 shape
def crop_center(images, cropx, cropy):
    _, y, x, _ = images.shape
    startx = x // 2 -(cropx // 2)
    starty = y // 2 -(cropy // 2)    
    return images[:, starty:starty+cropy, startx:startx+cropx, :]

images = crop_center(images=images, cropx=64, cropy=64)

# shuffle once
rng, key = jax.random.split(rng)
shuffle_idx = jax.random.randint(key, (len(images),), 0, len(images))
images = images[shuffle_idx, ...]
labels = labels[shuffle_idx, ...]

# show shape
print("Data Shape:",images.shape)



#set up the callbacks
start_time =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
dir_base = config_dict["save_model_folder"] + start_time + "_" + run_name

save_dir = os.path.join(dir_base, "models")
plot_dir = os.path.join(dir_base, "plots")

os.makedirs(save_dir, exist_ok=True) 
os.makedirs(plot_dir, exist_ok=True) 

print("All Functions and Classes defined! Start with the Training!")


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_dir,
    monitor='loss',
    mode='min',
    save_best_only=True
    )


history = vae.fit(images,
    verbose=1,
    epochs=config_dict["epochs"],
    callbacks=[
        WandbCallback(),
        model_checkpoint_callback
        ],
    )