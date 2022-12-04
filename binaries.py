from autoencoder.model import Sampling, VAE
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import h5py


# use Pythagorean norm (maybe weighted features in future?)
def calculate_distance(features_1, features_2):
    return np.linalg.norm(np.array(features_1) - np.array(features_2))

def load_training_dataset():

    import h5py
    global labels, images

    with h5py.File('Galaxy10.h5', 'r') as F:    

        labels = np.array(F['ans'])
        (milkywaylikes_idx,) = np.where(labels > 3)
        images = np.array(F['images'][milkywaylikes_idx])
        labels = np.array(F['ans'][milkywaylikes_idx]) - 4
        images = images.astype(np.float32) / 255.

    _, y, x, _ = images.shape
    startx = x // 2 -(64 // 2)
    starty = y // 2 -(64 // 2)    

    images = images[:, starty:starty + 64, startx:startx + 64, :]

def get_closest_neighbour(NN, reference_image):

    if isinstance(reference_image, str):
        reference_image = np.array(Image.open(reference_image))
        reference_latent_position = NN.encode(np.array([reference_image]).reshape(1,64,64,3))[0][0]

    elif isinstance(reference_image, np.ndarray):
        reference_latent_position = NN.encode(np.array([reference_image]).reshape(1,64,64,3))[0][0]

    try:
        images == None
    except NameError:
        load_training_dataset()

    predictions = np.array(NN.encoder.predict(images))

    min_index, min_label = 0, 0
    min_distance = np.inf

    for i, prediction in enumerate(predictions[0, :, :]):

        print(f"{i}/{len(images)} -> Minimal distance: {min_distance:.4f} for image {min_index} in cluster {min_label}", end = "\r")

        assert len(reference_latent_position) == len(prediction), "Dimensionality check failed"
        current_distance = calculate_distance(prediction, reference_latent_position)

        if current_distance < min_distance:

            min_prediction = prediction
            min_distance = current_distance
            min_index = i

    print(f"\n Minimum distance = {min_distance} for image {min_index} found!")

    return images[min_index]


if __name__ == "__main__":

    Network = VAE("./trained_models/vae_encoder_epoch_24.h5", "./trained_models/vae_decoder_epoch_24.h5")
    min_neighbour = get_closest_neighbour(Network, "./images/cropped/example_04.png")