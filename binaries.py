import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import h5py


# use Pythagorean norm (maybe weighted features in future?)
def calculate_distance(features_1, features_2):
    return np.linalg.norm(features_1 - features_2)

def load_training_dataset():

    import h5py
    global labels, images

    with h5py.File('Galaxy10.h5', 'r') as F:    

        labels = np.array(F['ans'])
        (milkywaylikes_idx,) = np.where(labels > 3)
        images = np.array(F['images'][milkywaylikes_idx])
        labels = np.array(F['ans'][milkywaylikes_idx]) - 4
        images = images.astype(np.float32) / 255.


def get_closest_neighbour(NN, reference_image):

    if isinstance(reference_image, str):
        reference_image = np.array(Image.open(reference_image))
        reference_latent_position = NN.encode(reference_image)

    elif isinstance(reference_image, np.ndarray):
        reference_latent_position = NN.encode(reference_image)

    try:
        images == None
    except NameError:
        load_training_dataset()

    min_distance = np.inf
    min_index = 0

    for i, (image, label) in enumerate(zip(images, labels)):    
        current_distance = calculate_distance(NN.encode(image), reference_latent_position)

        if current_distance < min_distance:

            print(f"Minimal distance: {min_distance:.4f} for image {i} in cluster {label}")
            min_distance = current_distance
            min_index = i

    return images[min_index]


# if __name__ == "__main__":
# 
    # plot_class_distances(0, "./images/cropped/example_04.png")