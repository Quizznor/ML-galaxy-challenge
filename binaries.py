import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import h5py


# use Pythagorean norm (maybe weighted features in future?)
def calculate_distance(features_1, features_2):
    return np.linalg.norm(features_1, features_2)

def load_training_dataset():

    import h5py
    global labels, images

    with h5py.File('Galaxy10.h5', 'r') as F:    

        labels = np.array(F['ans'])
        (milkywaylikes_idx,) = np.where(labels > 3)
        images = np.array(F['images'][milkywaylikes_idx])
        labels = np.array(F['ans'][milkywaylikes_idx]) - 4
        images = images.astype(np.float32) / 255.


def plot_class_distances(NN, reference_image_str):

    try:
        images == None
    except NameError:
        load_training_dataset()

    unique_labels = np.unique(labels)
    classwise_distances = [[] for _ in range(len(unique_labels))]

    for l in unique_labels:

        candidates = images[labels == l]
        

    # reference_image = np.array(Image.open(reference_image_str))
    # reference_latent_position = NN.encoder(reference_image)



    # for image in images:

if __name__ == "__main__":

    plot_class_distances(0, "./images/cropped/example_04.png")