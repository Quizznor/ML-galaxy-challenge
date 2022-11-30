import matplotlib.pyplot as plt
import tensorflor.keras as tf
import numpy as np

# use Pythagorean norm (maybe weighted features in future?)
def calculate_distance(features_1, features_2):
    return np.linalg.norm(features_1, features_2)

