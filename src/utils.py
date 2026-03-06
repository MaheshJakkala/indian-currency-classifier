import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from fastai.vision.all import *

import os

main_train_dir = os.path.join("Train/")
main_test_dir = os.path.join("Test/")

print(main_train_dir)
print(main_test_dir)