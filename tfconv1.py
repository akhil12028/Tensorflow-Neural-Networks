# Demonstrate tensorflow convolution 1
# from 13_convolutional_network.ipynb from Geron's github


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import tensorflow as tf

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
# matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "cnn"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
	# make sure the directories are made before saving!
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

def plot_color_image(image):
    plt.imshow(image.astype(np.uint8),interpolation="nearest")
    plt.axis("off")    

from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
image = china[150:220, 130:250]
height, width, channels = image.shape
image_grayscale = image.mean(axis=2).astype(np.float32)
images = image_grayscale.reshape(1, height, width, 1)

print(images.shape," ")

# create two convolutive maps, a vertical bar and a horizontal bar

fmap = np.zeros(shape=(7, 7, 1, 2), dtype=np.float32)
fmap[:, 3, 0, 0] = 1   # make a column vector 
fmap[3, :, 0, 1] = 1   # make a row vector
fmap[:, :, 0, 0]
plot_image(fmap[:, :, 0, 0])
plt.show()
input('press return 1')
plot_image(fmap[:, :, 0, 1])
plt.show()
input('press return 2')


reset_graph()

X = tf.placeholder(tf.float32, shape=(None, height, width, 1))
feature_maps = tf.constant(fmap)
convolution = tf.nn.conv2d(X, feature_maps, strides=[1,1,1,1], padding="SAME", use_cudnn_on_gpu=False)

with tf.Session() as sess:
    output = convolution.eval(feed_dict={X: images})

plot_image(images[0, :, :, 0])
save_fig("china_original", tight_layout=False)
plt.show()
input('press return 3')

plot_image(output[0, :, :, 0])
save_fig("china_vertical", tight_layout=False)
plt.show()
input('press return 4')

plot_image(output[0, :, :, 1])
save_fig("china_horizontal", tight_layout=False)
plt.show()

