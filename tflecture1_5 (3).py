# Linear regression with TF
# from '09_up_and_running_with_tensorflow.jpynb'
# 

# This is the same as the last (gradient descent), except
# that it uses TF's ability to compute gradients

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

from sklearn.datasets import fetch_california_housing

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


reset_graph()
housing = fetch_california_housing()
m,n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

#%%

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

#%%
# Use batch gradient descent --------------------------------------------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# scale the data
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

reset_graph()

n_epochs = 1000             # number of iterations of gradient descent algorithm
learning_rate = 0.01        # gradient descent step size

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y

# Computes the mean (reduces to a scalar value)
mse = tf.reduce_mean(tf.square(error), name="mse")

# Compute the gradient (2/M)*X'X
gradients = tf.gradients(mse, [theta])[0]
#  compute the gradient of the mse with respect to theta
#  The tf.gradients() function actually returns a list, so the [0] "unbrackets"
#      the list to just return the element of the list

# create a node that assigns a new value to a variable
# (This implements the gradient descent algorithm
training_op = tf.assign(theta, theta - learning_rate * gradients)
#  This is equivalent to writing
#  theta <-- theta - learning_rate * gradients


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)             # Do this once at the beginning of all the iterations

    for epoch in range(n_epochs):    # iterate over the training epochs
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()


    


