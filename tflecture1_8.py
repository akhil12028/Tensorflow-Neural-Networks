# Linear regression with TF
# from '09_up_and_running_with_tensorflow.jpynb'
# 

# Tensorboard
# 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

from sklearn.datasets import fetch_california_housing

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


housing = fetch_california_housing()
m,n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

reset_graph()

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)



#%%

X = tf.placeholder(tf.float32, shape = (None, n+1), name='X')
y = tf.placeholder(tf.float32, shape = (None, 1), name='y')

#%%
# --------------------------------------------


def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)   # set a seed 
    indices = np.random.randint(m, size=batch_size)   # draw batch_size variables in
                                                      # the range [0,m-1]
    X_batch = scaled_housing_data_plus_bias[indices]  # pull out the data for X at these indices
    y_batch = housing.target.reshape(-1, 1)[indices]  # pull out the data for y at these indices
    return X_batch, y_batch

learning_rate = 0.01        # gradient descent step size
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
n_epochs = 10

# scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y

# Computes the mean (reduces to a scalar value)
mse = tf.reduce_mean(tf.square(error), name="mse")

# Using TF's optimizer (here a gradient descent optimizer)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()   # create a saver node in the graph

# Set up the data to be saved
mse_summary = tf.summary.scalar('MSE',mse)
file_write = tf.summary.FileWriter(logdir,tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)             # Do this once at the beginning of all the iterations

    for epoch in range(n_epochs):    # iterate over the training epochs
        for batch_index in range(n_batches):  # for each batch
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X:X_batch,y:y_batch})
                step = epoch*n_batches + batch_index
                file_write.add_summary(summary_str,step)
            sess.run(training_op, feed_dict={X: X_batch, y:y_batch})

    
    best_theta = theta.eval()

file_write.close()



    


