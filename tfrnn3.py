# from Geron, 14_recurrent_neural_networks

# Demonstrate dynamic_rnn
# 
# 

import numpy as np
import tensorflow as tf

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


reset_graph()



n_steps = 2
n_inputs = 3
n_neurons = 5

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

# X_seqs is a python list of n_steps tensors of shape [None, n_inputs]
#  where the first dimension here is the minibatch size.
#  It does this by swapping the first dimensions tf.transpose(X, perm=[1, 0, 2])
#
#  After transpose have a tensor of shape [n_steps, None, n_inputs]
#
# Then extract a python list of tensors along the first dimension,
#  (one tensor per time step) using the unstack function.

X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))

# Then the line is much as they were before, but we
# pass in X_seqs, rather than [X0, X1] (so the data determines the
#   number of steps)

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons)

output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell,X_seqs,
                                                    dtype=tf.float32)

# Merge all the output tensors into a single tensor using stack,
#   swapping the first two dimensions to get an outputs tensor of shape
#    [None, n_steps, n_neurons] (first dimension is the minibatch size)

outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])


    


X_batch = np.array([
        # t = 0      t = 1 
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})
 
print(outputs_val)
