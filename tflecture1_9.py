# Another example: logistic regression
# See Hope, Reshkoff, Lieder, Learning TensorFlow, Ch. 3.
# 03__tensorflow_basics

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.reset_default_graph()


N = 20000            # number of training points

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# === Create data and simulate results =====
x_data = np.random.randn(N,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2
wxb = np.matmul(w_real,x_data.T) + b_real


y_data_pre_noise = sigmoid(wxb)
y_data = np.random.binomial(1,y_data_pre_noise)

plt.close('all')
plt.subplot(121)
plt.plot(wxb,y_data_pre_noise,'.')
plt.subplot(122)
plt.plot(wxb,y_data,'.')
plt.show()




NUM_STEPS = 50

g = tf.Graph()    # get a graph

wb_ = []
with g.as_default():         # use the graph just made as the default
    
    x = tf.placeholder(tf.float32,shape=[None,3])
    y_true = tf.placeholder(tf.float32,shape=None)
    
    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')  # create parameters
        b = tf.Variable(0,dtype=tf.float32,name='bias')
        y_pred = tf.matmul(w,tf.transpose(x)) + b

    with tf.name_scope('loss') as scope:
        # create loss function: cross entropy
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred) 
        loss = tf.reduce_mean(loss)
  
    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

   # Before starting, initialize the variables.  We will 'run' this first.
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)      
        for step in range(NUM_STEPS):
            sess.run(train,{x: x_data, y_true: y_data})
            if (step % 5 == 0):
                print(step, sess.run([w,b]))
                wb_.append(sess.run([w,b]))

        print(50, sess.run([w,b]))

        
