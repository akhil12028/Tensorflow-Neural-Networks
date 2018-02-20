# from '09_up_and_running_with_tensorflow.jpynb'
# Demonstrate global_variables_initalizer

import tensorflow as tf

tf.reset_default_graph()       # always a good idea

x = tf.Variable(3, name='X')   # create a variable, set its value to 3
y = tf.Variable(4, name='Y')

f = x*x*y + y + 2              # f is also a TF variable (overloaded operators)

init = tf.global_variables_initializer()  # this creates a special node to initiliaze stuff

with tf.Session() as sess:
    init.run()        # initially all the variables in the node
    result = f.eval()

print(result)       # Python v. 3.x print statement





