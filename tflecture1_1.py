# from '09_up_and_running_with_tensorflow.jpynb'

import tensorflow as tf

x = tf.Variable(3, name='X')   # create a variable, set its value to 3
y = tf.Variable(4, name='Y')

f = x*x*y + y + 2              # f is also a TF variable (overloaded operators)

# variables are not initialized yet

# a better way (which makes sure session is closed when done)

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

print(result)       # Python v. 3.x print statement





