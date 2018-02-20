# from '09_up_and_running_with_tensorflow.jpynb'
# A first introduction to TensorFlow

import tensorflow as tf

x = tf.Variable(3, name='X')   # create a variable, set its value to 3
y = tf.Variable(4, name='Y')

f = x*x*y + y + 2              # f is also a TF variable (overloaded operators)

# variables are not initialized yet

sess = tf.Session()             # create a TF session
# One way is to initialize all the tf varibles individually
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
sess.close()                    # Need to close the session after it is over

print(result)       # Python v. 3.x print statement






