# reinforcement learning on Ms PacMan
# Derived from 16_reinforcement_learning.ipynb

# Setup

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import sys
import numpy.random as rnd


# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures and animations
# matplotlib nbagg
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
# plt.rcParams['axes.labelsize'] = 14
# plt.rcParams['xtick.labelsize'] = 12
# plt.rcParams['ytick.labelsize'] = 12

# A little helper function to plot an environment:
def plot_environment(env, figsize=(5,4)):
    # An environment can be visualized by calling its render() method, 
    #  and you can pick the rendering mode (the rendering options depend 
    #  on the environment). In this example we will set 
    #  mode="rgb_array" to get an image of the environment as a NumPy array:
    plt.close()  # or else nbagg sometimes plots in the previous cell
    plt.figure(figsize=figsize)
    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.axis("off")
    plt.show()

#%%
# OpenAI gym
import gym
env = gym.make('MsPacman-v0')
obs = env.reset()

# 
print("obs.shape=",obs.shape)
print("env.action_space=",env.action_space)

# Discrete(9) means that the possible actions are integers 0 through 8, 
#  which represents the 9 possible positions of the joystick 
#  (0=center, 1=up, 2=right, 3=left, 4=down, 5=upper-right, 
#   6=upper-left, 7=lower-right, 8=lower-left).

print("Basic game board")
plot_environment(env);
input('Press return 1')

#%%

# Next we need to tell the environment which action to play, 
#    and it will compute the next step of the game. 
#    Let's go left for 110 steps, then lower left for 40 steps:
env.reset()
for step in range(110):
    env.step(3) #left
for step in range(40):
    env.step(8) #lower-left

# where are we now?

print("Go left 110, then lower left for 40")
plot_environment(env)
input("Press return 2")

#  The step() function actually returns several important objects:
#    obs, reward, done, info = env.step(0)

#  The observation tells the agent what the environment looks like, 
#      as discussed earlier. This is a 210x160 RGB image:

# The environment also tells the agent how much reward it got 
#     during the last step:

#  When the game is over, the environment returns done=True:

#  Finally, info is an environment-specific dictionary that can provide 
#    some extra information about the internal state of the environment. 
#    This is useful for debugging, but your agent should not use this 
#    information for learning (it would be cheating).
#%%

# Let's play one full game (with 3 lives), by moving in random directions 
#    for 10 steps at a time, recording each frame:

frames = []

n_max_steps = 1000
n_change_steps = 10

obs = env.reset()
for step in range(n_max_steps):
    img = env.render(mode="rgb_array")
    frames.append(img)
    if step % n_change_steps == 0:
        action = env.action_space.sample() # play randomly
    obs, reward, done, info = env.step(action)
    if done:
        break

# Now show the animation 

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat, interval=interval)

print("Random motion for 1000 steps")
video = plot_animation(frames)
plt.show()
input("Press return 3")


#  Once you have finished playing with an environment, 
#     you should close it to free up resources:

env.close()

#%%	

# learning to play MS-Pacmac using deep Q-learning
		
		
# Preprocessing the images is optional but greatly speeds up training.
		
mspacman_color = np.array([210, 164, 74]).mean()

def preprocess_observation(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.mean(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img - 128) / 128 - 1 # normalize from -1. to 1.
    return img.reshape(88, 80, 1)

img = preprocess_observation(obs)

plt.figure(figsize=(11, 7))
plt.subplot(121)
plt.title("Original observation (160×210 RGB)")
plt.imshow(obs)
plt.axis("off")
plt.subplot(122)
plt.title("Preprocessed observation (88×80 greyscale)")
plt.imshow(img.reshape(88, 80), interpolation="nearest", cmap="gray")
plt.axis("off")
# save_fig("preprocessing_plot")
plt.show()
print("Preprocessing of data")
input("Press return 4")

#%%

# Build DQN

import tensorflow as tf

reset_graph()

input_height = 88    #describe the size of the input (using cropped images)
input_width = 80
input_channels = 1
# there are 3 convolutional layers
conv_n_maps = [32, 64, 64]   # number of maps in each layer
conv_kernel_sizes = [(8,8), (4,4), (3,3)]   # kernel size in each layer
conv_strides = [4, 2, 1]   # strides in each layer
conv_paddings = ["SAME"]*3   # padding in eacy layer
conv_activation = [tf.nn.relu]*3   # activation function in each layer
n_hidden_inputs = 64 * 11 * 10  # conv3 has 64 maps of 11x10 each
n_hidden = 512    # number of hidden units in penultimate layer
hidden_activation = tf.nn.relu  
n_outputs = env.action_space.n
initializer = tf.contrib.layers.variance_scaling_initializer()

learning_rate = 0.01

# there are two DQNs with the same architecture (but different parameters)
#  The actor, which drives MsPacMan
# and the critic, which watches the actor and learns from its mistakes
# At regular intervals, copy the critic to the actor

# this is a function which builds a set of convolutional layers
def q_network(X_state, scope):
    prev_layer = X_state
    conv_layers = []
    with tf.variable_scope(scope) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(conv_n_maps, conv_kernel_sizes, conv_strides, conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(prev_layer, filters=n_maps, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=initializer)
            conv_layers.append(prev_layer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_inputs])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden, activation=hidden_activation, kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, n_outputs)
    trainable_vars = {var.name[len(scope.name):]: var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}
    return outputs, trainable_vars

# input
X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])


actor_q_values, actor_vars = q_network(X_state, scope="q_networks/actor")    # the actor network
critic_q_values, critic_vars = q_network(X_state, scope="q_networks/critic") # the critic network

# copy the critic DQN to the actor DQN:
copy_ops = [actor_var.assign(critic_vars[var_name])
            for var_name, actor_var in actor_vars.items()]
copy_critic_to_actor = tf.group(*copy_ops)

# By now we have two DQNs, both capable of taking an environment (a preprocessed observation)
#  as input and outputting an estimated Q value for each possible action in that state.
# We also have an operation copy_criti_to_actor for copying all the trainable
#  values of the critic DQN to the actor DQN.

# the Actor DQN plays the game.  

# The critic DQN tries to make its Q-value predictions match the Q-value estimates
# of the actor through its experience.  The actor plays for a while, storing
# experiences in a replay memory.  This is a 5-tuple with values
#  (state, action, next state, reward, continue).
# At regular intervals we sample a batch of memories from the replay memory
# and estimate Q-values from these memories.  Then the critic DQN is trained
# to predict these Q-values using regular supervised learning.
# Every few iteration we copy the critic DQN to the actor DQN.

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(critic_q_values * tf.one_hot(X_action, n_outputs),
                            axis=1, keep_dims=True)
    cost = tf.reduce_mean(tf.square(y - q_value))
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(cost, global_step=global_step)
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

actor_vars

#%%
from collections import deque

replay_memory_size = 10000
replay_memory = deque([], maxlen=replay_memory_size)

def sample_memories(batch_size):
    indices = rnd.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)
				
eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 50000
import sys

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if rnd.rand() < epsilon:
        return rnd.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action
								
#%%
				
n_steps = 100000  # total number of training steps
training_start = 1000  # start training after 1,000 game iterations
training_interval = 3  # run a training step every 3 game iterations
save_steps = 50  # save the model every 50 training steps
copy_steps = 25  # copy the critic to the actor every 25 training steps
discount_rate = 0.95
skip_start = 90  # Skip the start of every game (it's just waiting time).
batch_size = 50
iteration = 0  # game iterations
checkpoint_path = "./my_dqn.ckpt"
done = True # env needs to be reset

with tf.Session() as sess:
    if os.path.isfile(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
    while True:
        step = global_step.eval()
        if step >= n_steps:
            break
        iteration += 1
        print("\rIteration {}\tTraining step {}/{} ({:.1f}%)".format(iteration, step, n_steps, step * 100 / n_steps), end="")
        if done: # game over, start again
            obs = env.reset()
            for skip in range(skip_start): # skip boring game iterations at the start of each game
                obs, reward, done, info = env.step(0)
            state = preprocess_observation(obs)

        # Actor evaluates what to do
        q_values = actor_q_values.eval(feed_dict={X_state: [state]})
        action = epsilon_greedy(q_values, step)

        # Actor plays
        obs, reward, done, info = env.step(action)
        next_state = preprocess_observation(obs)

        # Let's memorize what happened
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        if iteration < training_start or iteration % training_interval != 0:
            continue
        
        # Critic learns
        X_state_val, X_action_val, rewards, X_next_state_val, continues = sample_memories(batch_size)
        next_q_values = actor_q_values.eval(feed_dict={X_state: X_next_state_val})
        y_val = rewards + continues * discount_rate * np.max(next_q_values, axis=1, keepdims=True)
        training_op.run(feed_dict={X_state: X_state_val, X_action: X_action_val, y: y_val})

        # Regularly copy critic to actor
        if step % copy_steps == 0:
            copy_critic_to_actor.run()

        # And save regularly
        if step % save_steps == 0:
            saver.save(sess, checkpoint_path)
												
												
