import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.9 # discount factor
INITIAL_EPSILON = 0.8 # starting value of epsilon
FINAL_EPSILON =  0.1 # final value of epsilon
EPSILON_DECAY_STEPS = 60 # decay period
HIDDEN_UNITS = 64
NUM_LAYERS = 2
learning_rate = 0.00025
BATCH_SIZE = 128
MEMORY_SIZE = 1000
replay_buffer = []

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph
next_state_in = tf.placeholder("float", [None, STATE_DIM])
actions_next = tf.placeholder("float", [None, ACTION_DIM])
done_flags = tf.placeholder("float", [None])
rewards = tf.placeholder("float", [None])

w, b = tf.random_normal_initializer(0., 0.05), tf.random_normal_initializer(0., 0.05)
curr_state = state_in
next_state = next_state_in

for i in range(NUM_LAYERS):
    curr_state = tf.layers.dense(inputs=curr_state, units=HIDDEN_UNITS, activation=tf.nn.relu, kernel_initializer= w, bias_initializer=b)
for i in range(NUM_LAYERS):
    next_state = tf.layers.dense(inputs=next_state, units=HIDDEN_UNITS, activation=tf.nn.relu, kernel_initializer= w, bias_initializer=b)

# TODO: Network outputs
q_values = tf.layers.dense(inputs=curr_state, units=ACTION_DIM, activation=None)
q_target = tf.layers.dense(inputs=next_state, units=ACTION_DIM, activation=None)

#action = tf.transpose(action_in)
action_selected = tf.reduce_sum(tf.multiply(q_values, action_in), axis=1)
max_q_next_by_target = tf.reduce_max(q_target, axis=1)

y = rewards + (1. - done_flags) * GAMMA * max_q_next_by_target

# TODO: Loss/Optimizer Definition
loss = tf.reduce_mean(tf.square(action_selected - tf.stop_gradient(y)))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

def update_memory(next_state, reward, done, _, state, action):
    replay_buffer.append([next_state, reward, done, _, state, action])
    if len(replay_buffer) > MEMORY_SIZE:
        replay_buffer.pop(0)

# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS
    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated

        update_memory(next_state, reward, done, _, state, action)

        if len(replay_buffer) >= BATCH_SIZE:

            memory = random.sample(replay_buffer, BATCH_SIZE)

            # index = random.randint(0, len(replay_buffer) - BATCH_SIZE)
            # memory = replay_buffer[index : index+BATCH_SIZE]
            # memory with rnn
            ns_batch = [data[0] for data in memory]
            r_batch = [data[1] for data in memory]
            a_batch = [data[5] for data in memory]
            s_batch = [data[4] for data in memory]
            d_batch = [1. if data[2] else 0. for data in memory]
        
            # Do one training step
            session.run([optimizer], feed_dict={
                action_in: a_batch,
                state_in: s_batch,
                next_state_in : ns_batch,
                done_flags : d_batch,
                rewards : r_batch 
            })

        # Update
        state = next_state
        if done:
            ep_reward = 0
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
