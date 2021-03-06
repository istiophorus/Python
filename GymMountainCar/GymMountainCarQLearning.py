import numpy as np
import gym
import math
import matplotlib.pyplot as plt
import pandas as pd
import collections as col
import pickle as p

env = gym.make('MountainCar-v0')
env._max_episode_steps = 10000

max_reward_def = 3
max_reward_treshold = 0.55
gamma = 0.9
alpha = 0.1
epsilon = 0.25

# location
# speed
buckets_counts_for_observations = (50, 50)
ranges = ((-2, 1), (-1, 1))

q_table = np.zeros(buckets_counts_for_observations + (env.action_space.n,))

def min(a,b):
    if a <= b:
        return a
    else:
        return b

def min_abs(a,b):
    aa = math.fabs(a)
    bb = math.fabs(b)
    return min(aa,bb)

# is random vallue is less than epsilon then choose random action
# othwise choose an action using Q table
def select_action(current_state, epsilon):
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[current_state])
    return action

# assigns continues value of observed parameter to discrete bucket index
def assign_to_buckets(observation):
    res = []
    for ix, x in enumerate(observation):
        buckets_count = buckets_counts_for_observations[ix]
        item_ranges = ranges[ix]
        if buckets_count == 1:
            bucket = 0
        else:
            bucket = int((x - item_ranges[0]) / (item_ranges[1] - item_ranges[0]) * buckets_count)
            if bucket < 0:
                bucket = 0
            if bucket >= buckets_count:
                bucket = buckets_count - 1

        res.append(bucket)
    return tuple(res)

# updates Q table basing on previous and current state, performed action and received reward
def update_q_table(state_old, action, reward, state_new, alpha):
    q_table[state_old][action] = (1 - alpha) * q_table[state_old][action] + alpha * (reward + gamma * np.max(q_table[state_new]))

def analyse_observation(observation, previous_results):
    max_0, min_1, max_1 = previous_results

    if observation[0] > max_0:
        max_0 = observation[0]

    if observation[1] < min_1:
        min_1 = observation[1]

    if observation[1] > max_1:
        max_1 = observation[1]

    return (max_0, min_1, max_1)

def calcualte_reward_for_observation(observation_analysed,  observation):
    if observation[0] >= max_reward_treshold: # position of cart is equal to flag's position
        return max_reward_def

    max_0, min_1, max_1 = observation_analysed
    reward = max_0 + min_abs(min_1, max_1)

    return reward

def run_episode(env, render_env, range_value, epsilon):  
    observation = env.reset()
    current_state = assign_to_buckets(observation)
    reward = 0
    observation_analysed = analyse_observation(observation, (0.0, 0.0, 0.0))
    max_reward = 0
    for ix in range(range_value):
        action = select_action(current_state, epsilon)
        observation, reward, done, info = env.step(action)
        new_state = assign_to_buckets(observation)
        
        observation_analysed = analyse_observation(observation, observation_analysed)

        reward = calcualte_reward_for_observation(observation_analysed, observation)

        if reward >= max_reward_def:
            return reward

        update_q_table(current_state, action, reward, new_state, alpha)

        if render_env:
            env.render()

        if reward > max_reward:
            max_reward = reward

        current_state = new_state

    return max_reward  

rewards = []
steps = []

def find_best_policy(env, cycles_count, epsilon):
    ix = 0
    reward = 0
    queue_length = 10
    scores = col.deque(maxlen = queue_length)

    while True:  
        ix += 1
        reward = run_episode(env, False, cycles_count, epsilon)
        scores.append(reward)

        rewards.append(reward)
        steps.append(ix)

        if ix % 10 == 0:
            epsilon *= 0.99            
            print(ix)
            print(scores)
            print(reward)

        if np.sum(scores) == max_reward_def * queue_length:
            break

    rewards.append(reward)
    steps.append(ix)
    print(scores)
    print(reward)
    print(ix)

cycles_count = 2000

find_best_policy(env, cycles_count, epsilon)

input("Press Enter to continue...")

run_episode(env, True, 5000, 0.0)

with open("d:/_py_/GymMountainCarQLearnPolicy.pickle", "wb") as s:
    p.dump(q_table, s)

plt.plot(steps, rewards)
plt.show()

input("Press Enter to continue...")

env.close()