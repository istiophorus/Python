import numpy as np
import gym
import math
import matplotlib.pyplot as plt
import pandas as pd
import collections as col
import pickle as p

env = gym.make('Pendulum-v0')
env._max_episode_steps = 10000

max_reward_def = 3
max_reward_treshold = 0.55
gamma = 0.99
alpha = 0.1
epsilon = 0.25

reward_offset = 20
observations_buckets_counts = (50, 50, 50)
os = env.observation_space
ranges = ((os.low[0], os.high[0]), (os.low[1], os.high[1]), (os.low[2], os.high[2]))
acsp = env.action_space
action_ranges = ((acsp.low[0], acsp.high[0]),)
action_buckets_counts = (50,)

q_table = np.zeros(observations_buckets_counts + action_buckets_counts)

def min(a,b):
    if a <= b:
        return a
    else:
        return b

def min_abs(a,b):
    aa = math.fabs(a)
    bb = math.fabs(b)
    return min(aa,bb)

def get_value_for_bucket(bucket_index, buckets_count, item_ranges):
    return bucket_index * (item_ranges[1] - item_ranges[0]) / buckets_count + item_ranges[0]

# is random vallue is less than epsilon then choose random action
# othwise choose an action using Q table
def select_action(current_state, epsilon):
    if np.random.random() < epsilon:
        action = env.action_space.sample()
        action_bucket = assign_to_buckets((action,), action_buckets_counts, action_ranges)
        action_index = action_bucket[0]
    else:
        action_index = np.argmax(q_table[current_state])
    return action_index

# assigns continues value of observed parameter to discrete bucket index
def assign_to_buckets(observation, buckets_counts, items_ranges):
    res = []
    for ix, x in enumerate(observation):
        buckets_count = buckets_counts[ix]
        item_ranges = items_ranges[ix]
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

def run_episode(env, render_env, range_value, epsilon):  
    observation = env.reset()
    current_state = assign_to_buckets(observation, observations_buckets_counts, ranges)
    total_reward = 0
    for ix in range(range_value):
        action_index = select_action(current_state, epsilon)
        action_value = get_value_for_bucket(action_index, action_buckets_counts[0], action_ranges[0])

        observation, reward, done, info = env.step([action_value])
        new_state = assign_to_buckets(observation, observations_buckets_counts, ranges)

        #reward += reward_offset

        update_q_table(current_state, action_index, reward, new_state, alpha)

        if render_env:
            env.render()

        current_state = new_state

        total_reward += reward

    return total_reward  

rewards = []
steps = []

def find_best_policy(env, cycles_count, epsilon, episodes_count):
    ix = 0
    reward = 0
    queue_length = 10
    scores = col.deque(maxlen = queue_length)

    for _ in range(episodes_count):
        ix += 1
        total_reward = run_episode(env, False, cycles_count, epsilon)
        scores.append(total_reward)

        rewards.append(total_reward)
        steps.append(ix)

        if ix % 100 == 0:
            epsilon *= 0.99            
            print(ix)
            print(epsilon)
            print(scores)
            print(reward)

        # if np.sum(scores) > max_reward_def * queue_length:
        #     break

    rewards.append(reward)
    steps.append(ix)
    print(scores)
    print(reward)
    print(ix)

cycles_count = 1000
episodes_count = 20000

find_best_policy(env, cycles_count, epsilon, episodes_count)

input("Press Enter to continue...")

run_episode(env, True, 5000, 0.0)

with open("d:/_py_/GymPendulumQLearnPolicy.pickle", "wb") as s:
    p.dump(q_table, s)

plt.plot(steps, rewards)
plt.show()

input("Press Enter to continue...")

env.close()