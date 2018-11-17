import numpy as np
import gym
import math
import matplotlib.pyplot as plt
import pandas as pd
import collections as col
import pickle as p

env = gym.make('CartPole-v0')
env._max_episode_steps = 10000
gamma = 0.99
alpha = 0.1
epsilon = 0.25

# location - does not really matter
# speed
# stick angle
# stick angular speed
buckets_counts_for_observations = (2, 10, 50, 50)
ranges = ((env.observation_space.low[0], env.observation_space.high[0]), (-2, 2), (-0.4, 0.4), (-3.5, 3.5) )

q_table = np.zeros(buckets_counts_for_observations + (env.action_space.n,))

def select_action(current_state, epsilon):
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[current_state])
    return action

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

def update_q_table(state_old, action, reward, state_new, alpha):
    q_table[state_old][action] = (1 - alpha) * q_table[state_old][action] + alpha * (reward + gamma * np.max(q_table[state_new]))

def run_episode(env, render_env, range_value, epsilon):  
    observation = env.reset()
    current_state = assign_to_buckets(observation)
    totalreward = 0
    #history = []    
    for ix in range(range_value):
        action = select_action(current_state, epsilon)
        observation, reward, done, info = env.step(action)
        new_state = assign_to_buckets(observation)
        if done:
            reward = -10
        #history.append((current_state, action, reward, new_state))            
        update_q_table(current_state, action, reward, new_state, alpha)   
        if render_env:
            env.render()
        totalreward += reward
        current_state = new_state
        if done:
            break

    # history = history[0:len(history) // 2]

    # for item in history:
    #     current_state, action, reward, new_state = item
    #     update_q_table(current_state, action, reward, new_state, alpha)

    return totalreward  

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
        if ix % 200 == 0:
            rewards.append(reward)
            steps.append(ix)

        if ix % 1000 == 0:
            epsilon *= 0.99            
            print(ix)
            print(scores)
            print(reward)

        if np.sum(scores) == cycles_count * queue_length:
            break

    rewards.append(reward)
    steps.append(ix)
    print(scores)
    print(reward)
    print(ix)

cycles_count = 1000

find_best_policy(env, cycles_count, epsilon)

input("Press Enter to continue...")

run_episode(env, True, 5000, 0.0)

with open("d:/_py_/GymCartQLearnPolicy.pickle", "wb") as s:
    p.dump(q_table, s)

plt.plot(steps, rewards)
plt.show()

input("Press Enter to continue...")

env.close()