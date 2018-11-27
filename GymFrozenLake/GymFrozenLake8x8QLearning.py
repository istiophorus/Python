import numpy as np
import gym
import math
import collections as col
import pickle as p

env = gym.make('FrozenLake8x8-v0')
env._max_episode_steps = 10000

max_reward_def = 1000
gamma = 0.8
alpha = 0.1
epsilon = 0.5

# location
buckets_counts_for_observations = (64,)

q_table = np.zeros(buckets_counts_for_observations + (env.action_space.n,))

# is random vallue is less than epsilon then choose random action
# othwise choose an action using Q table
def select_action(current_state, epsilon):
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[current_state])
    return action

# updates Q table basing on previous and current state, performed action and received reward
def update_q_table(state_old, action, reward, state_new, alpha):
    q_table[state_old][action] = (1 - alpha) * q_table[state_old][action] + alpha * (reward + gamma * np.max(q_table[state_new]))

def calcualte_reward_for_observation_a(observation, done: bool, reward):
    if done:
        if reward > 0:
            return max_reward_def
        else:
            return -max_reward_def
    else:
        return 0

def calcualte_reward_for_observation_b(observation, done: bool, reward):
    x = observation // 8
    y = observation % 8

    if done:
        if reward > 0:
            return max_reward_def
        else:
            return -max_reward_def
    else:
        return x + y

def run_episode(env, render_env, range_value, epsilon):  
    observation = env.reset()
    reward = 0
    max_reward = 0
    for ix in range(range_value):
        current_state = observation
        action = select_action(current_state, epsilon)
        observation, reward, done, info = env.step(action)

        reward = calcualte_reward_for_observation_b(observation, done, reward)

        update_q_table(current_state, action, reward, observation, alpha)

        if reward > max_reward:
            max_reward = reward

        if render_env:
            env.render()

        if done:
            break

    return (reward, max_reward)

def find_best_policy(env, cycles_count, epsilon):
    ix = 0
    reward = 0
    queue_length = 30
    scores = col.deque(maxlen = queue_length)
    rewards = []
    steps = []

    while True:  
        ix += 1
        reward, max_episode_reward = run_episode(env, False, cycles_count, epsilon)
        scores.append(reward)

        rewards.append(max_episode_reward)
        steps.append(ix)

        if ix % 10 == 0:
            epsilon *= 0.99            
            print(ix)
            print(reward)

        if np.sum(scores) == max_reward_def * queue_length:
            break

    rewards.append(max_episode_reward)
    steps.append(ix)
    print(reward)
    print(ix)

    return (steps, rewards)

cycles_count = 2000

steps, rewards = find_best_policy(env, cycles_count, epsilon)

run_episode(env, True, 5000, 0.0)

with open("d:/_py_/GymFrozenLake8x8QLearning.pickle", "wb") as s:
    p.dump(q_table, s)

input("Press Enter to continue...")

env.close()