import numpy as np
import gym
import math

env = gym.make('MountainCar-v0')
env._max_episode_steps = 10000
max_reward_def = 10
max_reward_treshold = 0.55

def select_action(parameters, observation):
    a1, a2, b = parameters
    parameters_a = [a1, a2]
    parameters_b = b
    res = np.matmul(parameters_a, observation) + parameters_b
    if res > 0.5:
        return 2
    elif res < -0.5:
        return 0
    else:
        return 1

def min(a,b):
    if a <= b:
        return a
    else:
        return b

def min_abs(a,b):
    aa = math.fabs(a)
    bb = math.fabs(b)
    return min(aa,bb)

def analyse_observation(observation, previous_results):
    max_0, min_1, max_1 = previous_results

    if observation[0] > max_0:
        max_0 = observation[0]

    if observation[1] < min_1:
        min_1 = observation[1]

    if observation[1] > max_1:
        max_1 = observation[1]

    return (max_0, min_1, max_1)

def calcualte_reward_for_observation(observation_analysed):
    max_0, min_1, max_1 = observation_analysed
    reward = max_0 + min_abs(min_1, max_1)
    return reward    

def run_episode(env, parameters, render_env, range_value):  
    observation = env.reset()
    observation_analysed = analyse_observation(observation, (0.0, 0.0, 0.0))
    for ix in range(range_value):
        action = select_action(parameters, observation)
        observation, reward, done, info = env.step(action)

        observation_analysed = analyse_observation(observation, observation_analysed)

        reward = calcualte_reward_for_observation(observation_analysed)

        if render_env:
            env.render()
            print(ix)
            print(reward)
            print(observation)
            print(done)

        if done:
            break

        # if done:
        #     # reward returned by Gym is always -1 even if cart passes the flag
        #     # therefore I could not  rely on this value and had to prepare my own reward
        #     if observation[0] >= max_reward_treshold: # position of cart is equal to flag's position
        #         return max_reward_def

    reward = calcualte_reward_for_observation(observation_analysed)

    return reward

def find_bestparams_random_increment(env, cycles_count, episodes_count):
    bestparams = None  
    bestreward = 0  
    params_count = 3
    ix = 0

    while bestreward < max_reward_def and ix < episodes_count:
        ix += 1
        if ix < 50:        
            parameters = np.random.rand(params_count) * 2 - 1
            parameters_temp = parameters.copy()
        else:
            parameters_temp = parameters.copy()
            # diff = (np.random.rand() * 2 - 1)
            # index = np.random.randint(params_count)
            # parameters_temp[index] = parameters_temp[index] + diff
            diff = np.random.rand(params_count) * 2 - 1
            parameters_temp = parameters_temp + diff

        reward = run_episode(env,parameters_temp, False, cycles_count)

        if reward > bestreward:
            select_index = False
            parameters = parameters_temp.copy()
            bestreward = reward
            bestparams = parameters.copy()
            print(ix)
            print(bestreward)
            print(parameters)

        if ix % 500 == 0:
            print(ix)

    return bestparams

cycles_count = 1000

episodes_count = 5000

bestparams = find_bestparams_random_increment(env, cycles_count, episodes_count)

print(bestparams)

input("Press enter...")

reward = run_episode(env, bestparams, True, 10000)

print(reward)

input("Press enter...")

env.close()

