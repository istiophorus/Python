import numpy as np
import gym
env = gym.make('CartPole-v0')
env._max_episode_steps = 4000

def run_episode(env, parameters, render_env, range_value):  
    observation = env.reset()
    totalreward = 0
    for ix in range(range_value):
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        if render_env:
            env.render()
        totalreward += reward
        if done:
            break

        if ix % 500 == 0:
            print(ix)

    return totalreward  

def find_bestparams_random(env, cycles_count):
    bestparams = None  
    bestreward = 0  
    for _ in range(cycles_count):  
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env,parameters, False, 500)
        if reward > bestreward:
            bestreward = reward
            bestparams = parameters
            # considered solved if the agent lasts 200 timesteps
            if reward == 500:
                break    

    return bestparams

def find_bestparams_random_increment(env, cycles_count):
    bestparams = None  
    bestreward = 0  
    parameters = np.random.rand(4) * 2 - 1
    ix = 0
    while bestreward < cycles_count:  
        ix += 1
        index = np.random.randint(4)
        diff = np.random.rand() * 2 - 1.0
        parameters_temp = parameters.copy()
        parameters_temp[index] = parameters_temp[index] + diff
        reward = run_episode(env,parameters_temp, False, cycles_count)
        print(reward)
        if reward > bestreward:
            parameters = parameters_temp.copy()
            bestreward = reward
            bestparams = parameters.copy()
            print(ix)
            print(bestreward)
            print(parameters)

        if ix % 500 == 0:
            print(ix)

    return bestparams    

cycles_count = 2000

#bestparams = find_bestparams_random(env, 20000)

bestparams = find_bestparams_random_increment(env, cycles_count)

print(bestparams)

reward = run_episode(env, bestparams, True, cycles_count)

print(reward)

env.close()