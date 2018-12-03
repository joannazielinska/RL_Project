from pong import Pong
from skimage.transform import resize


import matplotlib.pyplot as plt
from random import randint
import pickle
import numpy as np
from simple_ai import PongAi
from agent_test import Agent, Policy


def plot(observation):
    plt.imshow(observation/255)
    plt.show()
    
def preprocess(observation):
    observation = resize(observation, ((observation.shape[0]//2), (observation.shape[1]//2)))
    #processed_im = np.zeros((observation.shape[0],observation.shape[1]))
    #for ix in range(observation.shape[0]) :
    #    for iy in range(observation.shape[1]) :
    #        if observation[ix,iy,0] != 0 or observation[ix,iy,1] != 0 or observation[ix,iy,2] != 0:
    #            processed_im[ix,iy] = 255
    #return processed_im
    return observation


env = Pong()
episodes = 1

player_id = 1
opponent_id = 3 - player_id
opponent = PongAi(env, opponent_id)

# Get dimensionalities of actions and observations
action_space_dim = (200//2)*(210//2)*3
observation_space_dim = 3

# Create the agent and the policy
policy = Policy(observation_space_dim, action_space_dim)
player = Agent(env, policy, player_id)

env.set_names(player.get_name(), opponent.get_name())

reward_history, timestep_history = [], []
average_reward_history = []

for i in range(0,episodes):
    reward_sum, timesteps = 0, 0
    done = False
    while not done:
        if(i==0):
            observation=env.reset()
            #print(observation[0])
        print("0")
        action1, action_distri = player.get_action(preprocess(observation[0]))
        print("1")
        action2 = opponent.get_action()
        print("2")
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        print("3")
        observation = (ob1, ob2)
        print("4")
        player.store_outcome(previous_observation, action_distri, action, reward)
        print("5")
        #plot(ob1) # plot the observation of this frame
        env.render()
        
        reward_sum += reward
        timesteps += 1
            
        if done:
            observation= env.reset()
            #plot(ob1) # plot the reset observation
            print("episode {} over".format(i))
            
    # Bookkeeping (mainly for generating plots)
    reward_history.append(reward_sum)
    timestep_history.append(timesteps)
    if episode_number > 100:
        avg = np.mean(reward_history[-100:])
    else:
        avg = np.mean(reward_history)
    average_reward_history.append(avg)
        
    player.episode_finished(episode_number)
        
    # Training is finished - plot rewards
    plt.plot(reward_history)
    plt.plot(average_reward_history)
    plt.legend(["Reward", "100-episode average"])
    #plt.title("Reward history (sig=%f, net 18)" % agent.policy.sigma.item())
    plt.title("Reward history (%s)" % args.env)
    plt.show()
    print("Training finished.")

# Needs to be called in the end to shut down pygame
env.end()



