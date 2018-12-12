from pong import Pong
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from simple_ai import PongAi
from agent import Agent, Policy
import argparse

import math
from skimage.transform import resize
from skimage.color import rgb2gray
import random
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
args = parser.parse_args()

'''
Parameters
'''
episodes = 1000000
epsilon_start = 0.7
epsilon_end = 0.0001
epsilon_decay = 0.000001

parameters = (episodes, epsilon_start, epsilon_end, epsilon_decay)

def plot(observation):
    plt.imshow(observation/255)
    plt.show()

'''
Preprocessing the frame the reduce the load on system :
    - transform in grayscale
    - resize it to half
'''
def preprocess(frame):
    gray_frame = rgb2gray(frame)
    processed_frame = resize(gray_frame, ((frame.shape[0]//2), (frame.shape[1]//2)))
    return processed_frame

def train(player, opponent, parameters):
    '''Initialization'''
    timesteps_history = []
    reward_history = []
    average_reward_history = []
        
        #parameters
    episodes_nb = parameters[0]
    explore_start = parameters[1]
    epsilon = explore_start
    explore_end = parameters[2]
    #decay = parameters[3]
    explore_rate = 200000
    
    for i in range(0,episodes_nb):
        done = False
        timesteps = 0
        observation = env.reset()
        
        if epsilon > explore_end:
            #epsilon = epsilon-(decay*i)
            epsilon -= (explore_start-explore_end)/explore_rate
            
        while not done:
            
            action1, aprob = player.get_action(preprocess(observation[0]), epsilon)
            action2 = opponent.get_action()
            
            next_obs, rewards, done, info = env.step((action1, action2))
            player.store_outcome(observation, aprob, action1, rewards[0])
            
            observation = next_obs
            timesteps += 1
            
            env.render()
            
        print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(i, rewards[0], timesteps))
        player.episode_finished(i)
        
        #Saving history for plot purposes
        timesteps_history.append(timesteps)
        reward_history.append(rewards[0])
        if i > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)
        
        #Regular plot and model saving
        if(i % 500 == 0 and i!= 0):
            fig1 = plt.figure(1)
            plt.plot(reward_history)
            plt.plot(average_reward_history)
            plt.legend(["Reward", "100-episode average"])
            plt.title("Reward history")
            #plt.show()
            fig1.savefig("save/reward%s.png" %i)
            
            fig2 = plt.figure(2)
            plt.plot(timesteps_history)
            plt.legend("Timesteps")
            plt.title("Timesteps history")
            #plt.show()
            fig2.savefig("save/timesteps%s.png" %i)
            
            player.save_model()

def test(episodes, player, opponent):
    for i in range(0,episodes):
        done = False
        observation = env.reset()
        timesteps = 0
        reward_sum = 0
        
        while not done:
            action1 = player.get_action(preprocess(observation[0]))
            action2 = opponent.get_action()

            next_obs, rewards, done, info = env.step((action1, action2))

            reward_sum += rewards[0]

            obs = next_obs
            state = next_state

            env.render()
            
'''
Initialisation
'''
env = Pong(headless=args.headless)

#Players
player_id = 1
opponent_id = 3 - player_id
opponent = PongAi(env, opponent_id)

#Model
action_space_dim = 3
observation_space_dim = [105,100]

#Classes
policy = Policy(observation_space_dim, action_space_dim)
player = Agent(env, policy, player_id)

env.set_names(player.get_name(), opponent.get_name())
    
# If no model was passed, train a policy from scratch.
# Otherwise load the policy from the file and go directly to testing.
if args.test is None:
    try:
        train(player, opponent, parameters)
    # Handle Ctrl+C - save model and go to tests
    except KeyboardInterrupt:
        print("Interrupted!")
    player.save_model()
else:
    player.load_model(args.test)
    print("Testing...")
    test(100, player, opponent)
    
    
    
env.end()