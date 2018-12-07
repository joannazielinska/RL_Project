from pong import Pong
import matplotlib.pyplot as plt
import numpy as np
from simple_ai import PongAi
from agent import Agent, Policy, Memory
import argparse

from skimage.transform import resize
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

'''
Parameters
'''
episodes = 10
epsilon = 0.5
decay = 0.00001
memory_size = 10000


def plot(observation):
    plt.imshow(observation/255)
    plt.show()

'''
Preprocessing the frame the reduce the load on system :
    - transform in black and white
    - resize it to half
'''
def preprocess(frame):
    frame[frame != 0] = 1
    frame = resize(frame, ((frame.shape[0]//2), (frame.shape[1]//2)))
    return frame

'''
Stacking 4 frames. The stacked frames will be used for learning.
Stacking them allows to base itself on the previous timesteps and so getting the motion.

Each time a new frame is stacked, the oldest one is removed.
In the beginning of an episode, the stacked frames are reset ; the frame transmitted is used 4 times.
'''
def stack_frame(stacked_frames, new_frame, reset=False):
    frame = preprocess(new_frame)
    
    if reset :
        stacked_frames.clear()
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
    else :
        stacked_frames.append(frame)
    state = np.stack(stacked_frames, axis=2)
    return state, stacked_frames

'''
Initialisation
'''

env = Pong(headless=args.headless)

#Players
player_id = 1
opponent_id = 3 - player_id
opponent = PongAi(env, opponent_id)

env.set_names(player.get_name(), opponent.get_name())

#Model
action_space_dim = 3
observation_space_dim = 4

#Classes
policy = Policy(observation_space_dim, action_space_dim)
player = Agent(env, policy, player_id)


#Stacked preprocessed frames
stacked_frames = deque(np.zeros(200,210), maxlen=4)

#Memory Initialisation
# take random actions to fill the memory
memory = Memory(memory_size)
for i in range(memory_size):
    if (i==0):
        obs = env.reset()
        state, stacked_frames = stack_frame(stacked_frames, obs[0], True)
    action1 = random.randint(0,3)
    action2 = random.randint(0,3)
    next_obs, reward, done, info = env.step((action1,action2))
    next_state, stacked_frames = stack_frame(stacked_frames, next_obs[0])
    memory.store((next_state, action1, reward[0], done))
    #state = next_state

'''
Training
'''

for i in range(0,episodes):
    done = False
    obs = env.reset()
    state, stacked_frames = stack_frame(stacked_frames, obs[0], True)
    
    while not done:
        action1 = player.get_action(state, epsilon)
        action2 = opponent.get_action()
        
        next_obs, rewards, done, info = env.step((action1, action2))
        next_state, stacked_frames = stack_frame(stacked_frames, next_obs[0])
        
        memory.store((next_state, action1, reward[0], done))

        obs = next_obs
        state = next_state
        
    #Updating policy
    
    
    
env.end()