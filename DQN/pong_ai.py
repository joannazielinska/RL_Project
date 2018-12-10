from pong import Pong
import matplotlib.pyplot as plt
import numpy as np
from simple_ai import PongAi
from agent import Agent, Policy, Memory
import argparse

from skimage.transform import resize
from collections import deque
import random
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
args = parser.parse_args()

'''
Parameters
'''
episodes = 10000
epsilon = 0.6
gamma = 0.9
decay = 0.00001
memory_size = 10000
batch_size = 64
update_step = 1000


def plot(observation):
    plt.imshow(observation/255)
    plt.show()

'''
Preprocessing the frame the reduce the load on system :
    - transform in black and white
    - resize it to half
'''
def preprocess(frame):
    processed_frame = np.zeros((frame.shape[0],frame.shape[1]))
    for ix in range(frame.shape[0]):
        for iy in range(frame.shape[1]):
            if (frame[ix,iy].any != 0):
                processed_frame[ix,iy] = 255
    processed_frame = resize(processed_frame, ((frame.shape[0]//2), (frame.shape[1]//2)))
    return processed_frame

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
    
    state = np.zeros((1,4,frame.shape[0],frame.shape[1]))
    state[:,0] = stacked_frames[0]
    state[:,1] = stacked_frames[1]
    state[:,2] = stacked_frames[2]
    state[:,2] = stacked_frames[2]
    return state, stacked_frames

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
observation_space_dim = 4

#Classes
policy = Policy(observation_space_dim, action_space_dim)
player = Agent(env, policy, player_id)

env.set_names(player.get_name(), opponent.get_name())

def train(episodes, player, opponent):
    
    target_dqn = Policy(observation_space_dim, action_space_dim)
    target_dqn.load_state_dict(policy.state_dict())
    #Stacked preprocessed frames
    stacked_frames = deque(np.zeros((200,210)), maxlen=4)

    #Updates
    update_counter = 0

    #Memory Initialisation
    # take random actions to fill the memory
    memory = Memory(memory_size, batch_size)
    for i in range(memory_size):
        if (i==0):
            obs = env.reset()
            state, stacked_frames = stack_frame(stacked_frames, obs[0], True)
        action1 = random.randint(0,3)
        action2 = random.randint(0,3)
        next_obs, rewards, done, info = env.step((action1,action2))
        next_state, stacked_frames = stack_frame(stacked_frames, next_obs[0])
        memory.store((state, action1, rewards[0], next_state, done))
        state = next_state

    player.reset_score()
    opponent.reset_score()

    '''
    Training
    '''

    for i in range(0,episodes):
        done = False
        obs = env.reset()
        state, stacked_frames = stack_frame(stacked_frames, obs[0], True)
        timesteps = 0
        reward_sum = 0

        while not done:
            action1 = player.get_action(state, epsilon)
            action2 = opponent.get_action()

            next_obs, rewards, done, info = env.step((action1, action2))
            next_state, stacked_frames = stack_frame(stacked_frames, next_obs[0])

            memory.store((state, action1, rewards[0], next_state, done))
            reward_sum += rewards[0]

            obs = next_obs
            state = next_state

            env.render()

            #Updating policy
                #Loading from memory
            samples = memory.sample()
            batch_states = np.asarray([x[0] for x in samples])
            batch_actions = np.asarray([x[1] for x in samples])
            batch_rewards = np.asarray([x[2] for x in samples])
            batch_next_states = np.asarray([x[3] for x in samples])
            batch_done = np.asarray([x[4] for x in samples])

                #Target network
            batch = torch.from_numpy(batch_next_states.squeeze()).float().to(player.train_device)
            batch_t_q_values = target_dqn.forward(batch)

                #Q Learning
            batch_t_q_max,_ = batch_t_q_values.max(dim=1)
            y = torch.empty(batch_size, 1)
            batch_rewards = torch.from_numpy(batch_rewards).float().to(player.train_device)

            for j in range(batch_size):
                #.any() ?
                if batch_done[j].any():
                    y[j] = batch_rewards[j]
                else:
                    y[j] = batch_rewards[j] + batch_t_q_max[j].mul(gamma)
            y.detach()

                #Gradient_descent
            batch_q_values = policy.forward(torch.from_numpy(batch_states.squeeze()).float().to(player.train_device))
            loss = torch.mean(y.sub(batch_q_values)**2)
            loss.backward()

            player.update_policy()

            update_counter += 1
            if (update_counter % update_step == 0):
                target_dqn.load_state_dict(policy.state_dict())
            timesteps += 1

        epsilon = epsilon*decay
        print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(i, reward_sum, timesteps))
    
def test(episodes, player, opponent):
    for i in range(0,episodes):
        done = False
        obs = env.reset()
        state, stacked_frames = stack_frame(stacked_frames, obs[0], True)
        timesteps = 0
        reward_sum = 0
        
        while not done:
            action1 = player.get_action(state, epsilon)
            action2 = opponent.get_action()

            next_obs, rewards, done, info = env.step((action1, action2))
            next_state, stacked_frames = stack_frame(stacked_frames, next_obs[0])

            reward_sum += rewards[0]

            obs = next_obs
            state = next_state

    env.render()
    
# If no model was passed, train a policy from scratch.
# Otherwise load the policy from the file and go directly to testing.
if args.test is None:
    try:
        train(episodes, player, opponent)
    # Handle Ctrl+C - save model and go to tests
    except KeyboardInterrupt:
        print("Interrupted!")
    model_file = "%dqn.mdl" % args.env
    torch.save(policy.state_dict(), model_file)
    print("Model saved to", model_file)
else:
    state_dict = torch.load(args.test)
    policy.load_state_dict(state_dict)
    print("Testing...")
    test(100, player, opponent)
    
    
    
env.end()