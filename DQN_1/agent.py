from pong import Pong
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import numpy as np
import math

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 100000

model_file = "dqn_model"


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space

        #Layers
        self.conv1 = torch.nn.Conv2d(state_space, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(7744, 256)
        self.fc2 = torch.nn.Linear(256, action_space)
        self.fc3 = torch.nn.Linear(256, action_space)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class Agent(object):
    def __init__(self, env, policy, player_id=1):
        if type(env) is not Pong:
            raise TypeError("I'm not a very smart AI. All I can play is Pong.")
        self.env = env
        self.player_id = player_id
        self.name = "DQN_AI"

        self.train_device = "cpu"  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        
        self.batch_size = 1
        self.gamma = 0.95
        self.epsilon = EPSILON_START
        self.step = 0
        self.optimizer = optim.Adam(policy.parameters(), lr=0.0001, eps=1e-02)
        #self.optimizer = optim.RMSprop(policy.parameters(), lr=0.0025, alpha=0.9, eps=1e-02, momentum=0.0)
    
    def get_name(self):
        return self.name

    def load_model(self, filename):
        #todo: check if filename exists
        state_dict = torch.load(filename)
        self.policy.load_state_dict(state_dict)

    def get_action(self, frame, epsilon=0.5):
        rnd = random.random()

        # Decrease epsilon value
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                       math.exp(-1. * self.step / EPSILON_DECAY)

        if self.epsilon <= rnd:
            x = torch.from_numpy(frame).float().to(self.train_device)
            q_values = self.policy.forward(x)
            action = q_values.argmax()
            action = np.asscalar(action.data.to(self.train_device).numpy())
        else:
            action = random.randint(0, 3)
        return action
    
    def update_policy(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def reset(self):
        self.step = 0;
        #self.epsilon = EPSILON_START


class Memory(object):
    def __init__(self, max_size=10000, batch_size=64):
        self.data = deque(maxlen=max_size)
        self.batch_size = batch_size
        
    def store(self, outcome):
        self.data.append(outcome)
        
    def sample(self):
        index = random.sample(range(0,len(self.data)), self.batch_size)
        samples = []
        for i in index:
            samples.append(self.data[i])
        return samples