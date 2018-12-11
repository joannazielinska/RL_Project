from pong import Pong
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import numpy as np

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        
        #Layers
        self.conv1 = torch.nn.Conv2d(state_space, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = torch.nn.Linear(9*9*64, 256)
        self.fc2 = torch.nn.Linear(256, action_space)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(-1,9*9*64)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

class Agent(object):
    def __init__(self, env, policy, player_id=1):
        if type(env) is not Pong:
            raise TypeError("I'm not a very smart AI. All I can play is Pong.")
        self.env = env
        self.player_id = player_id
        self.bpe = 4
        self.name = "DQN_AI"
        
        self.train_device = "cpu"  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        
        self.batch_size = 1
        self.gamma = 0.98
        
        self.optimizer = optim.RMSprop(policy.parameters(), lr=0.0025, alpha=0.9, eps=1e-02, momentum=0.0)
    
    def get_name(self):
        return self.name

    def load_model(self,filename):
	    print("Loading")
    
    def get_action(self, frame, epsilon):
        rnd = random.random()
        if rnd < 1-epsilon :
            x = torch.from_numpy(frame).float().to(self.train_device)
            q_values = self.policy.forward(x)
            action = q_values.argmax()
            action = np.asscalar(action.data.to(self.train_device).numpy())
        else:
            action = random.randint(0,2)
        return action
    
    def update_policy(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def reset(self):
        print("Reset")
        
    def reset_score(self):
        self.score = 0
        
    def load_model(self, filename):
        #todo: check if filename exists
        state_dict = torch.load(filename)
        self.policy.load_state_dict(state_dict)
        
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
