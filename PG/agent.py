from pong import Pong
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from utils import discount_rewards, softmax_sample

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        
        #Layer
        self.fc1 = torch.nn.Linear(state_space[0]*state_space[1], 256)
        self.fc2 = torch.nn.Linear(256, action_space)
        
        if torch.cuda.is_available():
            self.fc1.cuda()
            self.fc2.cuda()
		
        # Initialize neural network weights
        self.init_weights()
        
        #self.train_device = "cuda" if torch.cuda.is_available() else "cpu"

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    #def sigmoid(self,x):
    #    return 1.0 / (1.0 + np.exp(-x.cpu().detach().numpy()))
    
    def forward(self, x):
        x = x.reshape(100*105)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
        #return torch.from_numpy(self.sigmoid(x)).float().to(self.train_device)
    
    
class Agent(object):
    def __init__(self, env, policy, player_id=1):
        self.env = env
        self.player_id = player_id
        self.name = "PG_AI"
        
        self.train_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-4)
        
        self.batch_size = 10
        self.gamma = 0.98    
        
        self.observations = []
        self.actions = []
        self.rewards = []
        
        
        
    def get_name(self):
        return self.name
    
    def get_action(self, frame, epsilon=0):
        x = torch.from_numpy(frame).float().to(self.train_device)
        aprob = self.policy.forward(x)
        #aprob = aprob.data.cpu().numpy()
        #print(aprob)
        rnd = random.random()
        if rnd < 1-epsilon :
            action = softmax_sample(aprob)
        else:
            action = random.randint(0,2)
        return action, aprob
        
    def update_policy(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def episode_finished(self, episode_number):
        all_actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.observations, self.actions, self.rewards = [], [], []
        discounted_rewards = discount_rewards(all_rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)
        
        if (episode_number+1) % self.batch_size == 0:
            self.optimizer.zero_grad()
            
        weighted_probs = all_actions * discounted_rewards
        loss = torch.sum(weighted_probs)
        loss.backward()

        if (episode_number+1) % self.batch_size == 0:
            self.optimizer.step()
        
    def reset(self):
        print("Resetting")
        
    def load_model(self, filename):
        #todo: check if filename exists
        print("Loading model")
        state_dict = torch.load(filename)
        self.policy.load_state_dict(state_dict)
       
    def save_model(self):
        model_file = "save/pg.mdl"
        torch.save(self.policy.state_dict(), model_file)
        print("Model saved to", model_file)
        
    def store_outcome(self, observation, action_output, action_taken, reward):
        dist = torch.distributions.Categorical(action_output)
        action_taken = torch.Tensor([action_taken]).to(self.train_device)
        log_action_prob = -dist.log_prob(action_taken)

        self.observations.append(observation)
        self.actions.append(log_action_prob)
        self.rewards.append(torch.Tensor([reward]))