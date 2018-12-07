from pong import Pong
import torch.nn.functional as F

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        
        #Layers
        self.conv1 = torch.nn.Conv2d(state_space, 32, 8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, stride=2)
        self.fc1 = torch.nn.Linear(64, 256)
        self.fc2 = torch.nn.Linear(256, action_space)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        print(x.shape)
        return x
    

class Agent(object):
    def __init__(self, env, policy, player_id=1):
        if type(env) is not Pong:
            raise TypeError("I'm not a very smart AI. All I can play is Pong.")
        self.env = env
        self.player_id = player_id
        self.bpe = 4
        self.name = "DQN-AI-trial"
        
        self.train_device = "cpu"  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        
        self.batch_size = 1
        self.gamma = 0.98
    
    def get_name():
        return self.name

	def load_model(filename):
	    print("Loading")
    
    def get_action(frame, epsilon=0.5):
        rnd = random.random()
        if rnd < 1-epsilon :
            x = torch.from_numpy(frame).float().to(self.train_device)
            q_values = self.policy.forward(x)
            action = q_values.max(dim=0)
            action = np.asscalar(action.data.to(self.train_device).numpy())
        else:
            action = random.randint(0,3)
        return action
    
    def reset():
        print("Reset")
        
class Memory(object)
    def __init__(self, max_size=10000, batch_size=64):
        self.data = deque(max_len=max_size)
        self.batch_size = batch_size
        
    def store(self, outcome):
        self.data.append(outcome)
        
    def sample(self):
        index = random.sample(range(0,len(self.data)), self.batch_size)
        samples = []
        for i in index:
            samples.append(self.data[index])
        return samples
