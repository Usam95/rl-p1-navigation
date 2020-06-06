import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Network(nn.Module):
	def __self__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
		 """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
        """ 
		super(Network, self).__init__()

		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size, fc1_units)
		self.fc2 = nn.Linear(fc1_units, fc2_units)
		self.fc3 = nn.Linear(fc2_units, action_size)

	def forward(self, state)
		""" Forward function that maps states to actions. """
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		actions = self.fc3(x)
		return actions