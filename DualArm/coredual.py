from certifi import core
import numpy as np
import scipy.signal

import torch
import torch.nn as nn
torch.manual_seed(0)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    print(sizes)
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        
        core_size =  [hidden_sizes[0]] + [hidden_sizes[0]]  + [hidden_sizes[0]]
        input_size = [obs_dim]+[hidden_sizes[0]]
        output_size = [hidden_sizes[0]]+[act_dim]
       
        self.act_limit = act_limit
        
        self.input = mlp(input_size,activation,activation)
        self.core =  mlp(core_size,activation,activation)
        self.output = mlp(output_size,activation,nn.Tanh)
        print(self.input)
        print(self.core)
        print(self.output)

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        input_value = self.input(obs)
        core_value = self.core(input_value)
        output_value = self.output(core_value)
        return self.act_limit * output_value


class MLPQFunction(nn.Module):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        input_size = [obs_dim + act_dim] +  [hidden_sizes[0]] 
        core_size =   [hidden_sizes[0]] + [hidden_sizes[0]] + [hidden_sizes[0]]
        output_size = [hidden_sizes[0]] + [1]
        
        self.input = mlp(input_size,activation,activation)
        self.core =  mlp(core_size,activation,activation)
        self.output = mlp(output_size,activation)
        print(self.input)
        print(self.core)
        print(self.output)
        
        

    def forward(self, obs, act):
        input_value = self.input(torch.cat([obs, act], dim=-1))
        core_value = self.core(input_value)
        output_value = self.output(core_value)
        return torch.squeeze(output_value,-1)


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(968,968),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0] + 6
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
