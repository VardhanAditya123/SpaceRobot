import copy
import gym
from pip import main
import numpy as np
import core
import torch
from torch.optim import Adam
import random

from memory import ReplayBuffer
torch.manual_seed(0)
np.random.seed(0)


env = gym.make("FetchReach-v1",reward_type="dense")
anetwork = torch.load('anetwork.pth')
def main():
    while(1):
        test_agent()
def get_action(o, noise_scale):
    act_limit = env.action_space.high[0]
    act_dim = env.action_space.shape[0]
    a = anetwork(torch.as_tensor(o, dtype=torch.float32)).detach().numpy()
    # a += noise_scale * np.random.randn(act_dim)
    # a = (1-noise_scale)*a  + noise_scale * np.random.randn(act_dim)
    return np.clip(a, -act_limit, act_limit)

def test_agent():
    
    avg_score_test = []
    o, d, ep_ret, ep_len = env.reset(), False, 0, 0
    g = o['desired_goal']
    o = o['observation']
    while not (d):
        # Take deterministic actions at test time (noise_2scale=0)
        env.render()
        newo = np.append(o,g)
        action = get_action(newo, 0)
        o, r, d, info = env.step(action)
        print(r,info,action)
        o = o['observation']
        ep_ret += r
        ep_len += 1
    avg_score_test.append(ep_ret)
    print(ep_ret)
   
if __name__ == '__main__':
    main()