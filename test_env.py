from cmath import inf, polar
import copy
from distutils.log import info
from math import gamma
from os import stat
import gym
from markupsafe import re
from pip import main

import SpaceRobotEnv
import numpy as np
import core
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from copy import deepcopy
from torch.optim import Adam

from scipy.stats import entropy

torch.manual_seed(0)
np.random.seed(0)
    
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

qdict = {}
fdict = {}
# env = gym.make("SpaceRobotState-v0")
env = gym.make("FetchReach-v1",reward_type="dense")
obs2 = env.reset()
env2 = obs2
actor_critic=core.MLPActorCritic
ac_kwargs=dict()
ac_kwargs = dict()
ac = actor_critic(env.observation_space['observation'], env.action_space, **ac_kwargs)
alpha = 0.9
discount = 0.9
qnetwork =  copy.deepcopy(ac.q)
anetwork =  copy.deepcopy(ac.pi)
qtarget =  copy.deepcopy(qnetwork)
atarget = copy.deepcopy(anetwork)
# Set up optimizers for policy and q-function
pi_lr=1e-3
q_lr=1e-3
aoptimizer = Adam(anetwork.parameters(), lr=pi_lr)
qoptimizer = Adam(qnetwork.parameters(), lr=q_lr)

def get_action(o, noise_scale):
   
    act_limit = env.action_space.high[0]
    act_dim = env.action_space.shape[0]
    a = anetwork(torch.as_tensor(o, dtype=torch.float32)).detach().numpy()
    a += noise_scale * np.random.randn(act_dim)
    return np.clip(a, -act_limit, act_limit)

def compute_loss_q(data):
        gamma = 0.99
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = qnetwork(o,a)
        # Bellman backup for Q function
        
        # etr = entropy(o.detach().numpy(),base=2,axis=1)
        var = a.detach().numpy().var()
        with torch.no_grad():
            q_pi_targ = qtarget(o2, atarget(o2))
            backup = r + gamma * (1 - d) * (q_pi_targ )

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()
        return loss_q

# Set up function for computing DDPG pi loss
def compute_loss_pi(data):
    o = data['obs']
    a = data['act']
    q_pi = qnetwork(o, anetwork(o))
    return -q_pi.mean()

def main():
    dim_u = env.action_space.shape[0]
    print(dim_u)
    dim_o = env.observation_space["observation"].shape[0]
    print(dim_o)
    observation = env.reset()
    max_action = env.action_space.high
    print("max_action:", max_action)
    print("mmin_action", env.action_space.low)
    
    score_history = []
    n_played_games = 0
    start_episodes = 100
    
    # Experience buffer
    replay_size=int(1e6)
    obs_dim = env.observation_space['observation'].shape[0] + 3
    act_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    goals = []
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    observation,ep_ret,ep_len = env.reset(),0,0
    
    for episode in range(350):
        # if(episode < start_episodes):
        print(episode)
        if(episode == start_episodes):
            observation,ep_ret,ep_len = env.reset(),0,0
        goals = []
        for steps in range (1,100):
            # print(observation)
            
            state = observation
            mstate = np.append(state['observation'],state['desired_goal'])
            
            a = env.action_space.sample()
            if episode > start_episodes:
                a = get_action(mstate, 0.1)
            else:
                a = env.action_space.sample()

            action = a
            observation, reward, done, info = env.step(action)
            
            sprime = observation
            msprime = np.append(observation['observation'],observation['desired_goal'])
            
            goals.append(sprime['achieved_goal'])
        
            ep_ret += reward
            ep_len += 1
            
            # done = info['is_success']
            replay_buffer.store(mstate,action,reward,msprime,done)
            
            for goal in goals:
                nstate = np.append(state['observation'],goal)
                nsprime =  np.append(sprime['observation'],goal)
                reward = env.compute_reward(state['achieved_goal'], goal , info)
                # print(reward)
                # done = 0
                replay_buffer.store(nstate,action,reward,nsprime,0)
           
            if (done or steps >= 999):
                n_played_games += 1
                score_history.append(ep_ret)
                avg_score = np.mean(score_history[-100:])
                print( 'score %.1f' %ep_ret, 'avg_score %.1f' %avg_score,'num_games', n_played_games, action )
                observation,ep_ret,ep_len= env.reset(), 0, 0
                break
            
            if(steps  % 1 == 0 and episode > 200):
                batch_size = 300
                data = replay_buffer.sample_batch(batch_size)
                qoptimizer.zero_grad()
                loss = compute_loss_q(data)
                loss.backward()
                qoptimizer.step()
                
                
                for p in qnetwork.parameters():
                    p.requires_grad = False

                # Next run one gradient descent step for pi.
                aoptimizer.zero_grad()
                loss_pi = compute_loss_pi(data)
                loss_pi.backward()
                aoptimizer.step()
                
                # Unfreeze Q-network so you can optimize it at next DDPG step.
                for p in qnetwork.parameters():
                    p.requires_grad = True
            
                polyak = 0.995
                for t, n in zip(qtarget.parameters(),qnetwork.parameters()):
                    t.data.mul_(polyak)
                    t.data.add_((1 - polyak) * n.data)
            
                for t, n in zip(atarget.parameters(),anetwork.parameters()):
                    t.data.mul_(polyak)
                    t.data.add_((1 - polyak) * n.data)
            
    while(1):
        test_agent()
            


def test_agent():
        num_test_episodes=10
        avg_score_test = []
        max_ep_len = 100000
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            g = o['desired_goal']
            o = o['observation']
            while not (d or (ep_len == max_ep_len)):
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
        
        
def test2(model):
    observation = env.reset()
    observation = env.reset()
    observation = env.reset()
    while(1):
        env.render()
        s = observation['achieved_goal']
        goal = observation['desired_goal']
        arr = np.array(observation['observation'])
        action = model( torch.tensor(np.append(arr,goal)).float())
        max_action = env.action_space.high
        observation, reward, done, info = env.step(action.detach().numpy())
        print(observation, done)


    
def roundTuple(arr):
    return tuple(np.round(arr,1))

def checkState(s , action):
    if(s not in qdict):
        qdict[s] = {}
    if(action not in qdict[s]):
        qdict[s][action] = 0
    
def maxQval(q1dict):
    return max(q1dict.values())
    
if __name__ == '__main__':
    main()