import copy
import gym
from pip import main
import numpy as np
import coredual as core_normal
import torch
from torch.optim import Adam
import random
import time 

import SpaceRobotEnv

from memorydual import ReplayBuffer

from bokeh.plotting import figure, show
from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource, figure, show

start_time = time.time()
torch.manual_seed(0)
np.random.seed(0)
env = gym.make("SpaceRobotDualArm-v0",reward_type="distance")

# actor_critic=core_transfer.MLPActorCritic
actor_critic=core_normal.MLPActorCritic
ac_kwargs=dict()
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
    noise_scale = 0.25
    act_limit = env.action_space.high[0]
    act_dim = env.action_space.shape[0]
    a = anetwork(torch.as_tensor(o, dtype=torch.float32)).detach().numpy()
    # a += noise_scale * np.random.randn(act_dim)
    a = (1-noise_scale)*a  + (noise_scale * np.random.randn(act_dim))
    return np.clip(a, -act_limit, act_limit)

def compute_loss_q(data):
        gamma = 0.99
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        # print(torch.var(a))
        q = qnetwork(o,a)
        # Bellman backup for Q function
 
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
    var_noise = 0.2
    q_pi = (qnetwork(o, anetwork(o))) - (var_noise*torch.var(a,dim=1))
    # print(q_pi,torch.var(a,dim=1))
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
    time_history =[]
    n_played_games = 0
    start_episodes = 5
    
    episode_list = []
    reward_list = []
    
    # Experience buffer
    replay_size=int(1e6)
    obs_dim = env.observation_space['observation'].shape[0] + 6
    act_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    goals = []
    observation,ep_ret,ep_len = env.reset(),0,0
    
    for episode in range(450):
        print(episode)
        if(episode == start_episodes):
            observation,ep_ret,ep_len = env.reset(),0,0
        goals = []
        replay_buffer_2 = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        for steps in range (1,550):   
                 
            state = observation

            mstate = np.append(state['observation'], np.append(state['desired_goal'],state['desired_goal1']))
            
            a = env.action_space.sample()
            if episode > start_episodes:
                a = get_action(mstate, 0.1)
            else:
                a = env.action_space.sample()

            action = a
            observation, reward, done, info = env.step(action)
            if(done):
                print(steps,done)
            # env.render()
            reward = -reward
            sprime = observation
            msprime = np.append(observation['observation'],np.append(observation['desired_goal'],observation['desired_goal1']))
            
            goals.append( (sprime['achieved_goal'],sprime['achieved_goal1']) )
        
            ep_ret += reward
            ep_len += 1
             
            replay_buffer.store(mstate,action,reward,msprime,done)
            
            for goal in goals:
                g0 = goal[0]
                g1 = goal[1]
                nstate = np.append(state['observation'],np.append(g0,g1))
                nsprime =  np.append(sprime['observation'],np.append(g0,g1))
                
                ag = np.concatenate([state["achieved_goal"], state["achieved_goal1"]])
                g = np.concatenate([g0, g1])
                reward = compute_reward(ag, g, info) 
                replay_buffer.store(nstate,action,reward,nsprime,0)
                replay_buffer_2.store(nstate,action,reward,nsprime,0)

            
            
            if(episode >= start_episodes):
                batch_size = 800
                # update(replay_buffer_2,batch_size//4)
                update(replay_buffer, batch_size//4)
                update(replay_buffer, batch_size)
              
                
              
                
                
            if (done or steps >= 999 or info['is_success'] == 1.0 ):
                n_played_games += 1
                score_history.append(ep_ret)
                episode_list.append(episode)
                time_history.append(time.time() - start_time)
                avg_score = np.mean(score_history[-100:])
                print( 'score %.1f' %ep_ret, 'avg_score %.1f' %avg_score,'num_games', n_played_games, action )
                observation,ep_ret,ep_len= env.reset(), 0, 0
                break
            
               
    torch.save(anetwork, 'danetwork.pth') 
    print("SAVED")
    visualize(episode_list,score_history,time_history)
    print("--- %s seconds ---" % (time.time() - start_time))
          
    # while(1):
    #     test_agent()
            
def visualize(episode_list,reward_list,time_history):
    # create a new plot with a title and axis labels
    TOOLTIPS = [
        ('episode', "@x"),
        ('reward', "@y"),
        ('time_taken','@time')
    ]
    
    source = ColumnDataSource(data=dict(
    x=episode_list,
    y=reward_list,
    time = time_history,
    ))
    
    p = figure(title="Transfer Learning Space Robot",
               tools=[HoverTool()],
               tooltips=TOOLTIPS,
               x_axis_label="episode", 
               y_axis_label="reward" 
               )

    # add a line renderer with legend and line thickness
    p.line(x='x', y='y', line_width=2,source=source)

    # show the results
    show(p)


def update(replay_buffer, batch_size):
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
    


def test_agent():
        num_test_episodes=10
        avg_score_test = []
        max_ep_len = 550
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            g = o['desired_goal']
            o = o['observation']
            while not ( d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_2scale=0)
                env.render()
                newo = np.append(o,g)
                action = get_action(newo, 0)
                o, r, d, info = env.step(action)
                print(r,info,action)
                if(info['is_success']):
                    break
                o = o['observation']
                ep_ret += r
                ep_len += 1
            avg_score_test.append(ep_ret)
        print(ep_ret)
   
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)
         
def compute_reward(achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, desired_goal)
        d0 = goal_distance(achieved_goal[:3], desired_goal[:3])
        d1 = goal_distance(achieved_goal[3:], desired_goal[3:])

        reward = {
            "d0": d0,
            "d1": d1,
            "dense": -(0.001 * d ** 2 + np.log10(d ** 2 + 1e-6)),
        }

        return -d


if __name__ == '__main__':
    main()