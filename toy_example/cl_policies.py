# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from collections import deque, namedtuple
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt

Transition = namedtuple("Transition", ("state", "action", "reward", "new_state", "log_prob"))
class TeacherModel(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_layers): 
        super(TeacherModel, self).__init__()
        # 1 hidden layer of some dimension -- probably need to change ? 
        self.layers =nn.Sequential(nn.Linear(input_dim, hidden_layers ), 
                              nn.Tanh(), 
                              nn.Linear(hidden_layers,hidden_layers +1  ), 
                              nn.Tanh())
        self.actor = nn.Linear(hidden_layers +1, output_dim)
        self.critic = nn.Linear(hidden_layers +1, 1 )

        # Initialize training parameters
        self.timesteps_per_batch = 50               # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 200          # Max number of timesteps per episode
        self.n_updates_per_iteration = 10                # Number of times to update actor/critic per iteration
        self.lr = 0.005                                 # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.iters = 500

		# Miscellaneous parameters
        self.render = True                              # If we should render during rollout
        self.render_every_i = 10                        # Only render every n iterations
        self.save_freq = 10                             # How often we save in number of iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results
    
    def forward(self, x): 
        x = self.layers(x)
        actor = self.actor(x)
        critic = self.critic(x)
        return actor, critic
    
    def rtg(self, rewards):
        batch_rtg = []
        #print(rewards)
        discounted = 0
        #print(ep_rew)
        for rew in reversed(rewards): 
            discounted = rew + discounted *self.gamma 
            batch_rtg.insert(0, discounted)
        batch_rtg = torch.tensor(batch_rtg)
        
        return batch_rtg
    def rollout(self, env, device, model):
        rollouts = Transition([], [], [], [], [])
        with torch.no_grad(): 
            for i in range(self.timesteps_per_batch):
                states = env.reset()[0]
                done = False
                ep_rew = []
                for j in range(self.max_timesteps_per_episode):
                    '''
                    Get action from policy model
                    '''
                    states = torch.from_numpy(np.array(states)).float().to(device)
                    actor, critic = model(states)
                    action_dist = Categorical(logits=actor.unsqueeze(-2))
                    action = action_dist.probs.argmax(-1)
                    prob = action_dist.log_prob(action)
                    action = action.numpy()[0]
                    
                    
                    new_states, rewards, done, _, _ = env.step(action)
                    ep_rew = np.append(ep_rew, rewards)
                    rollouts.state.append(states)
                    rollouts.action.append(action)
                    #rollouts.reward.append(rewards)
                    rollouts.new_state.append(torch.tensor(new_states))
                    rollouts.log_prob.append(prob.numpy())

                    states = new_states
                    if done:
                        break
                rtg = self.rtg(ep_rew)
                
            
                for k in range(len(rtg)):
                    rollouts.reward.append(rtg[k])

        return rollouts

    def evaluate(self, batch_states, batch_acts):
        actor, _ = self.forward(batch_states)
        
        action_dist = Categorical(logits=actor.unsqueeze(-2))
        action = action_dist.probs.argmax(-1)
        log_prob = action_dist.log_prob(action)

        return log_prob

    
class StudentModel(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        super(StudentModel, self).__init__()
        hidden_layers = 200
        self.layers =nn.Sequential(nn.Linear(input_dim, hidden_layers ), 
                              nn.Tanh(), 
                              nn.Linear(hidden_layers,hidden_layers +1  ), 
                              nn.Tanh())
        self.actor = nn.Linear(hidden_layers +1, output_dim)
        self.critic = nn.Linear(hidden_layers +1, 1 )
        
        self.timesteps_per_batch = 50
        self.max_timesteps_per_episode = 200
        
   
    def forward(self, x): 
        x = self.layers(x)
        actor = self.actor(x)
        critic = self.critic(x)
        return actor, critic

    def validate(self, env):
        with torch.no_grad():
            rewards = []
            log_probs = []
            states = env.reset()[0]
            done = False
            
            for i in range(self.timesteps_per_batch): 
                for j in range(self.max_timesteps_per_episode):
                    states = torch.tensor(states)
                    actor, critic = self.forward(states)
                    action_dist = Categorical(logits=actor.unsqueeze(-2))
                    #action = action_dist.probs.argmax(-1)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                    action = action.numpy()[0]
                    new_states, reward, dones, infos, _ = env.step(action)
                    
                    rewards = np.append(rewards, reward)
                    log_probs = np.append(log_probs, log_prob)
                    states = new_states
                    if done:
                        break
                
        return rewards, log_probs

  
class TeacherReplayBuffer(object):
    def __init__(self, size, state_dim, action_dim):
        self.size = size 
        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.memory = Transition([],[],[],[],[])
        
    def insert(self, rollouts):
        for j in range(len(rollouts.state)):
            self.memory.state.append(rollouts.state[j])
            self.memory.action.append(rollouts.action[j])
            self.memory.reward.append(rollouts.reward[j])
            self.memory.new_state.append(rollouts.new_state[j])
            self.memory.log_prob.append(rollouts.log_prob[j])

        while len(self.memory.state)> self.size: 
            #indx = np.random.choice(len(self.memory.state), size = 1, replace = False)
            #indx = int(indx)
            del self.memory.state[0]
            del self.memory.action[0]
            del self.memory.reward[0]
            del self.memory.new_state[0]
            del self.memory.log_prob[0]

    def sample_batch(self, batch_size):
        samples = Transition([], [], [], [], [])
        indxs = np.random.choice(len(self.memory.state), size = batch_size, replace = True)
        if len(self.memory.state) < self.size: 
            indxs = indxs[0:len(self.memory.state)]
        
        for k in range(len(indxs)):
            samples.state.append(self.memory.state[indxs[k]])
            samples.action.append(self.memory.action[indxs[k]])
            samples.reward.append(self.memory.reward[indxs[k]])
            samples.new_state.append(self.memory.new_state[indxs[k]])
            samples.log_prob.append(self.memory.log_prob[indxs[k]])

        return samples

class Training():
    def __init__(self): 
        self.timesteps_per_batch = 50                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 200          # Max number of timesteps per episode
        self.n_updates_per_iteration = 10                # Number of times to update actor/critic per iteration
        self.lr = 0.005                                 # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.iters = 100
        
        self.render_every_i = 10                        # Only render every n iterations
        self.save_freq = 10                             # How often we save in number of iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results
        self.reset_every = 50
        self.update_every = 50
        
        self.eta0_t = 0.05
        self.eta0_s = 0.95

        #for plotting
        self.its = []
        self.surprise = []
        self.t_rew = []
        self.s_rew = []
        self.states_vis = torch.zeros([1,4])
        self.actions_vis = []

        self.t_surprise = []
        self.s_surprise = []
        
    def plot(self):
        fig1, ax1 = plt.subplots(2,1)
        #ax1[0].plot(self.its, self.surprise, label = "Reward w/ Surprise")
        ax1[0].plot(self.its, self.t_rew, label = "Teacher reward")
        #ax1[0].plot(self.its, self.s_rew, label = "Student reward")
        #ax1[1].plot(self.its, self.t_surprise, label = "Teacher surprise") 
        #ax1[1].plot(self.its, self.s_surprise, label = "Student surprise")
        ax1[0].legend()
        #ax1[1].legend()
        #ax1[0].set_yscale('symlog')
        #ax1[1].set_yscale('symlog')
        
        pos = self.states_vis[:,0]
        vel = self.states_vis[:,1]
        fig2, ax2 = plt.subplots(3,1) 
        ax2[0].hist(pos, bins = 200 )
        ax2[0].set_title('Position on x-axis')
        ax2[1].hist(vel, bins = 200)
        ax2[1].set_title('Velocity')
        ax2[2].hist(self.actions_vis, bins = 3)
        ax2[2].set_title('Actions')
        fig2.suptitle(" Frequency of States and Actions ")

        return
    
    def surprise_reward(self,teacher_reward, t_prob, student_reward, student_policy, states, actions, new_states, t_surprise_model, s_surprise_model): 
        with torch.no_grad(): 
            eta1 = self.eta0_t/(max(1, abs((1/teacher_reward.shape[0])*torch.sum(teacher_reward))))
            eta2 = self.eta0_s/(max(1, abs(1/student_reward.shape[0])*torch.sum(student_reward)))
            
            inp = torch.hstack((states, actions.reshape([actions.shape[0], 1])))
            #print(surprise_model.predict(inp, return_std = True))
            t_mean, t_std = t_surprise_model.predict(inp, return_std = True)
            t_dist = torch.distributions.normal.Normal(torch.tensor(t_mean), torch.tensor(t_std))
            t_prob = t_dist.log_prob(new_states)[:,0] + t_dist.log_prob(new_states)[:,1]
           
            #need student log -prob
            act, c = student_policy(states)
            action_dist = Categorical(logits = act.unsqueeze(-2))
            action = action_dist.probs.argmax(-1)
            
            inp = torch.hstack((states, action.reshape([action.shape[0], 1])))
            s_mean, s_std = s_surprise_model.predict(inp, return_std = True)
            s_dist = torch.distributions.normal.Normal(torch.tensor(s_mean), torch.tensor(s_std))
            s_prob = s_dist.log_prob(new_states)[:,0] + s_dist.log_prob(new_states)[:,1]
            surprise_reward = teacher_reward - eta1*t_prob.reshape([t_prob.shape[0], 1]) + eta2*(t_prob.reshape([t_prob.shape[0], 1]) -s_prob.reshape([s_prob.shape[0], 1]))
           
            self.t_surprise = np.append(self.t_surprise,t_prob.mean().item() )
            self.s_surprise = np.append(self.s_surprise, (t_prob - s_prob).mean().item())
        return  surprise_reward.float()
    
    
    def train_loop(self, env, teacher_model, student_model): 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        teacher_a_optim = torch.optim.Adam(teacher_model.parameters(), lr = self.lr)
        teacher_c_optim = torch.optim.Adam(teacher_model.parameters(), lr = self.lr)
        
        #student_a_optim = torch.optim.Adam(student_model.parameters(), lr = self.lr )
        #student_c_optim = torch.optim.Adam(student_model.parameters(), lr = self.lr )

        batch_size = self.timesteps_per_batch*self.max_timesteps_per_episode
        
        #student_rewards = torch.zeros(batch_size)
        
        
        states = env.reset()[0]

        state_dim = env.observation_space.shape[0]
        action_dim= env.action_space.n
        teacher_memory = TeacherReplayBuffer(2*batch_size, state_dim, action_dim)
        
        t_surprise_prob = GaussianProcessRegressor()
        #s_surprise_prob = GaussianProcessRegressor()

        reset_count = 0
        update_count = 0 
        teacher_model.train()
        student_model.train()
        
        
        for i in range(self.iters):
            print("Starting Iteration number: ", i)
            # if (i%self.render_every_i == 0):
            #     env.render()
            #collect teacher rollouts 
            rollouts = teacher_model.rollout(env, device, teacher_model)
            print("rollout ended")
            teacher_memory.insert(rollouts)
            cl_func = torch.nn.MSELoss()
        
            states = torch.vstack(rollouts.state)
            new_states = torch.vstack(rollouts.new_state)
            #new_states = torch.from_numpy(np.asarray(rollouts.new_state)).float()
            actions = torch.from_numpy(np.asarray(rollouts.action))
            rewards = torch.from_numpy(np.asarray(rollouts.reward)).float()
            probs =  torch.from_numpy(np.asarray(rollouts.log_prob))
            rewards = rewards.reshape([rewards.shape[0], 1])
            
            reg_input = torch.hstack((states, actions.reshape([actions.shape[0], 1])))
            real_out = new_states
            t_surprise_prob.fit(reg_input, real_out)
            '''
            s_rollouts = teacher_model.rollout(env, device, student_model)
            s_states = torch.vstack(s_rollouts.state)
            s_new_states = torch.vstack(s_rollouts.new_state)
            s_actions = torch.from_numpy(np.asarray(s_rollouts.action))
            s_rewards = torch.from_numpy(np.asarray(s_rollouts.reward)).float()
            s_probs =  torch.from_numpy(np.asarray(s_rollouts.log_prob))
            s_rewards = s_rewards.reshape([s_rewards.shape[0], 1])
            
            s_reg_input = torch.hstack((s_states, s_actions.reshape([s_actions.shape[0], 1])))
            s_real_out = s_new_states
            s_surprise_prob.fit(s_reg_input, s_real_out)
            '''
            
            actor, value = teacher_model.forward(states)
            # rew_sup = self.surprise_reward(rewards, probs, student_rewards, student_model, states, actions, new_states, t_surprise_prob, s_surprise_prob)        
            rew_sup = rewards
            A = rew_sup - value.detach()
            #normalize advantage ? 
            A = (A - A.mean()) / (A.std()+1e-10) 
            
            # critic_loss = nn.MSELoss()

            for j in range(self.n_updates_per_iteration):    
                if j == 0: 
                    print("teacher training")
                actor, value = teacher_model.forward(states)
                action_dist = Categorical(logits=actor.unsqueeze(-2))
                action = action_dist.probs.argmax(-1)
                curr_prob = action_dist.log_prob(action)
                
                
                ratio = torch.exp(curr_prob - probs)
                
                sur1 = (ratio*A)
                
                sur2 = (torch.clamp(ratio, .8, 1.2)*A)
                al = (-torch.min(sur1, sur2)).mean()
                #rewards = torch.reshape(rewards, (-1,))
                
                
                teacher_a_optim.zero_grad()
                al.backward(retain_graph = True)
                teacher_a_optim.step()
                
                actor, value = teacher_model.forward(states)
                
                cl = nn.MSELoss()(value, rew_sup)
                teacher_c_optim.zero_grad()
                cl.backward()
                teacher_c_optim.step
                
            '''
            for k in range(self.n_updates_per_iteration): 
                if k ==0: 
                    print("student training")
                student_loss = torch.nn.HuberLoss()
                batch = teacher_memory.sample_batch(batch_size )
                batch_states = torch.vstack(batch.state)

                batch_new_states = torch.vstack(batch.new_state)
                batch_actions = torch.from_numpy(np.asarray(batch.action))
                batch_rewards = torch.from_numpy(np.asarray(batch.reward)).float()
                #batch_probs =  torch.from_numpy(np.asarray(batch.log_prob))
                #q_curr = teacher_model(batch_states).gather(1, batch_actions)
                actor, value = student_model(batch_states)
                new_actor, new_value = student_model(batch_new_states)
                rewards = torch.reshape(batch_rewards, [batch_rewards.shape[0],1])
                q_value = rewards + self.gamma*new_value 
                cl = cl_func(value, q_value.detach())
                student_c_optim.zero_grad()
                cl.backward()
                student_c_optim.step()
                
                #advantage
                actor, value = student_model(batch_states)
                new_actor, new_value = student_model(batch_new_states)
                q_value = rewards + self.gamma*new_value 
                advantage = q_value - value 
                action_dist = Categorical(logits=actor.unsqueeze(-2))
                #action = action_dist.probs.argmax(-1)
                log_prob = action_dist.log_prob(batch_actions).sum(dim = -1)
                #lp = log_prob.log_prob(actions).sum(dim = -1)
                lp = torch.reshape(log_prob, [log_prob.shape[0], 1])
                al = -1*lp.T@(advantage.detach())
                student_a_optim.zero_grad()
                al.backward()
                student_a_optim.step()

          '''  
                
            #test student network -- rollouts? 
            #update student reward 
            student_rewards, _ = torch.tensor(student_model.validate(env))
            if i % 10 == 0: 
                print("Avg. Teacher extrinsic reward / batch: " + str((rewards.sum()/self.timesteps_per_batch).item()))
                #print("teacher surprise bonus reward: " + str(rew_sup.mean().item()))
                #print("average student reward:" + str(student_rewards.mean().item()))
            
            self.its = np.append(self.its, i)
            #self.surprise = np.append(self.surprise, rew_sup.mean().item())
            self.t_rew = np.append(self.t_rew, (rewards.sum()/self.timesteps_per_batch).item())
            #self.s_rew = np.append(self.s_rew, student_rewards.mean().item())
            self.states_vis = torch.cat([self.states_vis, states])
            self.actions_vis = np.append(self.actions_vis, actions.numpy())

           
            
            reset_count += 1 
            if reset_count == self.reset_every: 
                states = env.reset()
        self.plot()
