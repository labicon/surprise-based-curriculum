# -*- coding: utf-8 -*-

import gym 
import torch 
import torch.nn as nn
import numpy as np 
import math 
from collections import deque, namedtuple
from torch.distributions.categorical import Categorical 
import cl_policies
'''
initialize environment 
'''
env = gym.make('CartPole-v0', render_mode = 'human')
observation, infor = env.reset()

def evaluate(model, env):
    episodes = 1
    steps =  200
    performance_list = []
    for epi in range(episodes):
        s = env.reset()[0]
        done = False
        cum_reward = 0.0
        #while not done:
        for j in range(steps):
            states = torch.tensor(s)
            actor, critic = model(states)
            action_dist = Categorical(logits=actor.unsqueeze(-2))
            action = action_dist.probs.argmax(-1)
            log_prob = action_dist.log_prob(action)
            action = action.numpy()[0]
            new_states, reward, dones, infos, _ = env.step(action)
            s = new_states
            cum_reward += reward
        
        performance_list.append(cum_reward)
    
    return np.sum(performance_list)



if __name__ == "__main__":
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]

    teacher = cl_policies.TeacherModel(state_space, action_space, 200)
    student = cl_policies.StudentModel(state_space, action_space)

    print("Training start")
    train = cl_policies.Training()
    train.train_loop(env, teacher, student, student)
    print("Training ended, Evaluation start")
    t_perf = evaluate(teacher, env)
    env2 = gym.make('MountainCar-v0', render_mode = "human")    
    s_perf = evaluate(student, env2)
    
    print("Teacher rewards: " + str(t_perf))
    print("student rewards: "+ str(s_perf))
