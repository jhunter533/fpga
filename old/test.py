import ctypes
import os
import random
import re
import struct
import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
import serial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self,memory_capacity=int(2e6),batch_size=256,num_actions=4,num_states=24):
        self.memory_capacity=memory_capacity
        self.num_actions=num_actions
        self.num_states=num_states
        self.batch_size=batch_size
        self.buffer_counter=0
        self.state_buffer=np.zeros((self.memory_capacity,self.num_states))
        self.action_buffer=np.zeros((self.memory_capacity,self.num_actions))
        self.reward_buffer=np.zeros(self.memory_capacity)
        self.next_state_buffer=np.zeros((self.memory_capacity,self.num_states))
        self.done_buffer=np.zeros(self.memory_capacity)
    def store(self,state,action,reward,next_state,done):
        index=self.buffer_counter%self.memory_capacity
        self.state_buffer[index]=state
        self.action_buffer[index]=action
        self.reward_buffer[index]=reward
        self.next_state_buffer[index]=next_state
        self.done_buffer[index]=done
        self.buffer_counter+=1
    def sample(self):
        max_range=min(self.buffer_counter,self.memory_capacity)
        indices=np.random.randint(0,max_range,size=self.batch_size)
        states=torch.tensor(self.state_buffer[indices],dtype=torch.float32,device=device,requires_grad=False)
        actions=torch.tensor(self.action_buffer[indices],dtype=torch.float32,device=device,requires_grad=False)
        rewards=torch.tensor(self.reward_buffer[indices],dtype=torch.float32,device=device,requires_grad=False)
        next_states=torch.tensor(self.next_state_buffer[indices],dtype=torch.float32,device=device,requires_grad=False)
        dones=torch.tensor(self.done_buffer[indices],dtype=torch.float32,device=device,requires_grad=False)
        return states,actions,rewards,next_states,dones

class Critic(nn.Module):
    def __init__(self,num_states,num_actions,hidden_dim,hidden_dim2):
        super(Critic,self).__init__()
        self.num_actions=num_actions
        self.num_states=num_states
        self.hidden_dim=hidden_dim
        self.hidden_dim2=hidden_dim2
        self.fc1=nn.Linear(self.num_states+self.num_actions,self.hidden_dim)
        self.fc2=nn.Linear(self.hidden_dim,self.hidden_dim2)
        self.fc3=nn.Linear(self.hidden_dim2,1)
    def forward(self,state,action):
        x=torch.cat((state,action),-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

class Actor:
    def __init__(self,port,baudrate=115200):
        self.ser=serial.Serial(port,baudrate,timeout=5.0)
        self.last_state=None
        self.last_action=None
        print("connected?")
    #packet=[startbyte][cmd][cmdlen][data][checksum]
    def send_packet(self,cmd,data=b''):
        packet=bytearray()
        packet.append(0xAA)
        packet.append(cmd)
        packet.append(len(data))
        packet.extend(data)
        packet.append(sum(packet)&0xFF)
        print(f"sending packet {packet}")
        self.ser.write(packet)
    def recieve_packet(self,expected_cmd):
        print("starting recieve packet")
        start_byte=b''
        start_time=time.time()
        while start_byte!=b'\xAA':
            if time.time()-start_time>5.0:
                raise TimeoutError("start byte not found")
            start_byte=self.ser.read(1)
        cmd_byte=self.ser.read(1)
        if not cmd_byte:
            raise TimeoutError("missing command byte")
        cmd=cmd_byte[0]
        if cmd!=expected_cmd:
            raise ValueError(f"Unexpected command {cmd:02X} (expected {expected_cmd:02X})")
        len_byte=self.ser.read(1)
        if not len_byte:
            raise TimeoutError("Missing length byte")
        length=len_byte[0]
        data=self.ser.read(length)
        if len(data)!=length:
            raise TimeoutError(f"Data incomplete for length")
        checksum_byte=self.ser.read(1)
        if not checksum_byte:
            raise TimeoutError("Missing checksum")
        checksum=checksum_byte[0]
        calc_checksum=(0xAA+cmd+length+sum(data))&0xFF
        if calc_checksum!=checksum:
            raise ValueError("Checksum error")
        
        print(f"recieve packet")
        return data
    def float_to_fixed(self,value):
        int_val=int(value*256)
        return np.uint16(int_val) if int_val>=0 else np.uint16((1<<16)+int_val)
    def fixed_to_float(self,value):
        i= value if value<32768 else value - (1<<16)
        return i/256.0
    def get_action(self,state):
        state_data=bytearray()
        for value in state:
            fixed=self.float_to_fixed(value)
            state_data.extend(fixed.to_bytes(2,'little'))
        self.send_packet(0x46,state_data)
        response=self.recieve_packet(0x46)#'F'
        mu_fixed=int.from_bytes(response[0:2],'little')
        log_std_fixed=int.from_bytes(response[2:4],'little')
        mu=self.fixed_to_float(mu_fixed)
        log_std=self.fixed_to_float(log_std_fixed)

        self.last_state=state
        self.last_action=(mu,log_std)
        return mu,log_std
    def reset_neurons(self):
        self.send_packet(0x52)#'R'
        self.recieve_packet(0x44)#'D'
    def send_backward(self,delta_mu,delta_log_std,reward):
        if self.last_state is None:
            raise RuntimeError("no forward pass record")
        data=bytearray()
        for value in [delta_mu,delta_log_std,reward]:
            fixed=self.float_to_fixed(value)
            data.extend(fixed.to_bytes(2,'little'))
        self.send_packet(0x42,data) #'B'
        self.recieve_packet(0x44)
    def action(self,state):
        mu,log_std=self.get_action(state)
        std=torch.exp(log_std)
        normal=torch.distributions.Normal(mu,std)
        raw_action=normal.rsample()
        action=self.action_bound*torch.tanh(raw_action)
        log_probs=normal.log_prob(raw_action).sum(axis=-1,keepdim=True)
        transform=2*(np.log(2)-raw_action-self.sp(-2*raw_action)).sum(axis=-1,keepdim=True)
        log_probs-=transform
        return action,log_probs
class Agent:
    def __init__(self,env,hidden_dim,hidden_dim2,seed=None):
        self.env=env
        self.arch="SANSAC"
        if seed is not None:
            self.seed=seed
            self.set_seed(seed)
        self.state_dim=self.env.observation_space.shape[0]
        self.action_dim=self.env.action_space.shape[0]
        self.action_bound=self.env.action_space.high[0]
        self.buffer=ReplayBuffer(num_actions=1,num_states=3)
        self.learning_rate=2e-4
        self.tau=.005
        self.gamma=.975
        self.alpha=.2
        self.hidden_dim2=hidden_dim2
        self.hidden_dim=hidden_dim
        self.actor=Actor('/tmp/virtual2')
        self.critic=Critic(self.state_dim,self.action_dim,hidden_dim,hidden_dim2).to(device)
        self.target_critic=deepcopy(self.critic).to(device)
        self.critic_optimizer=optim.Adam(self.critic.parameters(),lr=self.learning_rate)
        self.critic2=Critic(self.state_dim,self.action_dim,hidden_dim,hidden_dim2).to(device)
        self.target_critic2=deepcopy(self.critic2).to(device)
        self.critic2_optimizer=optim.Adam(self.critic2.parameters(),lr=self.learning_rate)
    def soft_update(self):
        for target_param,param in zip(self.target_critic.parameters(),self.critic.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
        for target_param,param in zip(self.target_critic2.parameters(),self.critic2.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
    def set_seed(self,seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        self.env.action_space.seed(seed_value)
        torch.backends.cudnn.deterministic=True
        self.seed=seed_value
    def train(self,max_episode):
        i=0
        for episode in (range(max_episode)):
            state,_=self.env.reset(seed=self.seed)
            done=False
            trunc=False
            self.actor.reset_neurons()
            while not (done or trunc):
                if i > 1000:
                    mu,log_std=self.actor.action(state)
                    action=mu
                else:
                    action=self.env.action_space.sample()
                i+=1
                next_state,reward,done,trunc,info=self.env.step(action)
                self.buffer.store(state,action,reward,next_state,done)

                states,actions,rewards,next_states,dones=self.buffer.sample()
                rewards=torch.unsqueeze(rewards,1)
                dones=torch.unsqueeze(dones,1)

                q1=self.critic(states,actions)
                q2=self.critic2(states,actions)

                with torch.no_grad():
                    next_actions,next_log_probs=self.actor.get_action(next_states) 
                    q1_next_target=self.target_critic(next_states,next_actions)
                    q2_next_target=self.target_critic2(next_states,next_actions)
                    q_next_target=torch.min(q1_next_target,q2_next_target)
                    value_target=rewards+(1-dones)*self.gamma*(q_next_target-self.alpha*next_log_probs)
                q1_loss=((q1-value_target)**2).mean()
                q2_loss=((q2-value_target)**2).mean()
                self.critic_optimizer.zero_grad()
                self.critic2_optimizer.zero_grad()
                q1_loss.backward()
                q2_loss.backward()
                self.critic_optimizer.step()
                self.critic2_optimizer.step()
                #insert actor backward surrogate gradient here
                '''
                self.actor_optimizer.zero_grad()
                actions_pred,log_pred=self.actor.action(states)
                q1_pred=self.critic(states,actions_pred)
                q2_pred=self.critic2(states,actions_pred)
                q_pred=torch.min(q1_pred,q2_pred)
                actor_loss=(self.alpha*log_pred-q_pred).mean()
                actor_loss.backward()
                self.actor_optimizer.step()

                '''
                print("pre soft update")
                self.soft_update()
                if done or trunc:
                    break
                state=next_state
def main():
    env=gym.make("Pendulum-v1",render_mode="human")
    agent=Agent(env=env,hidden_dim=64,hidden_dim2=32,seed=42)
    agent.train(max_episode=1000)
    env.close()
if __name__=="__main__":
    main()


