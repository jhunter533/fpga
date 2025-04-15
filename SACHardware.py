#####Includes######
import logging
import os
import random
import re
import struct
import threading
import time
from copy import deepcopy
from queue import Queue
from threading import Lock

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pynvml
import serial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm import tqdm

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
pynvml.nvmlInit()
gpu_index = 0

def elapsed_time(start_time):
    return time.time()-start_time

def compute_phase_shift(back_leg,front_leg,back_contact,front_contact):
    back_contact_event=np.where(np.diff(back_contact)==1)
    front_contact_event=np.where(np.diff(front_contact)==1)
    phi_b=np.mean(np.diff(back_contact_event))
    phi_f=np.mean(np.diff(front_contact_event))
    def nrmse(y_t,y_p):
        mse=np.mean((y_t-y_p)**2)
        return np.sqrt(mse)/(np.max(y_t)-np.min(y_t))
    E_f=nrmse(back_leg,np.roll(front_leg,-int(phi_f//2)))
    E_b=nrmse(np.roll(back_leg,int(phi_b//2)),front_leg)
    E_ps=.5*E_f+.5*E_b
    return E_ps
###FIFO replay buffer to store and sample data for policy
class ReplayBuffer:
    def __init__(self,memory_capacity=int(2e6),batch_size=256,num_actions=4,num_states=24):
        self.memory_capacity=memory_capacity
        self.num_states=num_states
        self.num_actions=num_actions
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
        states = torch.tensor(self.state_buffer[indices], dtype=torch.float32, device=device, requires_grad=False)
        actions = torch.tensor(self.action_buffer[indices], dtype=torch.float32, device=device, requires_grad=False)
        rewards = torch.tensor(self.reward_buffer[indices], dtype=torch.float32, device=device, requires_grad=False)
        next_states = torch.tensor(self.next_state_buffer[indices], dtype=torch.float32, device=device, requires_grad=False)
        dones = torch.tensor(self.done_buffer[indices], dtype=torch.float32, device=device, requires_grad=False)
        return states,actions,rewards,next_states,dones


### FIFO Buffer for circuit stuff
class CircuitCommunicator:
    def __init__(self,port,baudrate=115200,timeout=1.0):### change port and baudrate
        self.serial_lock=Lock()
        self.input_buffer=Queue(maxsize=1024) #FIFO for pc to circuit states
        self.output_buffer=Queue(maxsize=1024) #FIFO for circuit to pc action result
        self.port=port
        self.baudrate=baudrate
        self.timeout=timeout
        self.logger=logging.getLogger('CircuitComms')

        self.ser=serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=.1,
        )
        self._running=threading.Event()
        self._thread=None
    def start(self):
        if not self.ser.is_open:
            self.ser.open()
        self._running.set()
        self._thread=threading.Thread(target=self._communication_loop)
        self._thread.daemon=True
        self._thread.start()
    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join()
        self.ser.close()
    def _pack_state(self,state):
        ##convert numpy arr to byte stream
        return struct.pack(f'!I?{len(state)}f',len(state),training,*state)
    def _unpack_action(self,data):
        action_dim = struct.unpack('!I', data[:4])[0]
        training_flag = struct.unpack('!?', data[4:5])[0]
        
        if training_flag:
            log_prob = struct.unpack('!f', data[5:9])[0]
            action = struct.unpack(f'!{action_dim}f', data[9:])
            return np.array(action), log_prob
        else:
            action = struct.unpack(f'!{action_dim}f', data[5:])
            return np.array(action), 0.0

    def _communication_loop(self):
       while self._running.is_set():
            try:
                # Send data
                if not self.input_buffer.empty():
                    state, training = self.input_buffer.get()
                    packed = self._pack_state(state, training)
                    self.ser.write(packed)

                # Receive data
                if self.ser.in_waiting >= 4:
                    header = self.ser.read(4)
                    action_dim = struct.unpack('!I', header)[0]
                    
                    # Determine remaining bytes
                    if self.ser.in_waiting >= 1:
                        flag_byte = self.ser.read(1)
                        training_flag = struct.unpack('!?', flag_byte)[0]
                        bytes_needed = action_dim * 4 + (4 if training_flag else 0)
                        
                        # Read remaining data
                        data = bytearray()
                        start_time = time.time()
                        while len(data) < bytes_needed:
                            if time.time() - start_time > self.timeout:
                                raise TimeoutError
                            data += self.ser.read(bytes_needed - len(data))
                            
                        # Unpack and store
                        action, log_prob = self._unpack_action(header + flag_byte + data)
                        self.output_buffer.put((action, log_prob))
                        
            except Exception as e:
                self.logger.error(f"Communication error: {e}")
                self.reset_connection()

    def send_state(self, state, training=False):
        """Queue state for transmission with training flag"""
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        self.input_buffer.put((state.astype(np.float32), training))

    def get_action(self):
        """Get action and log_prob from FPGA"""
        try:
            return self.output_buffer.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError("No response from FPGA")

    def reset_connection(self):
        self.stop()
        time.sleep(1)
        self.ser = serial.Serial(self.port, self.baudrate)
        self.start()

class HardwareActor(nn.Module):
    def __init__(self,communicator,action_dim=4,action_bound=1.0):
        super().__init__()
        self.communicator=communicator
        self.action_dim=action_dim
        self.action_bound=action_bound
        self.timeout=1
        self.training_mode=False
        self.logger=logging.getLogger('HardwareActor')
    def forward(self,state):
        self.communicator.send_state(state, training=True)
        action, log_prob = self.communicator.get_action()
        
        return (
            torch.tensor(action, device=device) * self.action_bound,
            torch.tensor(log_prob, device=device)
        )

    def action(self, state, det=False, SAN=False):
        """For inference: returns deterministic action"""
        if det:
            self.communicator.send_state(state, training=False)
            action, _ = self.communicator.get_action()
            return torch.tensor(action, device=device).cpu().numpy()
        else:
            return self.forward(state)

##Asume that 4 bytes unint32-state dimension, 1 byte bool training flag,N*4 float32 state values for pc to FPGA
##Assumefor fpga to pc if training flag=true,  4 bytes uint32 action dim,1byte bool training flag,4 bytes float32 log prob,M*4 bytes float 32 action values
##same for flag = false but no logstd

###Critic class with 3 fully connected layers
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
        '''
            'action'| Input action-[(batch,num_actions)]
            'state'| Input state-[(batch,num_states)]
            'x'| The output of each layer-[(batch,1)]
        '''
        x=torch.cat((state,action),-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

class Agent:
    def __init__(self, env, communicator, hidden_dim=256, hidden_dim2=256, alpha=0.2, seed=None,SAN=False):
        self.env = env
        self.SAN=SAN
        if self.SAN==True:
            self.arch="SANSAC"
        else:
            self.arch="SAC"
        if seed is not None:
            self.seed=seed
            self.set_seed(seed)
        
        # Environment parameters
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        
        # Hardware integration
        self.communicator = communicator
        self.actor = HardwareActor(communicator, self.action_dim, self.action_bound)
        
        # Dual critics
        self.critic1 = Critic(self.state_dim, self.action_dim, hidden_dim, hidden_dim2).to(device)
        self.critic2 = Critic(self.state_dim, self.action_dim, hidden_dim, hidden_dim2).to(device)
        self.target_critic1 = deepcopy(self.critic1).to(device)
        self.target_critic2 = deepcopy(self.critic2).to(device)
        
        # Optimizers
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=3e-4
        )
        
        # SAC parameters
        self.alpha = alpha
        self.tau = 0.005
        self.gamma = 0.99
        self.buffer = ReplayBuffer()
        self.total_steps = 0

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            if deterministic:
                return self.actor.action(state, det=True)
            return self.actor(state)[0].cpu().numpy()

    def train(self, total_timesteps=1e6):
        state, _ = self.env.reset()
        for _ in tqdm(range(int(total_timesteps))):
            self.total_steps += 1
            
            # Collect experience
            action, log_prob = self.actor(torch.FloatTensor(state).to(device))
            next_state, reward, done, trunc, _ = self.env.step(action.cpu().numpy())
            self.buffer.store(state, action.cpu().numpy(), reward, next_state, done)
            state = next_state
            
            # Train after warmup
            if self.total_steps > 1000 and self.total_steps % 50 == 0:
                self._update_parameters()
                
            if done or trunc:
                state, _ = self.env.reset()

    def _update_parameters(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()
        
        with torch.no_grad():
            # Get actions from FPGA policy
            next_actions, next_log_probs = self.actor(next_states)
            
            # Target Q-values
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Critic loss
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss (handled implicitly through FPGA)
        with torch.no_grad():
            # Get new actions for logging
            new_actions, new_log_probs = self.actor(states)
            q1 = self.critic1(states, new_actions)
            q2 = self.critic2(states, new_actions)
            q = torch.min(q1, q2)
            actor_loss = (self.alpha * new_log_probs - q).mean()

        # Logging
        wandb.log({
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "log_probs": new_log_probs.mean().item()
        })
        
        # Soft updates
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    def sync_weights_to_fpga(self):
    #implement fpga specific weight update protocol
        pass
    def set_seed(self,seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        self.env.action_space.seed(seed_value)
        torch.backends.cudnn.deterministic=True
        self.seed=seed_value

##example usage:
#comms=CircuitCommunicator(port=)
#comms.start()
#env=gym.make("")
#agent=Agent(env,comms)
#agent.train(total_timesteps=)
#comms.stop()
