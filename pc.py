import pickle
import socket
import threading
import time
from collections import deque
from sys import byteorder

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Critic(nn.Module):
    def __init__(self,num_states,num_actions,hidden_dim=128,hidden_dim2=128):
        super(Critic,self).__init__()
        self.fc1=nn.Linear(num_states+num_actions,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim2)
        self.fc3=nn.Linear(hidden_dim2,1)
    def forward(self,state,action):
        x=torch.cat([state,action],dim=-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
class ReplayBuffer:
    def __init__(self, memory_capacity=int(2e6), batch_size=256, num_actions=4, num_states=24):
        self.memory_capacity = memory_capacity
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.memory_capacity, self.num_states))
        self.action_buffer = np.zeros((self.memory_capacity, self.num_actions))
        self.reward_buffer = np.zeros(self.memory_capacity)
        self.next_state_buffer = np.zeros((self.memory_capacity, self.num_states))
        self.done_buffer = np.zeros(self.memory_capacity)
    
    def store(self, state, action, reward, next_state, done):
        index = self.buffer_counter % self.memory_capacity
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        self.done_buffer[index] = done
        self.buffer_counter += 1
    
    def sample(self):
        max_range = min(self.buffer_counter, self.memory_capacity)
        indices = np.random.randint(0, max_range, size=self.batch_size)
        states = torch.tensor(self.state_buffer[indices], dtype=torch.float32)
        actions = torch.tensor(self.action_buffer[indices], dtype=torch.float32)
        rewards = torch.tensor(self.reward_buffer[indices], dtype=torch.float32)
        next_states = torch.tensor(self.next_state_buffer[indices], dtype=torch.float32)
        dones = torch.tensor(self.done_buffer[indices], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

class Trainer:
    def __init__(self,host='192.168.1.10',port=12345):
        self.host=host
        self.port=port
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.socket.connect((host,port))
        
        print(f"Training listening on {host}:{port}")

        self.env=gym.make('Pendulum-v1',render_mode='human')
        self.num_states=3
        self.env.metadata['render_fps']=0
        self.num_actions=1
        self.action_bound=self.env.action_space.high

        self.critic1=Critic(self.num_states,self.num_actions)
        self.critic2=Critic(self.num_states,self.num_actions)
        self.target_critic1=Critic(self.num_states,self.num_actions)
        self.target_critic2=Critic(self.num_states,self.num_actions)
        self.critic_optimizer1=optim.Adam(self.critic1.parameters(),lr=3e-4)
        self.critic_optimizer2=optim.Adam(self.critic2.parameters(),lr=3e-4)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.replay_buffer=ReplayBuffer(num_actions=self.num_actions,num_states=self.num_states)
        
        self.gamma=0.975
        self.tau=0.005
        self.alpha=0.2
    
    def handle_fpga_actor(self):
        
        state,_=self.env.reset()
        episode_reward=0
        episode_step=0
        episode_count=0
        done=False
        truncated=False
        while True:
            try:
                send_data = np.concatenate([state.astype(np.float32), [float(done or truncated)]])
                self.socket.sendall(send_data.tobytes())
                
                # Receive action and log_prob from FPGA (raw bytes)
                response_data = b''
                expected_size = 4 * (self.num_actions + 1)  # action + log_prob
                while len(response_data) < expected_size:
                    chunk = self.socket.recv(expected_size - len(response_data))
                    if not chunk:
                        raise ConnectionError("Connection closed")
                    response_data += chunk
                
                # Parse action and log_prob
                action_bytes = response_data[:4*self.num_actions]
                log_prob_bytes = response_data[4*self.num_actions:]
                
                action = np.frombuffer(action_bytes, dtype=np.float32)
                log_prob = np.frombuffer(log_prob_bytes, dtype=np.float32)[0]  # Single value
                
                next_state,reward,terminated,truncated,info=self.env.step(action)
                done=terminated or truncated
                total_reward=reward*5
                #total_reward=max(total_reward,0)
                self.replay_buffer.store(state,action,total_reward,next_state,done)
                
                self.train_step()
                state=next_state
                episode_reward+=total_reward
                episode_step+=1
                if done:
                    episode_count+=1
                    print(f"Episode {episode_count} finished: reward={episode_reward:.2f},steps={episode_step}")
                    state,_=self.env.reset()
                    episode_reward=0
                    episode_step=0

            except Exception as e:
                print(f"Error with fpga connection: {e}")
                break
     
    def train_step(self):
        if self.replay_buffer.buffer_counter < self.replay_buffer.batch_size:
            return  # Not enough samples yet
            
        states,actions,rewards,next_states,dones=self.replay_buffer.sample()

        # Convert lists to numpy arrays first for efficiency
        next_actions_list = []
        next_log_probs_list = []
        
        for next_s in next_states:
            send_data=np.concatenate([next_s.detach().cpu().numpy().astype(np.float32),[0.0]])
            self.socket.sendall(send_data.tobytes())
            response_data=b''
            expected_size=4*(self.num_actions+1)
            while len(response_data)<expected_size:
                chunk=self.socket.recv(expected_size-len(response_data))
                if not chunk:
                    raise ConnectionError("Connection closed")
                response_data+=chunk
            next_action_bytes=response_data[:4*self.num_actions]
            next_log_prob_bytes=response_data[4*self.num_actions:]
            next_action=np.frombuffer(next_action_bytes,dtype=np.float32)
            next_log_prob=np.frombuffer(next_log_prob_bytes,dtype=np.float32)  # Single value
            next_actions_list.append(next_action)
            next_log_probs_list.append(next_log_prob)
        
        # Convert to numpy arrays first, then to tensor
        next_actions_np = np.array(next_actions_list, dtype=np.float32)
        next_log_probs_np = np.array(next_log_probs_list, dtype=np.float32)
        
        # Ensure correct shapes: [batch_size, num_actions] and [batch_size, 1]
        next_actions_tensor = torch.tensor(next_actions_np, dtype=torch.float32).view(-1, self.num_actions)
        next_log_probs_tensor = torch.tensor(next_log_probs_np, dtype=torch.float32).view(-1, 1)
        
        rewards=torch.unsqueeze(rewards,1)
        dones=torch.unsqueeze(dones,1)

        q1=self.critic1(states,actions)
        q2=self.critic2(states,actions)

        with torch.no_grad():
            q1_next_target=self.target_critic1(next_states,next_actions_tensor)
            q2_next_target=self.target_critic2(next_states,next_actions_tensor)
            q_next_target=torch.min(q1_next_target,q2_next_target)
            value_target=rewards+(1-dones)*self.gamma*(q_next_target-self.alpha*next_log_probs_tensor)
        q1_loss=((q1-value_target)**2).mean()
        q2_loss=((q2-value_target)**2).mean()
        self.critic_optimizer1.zero_grad()
        q1_loss.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        q2_loss.backward()
        self.critic_optimizer2.step()

        self.soft_update()
        
    def soft_update(self):
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def training_loop(self):
        while True:
            time.sleep(0.01)
if __name__=="__main__":
    trainer=Trainer()
    trainer.handle_fpga_actor()
