####Includes####
import os
import random
import re
import time
from copy import deepcopy

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pynvml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from spikingjelly.activation_based import (functional, layer, learning, neuron,
                                           surrogate)
from tqdm import tqdm

##cude
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
##global declarations for logging
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

###Actor class with 3 fully connected layers
#can be used as SAC if san variable is false, can be used as SANSAC if san variable is true
class Actor(nn.Module):
    def __init__(self,num_states,num_actions,action_bound,hidden_dim,hidden_dim2,SAN=False):
        super(Actor,self).__init__()
        self.num_states=num_states
        self.num_actions=num_actions
        self.action_bound=action_bound
        self.hidden_dim=hidden_dim
        self.hidden_dim2=hidden_dim2
        if SAN==True:
            self.L1=nn.Sequential(
                nn.Linear(self.num_states,self.hidden_dim),
                neuron.LIFNode(surrogate_function=surrogate.Sigmoid(),backend='cupy'),
                nn.Linear(self.hidden_dim,self.hidden_dim2),
                neuron.LIFNode(surrogate_function=surrogate.Sigmoid(),backend='cupy'),
            )
            self.L2=nn.Sequential(
                nn.Linear(self.hidden_dim2,self.num_actions),
                neuron.NonSpikingLIFNode(),
            )
            self.L3=nn.Sequential(
                nn.Linear(self.hidden_dim2,self.num_actions),
                neuron.NonSpikingLIFNode(),
            )
            functional.set_step_mode(self.L1,step_mode='m')
            functional.set_step_mode(self.L2,step_mode='m')
            functional.set_step_mode(self.L3,step_mode='m')
            self.T=16
        else:
            self.fc1=nn.Linear(self.num_states,self.hidden_dim)
            self.fc2=nn.Linear(self.hidden_dim,self.hidden_dim2)
            self.mu=nn.Linear(self.hidden_dim2,self.num_actions)
            self.log_std=nn.Linear(self.hidden_dim2,self.num_actions)
        self.min_log_std=-20
        self.max_log_std=2
        self.sp=nn.Softplus()
    def forward(self,state,SAN=False):
        '''
            'state'| Input state-[(batch,num_states)]
            'mu'| Output mean actions-[(batch,num_action)]
            'log_std'| Output log probability-[(batch,num_action)]
        '''
        if isinstance(state,np.ndarray):
            state=torch.from_numpy(state).float().to(device)
        else:
            state=state.to(device)
        if SAN==True:
            state=state.unsqueeze(0).repeat(self.T,1,1)
            x=self.L1(state)
            mu=self.L2(x)
            log_std=self.L3(x)
        else:
            x=F.relu(self.fc1(state))
            x=F.relu(self.fc2(x))
            mu=self.mu(x)
            log_std=self.log_std(x)
        log_std=torch.clamp(log_std,self.min_log_std,self.max_log_std)
        return mu,log_std
    def action(self,state,det=False,SAN=False):
        '''
            'state'| Input state -[(batch,state_dim)]
            'det'| User input if testing to utilize deterministic not stochastic policy-False by default
            'action'| pi Output action to take as sampled from policy-[(batch,action_dim)]
                ///must take just the most recent for env step
            'log_probs| Outputs the final log probability of policy of sampled action-[(batch,action_dim)]'
            
        '''
        if isinstance(state,np.ndarray):
            state=torch.from_numpy(state).float().to(device)
        else:
            state=state.to(device)
        mu,log_std=self(state,SAN)
        if det:
            state=torch.tensor(state,dtype=torch.float32).to(device)
            action=self.action_bound*torch.tanh(mu)
            return action.cpu().detach().numpy()
        std=torch.exp(log_std)
        normal=torch.distributions.Normal(mu,std)
        raw_action=normal.rsample()
        action=self.action_bound*torch.tanh(raw_action)
        log_probs=normal.log_prob(raw_action).sum(axis=-1,keepdim=True)
        transform=2*(np.log(2)-raw_action-self.sp(-2*raw_action)).sum(axis=-1,keepdim=True)
        log_probs-=transform
        if SAN==True:
            functional.reset_net(self)
        return action,log_probs

###Agent class utilizing actor and crtic classes
class Agent:
    def __init__(self,env,hidden_dim,hidden_dim2,seed=None,SAN=False):
        self.env=env
        self.SAN=SAN
        if self.SAN==True:
            self.arch="SANSAC"
        else:
            self.arch="SAC"
        if seed is not None:
            self.seed=seed
            self.set_seed(seed)
        self.state_dim=self.env.observation_space.shape[0]
        self.action_dim=self.env.action_space.shape[0]
        self.action_bound=self.env.action_space.high[0]
        self.buffer=ReplayBuffer(num_actions=1,num_states=3)
        self.learning_rate=2e-4
        self._tau=.005
        self._gamma=.975
        self._alpha=.2
        self.hidden_dim=hidden_dim 
        self.hidden_dim2=hidden_dim2
        self.actor=Actor(self.state_dim,self.action_dim,self.action_bound,hidden_dim,hidden_dim2,SAN=self.SAN).to(device)
        self.critic=Critic(self.state_dim,self.action_dim,hidden_dim,hidden_dim2).to(device)
        self.target_critic=deepcopy(self.critic).to(device)
        self.actor_optimizer=optim.Adam(self.actor.parameters(),lr=self.learning_rate)
        self.critic_optimizer=optim.Adam(self.critic.parameters(),lr=self.learning_rate)
        self.critic2=Critic(self.state_dim,self.action_dim,hidden_dim,hidden_dim2).to(device)
        self.target_critic2=deepcopy(self.critic2).to(device)
        self.critic2_optimizer=optim.Adam(self.critic2.parameters(),lr=self.learning_rate)
        self._file_path='samplefile'
        self.totalsteps=0
        self.hip_front=[]
        self.hip_back=[]
        self.front_contact=[]
        self.back_contact=[]
    #######Getter and Setters for hyperparameters#######
    @property
    def tau(self):
        return self._tau
    @tau.setter
    def tau(self,value):
        self._tau=value
    @property
    def gamma(self):
        return self._gamma
    @gamma.setter
    def gamma(self,value):
        self._gamma=value
    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self, value):
        self._alpha = value
    @property
    def file_path(self):
        return self._file_path
    @file_path.setter
    def file_path(self, value):
        self._file_path = value

    def extract_number_from_filename(self, filename):
        match = re.search(r'_(\d+)\.pth', filename)
        if match:
            return match.group(1)
        return 'unknown' 

    def soft_update(self):
        for target_param,param in zip(self.target_critic.parameters(),self.critic.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
        for target_param,param in zip(self.target_critic2.parameters(),self.critic2.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

    def select_action(self,state):
        action,_=self.actor.action(state,SAN=self.SAN)
        action=action.cpu().detach().numpy()
        return action

    def save_checkpoint(self, episode):
        checkpoint_dir = os.path.join('checkpoints',self.file_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{self.seed}_{episode}.pth')
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at episode {episode}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")

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
        for episode in tqdm(range(max_episode)):
            state,_=self.env.reset(seed=self.seed)
            done=False
            trunc=False
            while not (done or trunc):
                self.totalsteps+=1
                if i >1000:
                    action=self.select_action(state)
                    if self.SAN==True:
                        action=action.squeeze(0)
                else:
                    action=self.env.action_space.sample()
                i+=1
                
                total_r=0
                for _ in range(1):
                    next_state,reward,done,trunc,info=self.env.step(action)
                    total_r+=reward
                    if done:
                        total_r=max(total_r,0.0)
                        break
                total_r*=5
                
                self.buffer.store(state,action,total_r,next_state,done)
                states,actions,rewards,next_states,dones=self.buffer.sample()
                rewards=torch.unsqueeze(rewards,1)
                dones=torch.unsqueeze(dones,1)

                q1=self.critic(states,actions)
                q2=self.critic2(states,actions)

                with torch.no_grad():
                    next_action,next_log_probs=self.actor.action(next_states,SAN=self.SAN)
                    q1_next_target=self.target_critic(next_states,next_action)
                    q2_next_target=self.target_critic2(next_states,next_action)
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

                self.actor_optimizer.zero_grad()
                actions_pred,log_pred=self.actor.action(states,SAN=self.SAN)
                q1_pred=self.critic(states,actions_pred)
                q2_pred=self.critic2(states,actions_pred)
                q_pred=torch.min(q1_pred,q2_pred)
                actor_loss=(self.alpha*log_pred-q_pred).mean()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.soft_update()

                if i%1000:
                    wandb.log({
                              "reward":total_r,
                              "critic_loss1":q1_loss,
                              "critic_loss2":q2_loss,
                              "actor_loss":actor_loss,
                        #"Front Hip Angle": state[9],
                        #     "Back Hip Angle": state[4],
                        #     "Front Knee Angle": state[11],
                        #      "Back Knee Angle": state[6],
                                         })

                if done or trunc:
                    break
                state=next_state
            if episode%50==0:
                self.save_checkpoint(episode)

    def test_digital(self):
        state,_=self.env.reset()
        total_reward=0
        start_time=time.time()
        done=False
        trunc=False
        step=0
        while not (done or trunc):
            action=self.actor.action(state,det=True,SAN=self.SAN)
            if self.SAN==True:
                action=action.squeeze(0)
            next_state,reward,done,trunc,_=self.env.step(action)
            #self.hip_back.append(state[4])
            #self.hip_front.append(state[9])
            #self.front_contact.append(state[13])
            #self.back_contact.append(state[8])
            state=next_state
            total_reward+=reward
            step+=1
            wandb.log({
                "reward":reward,
                #   "Front Hip Angle":state[9],
                #"Back Hip Angle":state[4],
                #"Front Knee Angle": state[11],
                #"Back Knee Angle": state[6],
            })
            if done or trunc:
                break
        elapsed_t=elapsed_time(start_time)
        try:
            E_ps=compute_phase_shift(self.hip_back,self.hip_front,self.back_contact,self.front_contact)
        except:
            E_ps=1
        return total_reward,step+1,elapsed_t,E_ps
    def eval_agent(self,num_tests=30):
        total_reward=0
        total_E_ps=0
        for test in range(num_tests):
            tot_r,tot_step,tot_time,tot_e=self.test_digital()
            total_E_ps+=tot_e
            total_reward+=tot_r
            wandb.log({
                "Total Reward":tot_r,
                "Total Phase Shift":tot_e,
            })
        avg_phase_shift=total_E_ps/num_tests
        avg_reward=total_reward/num_tests
        wandb.log({
            "average phase shift":avg_phase_shift,
            "average reward":avg_reward,
        })
        return avg_reward
    def _get_model_dir(self):
        model_dir=os.path.join("Models",self._file_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir
    def save_model(self):
        model_dir=self._get_model_dir()
        actor_path=os.path.join(model_dir,f'{self.seed}_actor.pth')
        critic1_path=os.path.join(model_dir,f'{self.seed}_critic1.pth')
        critic2_path=os.path.join(model_dir,f'{self.seed}_critic2.pth')
        torch.save(self.actor.state_dict(),actor_path,_use_new_zipfile_serialization=True)
        torch.save(self.critic.state_dict(),critic1_path,_use_new_zipfile_serialization=True)
        torch.save(self.critic2.state_dict(),critic2_path,_use_new_zipfile_serialization=True)
        print("Model Saved")

    def load_model(self):
        model_dir=self._get_model_dir()
        actor_path=os.path.join(model_dir,f'{self.seed}_actor.pth')
        critic1_path=os.path.join(model_dir,f'{self.seed}_critic1.pth')
        critic2_path=os.path.join(model_dir,f'{self.seed}_critic2.pth')
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic1_path))
        self.critic2.load_state_dict(torch.load(critic2_path))
        print("Model Loaded")
