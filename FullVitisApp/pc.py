import pickle
import queue
import socket
import struct
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

from frame_socket import FramedSocket
from tcp_framing import Message


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
        self.socket=FramedSocket(host,port)

        print(f"Training listening on {host}:{port}")

        if not self.socket.ping():
            raise RuntimeError("FPGA ping failed")
        print("PING successful")

        self.env=gym.make('Pendulum-v1',render_mode='human')
        self.num_states=3
        self.env.metadata['render_fps']=0
        self.num_actions=1
        self.action_bound=self.env.action_space.high


        self.transition_queue=queue.Queue(maxsize=50000)
        self.stop_event=threading.Event()

        self.critic1=Critic(self.num_states,self.num_actions)
        self.critic2=Critic(self.num_states,self.num_actions)
        self.target_critic1=Critic(self.num_states,self.num_actions)
        self.target_critic2=Critic(self.num_states,self.num_actions)
        self.critic_optimizer1=optim.Adam(self.critic1.parameters(),lr=3e-4)
        self.critic_optimizer2=optim.Adam(self.critic2.parameters(),lr=3e-4)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.replay_buffer=ReplayBuffer(num_actions=self.num_actions,num_states=self.num_states,batch_size=32)
        
        self.gamma=0.975
        self.tau=0.005
        self.alpha=0.2

    def actor_thread(self):
        state,_=self.env.reset()
        episode_reward=0
        episode_step=0
        episode_count=0
        while not self.stop_event.is_set():
            try:
                action,log_prob=self.socket.query_actor(state.tolist(),done=False)
                next_state,reward,term,trunc,_=self.env.step(action)
                done=term or trunc
                total_reward=reward*5
                print(f"[Actor] Step reward={reward}, done={done}")
                self.transition_queue.put((
                    state.copy(),          # current state
                    np.array(action),      # action
                    total_reward,          # reward
                    next_state.copy(),     # next state
                    done,                  # done
                    log_prob               # log_prob
                ))
                state=next_state
                episode_reward+=total_reward
                episode_step+=1
                if done:
                    print(f"Episode {episode_count:3d} | Steps: {episode_step:3d} | Reward: {episode_reward:8.2f}")
                    # Reset for next episode
                    state, _ = self.env.reset()
                    episode_reward = 0.0
                    episode_step = 0
                    episode_count += 1
            except Exception as e:
                print(f"[Actor] Error: {e}")
                time.sleep(1)
                try:
                    self.socket.connect()
                except Exception as ee:
                    print(f"Reconnect failed: {ee}")

    def trainer_thread(self):
        while not self.stop_event.is_set():
            while self.replay_buffer.buffer_counter<self.replay_buffer.batch_size:
                try:
                    trans=self.transition_queue.get(timeout=0.1)
                    self.replay_buffer.store(*trans[:5])
                except queue.Empty:
                    continue
            states,actions,rewards,next_states,dones=self.replay_buffer.sample()
            next_state_list = next_states.detach().cpu().numpy().tolist()
            try:
                batch_results = self.socket.query_minibatch(next_state_list)
            except Exception as e:
                print(f"[Trainer] Minibatch query failed: {e}")
                time.sleep(1)
                continue

            # Unpack: list of (action, log_prob)
            next_actions_list = [np.array(r[0]) for r in batch_results]
            next_log_probs_list = [r[1] for r in batch_results]

            next_actions_tensor = torch.tensor(np.stack(next_actions_list), dtype=torch.float32)
            next_log_probs_tensor = torch.tensor(next_log_probs_list, dtype=torch.float32).unsqueeze(1)

            # Rest of train_step (same as before)
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1)
            
            self.critic_optimizer1.zero_grad()
            self.critic_optimizer2.zero_grad()

            actions_det=actions.detach().requires_grad_(True)

            q1 = self.critic1(states, actions_det)
            q2 = self.critic2(states, actions_det)
            q_min=torch.min(q1,q2)
            
            q_min_sum=q_min.sum()
            q_min_sum.backward()
            dQ_da=actions_det.grad.clone().detach()

            with torch.no_grad():
                q1_next = self.target_critic1(next_states, next_actions_tensor)
                q2_next = self.target_critic2(next_states, next_actions_tensor)
                q_next = torch.min(q1_next, q2_next)
                targets = rewards + (1 - dones) * self.gamma * (q_next - self.alpha * next_log_probs_tensor)
            q1=self.critic1(states,actions)
            q2=self.critic2(states,actions)

            q1_loss = F.mse_loss(q1, targets)
            q2_loss = F.mse_loss(q2, targets)

            q1_loss.backward()
            self.critic_optimizer1.step()

            q2_loss.backward()
            self.critic_optimizer2.step()
        
            dL_da = -dQ_da.cpu().numpy().astype(np.float32)
            dL_dlogp = np.full((self.replay_buffer.batch_size,), self.alpha, dtype=np.float32)

            # === STEP 4: Send gradients to FPGA ===
            try:
                self.send_grad_update(dL_da, dL_dlogp)
            except Exception as e:
                print(f"[Trainer] Gradient update failed: {e}")
                time.sleep(1)
                continue
            self.soft_update()
            time.sleep(0.001)  # yield

    def run(self):
        actor_t = threading.Thread(target=self.actor_thread, daemon=True, name="ActorThread")
        trainer_t = threading.Thread(target=self.trainer_thread, daemon=True, name="TrainerThread")

        actor_t.start()
        trainer_t.start()

        try:
            while actor_t.is_alive() and trainer_t.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
            self.stop_event.set()
            actor_t.join(timeout=2)
            trainer_t.join(timeout=2)
    def send_grad_update(self, dL_da: np.ndarray, dL_dlogp: np.ndarray):
        # Convert to list of lists (Python-native)
        dL_da_list = dL_da.tolist()                 # [[a1], [a2], ..., [aN]]
        dL_dlogp_list = dL_dlogp.tolist()           # [s1, s2, ..., sN]
        self.socket.send_grad_update(dL_da_list, dL_dlogp_list)
    def evaluate(self, episodes=5):
        print(f"\n{'='*50}")
        print(f"Evaluation over {episodes} episodes")
        print(f"{'='*50}")

        total_rewards = []
      
        for ep in range(episodes):
            state, _ = self.env.reset()
            ep_reward = 0.0
            step = 0
            while True:
                    # Query actor — no done flag needed for eval (but pass False)
                action, _ = self.socket.query_actor(state.tolist(), done=False)
                state, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += reward
                step += 1

                   
                if terminated or truncated:
                    break

            total_rewards.append(ep_reward)
            print(f"Eval Episode {ep+1}/{episodes} | Reward: {ep_reward:8.2f} | Steps: {step}")

               
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"\nEvaluation Result: {avg_reward:.2f} ± {std_reward:.2f} (mean ± std)")
        return avg_reward, std_reward
    def soft_update(self):
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

if __name__=="__main__":
    trainer=Trainer()
    trainer.run()
    trainer.evaluate(episodes=10)

