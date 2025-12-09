import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import deque
import ale_py
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


env = gym.make("ALE/Pong-v5") 
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


MODEL_PATH = "./model_weights/PPO/pong"


class PPO(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PPO, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = self.conv(x).view(x.size()[0], -1)
        x = self.fc(x)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


class PPOBuffer:
    def __init__(self, size, gamma=0.99, lam=0.95):
        self.max_size = size
        self.gamma = gamma
        self.lam = lam
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def get(self, last_value=0.0):
        self.values.append(last_value)
        returns = []
        advantages = []
        advantage = 0
       
        for i in range(len(self.states) - 1, -1, -1):
            delta = self.rewards[i] + self.gamma * self.values[i + 1] * (1 - self.dones[i]) - self.values[i]
            advantage = delta + self.gamma * self.values[i + 1] * (1 - self.dones[i]) * advantage
            advantages.insert(0, advantage)
            returns.insert(0, advantage + self.values[i])

        return (torch.FloatTensor(np.float32(self.states)), torch.LongTensor(self.actions), torch.stack(self.log_probs), torch.FloatTensor(self.rewards), torch.FloatTensor(np.float32(advantages)), torch.FloatTensor(self.values))       


class PPOAgent:
    def __init__(self, state_shape, n_actions, device="cpu", lr=1e-4, gamma=0.99, lam=0.95, eps=0.2, minibatch_size=32, num_epochs=4, buffer_size=4096):
        self.device = device
        self.n_actions = n_actions
        self.state_shape = state_shape

        self.policy = PPO(state_shape, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = PPOBuffer(buffer_size, state_shape, gamma=gamma, lam=lam)
        self.gamma = gamma
        self.lam = lam
        self.eps = eps
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, value = self.policy(state)
        m = Categorical(logits=logits)
        action = m.sample()
        log_prob = m.log_prob(action)
       
    def update(self):
        states, actions, log_probs, returns, advantages = self.buffer.get()

        states = states.to(self.device)
        actions = actions.to(self.device)
        log_probs = log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        for i in range(self.num_epochs):
            logits, values = self.policy(states)
            m = Categorical(logits=logits)
            new_log_probs = m.log_prob(actions)
            entropy = m.entropy().mean()

            prob_ratio = (new_log_probs - old_log_probs).exp()

            val1 = prob_ratio * advantages
            val2 = torch.clamp(prob_ratio, 1 - self.eps, 1 + self.eps) * advantages
            policy_loss = -torch.min(val1, val2).mean()

            value_loss = F.mse_loss(values, returns)

            loss = policy_loss + 1.0 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.buffer.reset()

        return loss.item()
 

def preprocess_frame(frame):
    frame = frame[35:195]  
    frame = frame[::2, ::2]
    frame = frame.mean(axis=2)
    frame = frame.astype(np.float32) / 255.0
    return frame

def train(agent, env, num_episodes=1000, steps_per_epoch=4096, max_ep_len=10000):
    frame_stack = deque(maxlen=4)
    total_rewards = []
    losses = []
    total_steps = 0    

    for episode in range(num_episodes):
        frame, _ = env.reset()
        frame = preprocess_frame(frame)
        
        for _ in range(4):
            frame_stack.append(frame)
        
        state = np.array(frame_stack)
        total_reward = 0

        for t in range(max_ep_len):
            action, log_prob, value = agent.select_action(state)
            
            next_frame, reward, done, truncated, _ = env.step(action)
            next_frame = preprocess_frame(next_frame)
            frame_stack.append(next_frame)
            next_state = np.array(frame_stack)
            
            agent.buffer.store(state, action, log_prob, reward, done, value)
            
            state = next_state
            total_reward += reward
            total_steps += 1

            if done or truncated:
                break
            
            if len(agent.buffer.states) >= timesteps_per_batch:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    logits, last_value = agent.policy(state_t)
                    last_value = last_value.squeeze(0).item()
                loss = agent.update()
                print(f"Update: total_steps={total_steps}, loss={loss}")

        episode_rewards.append(total_reward)
        print(f"Episode {episode}, Total Reward: {episode_reward}, Total Steps: {total_steps}")

    return episode_rewards


if __name__ == "__main__":
  state_shape = (4, 80, 80)
  n_actions = env.action_space.n

  agent = PPOAgent(state_shape, n_actions, device)

  if os.path.exists(f"{MODEL_PATH}/pong_ppo_episode_end.pth"):
    agent.policy_net.load_state_dict(torch.load(f"{MODEL_PATH}/pong_ppo_episode_end.pth"))
    losses = np.load(f"{MODEL_PATH}/losses.npy")
  else:
    agent, losses = train(agent, env, num_episodes=1000)
    np.save(f"{MODEL_PATH}/losses.npy", np.array(losses))

  window = 20
  rolling_avg = np.convolve(losses, np.ones(window) / window, mode='valid')

  plt.figure(figsize=(10, 5))
  plt.plot(losses, label="Total Reward per Episode", alpha=0.4)
  plt.plot(np.arange(window - 1, len(losses)), rolling_avg, label=f"{window}-Episode Rolling Avg", color="red")
  plt.xlabel("Episode")
  plt.ylabel("Total Reward")
  plt.title("Pong PPO Training Progress")
  plt.legend()
  plt.savefig("./images/PPO/PPO_pong.pdf", bbox_inches="tight", format="pdf")
