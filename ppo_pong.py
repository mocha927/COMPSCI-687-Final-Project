import gymnasium as gym
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
        self.states.append(np.array(state, copy=False))
        self.actions.append(int(action))
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def get(self, last_value=0.0):
        values = self.values + [last_value]
        returns = []
        advantages = []
        advantage = 0
       
        for i in range(len(self.states) - 1, -1, -1):
            delta = self.rewards[i] + self.gamma * values[i + 1] * (1 - self.dones[i]) - values[i]
            advantage = delta + self.gamma * self.lam * (1 - self.dones[i]) * advantage
            advantages.insert(0, advantage)
            returns.insert(0, advantage + self.values[i])

        advantages = torch.FloatTensor(np.float32(advantages))
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return (torch.FloatTensor(np.float32(self.states)), torch.LongTensor(self.actions), torch.stack(self.log_probs), torch.FloatTensor(returns), advantages)       


class PPOAgent:
    def __init__(self, state_shape, n_actions, device="cpu", lr=3e-4, gamma=0.99, lam=0.95, eps=0.2, minibatch_size=256, num_epochs=4, buffer_size=4096):
        self.device = device
        self.n_actions = n_actions
        self.state_shape = state_shape

        self.policy = PPO(state_shape, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = PPOBuffer(buffer_size, gamma, lam)
        self.gamma = gamma
        self.lam = lam
        self.eps = eps
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.policy(state)
            m = Categorical(logits=logits)
            action = m.sample()
            log_prob = m.log_prob(action)
        
        return action.item(), log_prob, value.item()
        
    def update(self, last_value=0.0):
        states, actions, log_probs, returns, advantages = self.buffer.get(last_value)

        states = states.to(self.device)
        actions = actions.to(self.device)
        log_probs = log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        total_loss = 0
        batch_size = states.size(0)
        num_batches = 0       

        for i in range(self.num_epochs):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size        
                idx = indices[start:end]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_log_probs = log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]

                logits, values = self.policy(batch_states)
                m = Categorical(logits=logits)
                new_log_probs = m.log_prob(batch_actions)
                entropy = m.entropy().mean()

                prob_ratio = (new_log_probs - batch_log_probs).exp()
                val1 = prob_ratio * batch_advantages
                val2 = torch.clamp(prob_ratio, 1 - self.eps, 1 + self.eps) * batch_advantages
                policy_loss = -torch.min(val1, val2).mean()
                value_loss = F.mse_loss(values, batch_returns)

                loss = policy_loss + 1.0 * value_loss - 0.01 * entropy

                total_loss += loss.item()
                num_batches += 1

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.buffer.reset()

        return total_loss / max(num_batches, 1)

         

def preprocess_frame(frame):
    frame = frame[35:195]  
    frame = frame[::2, ::2]
    frame = frame.mean(axis=2)
    frame = frame.astype(np.float32) / 255.0
    return frame

def train(agent, env, num_episodes=1000, steps_per_epoch=16384, max_ep_len=10000):
    frame_stack = deque(maxlen=4)
    rewards = []
    losses = []
    total_steps = 0    

    for episode in range(num_episodes):
        steps = 0
        ep_rewards = []

        while steps < steps_per_epoch:
            frame, _ = env.reset()
            frame = preprocess_frame(frame)
            
            frame_stack.clear()
            for _ in range(4):
                frame_stack.append(frame)
        
            state = np.array(frame_stack)
            ep_return = 0
            ep_len = 0
            done = False
            truncated = False
        
            while not (done or truncated) and ep_len < max_ep_len and steps < steps_per_epoch:
                action, log_prob, value = agent.select_action(state)
            
                next_frame, reward, done, truncated, _ = env.step(action)
                next_frame = preprocess_frame(next_frame)
                frame_stack.append(next_frame)
                next_state = np.array(frame_stack)
            
                agent.buffer.store(state, action, log_prob, reward, float(done or truncated), value)
            
                state = next_state
                ep_return += reward
                steps += 1
                total_steps += 1
                ep_len += 1
              
            ep_rewards.append(ep_return)
            
        if not (done or truncated):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                logits, last_value = agent.policy(state_tensor)
                last_value = last_value.squeeze(0).item()
        else:
            last_value = 0.0
        
        loss = agent.update(last_value)
        losses.append(loss)         
        print(f"Update: total_steps={total_steps}, loss={loss}")
        
        avg_ep_rewards = np.mean(ep_rewards)
        rewards.append(avg_ep_rewards)
        print(f"Episode {episode}, Total Reward: {avg_ep_rewards}, Total Steps: {total_steps}")

    return rewards, losses


if __name__ == "__main__":
  env = gym.make("ALE/Pong-v5")
  device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
  )

  seed = 0
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  MODEL_PATH = "./model_weights/PPO/pong"
  PLOT_PATH = "./images/PPO"

  state_shape = (4, 80, 80)
  n_actions = env.action_space.n

  agent = PPOAgent(state_shape, n_actions, device)

  os.makedirs(MODEL_PATH, exist_ok=True)
  os.makedirs(PLOT_PATH, exist_ok=True)

  if os.path.exists(f"{MODEL_PATH}/pong_ppo_episode_end.pth"):
    agent.policy.load_state_dict(torch.load(f"{MODEL_PATH}/pong_ppo_episode_end.pth"))
    rewards = np.load(f"{MODEL_PATH}/rewards.npy")
    losses = np.load(f"{MODEL_PATH}/losses.npy")
  else:
    rewards, losses = train(agent, env, num_episodes=5000)
    torch.save(agent.policy.state_dict(), f"{MODEL_PATH}/pong_ppo_episode_end.pth")
    np.save(f"{MODEL_PATH}/rewards.npy", np.array(rewards))
    np.save(f"{MODEL_PATH}/losses.npy", np.array(losses))

  window = 20
  rolling_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')

  plt.figure(figsize=(10, 5))
  plt.plot(rewards, label="Total Reward per Episode", alpha=0.4)
  plt.plot(np.arange(window - 1, len(rewards)), rolling_avg, label=f"{window}-Episode Rolling Avg", color="red")
  plt.xlabel("Episode")
  plt.ylabel("Total Reward")
  plt.title("Pong PPO Training Progress")
  plt.legend()
  plt.savefig(f"{PLOT_PATH}/PPO_pong.pdf", bbox_inches="tight", format="pdf")
