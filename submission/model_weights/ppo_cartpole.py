import os
import random
from collections import deque

import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class PPO(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PPO, self).__init__()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        self.policy_head = nn.Linear(64, act_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.linear_tanh_stack(x)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


class PPOBuffer:
    def __init__(self, size, gamma=0.99, lam=0.95):
        self.max_size = size
        self.gamma = gamma
        self.lam = lam
        self.obs_buf = []
        self.act_buf = []
        self.log_prob_buf = []
        self.ret_buf = []
        self.don_buf = []
        self.val_buf = []

    def __len__(self):
        return len(self.obs_buf)

    def reset(self):
        self.obs_buf = []
        self.act_buf = []
        self.log_prob_buf = []
        self.ret_buf = []
        self.don_buf = []
        self.val_buf = []

    def store(self, state, action, log_prob, reward, done, value):
        if len(self.obs_buf) >= self.max_size:
            raise RuntimeError("PPOBuffer Overflow: Increase buffer_size or call update more often")

        self.obs_buf.append(np.array(state, copy=False))
        self.act_buf.append(int(action))
        self.log_prob_buf.append(float(log_prob))
        self.ret_buf.append(float(reward))
        self.don_buf.append(float(done))
        self.val_buf.append(float(value))

    def get(self, last_value=0.0):
        values = self.val_buf + [float(last_value)]
        returns = []
        advantages = []
        advantage = 0.0

        for i in range(len(self.obs_buf) - 1, -1, -1):
            delta = self.ret_buf[i] + self.gamma * values[i + 1] * (1.0 - self.don_buf[i]) - values[i]
            advantage = delta + self.gamma * self.lam * (1.0 - self.don_buf[i]) * advantage
            advantages.insert(0, advantage)
            returns.insert(0, advantage + self.val_buf[i])

        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.tensor(np.array(self.obs_buf, dtype=np.float32), dtype=torch.float32)
        actions = torch.tensor(self.act_buf, dtype=torch.long)
        log_probs = torch.tensor(self.log_prob_buf, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        return states, actions, log_probs, returns, advantages


class PPOAgent:
    def __init__(self, obs_dim, act_dim, device="cpu", gamma=0.99, clip_ratio=0.1, lr=2.5e-4, lam=0.95, minibatch_size=256, epochs=4, buffer_size=2048):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.policy = PPO(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = PPOBuffer(buffer_size, gamma, lam)
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.minibatch_size = minibatch_size
        self.epochs = epochs

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.policy(state_tensor)
            m = Categorical(logits=logits)
            action = m.sample()
            log_prob = m.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def select_greedy_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, value = self.policy(state)
        action = torch.argmax(logits, dim=-1)

        return action.item()

    def update(self, last_value=0.0):
        states, actions, log_probs, returns, advantages = self.buffer.get(last_value)

        states = states.to(self.device)
        actions = actions.to(self.device)
        log_probs = log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        total_loss = 0.0
        batch_size = states.size(0)
        num_batches = 0

        for _ in range(self.epochs):
            indices = torch.randperm(batch_size, device=self.device)
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

                prob_ratio = torch.exp(new_log_probs - batch_log_probs)
                val1 = prob_ratio * batch_advantages
                val2 = torch.clamp(
                    prob_ratio,
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio,
                ) * batch_advantages
                policy_loss = -torch.min(val1, val2).mean()

                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                loss = policy_loss + value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_loss += float(loss.item())
                num_batches += 1

        self.buffer.reset()

        return total_loss / max(num_batches, 1)


def play_game(agent, env, episodes=1):
    rewards = []
    with torch.no_grad():
        for t in range(episodes):
            state, _ = env.reset()
            total_reward = 0.0
            done = False
            truncated = False

            while not (done or truncated):
                action = agent.select_greedy_action(state)
                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward

            rewards.append(total_reward)
    return rewards


def train(agent, env, num_episodes=10000):
    rewards = []
    eval_rewards = []

    total_steps = 0
    buffer_max = agent.buffer.max_size

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        ep_rewards = []
        last_loss = 0.0

        while not (done or truncated):
            action, log_prob, value = agent.select_action(state)

            step_out = env.step(action)
            next_state, reward, done, truncated, _ = step_out
            terminated_flag = done or truncated

            ep_rewards.append(reward)
            total_steps += 1

            agent.buffer.store(state, action, log_prob, reward, float(terminated_flag), value)

            state = next_state

            if len(agent.buffer) >= buffer_max:
                with torch.no_grad():
                    if terminated_flag:
                        last_value = 0.0
                    else:
                        state_tensor = torch.tensor(state,dtype=torch.float32,device=agent.device,).unsqueeze(0)
                        logits, value_tensor = agent.policy(state_tensor)
                        last_value = float(value_tensor.item())

                last_loss = agent.update(last_value)

        avg_ep_reward = float(np.sum(ep_rewards))
        rewards.append(avg_ep_reward)

        if len(agent.buffer) > 0:
            last_loss = agent.update(last_value=0.0)

        eval_return = play_game(agent, env, episodes=1)[0]
        eval_rewards.append(eval_return)

        print(f"Episode {episode} | Train Return: {avg_ep_reward:.2f} | Eval Return: {eval_return:.2f} | Loss: {last_loss:.4f} | Total Steps: {total_steps}")

    return agent, rewards, eval_rewards


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    MODEL_PATH = "./model_weights/PPO/cartpole"
    PLOT_PATH = "./images/PPO"

    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PPOAgent(n_observations, n_actions, device=device)

    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(PLOT_PATH, exist_ok=True)

    eval_policy_rewards_path = os.path.join(MODEL_PATH, "eval_rewards.npy")
    final_weights_path = os.path.join(MODEL_PATH, "cartpole_ppo_final.pth")

    num_episodes = 5000

    if os.path.exists(final_weights_path) and os.path.exists(eval_policy_rewards_path):
        agent.policy.load_state_dict(torch.load(final_weights_path, map_location=device))
        eval_rewards = np.load(eval_policy_rewards_path)
    else:
        agent, _, eval_rewards = train(agent, env, num_episodes=num_episodes)
        torch.save(agent.policy.state_dict(), final_weights_path)
        np.save(eval_policy_rewards_path, np.array(eval_rewards))

    window = 50
    plt.figure(figsize=(10, 5))
    plt.plot(eval_rewards, label="Total Reward per Episode", alpha=0.4)

    if len(eval_rewards) >= window:
        rolling_avg = np.convolve(eval_rewards, np.ones(window) / window, mode="valid")
        rolling_x = np.arange(window - 1, len(eval_rewards))
        plt.plot(rolling_x, rolling_avg, label=f"{window}-Episode Rolling Avg", color="red")

    plt.xlabel("Episode")
    plt.ylabel("Eval Reward")
    plt.title("CartPole PPO Training Progress (Eval Rewards)")
    plt.legend()
    plt.savefig(f"{PLOT_PATH}/PPO_cartpole.pdf", bbox_inches="tight", format="pdf")
