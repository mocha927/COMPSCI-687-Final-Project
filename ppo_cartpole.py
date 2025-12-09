import gymnasium as gym
import matplotlib.pyplot as plt
import random
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class PPO(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(PPO, self).__init__()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        
        self.policy_head = nn.Linear(64, n_actions)
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
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

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
    def __init__(self, state_shape, n_actions, device="cpu", lr=3e-4, gamma=0.99, lam=0.95, eps=0.2, minibatch_size=64, num_epochs=10, buffer_size=4096):
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
        logits, value = self.policy(state)
        m = Categorical(logits=logits)
        action = m.sample()
        log_prob = m.log_prob(action)

        return action.item(), log_prob, value.item()

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

    for episode in range(num_episodes):
        total_steps = 0
        ep_rewards = []

        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        ep_len = 0

        while not (done or truncated):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.buffer.store(state, action, log_prob, reward, done or truncated, value)

            state = next_state
            total_reward += reward
            ep_len += 1
            total_steps += 1

        ep_rewards.append(total_reward)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        logits, last_value = agent.policy(state_tensor)
        last_value = last_value.item()

        loss = agent.update(last_value)

        avg_ep_reward = np.mean(ep_rewards)
        rewards.append(avg_ep_reward)

        env.reset()
        eval_rewards.append(play_game(agent, env, episodes=1)[0])

        print(
            f"Episode {episode} | "
            f"Train Avg Return: {avg_ep_reward} | "
            f"Eval Return: {eval_rewards[-1]} | "
            f"Loss: {loss}"
        )

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

    eps_policy_rewards_path = os.path.join(MODEL_PATH, "train_rewards.npy")
    eval_policy_rewards_path = os.path.join(MODEL_PATH, "eval_rewards.npy")
    final_weights_path = os.path.join(MODEL_PATH, "cartpole_ppo_final.pth")

    num_episodes = 1000

    if os.path.exists(final_weights_path) and os.path.exists(eps_policy_rewards_path):
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
