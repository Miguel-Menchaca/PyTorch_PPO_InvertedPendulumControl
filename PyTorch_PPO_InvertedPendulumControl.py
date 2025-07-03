import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym

# ===== Neural Network Architecture =====
class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network with shared feature extractor
    - Actor: Policy head with probability distribution over actions
    - Critic: Value estimator for state valuation
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)

# ===== PPO Core Implementation =====
class PPOAgent:
    """Proximal Policy Optimization agent with clipping and advantage normalization"""
    def __init__(self, env_name, lr=3e-4, gamma=0.99, clip_eps=0.2, k_epochs=4):
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        self.policy = ActorCritic(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.memory = []

    def select_action(self, state):
        """Sample action from policy with exploration"""
        state = torch.FloatTensor(state)
        logits, value = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def update(self):
        """PPO optimization with clipped surrogate objective"""
        # Unpack memory
        states, actions, log_probs, returns, advantages = zip(*self.memory)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple epochs of updates
        for _ in range(self.k_epochs):
            current_logits, state_values = self.policy(states)
            dist = Categorical(logits=current_logits)
            current_log_probs = dist.log_prob(actions)
            
            # Importance ratio
            ratios = torch.exp(current_log_probs - old_log_probs.detach())
            
            # Clipped surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss (MSE)
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)
            
            # Total loss with entropy bonus
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        self.memory = []  # Clear memory after update

    def compute_returns(self, rewards, dones, last_value):
        """Compute discounted returns with bootstrapping"""
        returns = []
        R = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return returns

    def train(self, episodes=1000, max_steps=200):
        """Main training loop with experience collection"""
        episode_rewards = []
        
        for ep in range(episodes):
            state = self.env.reset()
            ep_reward = 0
            step_data = []
            
            for t in range(max_steps):
                action, log_prob, value = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Store transition
                step_data.append((state, action, log_prob, reward, done, value))
                state = next_state
                ep_reward += reward
                
                if done:
                    break
            
            # Post-episode processing
            _, _, last_value = self.select_action(state)
            rewards = [x[3] for x in step_data]
            dones = [x[4] for x in step_data]
            returns = self.compute_returns(rewards, dones, last_value)
            
            # Calculate advantages
            values = [x[5] for x in step_data]
            advantages = [r - v for r, v in zip(returns, values)]
            
            # Store in memory
            for i, (s, a, lp, _, _, _) in enumerate(step_data):
                self.memory.append((s, a, lp, returns[i], advantages[i]))
            
            # Update policy
            self.update()
            
            episode_rewards.append(ep_reward)
            print(f"Episode {ep+1}/{episodes} | Reward: {ep_reward:.1f}")
        
        return episode_rewards

# ===== Execution =====
if __name__ == "__main__":
    agent = PPOAgent("CartPole-v1")
    rewards = agent.train(episodes=500)
    torch.save(agent.policy.state_dict(), "ppo_cartpole.pth")
