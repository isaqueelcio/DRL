import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, MADDPGCritic

BUFFER_SIZE = int(1e6)   # replay buffer size
BATCH_SIZE = 256          # minibatch size
GAMMA = 0.99              # discount factor
TAU = 1e-3                # soft update rate (reduced from 1e-2)
LR_ACTOR = 1e-4           # actor learning rate
LR_CRITIC = 3e-4          # critic learning rate
WEIGHT_DECAY = 1e-5       # L2 weight decay
UPDATES_PER_STEP = 2      # gradient updates per environment step (reduced from 4)
NOISE_DECAY = 0.9995      # multiplicative noise decay per episode
NOISE_MIN = 0.01          # minimum noise scale

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Shared buffer storing experiences for all agents together."""

    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["states", "actions", "rewards", "next_states", "dones"]
        )
        random.seed(seed)

    def add(self, states, actions, rewards, next_states, dones):
        """
        states, actions, next_states : (num_agents, size)
        rewards, dones               : (num_agents,)
        """
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        # Each tensor: (batch, num_agents, dim)
        states = torch.from_numpy(
            np.array([e.states for e in experiences])
        ).float().to(device)
        actions = torch.from_numpy(
            np.array([e.actions for e in experiences])
        ).float().to(device)
        rewards = torch.from_numpy(
            np.array([e.rewards for e in experiences])
        ).float().to(device)
        next_states = torch.from_numpy(
            np.array([e.next_states for e in experiences])
        ).float().to(device)
        dones = torch.from_numpy(
            np.array([e.dones for e in experiences]).astype(np.uint8)
        ).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class MADDPGAgent:
    """
    Multi-Agent DDPG.

    Each agent has its own Actor (decentralized execution) and a centralized
    Critic that receives the concatenated observations and actions of ALL agents.
    All agents share a single replay buffer.
    """

    def __init__(self, state_size, action_size, num_agents, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        full_state_size = state_size * num_agents    # 24 * 2 = 48
        full_action_size = action_size * num_agents  # 2  * 2 = 4

        # --- Per-agent networks ---
        self.actors_local = []
        self.actors_target = []
        self.actor_optimizers = []
        self.critics_local = []
        self.critics_target = []
        self.critic_optimizers = []
        self.noises = []

        for i in range(num_agents):
            seed_i = random_seed + i

            actor_local = Actor(state_size, action_size, seed_i).to(device)
            actor_target = Actor(state_size, action_size, seed_i).to(device)
            actor_target.load_state_dict(actor_local.state_dict())

            critic_local = MADDPGCritic(full_state_size, full_action_size, seed_i).to(device)
            critic_target = MADDPGCritic(full_state_size, full_action_size, seed_i).to(device)
            critic_target.load_state_dict(critic_local.state_dict())

            self.actors_local.append(actor_local)
            self.actors_target.append(actor_target)
            self.actor_optimizers.append(optim.Adam(actor_local.parameters(), lr=LR_ACTOR))

            self.critics_local.append(critic_local)
            self.critics_target.append(critic_target)
            self.critic_optimizers.append(
                optim.Adam(critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
            )

            self.noises.append(OUNoise(action_size, seed_i))

        # Shared replay buffer
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Noise scale (decays each episode)
        self.noise_scale = 1.0

    # ------------------------------------------------------------------
    def reset(self):
        for noise in self.noises:
            noise.reset()

    def decay_noise(self):
        """Call once per episode to decay exploration noise."""
        self.noise_scale = max(NOISE_MIN, self.noise_scale * NOISE_DECAY)

    # ------------------------------------------------------------------
    def act(self, states, add_noise=True):
        """
        states : np.ndarray (num_agents, state_size)
        returns: np.ndarray (num_agents, action_size)
        """
        actions = []
        for i, actor in enumerate(self.actors_local):
            state_t = torch.from_numpy(states[i]).float().unsqueeze(0).to(device)
            actor.eval()
            with torch.no_grad():
                action = actor(state_t).cpu().data.numpy().squeeze()
            actor.train()
            if add_noise:
                action += self.noise_scale * self.noises[i].sample()
            actions.append(np.clip(action, -1, 1))
        return np.array(actions)

    # ------------------------------------------------------------------
    def step(self, states, actions, rewards, next_states, dones):
        """Store experience and (possibly) learn."""
        self.memory.add(states, actions, rewards, next_states, dones)

        if len(self.memory) > BATCH_SIZE:
            for _ in range(UPDATES_PER_STEP):
                for agent_id in range(self.num_agents):
                    experiences = self.memory.sample()
                    self._learn(experiences, agent_id)

    # ------------------------------------------------------------------
    def _learn(self, experiences, agent_id):
        """Update actor and critic for a single agent using centralized training."""
        states, actions, rewards, next_states, dones = experiences
        # shapes: (batch, num_agents, dim)

        batch_size = states.shape[0]

        # Flattened views for the centralized critic
        states_all = states.view(batch_size, -1)           # (batch, num_agents*state_size)
        actions_all = actions.view(batch_size, -1)         # (batch, num_agents*action_size)
        next_states_all = next_states.view(batch_size, -1)

        # ---- Compute next actions from target actors ----
        with torch.no_grad():
            next_actions = [
                self.actors_target[i](next_states[:, i, :])
                for i in range(self.num_agents)
            ]
            next_actions_all = torch.cat(next_actions, dim=1)  # (batch, full_action_size)

        # ---- Update Critic ----
        Q_targets_next = self.critics_target[agent_id](next_states_all, next_actions_all)
        reward_i = rewards[:, agent_id].unsqueeze(1)
        done_i = dones[:, agent_id].unsqueeze(1)
        Q_targets = reward_i + GAMMA * Q_targets_next * (1 - done_i)

        Q_expected = self.critics_local[agent_id](states_all, actions_all)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizers[agent_id].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics_local[agent_id].parameters(), 1)
        self.critic_optimizers[agent_id].step()

        # ---- Update Actor ----
        # Only agent_id's actor is differentiable; others are detached
        pred_actions = []
        for i in range(self.num_agents):
            if i == agent_id:
                pred_actions.append(self.actors_local[i](states[:, i, :]))
            else:
                pred_actions.append(self.actors_local[i](states[:, i, :]).detach())
        pred_actions_all = torch.cat(pred_actions, dim=1)

        actor_loss = -self.critics_local[agent_id](states_all, pred_actions_all).mean()

        self.actor_optimizers[agent_id].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors_local[agent_id].parameters(), 1)
        self.actor_optimizers[agent_id].step()

        # ---- Soft update target networks ----
        self._soft_update(self.critics_local[agent_id], self.critics_target[agent_id], TAU)
        self._soft_update(self.actors_local[agent_id], self.actors_target[agent_id], TAU)

    # ------------------------------------------------------------------
    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    # ------------------------------------------------------------------
    def save(self, prefix='checkpoint_maddpg'):
        for i in range(self.num_agents):
            torch.save(self.actors_local[i].state_dict(), f'{prefix}_actor_{i}.pth')
            torch.save(self.critics_local[i].state_dict(), f'{prefix}_critic_{i}.pth')
