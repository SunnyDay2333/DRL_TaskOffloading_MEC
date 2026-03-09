"""
Baseline DRL 算法实现
=====================
包含用于对比实验的标准 DRL 基线算法:
- SAC (Soft Actor-Critic): 随机策略 + 熵正则化
- TD3 (Twin Delayed DDPG): 确定性策略 + 延迟更新 + 目标平滑
- DDPG (Deep Deterministic Policy Gradient): 确定性策略

所有 baseline 使用相同的网络容量 (256, 256) 以保证公平对比。
动作空间为 [0,1] (通过 sigmoid 映射), 与 FEAT 一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
import numpy as np
import copy
from typing import Dict, Tuple, Optional

from models.networks import init_weights


# ==================== 共用网络组件 ====================

class MLPActor(nn.Module):
    """
    随机策略网络 (用于 SAC)

    输出高斯分布参数 (mean, log_std), 通过 sigmoid 映射到 [0,1]
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 256),
                 log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        init_weights(self.mean_head, gain=0.01)
        init_weights(self.log_std_head, gain=0.01)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor,
               deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        if deterministic:
            action = torch.sigmoid(mean)
            log_prob = torch.zeros(state.shape[0], device=state.device)
            return action, log_prob

        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()
        action = torch.sigmoid(x)
        log_prob = normal.log_prob(x) - torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob


class DeterministicActor(nn.Module):
    """
    确定性策略网络 (用于 TD3 / DDPG)

    直接输出动作, 通过 sigmoid 映射到 [0,1]
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        init_weights(self.network[-1], gain=0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(state))


class TwinQNetwork(nn.Module):
    """双 Q 网络, 用于减少过估计"""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        super().__init__()
        input_dim = state_dim + action_dim

        def build_q():
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU()])
                prev = h
            layers.append(nn.Linear(prev, 1))
            return nn.Sequential(*layers)

        self.q1 = build_q()
        self.q2 = build_q()
        self._init_weights()

    def _init_weights(self):
        for net in [self.q1, self.q2]:
            for m in net:
                if isinstance(m, nn.Linear):
                    init_weights(m, gain=np.sqrt(2))
            init_weights(net[-1], gain=1.0)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state: torch.Tensor,
                   action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


# ==================== SAC Baseline ====================

class SACAgent:
    """
    Soft Actor-Critic 基线
    =======================
    标准单策略 SAC, 无 FEAT 的多策略结构。
    与 FEAT 共享相同的 SAC 核心 (自动熵调整, Twin Q, 软更新)。
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 256),
                 actor_lr: float = 3e-4, critic_lr: float = 3e-4,
                 alpha_lr: float = 3e-4, gamma: float = 0.95,
                 tau: float = 0.005, alpha: float = 0.2,
                 automatic_entropy_tuning: bool = True,
                 target_entropy: Optional[float] = None,
                 device: str = 'cuda'):
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.actor = MLPActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        self.automatic_entropy_tuning = automatic_entropy_tuning
        if automatic_entropy_tuning:
            self.target_entropy = target_entropy if target_entropy is not None else -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state_t, deterministic)
            return action.cpu().numpy()[0]

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)

        # ---- Critic ----
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            tq1, tq2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(tq1, tq2) - self.alpha * next_log_probs.unsqueeze(1)
            target_value = rewards + (1 - dones) * self.gamma * target_q

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---- Actor ----
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs.unsqueeze(1) - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ---- Alpha ----
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha.exp() *
                           (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ---- Target soft update ----
        for tp, sp in zip(self.target_critic.parameters(),
                          self.critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'q_value': q_new.mean().item(),
            'log_prob': log_probs.mean().item(),
        }

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.target_critic.load_state_dict(ckpt['target_critic'])


# ==================== TD3 Baseline ====================

class TD3Agent:
    """
    Twin Delayed DDPG 基线
    =======================
    确定性策略 + 延迟策略更新 + 目标策略平滑
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 256),
                 actor_lr: float = 3e-4, critic_lr: float = 3e-4,
                 gamma: float = 0.95, tau: float = 0.005,
                 exploration_noise: float = 0.1,
                 policy_noise: float = 0.2, noise_clip: float = 0.5,
                 policy_delay: int = 2, device: str = 'cuda'):
        self.gamma = gamma
        self.tau = tau
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.actor = DeterministicActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.update_count = 0

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_t).cpu().numpy()[0]
        if not deterministic:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, 0.0, 1.0)
        return action

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)

        self.update_count += 1

        # ---- Critic (with target policy smoothing) ----
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor(next_states) + noise).clamp(0.0, 1.0)
            tq1, tq2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(tq1, tq2)
            target_value = rewards + (1 - dones) * self.gamma * target_q

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_loss_val = 0.0

        # ---- Actor (delayed update) ----
        if self.update_count % self.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(
                states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            for tp, sp in zip(self.target_critic.parameters(),
                              self.critic.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
            for tp, sp in zip(self.target_actor.parameters(),
                              self.actor.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss_val,
            'q_value': q1.mean().item(),
        }

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.target_actor.load_state_dict(ckpt['target_actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.target_critic.load_state_dict(ckpt['target_critic'])


# ==================== DDPG Baseline ====================

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient 基线
    =========================================
    最基础的确定性策略梯度方法, 使用 Twin Q 减少过估计。
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 256),
                 actor_lr: float = 1e-4, critic_lr: float = 3e-4,
                 gamma: float = 0.95, tau: float = 0.005,
                 exploration_noise: float = 0.1, device: str = 'cuda'):
        self.gamma = gamma
        self.tau = tau
        self.exploration_noise = exploration_noise
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.actor = DeterministicActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_t).cpu().numpy()[0]
        if not deterministic:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, 0.0, 1.0)
        return action

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)

        # ---- Critic ----
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            tq1, tq2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(tq1, tq2)
            target_value = rewards + (1 - dones) * self.gamma * target_q

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---- Actor ----
        actor_loss = -self.critic.q1_forward(
            states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ---- Target soft update ----
        for tp, sp in zip(self.target_critic.parameters(),
                          self.critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        for tp, sp in zip(self.target_actor.parameters(),
                          self.actor.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': q1.mean().item(),
        }

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.target_actor.load_state_dict(ckpt['target_actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.target_critic.load_state_dict(ckpt['target_critic'])
