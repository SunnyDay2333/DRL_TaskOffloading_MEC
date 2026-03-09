"""
FEAT 智能体实现
=================
本文件实现了完整的 FEAT (Fast Environment-Adaptive Task offloading) 算法

核心训练策略 (论文 Table II 第5组设置):
- Steerer 更新: CMQ (Current Max Q) - 选择当前 Q 值最高的策略
- Meta-Policy 更新: HSD (Hard Selection with Detachment) - 只更新被选择的策略
- Q 网络更新: CMQ - 使用最大 Q 值计算目标

参考论文: FEAT: Towards Fast Environment-Adaptive Task Offloading 
         and Power Allocation in MEC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from typing import Dict, Tuple, Optional, List
import copy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.networks import MetaPolicyNetwork, SteererNetwork, QNetwork
from utils.replay_buffer import ReplayBuffer
from utils.helpers import soft_update, hard_update


class FEATAgent:
    """
    FEAT 智能体
    ============
    实现了完整的 FEAT 算法，包括:
    - Meta-Policies: K 个共享底层的策略网络
    - Steerer: 策略选择器
    - Q-Network: 双 Q 网络价值评估
    
    训练策略:
    - Steerer: 使用 CMQ 损失训练，学习选择当前 Q 值最高的策略
    - Meta-Policy: 使用 HSD 更新，只更新实际被选择的策略
    - Q-Network: 使用 CMQ 计算目标 Q 值
    
    Attributes:
        meta_policy: 元策略网络
        steerer: 策略选择器网络
        q_network: Q 网络
        target_q_network: 目标 Q 网络
    """
    
    def __init__(self,
                 env_state_dim: int,
                 time_varying_state_dim: int,
                 action_dim: int,
                 num_policies: int = 3,
                 shared_hidden_dims: Tuple[int, ...] = (256, 256),
                 policy_hidden_dim: int = 128,
                 steerer_hidden_dims: Tuple[int, ...] = (128, 64),
                 q_hidden_dims: Tuple[int, ...] = (256, 256),
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 steerer_lr: float = 3e-4,
                 alpha_lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 automatic_entropy_tuning: bool = True,
                 target_entropy: Optional[float] = None,
                 gumbel_temperature: float = 1.0,
                 temperature_decay: float = 0.9995,
                 # ===== 策略多样性参数 (方案 A+B) =====
                 steerer_entropy_coeff: float = 0.5,
                 forced_exploration_prob: float = 0.2,
                 forced_exploration_decay: float = 0.9995,
                 forced_exploration_min: float = 0.02,
                 device: str = 'cuda'):
        """
        初始化 FEAT 智能体
        
        Args:
            env_state_dim: 环境状态维度 (Steerer 输入)
            time_varying_state_dim: 时变状态维度 (Meta-Policy 输入)
            action_dim: 动作维度
            num_policies: 元策略数量 K
            shared_hidden_dims: 共享层维度
            policy_hidden_dim: 策略头隐藏层维度
            steerer_hidden_dims: Steerer 隐藏层维度
            q_hidden_dims: Q 网络隐藏层维度
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            steerer_lr: Steerer 学习率
            alpha_lr: 熵系数学习率
            gamma: 折扣因子
            tau: 目标网络软更新系数
            alpha: SAC 熵正则化系数
            automatic_entropy_tuning: 是否自动调整熵系数
            target_entropy: 目标熵
            gumbel_temperature: Gumbel-Softmax 初始温度
            temperature_decay: 温度衰减率
            steerer_entropy_coeff: [方案A] Steerer 熵正则化系数
            forced_exploration_prob: [方案B] 强制探索概率
            forced_exploration_decay: [方案B] 探索概率衰减率
            forced_exploration_min: [方案B] 最小探索概率
            device: 计算设备
        """
        # 保存参数
        self.env_state_dim = env_state_dim
        self.time_varying_state_dim = time_varying_state_dim
        self.action_dim = action_dim
        self.num_policies = num_policies
        self.gamma = gamma
        self.tau = tau
        self.temperature_decay = temperature_decay
        
        # ===== 策略多样性参数 =====
        self.steerer_entropy_coeff = steerer_entropy_coeff        # 方案A: 熵正则化系数
        self.forced_exploration_prob = forced_exploration_prob      # 方案B: 当前强制探索概率
        self.forced_exploration_decay = forced_exploration_decay    # 方案B: 衰减率
        self.forced_exploration_min = forced_exploration_min        # 方案B: 最小探索概率
        
        # 设置设备
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"FEAT 智能体使用设备: {self.device}")
        
        # ==================== 创建网络 ====================
        
        # Meta-Policy 网络 (包含 K 个策略)
        self.meta_policy = MetaPolicyNetwork(
            state_dim=time_varying_state_dim,
            action_dim=action_dim,
            num_policies=num_policies,
            shared_hidden_dims=shared_hidden_dims,
            policy_hidden_dim=policy_hidden_dim
        ).to(self.device)
        
        # Steerer 网络
        feature_dim = shared_hidden_dims[-1]  # 共享层输出维度
        self.steerer = SteererNetwork(
            env_state_dim=env_state_dim,
            feature_dim=feature_dim,
            num_policies=num_policies,
            hidden_dims=steerer_hidden_dims,
            initial_temperature=gumbel_temperature
        ).to(self.device)
        
        # Q 网络 (使用完整状态: env_state + time_varying_state)
        total_state_dim = env_state_dim + time_varying_state_dim
        self.q_network = QNetwork(
            state_dim=total_state_dim,
            action_dim=action_dim,
            hidden_dims=q_hidden_dims
        ).to(self.device)
        
        # 目标 Q 网络
        self.target_q_network = copy.deepcopy(self.q_network)
        
        # 冻结目标网络参数
        for param in self.target_q_network.parameters():
            param.requires_grad = False
            
        # ==================== 创建优化器 ====================
        
        # Actor 优化器 (Meta-Policy 的所有参数)
        self.actor_optimizer = Adam(
            self.meta_policy.parameters(), 
            lr=actor_lr
        )
        
        # Critic 优化器
        self.critic_optimizer = Adam(
            self.q_network.parameters(),
            lr=critic_lr
        )
        
        # Steerer 优化器
        self.steerer_optimizer = Adam(
            self.steerer.parameters(),
            lr=steerer_lr
        )
        
        # ==================== SAC 熵调整 ====================
        
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        if automatic_entropy_tuning:
            # 目标熵 (通常设为 -dim(A))
            if target_entropy is None:
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy
                
            # 可学习的 log(α)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
            
        # ==================== 训练统计 ====================
        
        self.training_step = 0
        self.update_count = 0
        
    def select_action(self,
                     env_state: np.ndarray,
                     time_varying_state: np.ndarray,
                     deterministic: bool = False,
                     return_info: bool = False) -> Tuple[np.ndarray, int]:
        """
        选择动作
        
        流程:
        1. Meta-Policy 提取特征
        2. Steerer 选择策略 (训练时有强制探索)
        3. 被选策略输出动作
        
        [方案B] 训练时以概率 ε 随机选择策略，保证所有策略都能被训练到
        
        Args:
            env_state: 环境状态
            time_varying_state: 时变状态
            deterministic: 是否使用确定性策略
            return_info: 是否返回额外信息
            
        Returns:
            action: 选择的动作
            selected_k: 选择的策略索引
        """
        with torch.no_grad():
            # 转换为张量
            env_state_t = torch.FloatTensor(env_state).unsqueeze(0).to(self.device)
            time_varying_state_t = torch.FloatTensor(time_varying_state).unsqueeze(0).to(self.device)
            
            # 获取共享特征
            features = self.meta_policy.get_features(time_varying_state_t)
            
            # Steerer 选择策略
            weights, selected_idx = self.steerer(
                env_state_t, features, 
                training=False  # 推理时使用 Softmax
            )
            selected_k = selected_idx.item()
            
            # [方案B] 强制探索: 训练(非确定性)模式下, 以概率 ε 随机选择策略
            # 确保所有策略都能获得训练数据, 打破马太效应
            if not deterministic and np.random.random() < self.forced_exploration_prob:
                selected_k = np.random.randint(0, self.num_policies)
            
            # 从选定策略采样动作
            action, log_prob = self.meta_policy.sample_action(
                time_varying_state_t, 
                selected_k, 
                deterministic=deterministic
            )
            
            action = action.cpu().numpy()[0]
            
            if return_info:
                info = {
                    'weights': weights.cpu().numpy()[0],
                    'log_prob': log_prob.item(),
                    'features': features.cpu().numpy()[0]
                }
                return action, selected_k, info
                
            return action, selected_k
    
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        执行一次参数更新
        
        实现论文中的训练策略:
        - Q 网络: CMQ 目标值计算
        - Steerer: CMQ 损失
        - Meta-Policy: HSD 更新
        
        Args:
            batch: 经验批次
            
        Returns:
            训练指标字典
        """
        # 转换为张量
        env_states = torch.FloatTensor(batch['env_states']).to(self.device)
        time_varying_states = torch.FloatTensor(batch['time_varying_states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
        next_env_states = torch.FloatTensor(batch['next_env_states']).to(self.device)
        next_time_varying_states = torch.FloatTensor(batch['next_time_varying_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)
        selected_ks = torch.LongTensor(batch['selected_ks']).to(self.device)
        
        # 拼接完整状态
        full_states = torch.cat([env_states, time_varying_states], dim=1)
        next_full_states = torch.cat([next_env_states, next_time_varying_states], dim=1)
        
        # ==================== 更新 Q 网络 ====================
        
        critic_loss, q_values = self._update_critic(
            full_states, actions, rewards, next_full_states, 
            next_env_states, next_time_varying_states, dones
        )
        
        # ==================== 更新 Steerer (CMQ 损失) ====================
        
        steerer_loss, max_q_indices = self._update_steerer(
            env_states, time_varying_states, full_states
        )
        
        # ==================== 更新 Meta-Policy (HSD 更新) ====================
        
        actor_loss, alpha_loss, log_probs = self._update_actor(
            env_states, time_varying_states, full_states, selected_ks
        )
        
        # ==================== 更新目标网络 ====================
        
        soft_update(self.target_q_network, self.q_network, self.tau)
        
        # ==================== 更新 Steerer 温度 ====================
        
        self.steerer.update_temperature(self.temperature_decay)
        
        # ==================== [方案B] 更新强制探索概率 ====================
        self.forced_exploration_prob = max(
            self.forced_exploration_min,
            self.forced_exploration_prob * self.forced_exploration_decay
        )
        
        self.update_count += 1
        
        # ==================== 计算策略多样性指标 ====================
        # 统计 max_q_indices 的分布 (Q 值最高的策略分布)
        q_policy_counts = torch.zeros(self.num_policies, device=self.device)
        for k in range(self.num_policies):
            q_policy_counts[k] = (max_q_indices == k).float().sum()
        q_policy_dist = q_policy_counts / q_policy_counts.sum()
        # 计算 Q 值策略分布的熵 (衡量 Q 值多样性)
        q_dist_entropy = -(q_policy_dist * (q_policy_dist + 1e-8).log()).sum().item()
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'steerer_loss': steerer_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha,
            'q_value': q_values.mean().item(),
            'log_prob': log_probs.mean().item() if log_probs is not None else 0.0,
            'steerer_temp': self.steerer.temperature,
            # [新增] 策略多样性指标
            'exploration_prob': self.forced_exploration_prob,
            'q_dist_entropy': q_dist_entropy,  # Q 值策略分布的熵 (越高越均匀)
        }
    
    def _update_critic(self,
                      full_states: torch.Tensor,
                      actions: torch.Tensor,
                      rewards: torch.Tensor,
                      next_full_states: torch.Tensor,
                      next_env_states: torch.Tensor,
                      next_time_varying_states: torch.Tensor,
                      dones: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        更新 Q 网络
        
        使用 CMQ 方法计算目标 Q 值:
        遍历所有 K 个策略，选择能让 Target Q 值最大的动作
        
        Args:
            full_states: 完整状态 (env + time_varying)
            actions: 执行的动作
            rewards: 奖励
            next_full_states: 下一完整状态
            next_env_states: 下一环境状态
            next_time_varying_states: 下一时变状态
            dones: 结束标志
            
        Returns:
            critic_loss: Critic 损失值
            q_values: 当前 Q 值
        """
        with torch.no_grad():
            # 从所有策略采样动作
            next_actions_list, next_log_probs_list = self.meta_policy.sample_all_actions(
                next_time_varying_states
            )
            
            # 计算每个策略的 Target Q 值
            target_q_values = []
            for k, (next_action, next_log_prob) in enumerate(zip(next_actions_list, next_log_probs_list)):
                target_q = self.target_q_network.q_min(next_full_states, next_action)
                # 减去熵项
                target_q = target_q - self.alpha * next_log_prob.unsqueeze(1)
                target_q_values.append(target_q)
                
            # 堆叠并取最大值 (CMQ: 选择最大 Q 值的动作)
            target_q_stack = torch.stack(target_q_values, dim=-1)  # (batch, 1, K)
            max_target_q = target_q_stack.max(dim=-1)[0]  # (batch, 1)
            
            # 计算 TD 目标
            td_target = rewards + (1 - dones) * self.gamma * max_target_q
            
        # 计算当前 Q 值
        q1, q2 = self.q_network(full_states, actions)
        
        # MSE 损失
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        
        # 更新
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.critic_optimizer.step()
        
        return critic_loss.item(), torch.min(q1, q2)
    
    def _update_steerer(self,
                       env_states: torch.Tensor,
                       time_varying_states: torch.Tensor,
                       full_states: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        更新 Steerer 网络
        
        使用 CMQ 损失 + [方案A] 熵正则化:
        Loss = CE(Steerer输出, OneHot(max_Q_policy)) - λ_entropy * H(softmax(logits))
        
        熵正则化阻止 Steerer 过早坍缩到单一策略，鼓励输出更均匀的分布
        
        Args:
            env_states: 环境状态
            time_varying_states: 时变状态
            full_states: 完整状态
            
        Returns:
            steerer_loss: Steerer 损失值
            max_q_indices: Q 值最大的策略索引
        """
        # 获取特征
        features = self.meta_policy.get_features(time_varying_states)
        
        # 从所有策略采样动作
        with torch.no_grad():
            actions_list, _ = self.meta_policy.sample_all_actions(time_varying_states)
            
            # 计算每个策略的 Q 值
            q_values = []
            for k, action in enumerate(actions_list):
                q = self.q_network.q_min(full_states, action)
                q_values.append(q)
                
            # 找到 Q 值最大的策略索引
            q_stack = torch.stack(q_values, dim=-1).squeeze(1)  # (batch, K)
            max_q_indices = q_stack.argmax(dim=-1)  # (batch,)
            
        # 计算 Steerer 输出
        steerer_logits = self.steerer.get_logits(env_states, features.detach())
        
        # ==================== 原始 CMQ 交叉熵损失 ====================
        ce_loss = F.cross_entropy(steerer_logits, max_q_indices)
        
        # ==================== [方案A] 熵正则化 ====================
        # H(p) = -Σ p_k * log(p_k)
        # 最大化熵 = 鼓励均匀分布 → 在损失中减去熵 (因为我们要最小化损失)
        steerer_probs = F.softmax(steerer_logits, dim=-1)
        steerer_log_probs = F.log_softmax(steerer_logits, dim=-1)
        steerer_entropy = -(steerer_probs * steerer_log_probs).sum(dim=-1).mean()
        
        # 总损失 = CE 损失 - λ_entropy * 熵
        # (减去熵 → 最小化损失时会最大化熵)
        steerer_loss = ce_loss - self.steerer_entropy_coeff * steerer_entropy
        
        # # [原始代码 - 无熵正则化]
        # steerer_loss = F.cross_entropy(steerer_logits, max_q_indices)
        
        # 更新
        self.steerer_optimizer.zero_grad()
        steerer_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.steerer.parameters(), 1.0)
        self.steerer_optimizer.step()
        
        return steerer_loss.item(), max_q_indices
    
    def _update_actor(self,
                     env_states: torch.Tensor,
                     time_varying_states: torch.Tensor,
                     full_states: torch.Tensor,
                     selected_ks: torch.Tensor) -> Tuple[float, float, Optional[torch.Tensor]]:
        """
        更新 Meta-Policy (Actor)
        
        使用 HSD (Hard Selection with Detachment) 更新:
        只更新实际被选择的策略，其他策略参数保持不变
        
        Args:
            env_states: 环境状态
            time_varying_states: 时变状态
            full_states: 完整状态
            selected_ks: 实际选择的策略索引
            
        Returns:
            actor_loss: Actor 损失值
            alpha_loss: Alpha 损失值 (如果启用自动调整)
            log_probs: 动作的 log 概率
        """
        # HSD 策略: 按策略分组更新
        batch_size = time_varying_states.shape[0]
        
        total_actor_loss = 0.0
        total_alpha_loss = 0.0
        all_log_probs = []
        
        # 对每个策略单独处理
        for k in range(self.num_policies):
            # 找到使用该策略的样本
            mask = (selected_ks == k)
            
            if mask.sum() == 0:
                continue
                
            # 获取该策略的样本
            k_time_varying_states = time_varying_states[mask]
            k_full_states = full_states[mask]
            
            # 从策略 k 采样动作
            action, log_prob = self.meta_policy.sample_action(k_time_varying_states, k)
            
            # 计算 Q 值
            q_value = self.q_network.q_min(k_full_states, action)
            
            # SAC Actor 损失
            actor_loss = (self.alpha * log_prob.unsqueeze(1) - q_value).mean()
            
            total_actor_loss += actor_loss * mask.float().mean()
            all_log_probs.append(log_prob)
            
            # 更新熵系数 (如果启用)
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha.exp() * 
                              (log_prob + self.target_entropy).detach()).mean()
                total_alpha_loss += alpha_loss * mask.float().mean()
                
        # 更新 Actor
        if total_actor_loss != 0:
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.meta_policy.parameters(), 1.0)
            self.actor_optimizer.step()
            
        # 更新 Alpha
        if self.automatic_entropy_tuning and total_alpha_loss != 0:
            self.alpha_optimizer.zero_grad()
            total_alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            
        # 合并 log_probs
        if all_log_probs:
            log_probs = torch.cat(all_log_probs)
        else:
            log_probs = None
            
        return (
            total_actor_loss.item() if isinstance(total_actor_loss, torch.Tensor) else 0.0,
            total_alpha_loss.item() if isinstance(total_alpha_loss, torch.Tensor) else 0.0,
            log_probs
        )
    
    def update_steerer_only(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        仅更新 Steerer (用于环境适应微调)
        
        在新环境中，只需要少量 Episode 微调 Steerer，
        而 Meta-Policy 和 Q 网络保持不变
        
        Args:
            batch: 经验批次
            
        Returns:
            训练指标
        """
        # 转换为张量
        env_states = torch.FloatTensor(batch['env_states']).to(self.device)
        time_varying_states = torch.FloatTensor(batch['time_varying_states']).to(self.device)
        full_states = torch.cat([env_states, time_varying_states], dim=1)
        
        # 只更新 Steerer
        steerer_loss, _ = self._update_steerer(
            env_states, time_varying_states, full_states
        )
        
        return {
            'steerer_loss': steerer_loss,
            'steerer_temp': self.steerer.temperature
        }
    
    def train(self):
        """设置为训练模式"""
        self.meta_policy.train()
        self.steerer.train()
        self.q_network.train()
        
    def eval(self):
        """设置为评估模式"""
        self.meta_policy.eval()
        self.steerer.eval()
        self.q_network.eval()
        
    def save(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'meta_policy': self.meta_policy.state_dict(),
            'steerer': self.steerer.state_dict(),
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'steerer_optimizer': self.steerer_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'steerer_temperature': self.steerer.temperature,
            'update_count': self.update_count
        }, path)
        
    def load(self, path: str, load_optimizer: bool = True):
        """
        加载模型
        
        Args:
            path: 模型路径
            load_optimizer: 是否加载优化器状态
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.meta_policy.load_state_dict(checkpoint['meta_policy'])
        self.steerer.load_state_dict(checkpoint['steerer'])
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network'])
        
        if load_optimizer:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.steerer_optimizer.load_state_dict(checkpoint['steerer_optimizer'])
            
            if checkpoint['log_alpha'] is not None and self.automatic_entropy_tuning:
                self.log_alpha.data.copy_(checkpoint['log_alpha'])
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
                self.alpha = self.log_alpha.exp().item()
                
        self.steerer.temperature = checkpoint['steerer_temperature']
        self.update_count = checkpoint['update_count']
        
    def get_policy_distribution(self,
                               env_state: np.ndarray,
                               time_varying_state: np.ndarray) -> np.ndarray:
        """
        获取策略选择的概率分布
        
        用于分析和可视化
        
        Args:
            env_state: 环境状态
            time_varying_state: 时变状态
            
        Returns:
            策略权重分布
        """
        with torch.no_grad():
            env_state_t = torch.FloatTensor(env_state).unsqueeze(0).to(self.device)
            time_varying_state_t = torch.FloatTensor(time_varying_state).unsqueeze(0).to(self.device)
            
            features = self.meta_policy.get_features(time_varying_state_t)
            weights, _ = self.steerer(env_state_t, features, training=False)
            
            return weights.cpu().numpy()[0]


if __name__ == "__main__":
    # 测试 FEAT 智能体
    print("=== FEAT 智能体测试 ===")
    
    # 参数设置
    env_state_dim = 5
    time_varying_state_dim = 49  # 12 * 4 + 1
    action_dim = 24  # 12 * 2
    batch_size = 64
    
    # 创建智能体
    agent = FEATAgent(
        env_state_dim=env_state_dim,
        time_varying_state_dim=time_varying_state_dim,
        action_dim=action_dim,
        num_policies=3,
        device='cpu'  # 使用 CPU 进行测试
    )
    
    print(f"Meta-Policy 参数数量: {sum(p.numel() for p in agent.meta_policy.parameters())}")
    print(f"Steerer 参数数量: {sum(p.numel() for p in agent.steerer.parameters())}")
    print(f"Q-Network 参数数量: {sum(p.numel() for p in agent.q_network.parameters())}")
    
    # 测试动作选择
    print("\n--- 动作选择测试 ---")
    env_state = np.random.randn(env_state_dim)
    time_varying_state = np.random.randn(time_varying_state_dim)
    
    action, selected_k = agent.select_action(env_state, time_varying_state)
    print(f"动作形状: {action.shape}")
    print(f"选择的策略: {selected_k}")
    print(f"动作范围: [{action.min():.3f}, {action.max():.3f}]")
    
    # 测试策略分布
    weights = agent.get_policy_distribution(env_state, time_varying_state)
    print(f"策略权重分布: {weights}")
    
    # 测试更新
    print("\n--- 更新测试 ---")
    
    # 创建假批次
    batch = {
        'env_states': np.random.randn(batch_size, env_state_dim).astype(np.float32),
        'time_varying_states': np.random.randn(batch_size, time_varying_state_dim).astype(np.float32),
        'actions': np.random.rand(batch_size, action_dim).astype(np.float32),
        'rewards': np.random.randn(batch_size).astype(np.float32),
        'next_env_states': np.random.randn(batch_size, env_state_dim).astype(np.float32),
        'next_time_varying_states': np.random.randn(batch_size, time_varying_state_dim).astype(np.float32),
        'dones': np.zeros(batch_size).astype(np.float32),
        'selected_ks': np.random.randint(0, 3, batch_size)
    }
    
    metrics = agent.update(batch)
    print(f"更新指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
        
    # 测试仅 Steerer 更新
    print("\n--- Steerer 微调测试 ---")
    steerer_metrics = agent.update_steerer_only(batch)
    print(f"Steerer 微调指标:")
    for key, value in steerer_metrics.items():
        print(f"  {key}: {value:.4f}")
        
    print("\n测试完成!")
