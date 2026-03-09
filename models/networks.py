"""
FEAT 神经网络架构实现
======================
本文件实现了 FEAT 算法的核心网络组件:
- Meta-Policies: 多个共享底层的元策略网络
- Steerer: 策略选择器网络
- Q-Network: 价值评估网络

参考论文: FEAT: Towards Fast Environment-Adaptive Task Offloading 
         and Power Allocation in MEC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, List, Optional
import numpy as np


def init_weights(m, gain: float = 1.0):
    """
    网络权重初始化
    
    使用正交初始化，有助于深度网络的训练稳定性
    
    Args:
        m: 网络层
        gain: 初始化增益
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class SharedFeatureExtractor(nn.Module):
    """
    共享特征提取器
    ===============
    所有元策略共享的底层特征提取网络
    
    架构: 2层 MLP，每层 256 节点，ReLU 激活
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 256),
                 activation: str = 'relu'):
        """
        初始化共享特征提取器
        
        Args:
            input_dim: 输入维度 (时变状态维度)
            hidden_dims: 隐藏层维度元组
            activation: 激活函数类型
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = hidden_dims[-1]
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.ReLU()
            
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
            
        self.network = nn.Sequential(*layers)
        
        # 应用权重初始化
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 时变状态, shape: (batch, input_dim)
            
        Returns:
            特征向量, shape: (batch, output_dim)
        """
        return self.network(state)


class PolicyHead(nn.Module):
    """
    策略输出头
    ===========
    每个元策略的独立输出层，输出动作分布参数 (μ, σ)
    """
    
    def __init__(self,
                 feature_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        """
        初始化策略头
        
        Args:
            feature_dim: 输入特征维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            log_std_min: log 标准差下界
            log_std_max: log 标准差上界
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 隐藏层
        self.hidden = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 输出均值和标准差
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # 初始化
        init_weights(self.hidden[0], gain=np.sqrt(2))
        init_weights(self.mean_layer, gain=0.01)
        init_weights(self.log_std_layer, gain=0.01)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: 特征向量, shape: (batch, feature_dim)
            
        Returns:
            mean: 动作均值, shape: (batch, action_dim)
            log_std: 动作 log 标准差, shape: (batch, action_dim)
        """
        hidden = self.hidden(features)
        mean = self.mean_layer(hidden)
        log_std = self.log_std_layer(hidden)
        
        # 限制 log_std 范围
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, 
               features: torch.Tensor,
               deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作
        
        Args:
            features: 特征向量
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 采样的动作 (经过 sigmoid 映射到 [0,1])
            log_prob: 动作的 log 概率
        """
        mean, log_std = self.forward(features)
        std = log_std.exp()
        
        if deterministic:
            # 确定性策略，直接使用均值
            action = torch.sigmoid(mean)
            log_prob = torch.zeros_like(action).sum(dim=-1)
        else:
            # 从高斯分布采样
            normal = Normal(mean, std)
            x = normal.rsample()  # 重参数化采样
            
            # 应用 sigmoid 变换将动作映射到 [0, 1]
            action = torch.sigmoid(x)
            
            # 计算 log 概率 (考虑 sigmoid 变换的雅可比)
            log_prob = normal.log_prob(x)
            # 雅可比修正: log|d(sigmoid)/dx| = log(sigmoid(x)(1-sigmoid(x)))
            log_prob -= torch.log(action * (1 - action) + 1e-6)
            log_prob = log_prob.sum(dim=-1)
            
        return action, log_prob


class MetaPolicyNetwork(nn.Module):
    """
    元策略网络组
    =============
    包含 K 个共享底层的策略网络
    
    架构:
    - 共享特征提取层 (2层 MLP, 256节点)
    - K 个独立的策略头
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_policies: int = 3,
                 shared_hidden_dims: Tuple[int, ...] = (256, 256),
                 policy_hidden_dim: int = 128,
                 activation: str = 'relu'):
        """
        初始化元策略网络
        
        Args:
            state_dim: 时变状态维度
            action_dim: 动作维度
            num_policies: 元策略数量 K
            shared_hidden_dims: 共享层维度
            policy_hidden_dim: 策略头隐藏层维度
            activation: 激活函数
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_policies = num_policies
        
        # 共享特征提取器
        self.shared_extractor = SharedFeatureExtractor(
            input_dim=state_dim,
            hidden_dims=shared_hidden_dims,
            activation=activation
        )
        
        # K 个独立的策略头
        self.policy_heads = nn.ModuleList([
            PolicyHead(
                feature_dim=self.shared_extractor.output_dim,
                action_dim=action_dim,
                hidden_dim=policy_hidden_dim
            )
            for _ in range(num_policies)
        ])
        
    def forward(self, 
                state: torch.Tensor,
                policy_idx: Optional[int] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 时变状态, shape: (batch, state_dim)
            policy_idx: 指定使用的策略索引 (可选)
            
        Returns:
            means: 各策略的动作均值列表
            log_stds: 各策略的 log 标准差列表
            features: 共享特征向量
        """
        # 提取共享特征
        features = self.shared_extractor(state)
        
        if policy_idx is not None:
            # 只使用指定的策略
            mean, log_std = self.policy_heads[policy_idx](features)
            return [mean], [log_std], features
        
        # 使用所有策略
        means = []
        log_stds = []
        
        for head in self.policy_heads:
            mean, log_std = head(features)
            means.append(mean)
            log_stds.append(log_std)
            
        return means, log_stds, features
    
    def sample_action(self,
                     state: torch.Tensor,
                     policy_idx: int,
                     deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从指定策略采样动作
        
        Args:
            state: 时变状态
            policy_idx: 策略索引
            deterministic: 是否确定性
            
        Returns:
            action: 采样的动作
            log_prob: log 概率
        """
        features = self.shared_extractor(state)
        action, log_prob = self.policy_heads[policy_idx].sample(features, deterministic)
        return action, log_prob
    
    def sample_all_actions(self,
                          state: torch.Tensor,
                          deterministic: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        从所有策略采样动作
        
        Args:
            state: 时变状态
            deterministic: 是否确定性
            
        Returns:
            actions: 所有策略的动作列表
            log_probs: 所有策略的 log 概率列表
        """
        features = self.shared_extractor(state)
        
        actions = []
        log_probs = []
        
        for head in self.policy_heads:
            action, log_prob = head.sample(features, deterministic)
            actions.append(action)
            log_probs.append(log_prob)
            
        return actions, log_probs
    
    def get_features(self, state: torch.Tensor) -> torch.Tensor:
        """
        获取共享特征 (用于 Steerer 输入)
        
        Args:
            state: 时变状态
            
        Returns:
            features: 共享特征向量
        """
        return self.shared_extractor(state)


class SteererNetwork(nn.Module):
    """
    Steerer 驾驶员网络
    ===================
    根据环境状态和特征选择最优的元策略
    
    输入: 环境状态 + Meta-Policy 提取的特征
    输出: K 个元策略的权重分布
    
    训练时使用 Gumbel-Softmax 保证梯度可导
    推理时使用普通 Softmax
    """
    
    def __init__(self,
                 env_state_dim: int,
                 feature_dim: int,
                 num_policies: int = 3,
                 hidden_dims: Tuple[int, ...] = (128, 64),
                 initial_temperature: float = 1.0,
                 min_temperature: float = 0.1):
        """
        初始化 Steerer 网络
        
        Args:
            env_state_dim: 环境状态维度
            feature_dim: Meta-Policy 特征维度
            num_policies: 元策略数量 K
            hidden_dims: 隐藏层维度
            initial_temperature: 初始 Gumbel-Softmax 温度
            min_temperature: 最小温度
        """
        super().__init__()
        
        self.env_state_dim = env_state_dim
        self.feature_dim = feature_dim
        self.num_policies = num_policies
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        
        # 输入维度 = 环境状态 + 特征
        input_dim = env_state_dim + feature_dim
        
        # 构建网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        self.hidden_layers = nn.Sequential(*layers)
        
        # 输出层
        self.output_layer = nn.Linear(prev_dim, num_policies)
        
        # 初始化
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        init_weights(self.output_layer, gain=0.01)
        
    def forward(self, 
                env_state: torch.Tensor,
                features: torch.Tensor,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            env_state: 环境状态, shape: (batch, env_state_dim)
            features: Meta-Policy 特征, shape: (batch, feature_dim)
            training: 是否处于训练模式
            
        Returns:
            weights: 策略权重, shape: (batch, num_policies)
            selected_idx: 选择的策略索引, shape: (batch,)
        """
        # 拼接输入
        x = torch.cat([env_state, features], dim=-1)
        
        # 隐藏层
        hidden = self.hidden_layers(x)
        
        # 输出 logits
        logits = self.output_layer(hidden)
        
        if training:
            # 训练时使用 Gumbel-Softmax
            weights = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
            # 使用 argmax 获取选择的索引，但使用 soft weights 进行梯度传播
            selected_idx = torch.argmax(weights, dim=-1)
        else:
            # 推理时使用普通 Softmax
            weights = F.softmax(logits, dim=-1)
            selected_idx = torch.argmax(weights, dim=-1)
            
        return weights, selected_idx
    
    def get_logits(self,
                   env_state: torch.Tensor,
                   features: torch.Tensor) -> torch.Tensor:
        """
        获取原始 logits (用于计算损失)
        
        Args:
            env_state: 环境状态
            features: Meta-Policy 特征
            
        Returns:
            logits: 原始 logits
        """
        x = torch.cat([env_state, features], dim=-1)
        hidden = self.hidden_layers(x)
        logits = self.output_layer(hidden)
        return logits
    
    def update_temperature(self, decay_rate: float = 0.9995):
        """
        更新 Gumbel-Softmax 温度
        
        Args:
            decay_rate: 温度衰减率
        """
        self.temperature = max(
            self.min_temperature,
            self.temperature * decay_rate
        )


class QNetwork(nn.Module):
    """
    Q 网络 (Critic)
    ================
    评估状态-动作对的价值
    
    使用双 Q 网络减少过估计
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        """
        初始化 Q 网络
        
        Args:
            state_dim: 状态维度 (环境状态 + 时变状态)
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 输入维度 = 状态 + 动作
        input_dim = state_dim + action_dim
        
        # Q1 网络
        q1_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            q1_layers.append(nn.Linear(prev_dim, hidden_dim))
            q1_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        q1_layers.append(nn.Linear(prev_dim, 1))
        self.q1_network = nn.Sequential(*q1_layers)
        
        # Q2 网络
        q2_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            q2_layers.append(nn.Linear(prev_dim, hidden_dim))
            q2_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        q2_layers.append(nn.Linear(prev_dim, 1))
        self.q2_network = nn.Sequential(*q2_layers)
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in [self.q1_network, self.q2_network]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    init_weights(layer, gain=np.sqrt(2))
        # 输出层使用较小的初始化
        init_weights(self.q1_network[-1], gain=1.0)
        init_weights(self.q2_network[-1], gain=1.0)
        
    def forward(self, 
                state: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态, shape: (batch, state_dim)
            action: 动作, shape: (batch, action_dim)
            
        Returns:
            q1: Q1 值, shape: (batch, 1)
            q2: Q2 值, shape: (batch, 1)
        """
        x = torch.cat([state, action], dim=-1)
        q1 = self.q1_network(x)
        q2 = self.q2_network(x)
        return q1, q2
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """只返回 Q1 值"""
        x = torch.cat([state, action], dim=-1)
        return self.q1_network(x)
    
    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """返回 Q1 和 Q2 的最小值 (用于目标值计算)"""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class PolicySelectorQNetwork(nn.Module):
    """
    策略选择 Q 网络
    =================
    为每个元策略单独评估 Q 值
    用于 CMQ (Current Max Q) 方法
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_policies: int = 3,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        """
        初始化策略选择 Q 网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            num_policies: 元策略数量
            hidden_dims: 隐藏层维度
        """
        super().__init__()
        
        self.num_policies = num_policies
        
        # 为每个策略创建独立的 Q 网络
        self.q_networks = nn.ModuleList([
            QNetwork(state_dim, action_dim, hidden_dims)
            for _ in range(num_policies)
        ])
        
    def forward(self,
                state: torch.Tensor,
                action: torch.Tensor,
                policy_idx: Optional[int] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            state: 状态
            action: 动作
            policy_idx: 策略索引 (可选，如果指定则只计算该策略的 Q 值)
            
        Returns:
            Q 值列表，每个元素是 (Q1, Q2) 元组
        """
        if policy_idx is not None:
            q1, q2 = self.q_networks[policy_idx](state, action)
            return [(q1, q2)]
        
        q_values = []
        for q_net in self.q_networks:
            q1, q2 = q_net(state, action)
            q_values.append((q1, q2))
            
        return q_values
    
    def get_max_q_policy(self,
                         state: torch.Tensor,
                         actions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取使 Q 值最大的策略索引
        
        Args:
            state: 状态
            actions: 各策略的动作列表
            
        Returns:
            max_idx: Q 值最大的策略索引
            max_q: 最大 Q 值
        """
        q_mins = []
        
        for i, action in enumerate(actions):
            q_min = self.q_networks[i].q_min(state, action)
            q_mins.append(q_min)
            
        # 堆叠并找最大值
        q_stack = torch.stack(q_mins, dim=-1).squeeze(-2)  # (batch, num_policies)
        max_q, max_idx = torch.max(q_stack, dim=-1)
        
        return max_idx, max_q


if __name__ == "__main__":
    # 测试网络
    print("=== 神经网络测试 ===")
    
    # 参数
    batch_size = 32
    state_dim = 49  # 12 * 4 + 1
    env_state_dim = 5
    action_dim = 24  # 12 * 2
    num_policies = 3
    
    # 创建网络
    meta_policy = MetaPolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        num_policies=num_policies
    )
    
    steerer = SteererNetwork(
        env_state_dim=env_state_dim,
        feature_dim=256,
        num_policies=num_policies
    )
    
    q_net = QNetwork(
        state_dim=env_state_dim + state_dim,
        action_dim=action_dim
    )
    
    # 测试数据
    state = torch.randn(batch_size, state_dim)
    env_state = torch.randn(batch_size, env_state_dim)
    action = torch.rand(batch_size, action_dim)
    
    # 测试 Meta-Policy
    print("\n--- Meta-Policy 测试 ---")
    means, log_stds, features = meta_policy(state)
    print(f"策略数量: {len(means)}")
    print(f"均值形状: {means[0].shape}")
    print(f"特征形状: {features.shape}")
    
    # 采样动作
    sampled_action, log_prob = meta_policy.sample_action(state, policy_idx=0)
    print(f"采样动作形状: {sampled_action.shape}")
    print(f"Log 概率形状: {log_prob.shape}")
    
    # 测试 Steerer
    print("\n--- Steerer 测试 ---")
    weights, selected_idx = steerer(env_state, features, training=True)
    print(f"权重形状: {weights.shape}")
    print(f"选择索引: {selected_idx[:5]}")
    print(f"当前温度: {steerer.temperature}")
    
    # 测试 Q 网络
    print("\n--- Q 网络测试 ---")
    full_state = torch.cat([env_state, state], dim=-1)
    q1, q2 = q_net(full_state, action)
    print(f"Q1 形状: {q1.shape}")
    print(f"Q2 形状: {q2.shape}")
    
    print("\n所有网络测试通过!")
