"""
经验回放池实现
================
本文件实现了用于强化学习的经验回放池 (Replay Buffer)

FEAT 算法需要存储额外的信息:
- 环境状态 (用于 Steerer)
- 选择的策略索引 (用于 HSD 更新)

参考论文: FEAT: Towards Fast Environment-Adaptive Task Offloading 
         and Power Allocation in MEC
"""

import numpy as np
from typing import Dict, Tuple, Optional, NamedTuple
from collections import deque
import random


class Transition(NamedTuple):
    """
    经验转换数据结构
    =================
    存储单次交互的完整信息
    """
    env_state: np.ndarray         # 环境状态 (Steerer 输入)
    time_varying_state: np.ndarray # 时变状态 (Meta-Policy 输入)
    action: np.ndarray            # 执行的动作
    reward: float                 # 获得的奖励
    next_env_state: np.ndarray    # 下一环境状态
    next_time_varying_state: np.ndarray  # 下一时变状态
    done: bool                    # Episode 是否结束
    selected_k: int               # 选择的策略索引


class ReplayBuffer:
    """
    经验回放池
    ===========
    使用循环缓冲区存储经验，支持随机采样
    
    特点:
    - 支持 FEAT 算法所需的额外信息存储
    - 高效的 NumPy 数组存储
    - 随机批量采样
    
    Attributes:
        capacity: 缓冲区容量
        buffer: 存储经验的列表
        position: 当前写入位置
    """
    
    def __init__(self, capacity: int):
        """
        初始化经验回放池
        
        Args:
            capacity: 缓冲区容量
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self,
             env_state: np.ndarray,
             time_varying_state: np.ndarray,
             action: np.ndarray,
             reward: float,
             next_env_state: np.ndarray,
             next_time_varying_state: np.ndarray,
             done: bool,
             selected_k: int):
        """
        添加一条经验
        
        Args:
            env_state: 环境状态
            time_varying_state: 时变状态
            action: 动作
            reward: 奖励
            next_env_state: 下一环境状态
            next_time_varying_state: 下一时变状态
            done: 是否结束
            selected_k: 选择的策略索引
        """
        transition = Transition(
            env_state=env_state,
            time_varying_state=time_varying_state,
            action=action,
            reward=reward,
            next_env_state=next_env_state,
            next_time_varying_state=next_time_varying_state,
            done=done,
            selected_k=selected_k
        )
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        随机采样一批经验
        
        Args:
            batch_size: 批量大小
            
        Returns:
            包含各类数据的字典
        """
        batch = random.sample(self.buffer, batch_size)
        
        # 转换为 NumPy 数组
        env_states = np.array([t.env_state for t in batch], dtype=np.float32)
        time_varying_states = np.array([t.time_varying_state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.float32)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_env_states = np.array([t.next_env_state for t in batch], dtype=np.float32)
        next_time_varying_states = np.array([t.next_time_varying_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        selected_ks = np.array([t.selected_k for t in batch], dtype=np.int64)
        
        return {
            'env_states': env_states,
            'time_varying_states': time_varying_states,
            'actions': actions,
            'rewards': rewards,
            'next_env_states': next_env_states,
            'next_time_varying_states': next_time_varying_states,
            'dones': dones,
            'selected_ks': selected_ks
        }
    
    def sample_by_policy(self, 
                        batch_size: int,
                        policy_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """
        按策略索引采样经验
        
        用于 HSD 更新，只采样使用特定策略的经验
        
        Args:
            batch_size: 批量大小
            policy_idx: 策略索引
            
        Returns:
            经验字典，如果样本不足则返回 None
        """
        # 筛选使用该策略的经验
        filtered = [t for t in self.buffer if t.selected_k == policy_idx]
        
        if len(filtered) < batch_size:
            return None
            
        batch = random.sample(filtered, batch_size)
        
        # 转换为 NumPy 数组
        env_states = np.array([t.env_state for t in batch], dtype=np.float32)
        time_varying_states = np.array([t.time_varying_state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.float32)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_env_states = np.array([t.next_env_state for t in batch], dtype=np.float32)
        next_time_varying_states = np.array([t.next_time_varying_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        selected_ks = np.array([t.selected_k for t in batch], dtype=np.int64)
        
        return {
            'env_states': env_states,
            'time_varying_states': time_varying_states,
            'actions': actions,
            'rewards': rewards,
            'next_env_states': next_env_states,
            'next_time_varying_states': next_time_varying_states,
            'dones': dones,
            'selected_ks': selected_ks
        }
    
    def __len__(self) -> int:
        """返回当前缓冲区大小"""
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """
        检查缓冲区是否有足够的样本
        
        Args:
            min_size: 最小样本数
            
        Returns:
            是否可以开始训练
        """
        return len(self.buffer) >= min_size
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.position = 0
        
    def get_statistics(self) -> Dict:
        """
        获取缓冲区统计信息
        
        Returns:
            统计信息字典
        """
        if len(self.buffer) == 0:
            return {'size': 0}
            
        rewards = [t.reward for t in self.buffer]
        policy_counts = {}
        for t in self.buffer:
            k = t.selected_k
            policy_counts[k] = policy_counts.get(k, 0) + 1
            
        return {
            'size': len(self.buffer),
            'mean_reward': np.mean(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'policy_distribution': policy_counts
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    优先级经验回放池 (可选扩展)
    ============================
    根据 TD 误差进行优先采样
    
    注意: FEAT 论文中使用的是普通回放池，
    这是一个可选的扩展实现
    """
    
    def __init__(self, 
                 capacity: int,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001):
        """
        初始化优先级回放池
        
        Args:
            capacity: 容量
            alpha: 优先级指数
            beta: 重要性采样指数
            beta_increment: beta 增长率
        """
        super().__init__(capacity)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
    def push(self, *args, **kwargs):
        """添加经验，使用最大优先级"""
        super().push(*args, **kwargs)
        self.priorities.append(self.max_priority)
        
    def sample(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        按优先级采样
        
        Returns:
            batch: 经验批次
            indices: 采样索引
            weights: 重要性权重
        """
        n = len(self.buffer)
        
        if n < batch_size:
            raise ValueError("缓冲区样本不足")
            
        # 计算采样概率
        priorities = np.array(list(self.priorities))[:n]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        
        # 计算重要性权重
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # 更新 beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 构建批次
        batch = [self.buffer[i] for i in indices]
        
        env_states = np.array([t.env_state for t in batch], dtype=np.float32)
        time_varying_states = np.array([t.time_varying_state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.float32)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_env_states = np.array([t.next_env_state for t in batch], dtype=np.float32)
        next_time_varying_states = np.array([t.next_time_varying_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        selected_ks = np.array([t.selected_k for t in batch], dtype=np.int64)
        
        batch_dict = {
            'env_states': env_states,
            'time_varying_states': time_varying_states,
            'actions': actions,
            'rewards': rewards,
            'next_env_states': next_env_states,
            'next_time_varying_states': next_time_varying_states,
            'dones': dones,
            'selected_ks': selected_ks
        }
        
        return batch_dict, indices, weights.astype(np.float32)
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        更新优先级
        
        Args:
            indices: 更新的索引
            priorities: 新的优先级 (通常是 |TD error| + ε)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


if __name__ == "__main__":
    # 测试回放池
    print("=== 经验回放池测试 ===")
    
    buffer = ReplayBuffer(capacity=1000)
    
    # 添加一些经验
    for i in range(100):
        env_state = np.random.randn(5)
        time_varying_state = np.random.randn(49)
        action = np.random.rand(24)
        reward = np.random.randn()
        next_env_state = np.random.randn(5)
        next_time_varying_state = np.random.randn(49)
        done = i % 10 == 9
        selected_k = np.random.randint(0, 3)
        
        buffer.push(
            env_state, time_varying_state, action, reward,
            next_env_state, next_time_varying_state, done, selected_k
        )
        
    print(f"缓冲区大小: {len(buffer)}")
    print(f"统计信息: {buffer.get_statistics()}")
    
    # 采样
    batch = buffer.sample(32)
    print(f"\n采样批次形状:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
        
    # 按策略采样
    for k in range(3):
        policy_batch = buffer.sample_by_policy(16, k)
        if policy_batch is not None:
            print(f"\n策略 {k} 采样成功，样本数: {len(policy_batch['rewards'])}")
        else:
            print(f"\n策略 {k} 样本不足")
