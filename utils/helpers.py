"""
辅助函数模块
=============
包含各种通用的辅助函数

功能包括:
- 随机种子设置
- 网络参数更新
- 模型保存和加载
- 训练指标计算
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
from typing import Dict, Any, Optional
import json
from datetime import datetime


def set_seed(seed: int):
    """
    设置全局随机种子
    
    确保实验的可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保 CUDA 操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def soft_update(target_net: nn.Module, 
                source_net: nn.Module, 
                tau: float):
    """
    软更新目标网络参数
    
    θ_target = τ * θ_source + (1 - τ) * θ_target
    
    Args:
        target_net: 目标网络
        source_net: 源网络
        tau: 更新系数 (通常很小，如 0.005)
    """
    for target_param, source_param in zip(target_net.parameters(), 
                                          source_net.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1 - tau) * target_param.data
        )


def hard_update(target_net: nn.Module, source_net: nn.Module):
    """
    硬更新目标网络参数 (直接复制)
    
    θ_target = θ_source
    
    Args:
        target_net: 目标网络
        source_net: 源网络
    """
    target_net.load_state_dict(source_net.state_dict())


def save_checkpoint(save_path: str,
                   agent,
                   episode: int,
                   metrics: Dict[str, float],
                   config: Any = None):
    """
    保存训练检查点
    
    Args:
        save_path: 保存路径
        agent: FEAT 智能体
        episode: 当前 Episode
        metrics: 训练指标
        config: 配置对象 (可选)
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'episode': episode,
        'meta_policy_state_dict': agent.meta_policy.state_dict(),
        'steerer_state_dict': agent.steerer.state_dict(),
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_q_network_state_dict': agent.target_q_network.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
        'steerer_optimizer_state_dict': agent.steerer_optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存熵系数（如果启用了自动调整）
    if hasattr(agent, 'log_alpha'):
        checkpoint['log_alpha'] = agent.log_alpha.item()
        checkpoint['alpha_optimizer_state_dict'] = agent.alpha_optimizer.state_dict()
        
    # 保存 Steerer 温度
    checkpoint['steerer_temperature'] = agent.steerer.temperature
    
    torch.save(checkpoint, save_path)
    
    # 保存配置为 JSON (如果提供)
    if config is not None:
        config_path = save_path.replace('.pt', '_config.json')
        config_dict = {
            'env_config': vars(config.env_config) if hasattr(config, 'env_config') else {},
            'net_config': vars(config.net_config) if hasattr(config, 'net_config') else {},
            'train_config': vars(config.train_config) if hasattr(config, 'train_config') else {}
        }
        # 过滤不可序列化的对象
        def filter_serializable(d):
            return {k: v for k, v in d.items() 
                   if isinstance(v, (int, float, str, bool, list, tuple, dict, type(None)))}
        config_dict = {k: filter_serializable(v) for k, v in config_dict.items()}
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def load_checkpoint(load_path: str,
                   agent,
                   load_optimizer: bool = True) -> Dict[str, Any]:
    """
    加载训练检查点
    
    Args:
        load_path: 检查点路径
        agent: FEAT 智能体
        load_optimizer: 是否加载优化器状态
        
    Returns:
        检查点信息字典
    """
    checkpoint = torch.load(load_path, map_location=agent.device)
    
    # 加载网络参数
    agent.meta_policy.load_state_dict(checkpoint['meta_policy_state_dict'])
    agent.steerer.load_state_dict(checkpoint['steerer_state_dict'])
    agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    agent.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
    
    # 加载优化器状态
    if load_optimizer:
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        agent.steerer_optimizer.load_state_dict(checkpoint['steerer_optimizer_state_dict'])
        
        if 'log_alpha' in checkpoint and hasattr(agent, 'log_alpha'):
            agent.log_alpha.data.fill_(checkpoint['log_alpha'])
            agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            
    # 恢复 Steerer 温度
    if 'steerer_temperature' in checkpoint:
        agent.steerer.temperature = checkpoint['steerer_temperature']
        
    return {
        'episode': checkpoint['episode'],
        'metrics': checkpoint['metrics'],
        'timestamp': checkpoint.get('timestamp', 'unknown')
    }


class RunningMeanStd:
    """
    运行时均值和标准差计算
    ======================
    用于观察值的归一化
    """
    
    def __init__(self, shape: tuple = (), epsilon: float = 1e-8):
        """
        初始化
        
        Args:
            shape: 数据形状
            epsilon: 防止除零的小常数
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon
        
    def update(self, x: np.ndarray):
        """
        更新统计量
        
        Args:
            x: 新的观察值
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
        
    def _update_from_moments(self, 
                             batch_mean: np.ndarray,
                             batch_var: np.ndarray,
                             batch_count: int):
        """使用增量更新公式"""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
        
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        归一化输入
        
        Args:
            x: 输入数据
            
        Returns:
            归一化后的数据
        """
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)


class MetricsLogger:
    """
    训练指标记录器
    ===============
    记录和管理训练过程中的各种指标
    """
    
    def __init__(self, log_dir: str = "./logs"):
        """
        初始化记录器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = {}
        self.episode_metrics = []
        
    def log(self, name: str, value: float, step: int):
        """
        记录单个指标
        
        Args:
            name: 指标名称
            value: 指标值
            step: 时间步
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((step, value))
        
    def log_episode(self, episode: int, metrics: Dict[str, float]):
        """
        记录一个 Episode 的指标
        
        Args:
            episode: Episode 编号
            metrics: 指标字典
        """
        record = {'episode': episode, **metrics}
        self.episode_metrics.append(record)
        
    def get_recent_average(self, name: str, window: int = 100) -> Optional[float]:
        """
        获取最近的平均值
        
        Args:
            name: 指标名称
            window: 窗口大小
            
        Returns:
            平均值，如果没有数据则返回 None
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return None
            
        recent = self.metrics[name][-window:]
        values = [v[1] for v in recent]
        return np.mean(values)
    
    def save(self, filename: str = "metrics.json"):
        """
        保存所有指标到文件
        
        Args:
            filename: 文件名
        """
        filepath = os.path.join(self.log_dir, filename)
        
        data = {
            'metrics': {name: list(values) for name, values in self.metrics.items()},
            'episode_metrics': self.episode_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def print_summary(self, episode: int, metrics: Dict[str, float]):
        """
        打印训练摘要
        
        Args:
            episode: Episode 编号
            metrics: 当前指标
        """
        print(f"\n{'='*50}")
        print(f"Episode {episode}")
        print(f"{'='*50}")
        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")
        
        # 打印最近平均值
        for name in ['reward', 'actor_loss', 'critic_loss']:
            avg = self.get_recent_average(name, 100)
            if avg is not None:
                print(f"  {name} (avg100): {avg:.4f}")


def compute_gae(rewards: np.ndarray,
                values: np.ndarray,
                dones: np.ndarray,
                gamma: float = 0.99,
                lambda_: float = 0.95) -> np.ndarray:
    """
    计算广义优势估计 (GAE)
    
    注意: FEAT 使用 SAC，不需要 GAE，
    这是一个可选的扩展函数
    
    Args:
        rewards: 奖励序列
        values: 价值估计序列
        dones: 结束标志序列
        gamma: 折扣因子
        lambda_: GAE 参数
        
    Returns:
        advantages: 优势估计
    """
    n = len(rewards)
    advantages = np.zeros(n)
    last_gae = 0
    
    for t in reversed(range(n)):
        if t == n - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
            
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lambda_ * (1 - dones[t]) * last_gae
        
    return advantages


if __name__ == "__main__":
    # 测试辅助函数
    print("=== 辅助函数测试 ===")
    
    # 测试随机种子
    set_seed(42)
    print(f"随机数测试: {np.random.rand()}")
    set_seed(42)
    print(f"重置后相同: {np.random.rand()}")
    
    # 测试 RunningMeanStd
    print("\n--- RunningMeanStd 测试 ---")
    rms = RunningMeanStd(shape=(3,))
    for _ in range(10):
        rms.update(np.random.randn(100, 3))
    print(f"均值: {rms.mean}")
    print(f"方差: {rms.var}")
    
    # 测试指标记录器
    print("\n--- MetricsLogger 测试 ---")
    logger = MetricsLogger("./test_logs")
    for i in range(10):
        logger.log('reward', np.random.randn(), i)
        logger.log_episode(i, {'reward': np.random.randn(), 'steps': i * 10})
    
    avg = logger.get_recent_average('reward', 5)
    print(f"最近5个奖励平均值: {avg}")
    
    print("\n测试完成!")
