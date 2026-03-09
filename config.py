"""
FEAT 算法配置文件
====================
本文件包含所有环境参数、网络架构参数和训练超参数。
通过修改此文件可以方便地调整实验设置。

参考论文: FEAT: Towards Fast Environment-Adaptive Task Offloading 
         and Power Allocation in MEC
"""

import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class EnvironmentConfig:
    """
    MEC 环境配置类
    ===============
    包含所有物理环境相关的参数设置
    """
    
    # ==================== 基础拓扑设置 ====================
    num_mobile_devices: int = 12          # 移动设备数量 M
    num_time_slots: int = 10              # 每个 Episode 的时隙数 N
    time_slot_duration: float = 0.5       # 单时隙长度 δ (秒)
    
    # 移动设备分布范围 (距离基站的距离，单位：米)
    min_distance: float = 20.0            # 最小距离
    max_distance: float = 100.0           # 最大距离
    
    # ==================== 通信参数 ====================
    bandwidth: float = 10e6               # 系统带宽 B = 10 MHz
    noise_power_dbm: float = -100.0       # 噪声功率 σ² = -100 dBm
    
    # 大尺度衰落参数
    path_loss_constant: float = -148.1    # 路径损耗常数
    path_loss_exponent: float = 37.6      # 路径损耗指数
    shadow_fading_std: float = 8.0        # 阴影衰落标准差 (dB)
    
    # Jakes 模型参数
    carrier_frequency: float = 2.4e9      # 载波频率 (Hz)
    mobile_speed: float = 3.0             # 移动设备速度 (m/s)，步行速度
    num_sinusoids: int = 20               # 正弦波叠加数量
    
    # ==================== 计算参数 ====================
    es_computing_capacity: float = 9e9    # 边缘服务器计算能力 f^ES = 9 GHz
    md_computing_capacity: float = 1e9    # 移动设备计算能力 f^MD = 1 GHz
    md_max_power: float = 0.5             # 移动设备最大发射功率 p^max = 0.5 W
    
    # 能耗系数
    energy_coefficient: float = 1e-27     # 能耗系数 ξ = 10^-27
    
    # ==================== 任务参数 ====================
    task_size_mean: float = 700e3         # 任务大小均值 (bits)，700 Kb
    task_size_variance: float = 1000e3    # 任务大小方差 (bits)
    
    # 计算密度范围 (cycles/bit)
    computation_density_min: float = 800  # 最小计算密度
    computation_density_max: float = 900  # 最大计算密度
    
    # 延迟容忍范围 (秒)
    delay_tolerance_min: float = 0.8      # 最小延迟容忍
    delay_tolerance_max: float = 0.9      # 最大延迟容忍
    
    # ==================== 电池参数 ====================
    initial_battery: float = 8.0          # 初始电池电量 ebMD = 8 J (论文 Table I)
    min_battery: float = 0.0              # 最小电池电量
    
    # ==================== 奖励参数 (严格按论文设定) ====================
    delay_weight: float = 0.5             # 延迟权重 η (论文 Eq.12)
    energy_weight: float = 0.5            # 能耗权重 (1-η)
    reward_coefficient: float = 1.0       # 奖励系数 α (论文 reward 定义)
    
    # 约束惩罚权重 β_j (论文 reward 定义中的各约束惩罚权重)
    # 注意: 惩罚不宜过大，否则模型会过于保守
    penalty_offloading: float = 1.0       # C1: 卸载决策约束惩罚 (二值化时的软约束)
    penalty_power: float = 1.0            # C2: 功率范围约束惩罚
    penalty_delay: float = 2.0            # C3: 延迟约束惩罚 (适度惩罚，避免过于保守)
    penalty_energy: float = 2.0           # C4: 能量预算约束惩罚 (适度惩罚)
    
    def __post_init__(self):
        """计算派生参数"""
        # 将噪声功率从 dBm 转换为 W
        self.noise_power = 10 ** ((self.noise_power_dbm - 30) / 10)
        
        # 计算最大多普勒频移
        speed_of_light = 3e8
        self.max_doppler_shift = (self.mobile_speed * self.carrier_frequency) / speed_of_light


@dataclass
class NetworkConfig:
    """
    神经网络架构配置类
    ==================
    包含 Meta-Policy、Steerer 和 Q-Network 的架构参数
    """
    
    # ==================== Meta-Policy 网络参数 ====================
    num_meta_policies: int = 3            # 元策略数量 K = 3
    
    # 共享特征提取层
    shared_hidden_dims: Tuple[int, ...] = (256, 256)  # 共享层维度
    
    # 策略输出层
    policy_hidden_dim: int = 128          # 策略独立头的隐藏层维度
    
    # ==================== Steerer 网络参数 ====================
    steerer_hidden_dims: Tuple[int, ...] = (128, 64)  # Steerer 隐藏层维度
    # # [原始设定]
    # gumbel_temperature: float = 0.2       # Gumbel-Softmax 温度参数 τ (论文: annealed from 0.2)
    # gumbel_temperature_min: float = 0.1   # 最小温度
    # temperature_decay: float = 0.9995     # 温度衰减率
    # [优化: 更慢的退火允许更多探索]
    gumbel_temperature: float = 1.0       # 初始温度 (较高以允许充分探索)
    gumbel_temperature_min: float = 0.2   # 最小温度 (论文目标值)
    temperature_decay: float = 0.9997     # 温度衰减率 (更慢的退火)
    
    # ==================== 策略多样性参数 ====================
    steerer_entropy_coeff: float = 0.5    # Steerer 熵正则化系数 λ_entropy
    diversity_coeff: float = 0.1          # 策略差异化损失系数 λ_diversity
    forced_exploration_prob: float = 0.2  # 强制探索概率 (训练时随机选策略)
    forced_exploration_decay: float = 0.9995  # 强制探索概率衰减率
    forced_exploration_min: float = 0.02  # 最小强制探索概率
    
    # ==================== Q-Network 参数 ====================
    q_hidden_dims: Tuple[int, ...] = (256, 256)  # Q 网络隐藏层维度
    
    # ==================== 激活函数 ====================
    activation: str = 'relu'              # 激活函数类型
    
    # ==================== 动作空间 ====================
    action_dim: int = 2                   # 动作维度 (卸载决策 + 功率分配)


@dataclass
class TrainingConfig:
    """
    训练配置类
    ===========
    包含所有训练相关的超参数
    """
    
    # ==================== 基础训练参数 ====================
    num_episodes: int = 5000              # 训练 Episode 数量
    batch_size: int = 64                  # 批量大小 (论文: batch size of experience sampling is 64)
    
    # ==================== 学习率 ====================
    actor_lr: float = 3e-4                # Actor (Meta-Policy) 学习率
    critic_lr: float = 3e-4               # Critic (Q-Network) 学习率
    steerer_lr: float = 3e-4              # Steerer 学习率
    alpha_lr: float = 3e-4                # SAC 温度参数学习率
    
    # ==================== SAC 参数 ====================
    gamma: float = 0.95                   # 折扣因子 γ (论文: empirically set as 0.95)
    tau: float = 0.005                    # 目标网络软更新系数
    alpha: float = 0.2                    # SAC 熵正则化系数 ζ (论文: auto-tuned)
    automatic_entropy_tuning: bool = True # 是否自动调整熵系数 (论文: auto-tuned)
    target_entropy: Optional[float] = None  # 目标熵，None 时自动计算
    
    # ==================== 经验回放 ====================
    buffer_size: int = 100000             # 经验回放池大小
    min_buffer_size: int = 1000           # 开始训练前的最小样本数
    
    # ==================== 更新频率 ====================
    update_frequency: int = 1             # 每隔多少步更新一次
    target_update_frequency: int = 1      # 目标网络更新频率
    
    # ==================== 探索参数 ====================
    exploration_episodes: int = 100       # 纯探索的 Episode 数量
    
    # ==================== 评估参数 ====================
    eval_frequency: int = 50              # 评估频率 (Episode)
    eval_episodes: int = 10               # 每次评估的 Episode 数量
    
    # ==================== 保存参数 ====================
    save_frequency: int = 100             # 模型保存频率
    log_frequency: int = 10               # 日志记录频率
    
    # ==================== 微调参数 (用于环境适应) ====================
    finetune_episodes: int = 200          # 微调 Episode 数量
    finetune_steerer_only: bool = True    # 微调时是否只更新 Steerer


@dataclass  
class ExperimentConfig:
    """
    实验配置类
    ===========
    整合所有配置，方便统一管理
    """
    env_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    net_config: NetworkConfig = field(default_factory=NetworkConfig)
    train_config: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 实验名称和路径 (默认使用时间戳，避免覆盖旧实验)
    experiment_name: str = field(default_factory=lambda: datetime.now().strftime("Run_%Y-%m-%d_%H-%M-%S"))
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # 随机种子
    seed: int = 42
    
    # 设备设置
    device: str = "cuda"  # 'cuda' 或 'cpu'
    
    def get_state_dims(self) -> dict:
        """
        计算状态空间维度
        
        Returns:
            dict: 包含各类状态维度的字典
        """
        M = self.env_config.num_mobile_devices
        
        # 环境状态维度 (Steerer 输入)
        # 包含: 任务分布参数(2), 带宽(1), MD计算能力(1), ES计算能力(1)
        env_state_dim = 5
        
        # 时变状态维度 (Meta-Policy 输入)
        # 每个设备: 任务大小(1), 电池电量(1), 本地队列(1), 边缘队列贡献(1)
        # 加上全局边缘队列长度(1)
        time_varying_state_dim = M * 4 + 1
        
        # 动作维度
        # 每个设备: 卸载决策(1) + 功率分配(1)
        action_dim = M * 2
        
        return {
            'env_state_dim': env_state_dim,
            'time_varying_state_dim': time_varying_state_dim,
            'action_dim': action_dim,
            'total_state_dim': env_state_dim + time_varying_state_dim
        }


def create_default_config() -> ExperimentConfig:
    """
    创建默认实验配置
    
    Returns:
        ExperimentConfig: 默认配置对象
    """
    return ExperimentConfig()


def create_test_config(bandwidth: float = 9e6, 
                       task_variance: float = 1200e3) -> ExperimentConfig:
    """
    创建测试配置 (用于环境适应性测试)
    
    Args:
        bandwidth: 测试带宽 (默认 9 MHz，与训练时的 10 MHz 不同)
        task_variance: 任务大小方差
        
    Returns:
        ExperimentConfig: 测试配置对象
    """
    config = ExperimentConfig()
    config.env_config.bandwidth = bandwidth
    config.env_config.task_size_variance = task_variance
    config.experiment_name = f"FEAT_test_bw{bandwidth/1e6:.1f}MHz"
    return config


if __name__ == "__main__":
    # 测试配置
    config = create_default_config()
    print("=== FEAT 默认配置 ===")
    print(f"移动设备数量: {config.env_config.num_mobile_devices}")
    print(f"时隙数量: {config.env_config.num_time_slots}")
    print(f"带宽: {config.env_config.bandwidth / 1e6} MHz")
    print(f"噪声功率: {config.env_config.noise_power:.2e} W")
    print(f"Meta-Policy 数量: {config.net_config.num_meta_policies}")
    print(f"状态维度: {config.get_state_dims()}")
