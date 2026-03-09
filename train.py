"""
FEAT 算法训练脚本
==================
本文件实现了完整的 FEAT 算法训练流程

训练流程:
1. 初始化环境和智能体
2. 收集经验并存储到回放池
3. 使用论文指定的更新策略训练网络
4. 定期评估和保存模型

参考论文: FEAT: Towards Fast Environment-Adaptive Task Offloading 
         and Power Allocation in MEC
"""

import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from config import (
    ExperimentConfig, 
    create_default_config, 
    create_test_config,
    EnvironmentConfig,
    NetworkConfig,
    TrainingConfig
)
from environment.mec_environment import MECEnvironment
from models.feat_agent import FEATAgent
from utils.replay_buffer import ReplayBuffer
from utils.helpers import set_seed, MetricsLogger, save_checkpoint


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FEAT 算法训练')
    
    # 实验设置
    parser.add_argument('--exp_name', type=str, default='FEAT_default',
                       help='实验名称')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备 (cuda/cpu)')
    
    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=5000,
                       help='训练 Episode 数量')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批量大小 (论文: 64)')
    parser.add_argument('--buffer_size', type=int, default=100000,
                       help='经验回放池大小')
    parser.add_argument('--min_buffer_size', type=int, default=1000,
                       help='开始训练前的最小样本数')
    
    # 学习率
    parser.add_argument('--actor_lr', type=float, default=3e-4,
                       help='Actor 学习率')
    parser.add_argument('--critic_lr', type=float, default=3e-4,
                       help='Critic 学习率')
    parser.add_argument('--steerer_lr', type=float, default=3e-4,
                       help='Steerer 学习率')
    
    # SAC 参数
    parser.add_argument('--gamma', type=float, default=0.95,
                       help='折扣因子 (论文: 0.95)')
    parser.add_argument('--tau', type=float, default=0.005,
                       help='目标网络软更新系数')
    parser.add_argument('--alpha', type=float, default=0.2,
                       help='SAC 熵系数')
    parser.add_argument('--auto_alpha', action='store_true', default=True,
                       help='是否自动调整熵系数')
    
    # 环境参数
    parser.add_argument('--num_devices', type=int, default=12,
                       help='移动设备数量')
    parser.add_argument('--bandwidth', type=float, default=10e6,
                       help='系统带宽 (Hz)')
    
    # 评估和保存
    parser.add_argument('--eval_freq', type=int, default=50,
                       help='评估频率 (Episode)')
    parser.add_argument('--save_freq', type=int, default=100,
                       help='保存频率 (Episode)')
    parser.add_argument('--log_freq', type=int, default=10,
                       help='日志频率 (Episode)')
    
    # 保存路径
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志保存目录')
    
    return parser.parse_args()


class FEATTrainer:
    """
    FEAT 训练器
    ============
    封装完整的训练流程
    """
    
    def __init__(self, config: ExperimentConfig, args: argparse.Namespace):
        """
        初始化训练器
        
        Args:
            config: 实验配置
            args: 命令行参数
        """
        self.config = config
        self.args = args
        
        # 设置随机种子
        set_seed(config.seed)
        
        # 创建保存目录
        self.save_dir = os.path.join(args.save_dir, args.exp_name)
        self.log_dir = os.path.join(args.log_dir, args.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 保存配置
        self._save_config()
        
        # 创建环境
        self.env = MECEnvironment(config=config.env_config, seed=config.seed)
        self.eval_env = MECEnvironment(config=config.env_config, seed=config.seed + 1000)
        
        # 获取状态维度
        state_dims = self.env.get_state_dims()
        self.env_state_dim = state_dims['env_state_dim']
        self.time_varying_state_dim = state_dims['time_varying_state_dim']
        self.action_dim = state_dims['action_dim']
        
        print(f"状态维度: 环境状态={self.env_state_dim}, 时变状态={self.time_varying_state_dim}")
        print(f"动作维度: {self.action_dim}")
        
        # 创建智能体
        self.agent = FEATAgent(
            env_state_dim=self.env_state_dim,
            time_varying_state_dim=self.time_varying_state_dim,
            action_dim=self.action_dim,
            num_policies=config.net_config.num_meta_policies,
            shared_hidden_dims=config.net_config.shared_hidden_dims,
            policy_hidden_dim=config.net_config.policy_hidden_dim,
            steerer_hidden_dims=config.net_config.steerer_hidden_dims,
            q_hidden_dims=config.net_config.q_hidden_dims,
            actor_lr=config.train_config.actor_lr,
            critic_lr=config.train_config.critic_lr,
            steerer_lr=config.train_config.steerer_lr,
            gamma=config.train_config.gamma,
            tau=config.train_config.tau,
            alpha=config.train_config.alpha,
            automatic_entropy_tuning=config.train_config.automatic_entropy_tuning,
            gumbel_temperature=config.net_config.gumbel_temperature,
            temperature_decay=config.net_config.temperature_decay,
            # ===== 策略多样性参数 (方案 A+B) =====
            steerer_entropy_coeff=config.net_config.steerer_entropy_coeff,
            forced_exploration_prob=config.net_config.forced_exploration_prob,
            forced_exploration_decay=config.net_config.forced_exploration_decay,
            forced_exploration_min=config.net_config.forced_exploration_min,
            device=config.device
        )
        
        # 创建经验回放池
        self.replay_buffer = ReplayBuffer(capacity=config.train_config.buffer_size)
        
        # 创建日志记录器
        self.logger = MetricsLogger(log_dir=self.log_dir)
        
        # 训练统计
        self.total_steps = 0
        self.episode_count = 0
        self.best_eval_reward = float('-inf')
        
    def _save_config(self):
        """保存实验配置"""
        config_path = os.path.join(self.save_dir, 'config.json')
        
        config_dict = {
            'experiment_name': self.args.exp_name,
            'seed': self.config.seed,
            'env_config': {
                'num_mobile_devices': self.config.env_config.num_mobile_devices,
                'num_time_slots': self.config.env_config.num_time_slots,
                'time_slot_duration': self.config.env_config.time_slot_duration,
                'bandwidth': self.config.env_config.bandwidth,
                'es_computing_capacity': self.config.env_config.es_computing_capacity,
                'md_computing_capacity': self.config.env_config.md_computing_capacity,
                'task_size_mean': self.config.env_config.task_size_mean,
                'task_size_variance': self.config.env_config.task_size_variance,
            },
            'net_config': {
                'num_meta_policies': self.config.net_config.num_meta_policies,
                'shared_hidden_dims': self.config.net_config.shared_hidden_dims,
                'policy_hidden_dim': self.config.net_config.policy_hidden_dim,
            },
            'train_config': {
                'num_episodes': self.config.train_config.num_episodes,
                'batch_size': self.config.train_config.batch_size,
                'gamma': self.config.train_config.gamma,
                'tau': self.config.train_config.tau,
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    def collect_experience(self, episode: int) -> Dict[str, float]:
        """
        收集一个 Episode 的经验
        
        Args:
            episode: 当前 Episode 编号
            
        Returns:
            Episode 统计信息
        """
        # 重置环境
        env_state, time_varying_state, _ = self.env.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        episode_costs = []  # 收集每个时隙的 cost
        policy_counts = {k: 0 for k in range(self.config.net_config.num_meta_policies)}
        
        done = False
        
        while not done:
            # 选择动作
            if self.total_steps < self.config.train_config.min_buffer_size:
                # 初始阶段使用结构化随机探索
                # 随机选择卸载比例，探索不同的混合策略
                M = self.config.env_config.num_mobile_devices
                
                # 随机选择卸载比例 (0%, 25%, 50%, 75%, 100%)
                offload_ratio = np.random.choice([0.0, 0.25, 0.5, 0.75, 1.0])
                num_offload = int(M * offload_ratio)
                
                # 随机选择哪些设备卸载
                offload_decisions = np.zeros(M)
                if num_offload > 0:
                    offload_indices = np.random.choice(M, num_offload, replace=False)
                    offload_decisions[offload_indices] = 1.0
                
                # 功率随机分配
                power_ratios = np.random.uniform(0.3, 1.0, size=M)
                
                action = np.concatenate([offload_decisions, power_ratios])
                selected_k = np.random.randint(0, self.config.net_config.num_meta_policies)
            else:
                action, selected_k = self.agent.select_action(
                    env_state, time_varying_state, 
                    deterministic=False
                )
                
            # 记录策略使用
            policy_counts[selected_k] += 1
            
            # 执行动作
            next_env_state, next_time_varying_state, reward, done, info = self.env.step(action)
            
            # 存储经验
            self.replay_buffer.push(
                env_state=env_state,
                time_varying_state=time_varying_state,
                action=action,
                reward=reward,
                next_env_state=next_env_state,
                next_time_varying_state=next_time_varying_state,
                done=done,
                selected_k=selected_k
            )
            
            # 更新状态
            env_state = next_env_state
            time_varying_state = next_time_varying_state
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            # 收集 cost
            if 'cost' in info:
                episode_costs.append(info['cost'])
            
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'policy_distribution': policy_counts,
            'avg_cost': np.mean(episode_costs) if episode_costs else 0.0
        }
    
    def train_step(self) -> Dict[str, float]:
        """
        执行一次训练更新
        
        Returns:
            训练指标
        """
        # 采样批次
        batch = self.replay_buffer.sample(self.config.train_config.batch_size)
        
        # 更新网络
        metrics = self.agent.update(batch)
        
        return metrics
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        评估智能体性能
        
        Args:
            num_episodes: 评估 Episode 数量
            
        Returns:
            评估指标
        """
        self.agent.eval()
        
        total_rewards = []
        total_delays = []
        total_energies = []
        success_rates = []
        
        for _ in range(num_episodes):
            env_state, time_varying_state, _ = self.eval_env.reset()
            episode_reward = 0.0
            episode_delays = []
            episode_energies = []
            episode_successes = []
            
            done = False
            while not done:
                action, _ = self.agent.select_action(
                    env_state, time_varying_state,
                    deterministic=True
                )
                
                next_env_state, next_time_varying_state, reward, done, info = self.eval_env.step(action)
                
                episode_reward += reward
                episode_delays.extend(info['delays'])
                episode_energies.extend(info['energies'])
                episode_successes.extend(info['successes'])
                
                env_state = next_env_state
                time_varying_state = next_time_varying_state
                
            total_rewards.append(episode_reward)
            total_delays.append(np.mean(episode_delays))
            total_energies.append(np.mean(episode_energies))
            success_rates.append(np.mean(episode_successes))
            
        self.agent.train()
        
        return {
            'eval_reward': np.mean(total_rewards),
            'eval_reward_std': np.std(total_rewards),
            'eval_delay': np.mean(total_delays),
            'eval_energy': np.mean(total_energies),
            'eval_success_rate': np.mean(success_rates)
        }
    
    def train(self):
        """执行完整的训练流程"""
        print(f"\n{'='*60}")
        print(f"开始训练 FEAT 算法")
        print(f"实验名称: {self.args.exp_name}")
        print(f"设备: {self.config.device}")
        print(f"Episode 数量: {self.config.train_config.num_episodes}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for episode in range(1, self.config.train_config.num_episodes + 1):
            self.episode_count = episode
            
            # 收集经验
            episode_stats = self.collect_experience(episode)
            
            # 记录 Episode 奖励和 cost
            self.logger.log('reward', episode_stats['episode_reward'], episode)
            self.logger.log('cost', episode_stats['avg_cost'], episode)
            
            # 训练更新
            if self.replay_buffer.is_ready(self.config.train_config.min_buffer_size):
                # 每个时隙更新一次
                for _ in range(episode_stats['episode_steps']):
                    train_metrics = self.train_step()
                    
                # 记录训练指标
                for key, value in train_metrics.items():
                    self.logger.log(key, value, episode)
                    
            # 定期日志输出
            if episode % self.args.log_freq == 0:
                avg_reward = self.logger.get_recent_average('reward', 100)
                elapsed = time.time() - start_time
                
                avg_cost = self.logger.get_recent_average('cost', 100)
                print(f"Episode {episode}/{self.config.train_config.num_episodes} | "
                      f"奖励: {episode_stats['episode_reward']:.3f} | "
                      f"平均奖励 (100): {(avg_reward if avg_reward else 0):.3f} | "
                      f"Cost: {episode_stats['avg_cost']:.4f} | "
                      f"平均Cost (100): {(avg_cost if avg_cost else 0):.4f} | "
                      f"步数: {self.total_steps} | "
                      f"时间: {elapsed/60:.1f}min")
                
                # 打印策略分布 + 多样性指标
                policy_dist = episode_stats['policy_distribution']
                total = sum(policy_dist.values())
                if total > 0:
                    dist_str = ', '.join([f"P{k}: {v/total:.1%}" for k, v in policy_dist.items()])
                    # 获取多样性相关指标
                    explore_prob = self.agent.forced_exploration_prob
                    q_entropy = self.logger.get_recent_average('q_dist_entropy', 10)
                    steerer_temp = self.agent.steerer.temperature
                    print(f"         策略分布: {dist_str} | "
                          f"探索ε: {explore_prob:.3f} | "
                          f"Q熵: {(q_entropy if q_entropy else 0):.3f} | "
                          f"τ: {steerer_temp:.4f}")
                    
            # 定期评估
            if episode % self.args.eval_freq == 0:
                eval_metrics = self.evaluate(self.config.train_config.eval_episodes)
                
                print(f"\n--- 评估结果 (Episode {episode}) ---")
                print(f"平均奖励: {eval_metrics['eval_reward']:.4f} ± {eval_metrics['eval_reward_std']:.4f}")
                print(f"平均延迟: {eval_metrics['eval_delay']*1000:.2f} ms")
                print(f"平均能耗: {eval_metrics['eval_energy']*1000:.4f} mJ")
                print(f"成功率: {eval_metrics['eval_success_rate']:.2%}\n")
                
                # 记录评估指标
                for key, value in eval_metrics.items():
                    self.logger.log(key, value, episode)
                    
                # 保存最佳模型
                if eval_metrics['eval_reward'] > self.best_eval_reward:
                    self.best_eval_reward = eval_metrics['eval_reward']
                    best_path = os.path.join(self.save_dir, 'best_model.pt')
                    self.agent.save(best_path)
                    print(f"保存最佳模型 (奖励: {self.best_eval_reward:.4f})")
                    
            # 定期保存模型
            if episode % self.args.save_freq == 0:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_ep{episode}.pt')
                self.agent.save(checkpoint_path)
                
        # 训练结束
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"训练完成!")
        print(f"总时间: {total_time/60:.1f} 分钟")
        print(f"总步数: {self.total_steps}")
        print(f"最佳评估奖励: {self.best_eval_reward:.4f}")
        print(f"{'='*60}")
        
        # 保存最终模型和日志
        final_path = os.path.join(self.save_dir, 'final_model.pt')
        self.agent.save(final_path)
        self.logger.save()
        
        return self.best_eval_reward


def main():
    """主函数"""
    args = parse_args()
    
    # 创建配置
    config = create_default_config()
    
    # 从命令行参数更新配置
    config.seed = args.seed
    config.device = args.device
    config.experiment_name = args.exp_name
    
    config.train_config.num_episodes = args.num_episodes
    config.train_config.batch_size = args.batch_size
    config.train_config.buffer_size = args.buffer_size
    config.train_config.min_buffer_size = args.min_buffer_size
    config.train_config.actor_lr = args.actor_lr
    config.train_config.critic_lr = args.critic_lr
    config.train_config.steerer_lr = args.steerer_lr
    config.train_config.gamma = args.gamma
    config.train_config.tau = args.tau
    config.train_config.alpha = args.alpha
    config.train_config.automatic_entropy_tuning = args.auto_alpha
    config.train_config.eval_frequency = args.eval_freq
    config.train_config.save_frequency = args.save_freq
    config.train_config.log_frequency = args.log_freq
    
    config.env_config.num_mobile_devices = args.num_devices
    config.env_config.bandwidth = args.bandwidth
    
    # 创建训练器并开始训练
    trainer = FEATTrainer(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
