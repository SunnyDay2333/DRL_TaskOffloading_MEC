"""
FEAT 算法评估脚本
==================
本文件实现了 FEAT 算法的评估功能，包括:
- 加载预训练模型进行评估
- 环境适应性测试 (修改带宽、任务分布等)
- 仅微调 Steerer 的快速适应
- 性能对比分析

参考论文: FEAT: Towards Fast Environment-Adaptive Task Offloading 
         and Power Allocation in MEC
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from config import (
    ExperimentConfig, 
    create_default_config, 
    create_test_config,
    EnvironmentConfig
)
from environment.mec_environment import MECEnvironment
from models.feat_agent import FEATAgent
from utils.replay_buffer import ReplayBuffer
from utils.helpers import set_seed, MetricsLogger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FEAT 算法评估')
    
    # 模型路径
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型检查点路径')
    
    # 评估设置
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='评估 Episode 数量')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    # 环境适应测试
    parser.add_argument('--test_adaptation', action='store_true',
                       help='是否进行环境适应性测试')
    parser.add_argument('--new_bandwidth', type=float, default=9e6,
                       help='新环境的带宽 (Hz)')
    parser.add_argument('--new_task_variance', type=float, default=1200e3,
                       help='新环境的任务大小方差')
    parser.add_argument('--finetune_episodes', type=int, default=200,
                       help='微调 Steerer 的 Episode 数量')
    
    # 对比测试
    parser.add_argument('--compare_baselines', action='store_true',
                       help='是否与基线算法对比')
    
    # 保存
    parser.add_argument('--save_results', action='store_true',
                       help='是否保存评估结果')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='结果保存目录')
    
    return parser.parse_args()


class Evaluator:
    """
    FEAT 评估器
    ============
    提供完整的评估功能
    """
    
    def __init__(self, 
                 model_path: str,
                 config: Optional[ExperimentConfig] = None,
                 device: str = 'cuda',
                 seed: int = 42):
        """
        初始化评估器
        
        Args:
            model_path: 模型检查点路径
            config: 实验配置 (如果为 None，使用默认配置)
            device: 计算设备
            seed: 随机种子
        """
        set_seed(seed)
        
        self.model_path = model_path
        self.config = config if config else create_default_config()
        self.device = device
        self.seed = seed
        
        # 创建环境
        self.env = MECEnvironment(config=self.config.env_config, seed=seed)
        
        # 获取维度
        state_dims = self.env.get_state_dims()
        self.env_state_dim = state_dims['env_state_dim']
        self.time_varying_state_dim = state_dims['time_varying_state_dim']
        self.action_dim = state_dims['action_dim']
        
        # 创建智能体
        self.agent = FEATAgent(
            env_state_dim=self.env_state_dim,
            time_varying_state_dim=self.time_varying_state_dim,
            action_dim=self.action_dim,
            num_policies=self.config.net_config.num_meta_policies,
            device=device
        )
        
        # 加载模型
        self.agent.load(model_path, load_optimizer=False)
        self.agent.eval()
        
        print(f"加载模型: {model_path}")
        
    def evaluate(self, 
                 num_episodes: int = 100,
                 verbose: bool = True) -> Dict[str, float]:
        """
        评估智能体性能
        
        Args:
            num_episodes: 评估 Episode 数量
            verbose: 是否打印详细信息
            
        Returns:
            评估指标字典
        """
        self.agent.eval()
        
        all_rewards = []
        all_delays = []
        all_energies = []
        all_success_rates = []
        all_costs = []  # 收集 cost
        policy_usage = {k: 0 for k in range(self.config.net_config.num_meta_policies)}
        total_steps = 0
        
        for ep in range(num_episodes):
            env_state, time_varying_state, _ = self.env.reset()
            episode_reward = 0.0
            episode_delays = []
            episode_energies = []
            episode_successes = []
            episode_costs = []  # 收集每个时隙的 cost
            
            done = False
            while not done:
                action, selected_k = self.agent.select_action(
                    env_state, time_varying_state,
                    deterministic=True
                )
                
                next_env_state, next_time_varying_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_delays.extend(info['delays'])
                episode_energies.extend(info['energies'])
                episode_successes.extend(info['successes'])
                if 'cost' in info:
                    episode_costs.append(info['cost'])
                policy_usage[selected_k] += 1
                total_steps += 1
                
                env_state = next_env_state
                time_varying_state = next_time_varying_state
                
            all_rewards.append(episode_reward)
            all_delays.append(np.mean(episode_delays))
            all_energies.append(np.mean(episode_energies))
            all_success_rates.append(np.mean(episode_successes))
            all_costs.append(np.mean(episode_costs) if episode_costs else 0.0)
            
            if verbose and (ep + 1) % 20 == 0:
                print(f"评估进度: {ep+1}/{num_episodes}, "
                      f"当前平均奖励: {np.mean(all_rewards):.4f}")
                
        # 计算策略使用比例
        total_usage = sum(policy_usage.values())
        policy_distribution = {k: v/total_usage for k, v in policy_usage.items()}
        
        results = {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_delay': np.mean(all_delays),
            'std_delay': np.std(all_delays),
            'mean_energy': np.mean(all_energies),
            'std_energy': np.std(all_energies),
            'mean_cost': np.mean(all_costs),
            'std_cost': np.std(all_costs),
            'mean_success_rate': np.mean(all_success_rates),
            'std_success_rate': np.std(all_success_rates),
            'policy_distribution': policy_distribution,
            'total_steps': total_steps
        }
        
        if verbose:
            self._print_results(results)
            
        return results
    
    def _print_results(self, results: Dict):
        """打印评估结果"""
        print(f"\n{'='*50}")
        print("评估结果")
        print(f"{'='*50}")
        print(f"平均奖励: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
        print(f"平均延迟: {results['mean_delay']*1000:.2f} ± {results['std_delay']*1000:.2f} ms")
        print(f"平均能耗: {results['mean_energy']*1000:.4f} ± {results['std_energy']*1000:.4f} mJ")
        print(f"平均Cost: {results['mean_cost']:.4f} ± {results['std_cost']:.4f}")
        print(f"成功率: {results['mean_success_rate']:.2%} ± {results['std_success_rate']:.2%}")
        print(f"\n策略使用分布:")
        for k, v in results['policy_distribution'].items():
            print(f"  策略 {k}: {v:.2%}")
        print(f"{'='*50}")
        
    def test_environment_adaptation(self,
                                    new_bandwidth: float = 9e6,
                                    new_task_variance: float = 1200e3,
                                    finetune_episodes: int = 200,
                                    eval_episodes: int = 100,
                                    verbose: bool = True) -> Dict:
        """
        测试环境适应性
        
        验证 FEAT 在新环境中只需微调 Steerer 即可快速适应
        
        流程:
        1. 在新环境中直接评估 (不微调)
        2. 仅微调 Steerer
        3. 微调后再次评估
        
        Args:
            new_bandwidth: 新环境的带宽
            new_task_variance: 新环境的任务大小方差
            finetune_episodes: 微调 Episode 数量
            eval_episodes: 评估 Episode 数量
            verbose: 是否打印详细信息
            
        Returns:
            适应测试结果
        """
        print(f"\n{'='*60}")
        print("环境适应性测试")
        print(f"新环境参数: 带宽={new_bandwidth/1e6}MHz, 任务方差={new_task_variance/1e3}Kb")
        print(f"{'='*60}")
        
        # 修改环境配置
        new_config = create_test_config(
            bandwidth=new_bandwidth,
            task_variance=new_task_variance
        )
        
        # 创建新环境
        new_env = MECEnvironment(config=new_config.env_config, seed=self.seed + 100)
        
        # ==================== 阶段1: 直接评估 (不微调) ====================
        print("\n--- 阶段1: 直接迁移评估 (不微调) ---")
        
        results_before = self._evaluate_on_env(
            new_env, eval_episodes, verbose=False
        )
        
        print(f"迁移后性能 (无微调):")
        print(f"  平均奖励: {results_before['mean_reward']:.4f}")
        print(f"  成功率: {results_before['mean_success_rate']:.2%}")
        
        # ==================== 阶段2: 仅微调 Steerer ====================
        print(f"\n--- 阶段2: 仅微调 Steerer ({finetune_episodes} Episodes) ---")
        
        # 创建微调用的回放池
        finetune_buffer = ReplayBuffer(capacity=10000)
        
        # 收集经验并微调
        self.agent.train()  # 设置为训练模式
        
        finetune_rewards = []
        for ep in range(finetune_episodes):
            env_state, time_varying_state, _ = new_env.reset()
            episode_reward = 0.0
            
            done = False
            while not done:
                action, selected_k = self.agent.select_action(
                    env_state, time_varying_state,
                    deterministic=False  # 探索模式
                )
                
                next_env_state, next_time_varying_state, reward, done, info = new_env.step(action)
                
                # 存储经验
                finetune_buffer.push(
                    env_state, time_varying_state, action, reward,
                    next_env_state, next_time_varying_state, done, selected_k
                )
                
                episode_reward += reward
                env_state = next_env_state
                time_varying_state = next_time_varying_state
                
            finetune_rewards.append(episode_reward)
            
            # 微调更新 (仅更新 Steerer)
            if len(finetune_buffer) >= 64:
                batch = finetune_buffer.sample(64)
                self.agent.update_steerer_only(batch)
                
            if verbose and (ep + 1) % 50 == 0:
                avg_reward = np.mean(finetune_rewards[-50:])
                print(f"  微调进度: {ep+1}/{finetune_episodes}, 平均奖励: {avg_reward:.4f}")
                
        # ==================== 阶段3: 微调后评估 ====================
        print("\n--- 阶段3: 微调后评估 ---")
        
        self.agent.eval()
        results_after = self._evaluate_on_env(
            new_env, eval_episodes, verbose=False
        )
        
        print(f"微调后性能:")
        print(f"  平均奖励: {results_after['mean_reward']:.4f}")
        print(f"  成功率: {results_after['mean_success_rate']:.2%}")
        
        # ==================== 结果汇总 ====================
        improvement = results_after['mean_reward'] - results_before['mean_reward']
        relative_improvement = improvement / abs(results_before['mean_reward']) * 100
        
        print(f"\n{'='*50}")
        print("适应性测试结果汇总")
        print(f"{'='*50}")
        print(f"直接迁移奖励: {results_before['mean_reward']:.4f}")
        print(f"微调后奖励: {results_after['mean_reward']:.4f}")
        print(f"绝对提升: {improvement:.4f}")
        print(f"相对提升: {relative_improvement:.1f}%")
        print(f"{'='*50}")
        
        return {
            'before_finetune': results_before,
            'after_finetune': results_after,
            'improvement': improvement,
            'relative_improvement': relative_improvement,
            'finetune_rewards': finetune_rewards,
            'new_env_config': {
                'bandwidth': new_bandwidth,
                'task_variance': new_task_variance
            }
        }
    
    def _evaluate_on_env(self, 
                         env: MECEnvironment,
                         num_episodes: int,
                         verbose: bool = False) -> Dict[str, float]:
        """在指定环境上评估"""
        all_rewards = []
        all_delays = []
        all_energies = []
        all_success_rates = []
        all_costs = []  # 收集 cost
        
        for ep in range(num_episodes):
            env_state, time_varying_state, _ = env.reset()
            episode_reward = 0.0
            episode_delays = []
            episode_energies = []
            episode_successes = []
            episode_costs = []  # 收集每个时隙的 cost
            
            done = False
            while not done:
                action, _ = self.agent.select_action(
                    env_state, time_varying_state,
                    deterministic=True
                )
                
                next_env_state, next_time_varying_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_delays.extend(info['delays'])
                episode_energies.extend(info['energies'])
                episode_successes.extend(info['successes'])
                if 'cost' in info:
                    episode_costs.append(info['cost'])
                
                env_state = next_env_state
                time_varying_state = next_time_varying_state
                
            all_rewards.append(episode_reward)
            all_delays.append(np.mean(episode_delays))
            all_energies.append(np.mean(episode_energies))
            all_success_rates.append(np.mean(episode_successes))
            all_costs.append(np.mean(episode_costs) if episode_costs else 0.0)
            
        return {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_delay': np.mean(all_delays),
            'mean_energy': np.mean(all_energies),
            'mean_cost': np.mean(all_costs),
            'mean_success_rate': np.mean(all_success_rates)
        }


class BaselineComparison:
    """
    基线算法对比
    =============
    实现多种基线算法用于性能对比
    """
    
    def __init__(self, env: MECEnvironment):
        """
        初始化基线对比器
        
        Args:
            env: MEC 环境
        """
        self.env = env
        
    def random_policy(self) -> Tuple[np.ndarray, int]:
        """随机策略"""
        action = np.random.uniform(0, 1, size=self.env.get_state_dims()['action_dim'])
        return action, 0
    
    def all_local_policy(self) -> Tuple[np.ndarray, int]:
        """全部本地执行策略"""
        M = self.env.config.num_mobile_devices
        action = np.zeros(M * 2)
        action[:M] = 0  # 不卸载
        action[M:] = 0  # 功率为0
        return action, 0
    
    def all_offload_policy(self) -> Tuple[np.ndarray, int]:
        """全部卸载策略"""
        M = self.env.config.num_mobile_devices
        action = np.zeros(M * 2)
        action[:M] = 1  # 全部卸载
        action[M:] = 1  # 最大功率
        return action, 0
    
    def threshold_policy(self, 
                        threshold: float = 0.5) -> Tuple[np.ndarray, int]:
        """
        阈值策略
        
        根据任务大小决定是否卸载
        """
        M = self.env.config.num_mobile_devices
        action = np.zeros(M * 2)
        
        for i, device in enumerate(self.env.devices):
            # 如果任务大小超过阈值，则卸载
            task_size_norm = device.current_task.size / self.env.config.task_size_mean
            if task_size_norm > threshold:
                action[i] = 1  # 卸载
                action[M + i] = 0.5  # 中等功率
            else:
                action[i] = 0  # 本地
                action[M + i] = 0
                
        return action, 0
    
    def evaluate_baseline(self,
                         policy_name: str,
                         num_episodes: int = 100) -> Dict[str, float]:
        """
        评估基线策略
        
        Args:
            policy_name: 策略名称
            num_episodes: 评估 Episode 数量
            
        Returns:
            评估结果
        """
        # 选择策略函数
        policy_fn = {
            'random': self.random_policy,
            'all_local': self.all_local_policy,
            'all_offload': self.all_offload_policy,
            'threshold': self.threshold_policy
        }.get(policy_name)
        
        if policy_fn is None:
            raise ValueError(f"未知的策略: {policy_name}")
            
        all_rewards = []
        all_delays = []
        all_energies = []
        all_success_rates = []
        all_costs = []  # 收集 cost
        
        for _ in range(num_episodes):
            env_state, time_varying_state, _ = self.env.reset()
            episode_reward = 0.0
            episode_delays = []
            episode_energies = []
            episode_successes = []
            episode_costs = []  # 收集每个时隙的 cost
            
            done = False
            while not done:
                action, _ = policy_fn()
                
                next_env_state, next_time_varying_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_delays.extend(info['delays'])
                episode_energies.extend(info['energies'])
                episode_successes.extend(info['successes'])
                if 'cost' in info:
                    episode_costs.append(info['cost'])
                
                env_state = next_env_state
                time_varying_state = next_time_varying_state
                
            all_rewards.append(episode_reward)
            all_delays.append(np.mean(episode_delays))
            all_energies.append(np.mean(episode_energies))
            all_success_rates.append(np.mean(episode_successes))
            all_costs.append(np.mean(episode_costs) if episode_costs else 0.0)
            
        return {
            'policy_name': policy_name,
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_delay': np.mean(all_delays),
            'mean_energy': np.mean(all_energies),
            'mean_cost': np.mean(all_costs),
            'mean_success_rate': np.mean(all_success_rates)
        }
    
    def compare_all(self, num_episodes: int = 100) -> List[Dict]:
        """
        比较所有基线策略
        
        Args:
            num_episodes: 评估 Episode 数量
            
        Returns:
            所有策略的评估结果
        """
        baselines = ['random', 'all_local', 'all_offload', 'threshold']
        results = []
        
        print("\n评估基线策略...")
        for policy_name in baselines:
            print(f"  评估 {policy_name}...")
            result = self.evaluate_baseline(policy_name, num_episodes)
            results.append(result)
            
        # 打印比较结果
        print(f"\n{'='*85}")
        print("基线策略对比")
        print(f"{'='*85}")
        print(f"{'策略名称':<15} {'平均奖励':>12} {'平均延迟(ms)':>15} {'平均Cost':>12} {'成功率':>10}")
        print(f"{'-'*85}")
        
        for r in results:
            print(f"{r['policy_name']:<15} {r['mean_reward']:>12.4f} "
                  f"{r['mean_delay']*1000:>15.2f} {r['mean_cost']:>12.4f} {r['mean_success_rate']:>10.2%}")
                  
        print(f"{'='*85}")
        
        return results


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建评估器
    config = create_default_config()
    config.device = args.device
    
    evaluator = Evaluator(
        model_path=args.model_path,
        config=config,
        device=args.device,
        seed=args.seed
    )
    
    # 标准评估
    print("\n" + "="*60)
    print("标准环境评估")
    print("="*60)
    
    results = evaluator.evaluate(num_episodes=args.num_episodes)
    
    # 环境适应性测试
    if args.test_adaptation:
        adaptation_results = evaluator.test_environment_adaptation(
            new_bandwidth=args.new_bandwidth,
            new_task_variance=args.new_task_variance,
            finetune_episodes=args.finetune_episodes,
            eval_episodes=args.num_episodes
        )
        
    # 基线对比
    if args.compare_baselines:
        baseline_compare = BaselineComparison(evaluator.env)
        baseline_results = baseline_compare.compare_all(num_episodes=args.num_episodes)
        
        # 添加 FEAT 结果进行对比
        print(f"\nFEAT 算法: 平均奖励={results['mean_reward']:.4f}, "
              f"成功率={results['mean_success_rate']:.2%}")
              
    # 保存结果
    if args.save_results:
        output_path = os.path.join(args.output_dir, 'evaluation_results.json')
        
        save_data = {
            'model_path': args.model_path,
            'standard_evaluation': {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in results.items() if k != 'policy_distribution'
            }
        }
        
        if args.test_adaptation:
            save_data['adaptation_test'] = {
                'improvement': adaptation_results['improvement'],
                'relative_improvement': adaptation_results['relative_improvement']
            }
            
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
