"""
FEAT 实验运行脚本
==================
支持两种实验模式:
1. ablation  - 消融实验: 对比 FEAT(A+B) / A only / B only / vanilla 四个变体
2. baselines - 基线对比: 对比 FEAT vs SAC / TD3 / DDPG
3. all       - 同时运行消融和基线实验

用法:
    python run_experiments.py ablation  [--num_episodes 5000]
    python run_experiments.py baselines [--num_episodes 5000]
    python run_experiments.py all       [--num_episodes 5000]

实验结果保存在 ./experiments/<模式>_<时间戳>/ 下, 包含:
    - checkpoints/  模型权重
    - logs/         训练指标 (metrics.json)
    - plots/        对比曲线图


    使用方法
# 消融实验 (默认 5000 episodes)
python run_experiments.py ablation
# 基线对比
python run_experiments.py baselines
# 全部运行
python run_experiments.py all
# 自定义参数
python run_experiments.py ablation --num_episodes 1000 --device cpu --seed 42
输出结构
运行后会自动在 ./experiments/ 下生成带时间戳的目录:

experiments/
  ablation_2026-03-09_15-30-45/
    checkpoints/          # 每个变体的模型权重
      FEAT_A+B/
      FEAT_A_only/
      FEAT_B_only/
      FEAT_vanilla/
    logs/                 # 每个变体的 metrics.json
    plots/                # 自动生成的对比图
      reward_curve.png       训练奖励曲线
      cost_curve.png         训练 Cost 曲线
      eval_reward_curve.png  评估奖励曲线
      eval_comparison.png    评估指标柱状图 (奖励/延迟/能耗/成功率)
    summary.json          # 汇总结果
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from config import ExperimentConfig, create_default_config
from environment.mec_environment import MECEnvironment
from models.feat_agent import FEATAgent
from models.baselines import SACAgent, TD3Agent, DDPGAgent
from utils.replay_buffer import ReplayBuffer
from utils.helpers import set_seed, MetricsLogger

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ==================== 通用训练工具 ====================

def structured_random_action(num_devices: int) -> np.ndarray:
    """与 FEAT 训练相同的结构化随机探索策略, 保证公平对比"""
    offload_ratio = np.random.choice([0.0, 0.25, 0.5, 0.75, 1.0])
    num_offload = int(num_devices * offload_ratio)
    offload = np.zeros(num_devices)
    if num_offload > 0:
        idx = np.random.choice(num_devices, num_offload, replace=False)
        offload[idx] = 1.0
    power = np.random.uniform(0.3, 1.0, size=num_devices)
    return np.concatenate([offload, power])


def evaluate_agent(agent, env, num_episodes: int = 10,
                   is_feat: bool = False) -> Dict[str, float]:
    """
    通用评估函数, 同时支持 FEAT 和 baseline agent

    Args:
        agent: 要评估的智能体
        env: 评估环境
        num_episodes: 评估轮数
        is_feat: 是否为 FEAT 智能体 (接口不同)
    """
    if hasattr(agent, 'eval'):
        agent.eval()

    rewards, delays, energies, successes = [], [], [], []

    for _ in range(num_episodes):
        env_state, time_varying_state, _ = env.reset()
        ep_reward = 0.0
        ep_delays, ep_energies, ep_successes = [], [], []
        done = False

        while not done:
            if is_feat:
                action, _ = agent.select_action(
                    env_state, time_varying_state, deterministic=True)
            else:
                state = np.concatenate([env_state, time_varying_state])
                action = agent.select_action(state, deterministic=True)

            next_env, next_tv, reward, done, info = env.step(action)
            ep_reward += reward
            ep_delays.extend(info['delays'])
            ep_energies.extend(info['energies'])
            ep_successes.extend(info['successes'])
            env_state, time_varying_state = next_env, next_tv

        rewards.append(ep_reward)
        delays.append(np.mean(ep_delays))
        energies.append(np.mean(ep_energies))
        successes.append(np.mean(ep_successes))

    if hasattr(agent, 'train'):
        agent.train()

    return {
        'eval_reward': float(np.mean(rewards)),
        'eval_reward_std': float(np.std(rewards)),
        'eval_delay': float(np.mean(delays)),
        'eval_energy': float(np.mean(energies)),
        'eval_success_rate': float(np.mean(successes)),
    }


# ==================== Baseline 训练循环 ====================

def train_baseline(agent_name: str, agent, config: ExperimentConfig,
                   save_dir: str, log_dir: str,
                   num_episodes: int, log_freq: int = 10,
                   eval_freq: int = 50, save_freq: int = 100):
    """
    通用 baseline 训练循环

    复用 FEAT 的 MEC 环境和经验回放池, 只在智能体交互接口上有差异:
    - baseline 将 env_state + time_varying_state 拼接为单一状态
    - 不涉及策略选择 (selected_k 固定为 0)
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = MECEnvironment(config=config.env_config, seed=config.seed)
    eval_env = MECEnvironment(config=config.env_config, seed=config.seed + 1000)
    buffer = ReplayBuffer(capacity=config.train_config.buffer_size)
    logger = MetricsLogger(log_dir=log_dir)

    num_devices = config.env_config.num_mobile_devices
    min_buffer = config.train_config.min_buffer_size
    batch_size = config.train_config.batch_size

    total_steps = 0
    best_eval_reward = float('-inf')
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"开始训练 {agent_name}")
    print(f"Episodes: {num_episodes} | Device: {config.device}")
    print(f"{'='*60}\n")

    for episode in range(1, num_episodes + 1):
        env_state, time_varying_state, _ = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        ep_costs = []
        done = False

        while not done:
            if total_steps < min_buffer:
                action = structured_random_action(num_devices)
            else:
                state = np.concatenate([env_state, time_varying_state])
                action = agent.select_action(state, deterministic=False)

            next_env, next_tv, reward, done, info = env.step(action)

            buffer.push(
                env_state=env_state,
                time_varying_state=time_varying_state,
                action=action, reward=reward,
                next_env_state=next_env,
                next_time_varying_state=next_tv,
                done=done, selected_k=0
            )

            env_state, time_varying_state = next_env, next_tv
            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            if 'cost' in info:
                ep_costs.append(info['cost'])

        logger.log('reward', ep_reward, episode)
        if ep_costs:
            logger.log('cost', float(np.mean(ep_costs)), episode)

        # 训练更新
        if buffer.is_ready(min_buffer):
            for _ in range(ep_steps):
                raw = buffer.sample(batch_size)
                batch = {
                    'states': np.concatenate(
                        [raw['env_states'], raw['time_varying_states']], axis=1),
                    'actions': raw['actions'],
                    'rewards': raw['rewards'],
                    'next_states': np.concatenate(
                        [raw['next_env_states'], raw['next_time_varying_states']], axis=1),
                    'dones': raw['dones'],
                }
                metrics = agent.update(batch)

            for k, v in metrics.items():
                logger.log(k, v, episode)

        # 日志
        if episode % log_freq == 0:
            avg_r = logger.get_recent_average('reward', 100)
            avg_c = logger.get_recent_average('cost', 100)
            elapsed = time.time() - start_time
            print(f"[{agent_name}] Ep {episode}/{num_episodes} | "
                  f"奖励: {ep_reward:.3f} | "
                  f"平均奖励(100): {(avg_r if avg_r else 0):.3f} | "
                  f"Cost: {(float(np.mean(ep_costs)) if ep_costs else 0):.4f} | "
                  f"平均Cost(100): {(avg_c if avg_c else 0):.4f} | "
                  f"步数: {total_steps} | "
                  f"时间: {elapsed/60:.1f}min")

        # 评估
        if episode % eval_freq == 0:
            eval_m = evaluate_agent(agent, eval_env, num_episodes=10,
                                    is_feat=False)
            print(f"\n--- [{agent_name}] 评估 (Ep {episode}) ---")
            print(f"  奖励: {eval_m['eval_reward']:.4f} ± "
                  f"{eval_m['eval_reward_std']:.4f}")
            print(f"  延迟: {eval_m['eval_delay']*1000:.2f} ms | "
                  f"能耗: {eval_m['eval_energy']*1000:.4f} mJ | "
                  f"成功率: {eval_m['eval_success_rate']:.2%}\n")

            for k, v in eval_m.items():
                logger.log(k, v, episode)

            if eval_m['eval_reward'] > best_eval_reward:
                best_eval_reward = eval_m['eval_reward']
                agent.save(os.path.join(save_dir, 'best_model.pt'))
                print(f"  >> 保存最佳模型 (奖励: {best_eval_reward:.4f})")

        # 定期保存
        if episode % save_freq == 0:
            agent.save(os.path.join(save_dir, f'checkpoint_ep{episode}.pt'))

    # 训练结束
    total_time = time.time() - start_time
    print(f"\n[{agent_name}] 训练完成 | "
          f"耗时: {total_time/60:.1f}min | "
          f"最佳奖励: {best_eval_reward:.4f}")

    agent.save(os.path.join(save_dir, 'final_model.pt'))
    logger.save()
    return best_eval_reward


# ==================== FEAT 训练循环 (消融 / 对比共用) ====================

def train_feat_variant(variant_name: str, config: ExperimentConfig,
                       save_dir: str, log_dir: str,
                       num_episodes: int, log_freq: int = 10,
                       eval_freq: int = 50, save_freq: int = 100):
    """
    训练一个 FEAT 变体 (消融实验或完整模型对比)
    直接使用 FEATAgent, 避免依赖 train.py 的 argparse
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    set_seed(config.seed)

    env = MECEnvironment(config=config.env_config, seed=config.seed)
    eval_env = MECEnvironment(config=config.env_config, seed=config.seed + 1000)

    dims = env.get_state_dims()
    env_state_dim = dims['env_state_dim']
    tv_state_dim = dims['time_varying_state_dim']
    action_dim = dims['action_dim']

    agent = FEATAgent(
        env_state_dim=env_state_dim,
        time_varying_state_dim=tv_state_dim,
        action_dim=action_dim,
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
        steerer_entropy_coeff=config.net_config.steerer_entropy_coeff,
        forced_exploration_prob=config.net_config.forced_exploration_prob,
        forced_exploration_decay=config.net_config.forced_exploration_decay,
        forced_exploration_min=config.net_config.forced_exploration_min,
        device=config.device,
    )

    buffer = ReplayBuffer(capacity=config.train_config.buffer_size)
    logger = MetricsLogger(log_dir=log_dir)

    num_devices = config.env_config.num_mobile_devices
    K = config.net_config.num_meta_policies
    min_buffer = config.train_config.min_buffer_size
    batch_size = config.train_config.batch_size

    total_steps = 0
    best_eval_reward = float('-inf')
    start_time = time.time()

    # 保存配置
    cfg_dict = {
        'variant': variant_name,
        'steerer_entropy_coeff': config.net_config.steerer_entropy_coeff,
        'forced_exploration_prob': config.net_config.forced_exploration_prob,
        'forced_exploration_min': config.net_config.forced_exploration_min,
        'seed': config.seed,
        'num_episodes': num_episodes,
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(cfg_dict, f, indent=2)

    print(f"\n{'='*60}")
    print(f"开始训练 {variant_name}")
    print(f"  熵正则化系数 λ_entropy = {config.net_config.steerer_entropy_coeff}")
    print(f"  强制探索概率 ε₀ = {config.net_config.forced_exploration_prob}")
    print(f"  Episodes: {num_episodes} | Device: {config.device}")
    print(f"{'='*60}\n")

    for episode in range(1, num_episodes + 1):
        env_state, time_varying_state, _ = env.reset()
        ep_reward, ep_steps = 0.0, 0
        ep_costs = []
        policy_counts = {k: 0 for k in range(K)}
        done = False

        while not done:
            if total_steps < min_buffer:
                action = structured_random_action(num_devices)
                selected_k = np.random.randint(0, K)
            else:
                action, selected_k = agent.select_action(
                    env_state, time_varying_state, deterministic=False)

            policy_counts[selected_k] += 1
            next_env, next_tv, reward, done, info = env.step(action)

            buffer.push(
                env_state=env_state,
                time_varying_state=time_varying_state,
                action=action, reward=reward,
                next_env_state=next_env,
                next_time_varying_state=next_tv,
                done=done, selected_k=selected_k,
            )

            env_state, time_varying_state = next_env, next_tv
            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            if 'cost' in info:
                ep_costs.append(info['cost'])

        logger.log('reward', ep_reward, episode)
        if ep_costs:
            logger.log('cost', float(np.mean(ep_costs)), episode)

        # 训练更新
        if buffer.is_ready(min_buffer):
            for _ in range(ep_steps):
                batch = buffer.sample(batch_size)
                train_metrics = agent.update(batch)
            for k, v in train_metrics.items():
                logger.log(k, v, episode)

        # 日志
        if episode % log_freq == 0:
            avg_r = logger.get_recent_average('reward', 100)
            avg_c = logger.get_recent_average('cost', 100)
            elapsed = time.time() - start_time
            total = sum(policy_counts.values())
            dist_str = ', '.join(
                [f"P{k}: {v/total:.0%}" for k, v in policy_counts.items()]
            ) if total > 0 else ''
            print(f"[{variant_name}] Ep {episode}/{num_episodes} | "
                  f"奖励: {ep_reward:.3f} | "
                  f"平均(100): {(avg_r if avg_r else 0):.3f} | "
                  f"Cost: {(float(np.mean(ep_costs)) if ep_costs else 0):.4f} | "
                  f"策略: [{dist_str}] | "
                  f"ε: {agent.forced_exploration_prob:.3f} | "
                  f"时间: {elapsed/60:.1f}min")

        # 评估
        if episode % eval_freq == 0:
            eval_m = evaluate_agent(agent, eval_env, num_episodes=10,
                                    is_feat=True)
            print(f"\n--- [{variant_name}] 评估 (Ep {episode}) ---")
            print(f"  奖励: {eval_m['eval_reward']:.4f} ± "
                  f"{eval_m['eval_reward_std']:.4f}")
            print(f"  延迟: {eval_m['eval_delay']*1000:.2f} ms | "
                  f"能耗: {eval_m['eval_energy']*1000:.4f} mJ | "
                  f"成功率: {eval_m['eval_success_rate']:.2%}\n")
            for k, v in eval_m.items():
                logger.log(k, v, episode)

            if eval_m['eval_reward'] > best_eval_reward:
                best_eval_reward = eval_m['eval_reward']
                agent.save(os.path.join(save_dir, 'best_model.pt'))
                print(f"  >> 保存最佳模型 (奖励: {best_eval_reward:.4f})")

        if episode % save_freq == 0:
            agent.save(os.path.join(save_dir, f'checkpoint_ep{episode}.pt'))

    total_time = time.time() - start_time
    print(f"\n[{variant_name}] 训练完成 | "
          f"耗时: {total_time/60:.1f}min | "
          f"最佳奖励: {best_eval_reward:.4f}")

    agent.save(os.path.join(save_dir, 'final_model.pt'))
    logger.save()
    return best_eval_reward


# ==================== 绘图 ====================

def plot_comparison(log_dirs: Dict[str, str], save_path: str,
                    title_prefix: str = ''):
    """
    加载各实验的 metrics.json 并绘制对比曲线

    Args:
        log_dirs: {方法名: log目录} 映射
        save_path: 图片保存路径
        title_prefix: 图标题前缀
    """
    if not HAS_MATPLOTLIB:
        print("[警告] 未安装 matplotlib, 跳过绘图。pip install matplotlib 后可生成图表。")
        return

    os.makedirs(save_path, exist_ok=True)
    all_data = {}

    for name, d in log_dirs.items():
        fpath = os.path.join(d, 'metrics.json')
        if not os.path.exists(fpath):
            print(f"[警告] 未找到 {fpath}, 跳过 {name}")
            continue
        with open(fpath, 'r') as f:
            all_data[name] = json.load(f)['metrics']

    if not all_data:
        print("[警告] 没有可用数据, 跳过绘图")
        return

    def smooth(values, window=50):
        if len(values) < window:
            return values
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(values[start:i+1]))
        return smoothed

    # ---------- 1. 训练奖励曲线 ----------
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for name, data in all_data.items():
        if 'reward' in data:
            episodes = [p[0] for p in data['reward']]
            values = [p[1] for p in data['reward']]
            smoothed = smooth(values)
            ax.plot(episodes, smoothed, label=name, linewidth=1.5)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward (smoothed)', fontsize=12)
    ax.set_title(f'{title_prefix}Training Reward Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'reward_curve.png'), dpi=150)
    plt.close(fig)

    # ---------- 2. 训练 Cost 曲线 ----------
    has_cost = any('cost' in d for d in all_data.values())
    if has_cost:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for name, data in all_data.items():
            if 'cost' in data:
                episodes = [p[0] for p in data['cost']]
                values = [p[1] for p in data['cost']]
                smoothed = smooth(values)
                ax.plot(episodes, smoothed, label=name, linewidth=1.5)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Cost (smoothed)', fontsize=12)
        ax.set_title(f'{title_prefix}Training Cost Comparison', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, 'cost_curve.png'), dpi=150)
        plt.close(fig)

    # ---------- 3. 评估指标柱状图 ----------
    eval_names, eval_rewards, eval_delays, eval_energies, eval_successes = \
        [], [], [], [], []
    for name, data in all_data.items():
        if 'eval_reward' in data and data['eval_reward']:
            eval_names.append(name)
            eval_rewards.append(data['eval_reward'][-1][1])
            eval_delays.append(
                data['eval_delay'][-1][1] * 1000 if 'eval_delay' in data else 0)
            eval_energies.append(
                data['eval_energy'][-1][1] * 1000 if 'eval_energy' in data else 0)
            eval_successes.append(
                data['eval_success_rate'][-1][1] * 100
                if 'eval_success_rate' in data else 0)

    if eval_names:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        metrics_data = [
            (eval_rewards, 'Reward', 'Reward'),
            (eval_delays, 'Delay (ms)', 'Average Delay'),
            (eval_energies, 'Energy (mJ)', 'Average Energy'),
            (eval_successes, 'Success Rate (%)', 'Success Rate'),
        ]
        colors = plt.cm.Set2(np.linspace(0, 1, len(eval_names)))
        for ax, (vals, ylabel, title) in zip(axes, metrics_data):
            bars = ax.bar(range(len(eval_names)), vals, color=colors)
            ax.set_xticks(range(len(eval_names)))
            ax.set_xticklabels(eval_names, rotation=30, ha='right', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f'{title_prefix}{title}', fontsize=11)
            ax.grid(True, axis='y', alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(save_path, 'eval_comparison.png'), dpi=150)
        plt.close(fig)

    # ---------- 4. 评估奖励曲线 ----------
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for name, data in all_data.items():
        if 'eval_reward' in data:
            episodes = [p[0] for p in data['eval_reward']]
            values = [p[1] for p in data['eval_reward']]
            ax.plot(episodes, values, 'o-', label=name, linewidth=1.5,
                    markersize=3)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Eval Reward', fontsize=12)
    ax.set_title(f'{title_prefix}Evaluation Reward Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'eval_reward_curve.png'), dpi=150)
    plt.close(fig)

    print(f"[绘图] 对比图已保存到 {save_path}/")


# ==================== 消融实验 ====================

def run_ablation(args):
    """
    消融实验: 验证方案 A (Steerer 熵正则化) 和方案 B (强制探索) 的各自贡献

    对比四个变体:
    +--------------+------------------+---------------------+
    | 变体         | 方案A (熵正则化) | 方案B (强制探索)    |
    +--------------+------------------+---------------------+
    | FEAT (A+B)   | ✓ λ=0.5          | ✓ ε₀=0.2           |
    | FEAT (A only)| ✓ λ=0.5          | ✗ ε=0              |
    | FEAT (B only)| ✗ λ=0            | ✓ ε₀=0.2           |
    | FEAT vanilla | ✗ λ=0            | ✗ ε=0              |
    +--------------+------------------+---------------------+
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join(args.output_dir, f"ablation_{timestamp}")

    variants = {
        'FEAT_A+B': {
            'steerer_entropy_coeff': 0.5,
            'forced_exploration_prob': 0.2,
            'forced_exploration_min': 0.02,
        },
        'FEAT_A_only': {
            'steerer_entropy_coeff': 0.5,
            'forced_exploration_prob': 0.0,
            'forced_exploration_min': 0.0,
        },
        'FEAT_B_only': {
            'steerer_entropy_coeff': 0.0,
            'forced_exploration_prob': 0.2,
            'forced_exploration_min': 0.02,
        },
        'FEAT_vanilla': {
            'steerer_entropy_coeff': 0.0,
            'forced_exploration_prob': 0.0,
            'forced_exploration_min': 0.0,
        },
    }

    log_dirs = {}
    results = {}

    for name, params in variants.items():
        config = create_default_config()
        config.seed = args.seed
        config.device = args.device
        config.net_config.steerer_entropy_coeff = params['steerer_entropy_coeff']
        config.net_config.forced_exploration_prob = params['forced_exploration_prob']
        config.net_config.forced_exploration_min = params['forced_exploration_min']

        save_d = os.path.join(base_dir, 'checkpoints', name)
        log_d = os.path.join(base_dir, 'logs', name)
        log_dirs[name] = log_d

        best = train_feat_variant(
            variant_name=name, config=config,
            save_dir=save_d, log_dir=log_d,
            num_episodes=args.num_episodes,
            log_freq=args.log_freq,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
        )
        results[name] = best

    # 汇总
    print(f"\n{'='*60}")
    print("消融实验结果汇总")
    print(f"{'='*60}")
    for name, reward in results.items():
        print(f"  {name:20s}  最佳评估奖励: {reward:.4f}")
    print(f"{'='*60}\n")

    # 绘图
    plot_comparison(log_dirs, os.path.join(base_dir, 'plots'),
                    title_prefix='Ablation: ')

    # 保存汇总
    summary = {
        'timestamp': timestamp,
        'mode': 'ablation',
        'args': vars(args),
        'results': results,
    }
    with open(os.path.join(base_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"实验目录: {base_dir}")
    return results


# ==================== Baseline 对比实验 ====================

def run_baselines(args):
    """
    基线对比实验: FEAT (A+B) vs SAC / TD3 / DDPG

    所有方法共享:
    - 相同的 MEC 环境和参数
    - 相同的隐藏层维度 (256, 256)
    - 相同的学习率和折扣因子
    - 相同的经验回放池和初始探索策略
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join(args.output_dir, f"baselines_{timestamp}")

    config = create_default_config()
    config.seed = args.seed
    config.device = args.device

    dims = config.get_state_dims()
    state_dim = dims['env_state_dim'] + dims['time_varying_state_dim']
    action_dim = dims['action_dim']

    device = args.device
    gamma = config.train_config.gamma
    tau = config.train_config.tau

    baseline_specs = {
        'SAC': lambda: SACAgent(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dims=(256, 256),
            actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
            gamma=gamma, tau=tau,
            automatic_entropy_tuning=True, device=device,
        ),
        'TD3': lambda: TD3Agent(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dims=(256, 256),
            actor_lr=3e-4, critic_lr=3e-4,
            gamma=gamma, tau=tau,
            exploration_noise=0.1, policy_noise=0.2,
            noise_clip=0.5, policy_delay=2, device=device,
        ),
        'DDPG': lambda: DDPGAgent(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dims=(256, 256),
            actor_lr=1e-4, critic_lr=3e-4,
            gamma=gamma, tau=tau,
            exploration_noise=0.1, device=device,
        ),
    }

    log_dirs = {}
    results = {}

    # 先训练 FEAT (A+B) 作为对照
    feat_config = create_default_config()
    feat_config.seed = args.seed
    feat_config.device = args.device
    feat_save = os.path.join(base_dir, 'checkpoints', 'FEAT')
    feat_log = os.path.join(base_dir, 'logs', 'FEAT')
    log_dirs['FEAT'] = feat_log

    results['FEAT'] = train_feat_variant(
        variant_name='FEAT', config=feat_config,
        save_dir=feat_save, log_dir=feat_log,
        num_episodes=args.num_episodes,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
    )

    # 训练各 baseline
    for name, agent_factory in baseline_specs.items():
        set_seed(args.seed)
        agent = agent_factory()
        save_d = os.path.join(base_dir, 'checkpoints', name)
        log_d = os.path.join(base_dir, 'logs', name)
        log_dirs[name] = log_d

        bl_config = create_default_config()
        bl_config.seed = args.seed
        bl_config.device = args.device

        results[name] = train_baseline(
            agent_name=name, agent=agent, config=bl_config,
            save_dir=save_d, log_dir=log_d,
            num_episodes=args.num_episodes,
            log_freq=args.log_freq,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
        )

    # 汇总
    print(f"\n{'='*60}")
    print("Baseline 对比实验结果汇总")
    print(f"{'='*60}")
    for name, reward in results.items():
        print(f"  {name:10s}  最佳评估奖励: {reward:.4f}")
    print(f"{'='*60}\n")

    plot_comparison(log_dirs, os.path.join(base_dir, 'plots'),
                    title_prefix='Baseline: ')

    summary = {
        'timestamp': timestamp,
        'mode': 'baselines',
        'args': vars(args),
        'results': results,
    }
    with open(os.path.join(base_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"实验目录: {base_dir}")
    return results


# ==================== 入口 ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description='FEAT 消融实验 & Baseline 对比',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_experiments.py ablation              # 消融实验 (默认5000 ep)
  python run_experiments.py baselines             # Baseline 对比
  python run_experiments.py all                   # 全部运行
  python run_experiments.py ablation --num_episodes 1000 --device cpu
        """)

    parser.add_argument('mode', choices=['ablation', 'baselines', 'all'],
                        help='实验模式: ablation=消融, baselines=基线对比, all=全部')
    parser.add_argument('--num_episodes', type=int, default=5000,
                        help='训练 Episode 数 (默认 5000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认 42)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (默认 cuda)')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                        help='实验输出根目录 (默认 ./experiments)')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='日志频率 (默认 10)')
    parser.add_argument('--eval_freq', type=int, default=50,
                        help='评估频率 (默认 50)')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='保存频率 (默认 100)')
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\n{'#'*60}")
    print(f"# FEAT 实验脚本")
    print(f"# 模式: {args.mode}")
    print(f"# Episodes: {args.num_episodes}")
    print(f"# 种子: {args.seed}")
    print(f"# 设备: {args.device}")
    print(f"{'#'*60}\n")

    if args.mode == 'ablation':
        run_ablation(args)
    elif args.mode == 'baselines':
        run_baselines(args)
    elif args.mode == 'all':
        run_ablation(args)
        run_baselines(args)


if __name__ == "__main__":
    main()
