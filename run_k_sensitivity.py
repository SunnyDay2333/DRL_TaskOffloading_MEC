"""
K-Sensitivity Experiment: Varying the Number of Expert Policies
================================================================
Trains FEAT variants with different K values (number of expert policies) and
evaluates the impact on:
  - Training performance (convergence)
  - Functional diversity (Experiment A from collapse_v2)
  - Adaptation capability (Experiment C from collapse_v2)

For each K value, two variants are trained:
  - Full PDM (A+B): the proposed method
  - No PDM (vanilla): baseline without diversity maintenance

This script also reuses existing K=3 checkpoints if available.

Usage:
    python run_k_sensitivity.py --k_values 2 3 5 --num_episodes 5000 --device cuda
    python run_k_sensitivity.py --k_values 2 5 --num_episodes 5000  # skip K=3 if exists
    python run_k_sensitivity.py --skip_training --k3_ablation_dir experiments/ablation_...

Output structure:
    experiments/k_sensitivity_<ts>/
        K2/  K3/  K5/
            checkpoints/Full_PDM/  No_PDM/
            logs/Full_PDM/  No_PDM/
        plots/
            plot_data/   <-- raw data for all figures
            k_sensitivity_training.png
            k_sensitivity_diversity.png
            k_sensitivity_adaptation.png
            k_sensitivity_summary.png
        k_sensitivity_report.json
"""

import os
import sys
import copy
import json
import time
import argparse
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (ExperimentConfig, EnvironmentConfig, NetworkConfig,
                    TrainingConfig, create_default_config)
from environment.mec_environment import MECEnvironment
from models.feat_agent import FEATAgent
from models.networks import MetaPolicyNetwork
from utils.replay_buffer import ReplayBuffer
from utils.helpers import set_seed, MetricsLogger

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

PLOT_DATA_SUBDIR = 'plot_data'

PDM_VARIANTS = {
    'Full_PDM': {
        'steerer_entropy_coeff': 0.5,
        'forced_exploration_prob': 0.2,
        'forced_exploration_min': 0.02,
    },
    'No_PDM': {
        'steerer_entropy_coeff': 0.0,
        'forced_exploration_prob': 0.0,
        'forced_exploration_min': 0.0,
    },
}

TEST_ENVS = {
    'Default':     {},
    'BW_5MHz':     {'bandwidth': 5e6},
    'BW_15MHz':    {'bandwidth': 15e6},
    'ES_4GHz':     {'es_computing_capacity': 4e9},
    'Task_Heavy':  {'task_size_mean': 1000e3, 'task_size_variance': 1500e3},
    'Combined':    {'bandwidth': 7e6, 'es_computing_capacity': 6e9},
}

ADAPT_ENVS = {
    'BW_5MHz':    {'bandwidth': 5e6},
    'ES_4GHz':    {'es_computing_capacity': 4e9},
    'Combined':   {'bandwidth': 7e6, 'es_computing_capacity': 6e9},
}

K_COLORS = {2: '#e74c3c', 3: '#2ecc71', 5: '#3498db', 7: '#9b59b6'}


def _save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_convert)


# ============================================================
#  Training
# ============================================================

def structured_random_action(num_devices: int) -> np.ndarray:
    offload_ratio = np.random.choice([0.0, 0.25, 0.5, 0.75, 1.0])
    num_offload = int(num_devices * offload_ratio)
    offload = np.zeros(num_devices)
    if num_offload > 0:
        idx = np.random.choice(num_devices, num_offload, replace=False)
        offload[idx] = 1.0
    power = np.random.uniform(0.3, 1.0, size=num_devices)
    return np.concatenate([offload, power])


def train_variant(K: int, variant_name: str, pdm_params: dict,
                  num_episodes: int, seed: int, device: str,
                  save_dir: str, log_dir: str,
                  log_freq: int = 10, eval_freq: int = 50,
                  save_freq: int = 100) -> float:
    """Train a single FEAT variant with the specified K."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    set_seed(seed)

    config = create_default_config()
    config.seed = seed
    config.device = device
    config.net_config.num_meta_policies = K
    config.net_config.steerer_entropy_coeff = pdm_params['steerer_entropy_coeff']
    config.net_config.forced_exploration_prob = pdm_params['forced_exploration_prob']
    config.net_config.forced_exploration_min = pdm_params['forced_exploration_min']

    env = MECEnvironment(config=config.env_config, seed=seed)
    eval_env = MECEnvironment(config=config.env_config, seed=seed + 1000)

    dims = env.get_state_dims()
    agent = FEATAgent(
        env_state_dim=dims['env_state_dim'],
        time_varying_state_dim=dims['time_varying_state_dim'],
        action_dim=dims['action_dim'],
        num_policies=K,
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
        device=device,
    )

    buffer = ReplayBuffer(capacity=config.train_config.buffer_size)
    logger = MetricsLogger(log_dir=log_dir)

    M = config.env_config.num_mobile_devices
    min_buffer = config.train_config.min_buffer_size
    batch_size = config.train_config.batch_size
    total_steps = 0
    best_eval_reward = float('-inf')
    start_time = time.time()

    tag = f'K={K} {variant_name}'
    print(f'\n{"="*60}')
    print(f'  Training: {tag}')
    print(f'  entropy_coeff={pdm_params["steerer_entropy_coeff"]}, '
          f'explore_prob={pdm_params["forced_exploration_prob"]}')
    print(f'  Episodes: {num_episodes}  |  Device: {device}')
    print(f'{"="*60}\n')

    for ep in range(1, num_episodes + 1):
        env_state, tv_state, _ = env.reset()
        ep_reward, ep_steps = 0.0, 0
        ep_costs = []
        done = False

        while not done:
            if total_steps < min_buffer:
                action = structured_random_action(M)
                sel_k = np.random.randint(0, K)
            else:
                action, sel_k = agent.select_action(env_state, tv_state,
                                                    deterministic=False)
            next_env, next_tv, reward, done, info = env.step(action)
            buffer.push(env_state=env_state, time_varying_state=tv_state,
                        action=action, reward=reward,
                        next_env_state=next_env,
                        next_time_varying_state=next_tv,
                        done=done, selected_k=sel_k)
            env_state, tv_state = next_env, next_tv
            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            if 'cost' in info:
                ep_costs.append(info['cost'])

        logger.log('reward', ep_reward, ep)
        if ep_costs:
            logger.log('cost', float(np.mean(ep_costs)), ep)

        if buffer.is_ready(min_buffer):
            for _ in range(ep_steps):
                batch = buffer.sample(batch_size)
                metrics = agent.update(batch)
            for k_name, v in metrics.items():
                logger.log(k_name, v, ep)

        if ep % log_freq == 0:
            avg_r = logger.get_recent_average('reward', 100)
            elapsed = time.time() - start_time
            print(f'  [{tag}] Ep {ep}/{num_episodes} | '
                  f'R={ep_reward:.2f} | Avg100={avg_r or 0:.2f} | '
                  f'{elapsed/60:.1f}min')

        if ep % eval_freq == 0:
            agent.eval()
            eval_rewards = []
            for _ in range(10):
                es, tv, _ = eval_env.reset()
                er = 0.0
                d = False
                while not d:
                    a, _ = agent.select_action(es, tv, deterministic=True)
                    es, tv, r, d, _ = eval_env.step(a)
                    er += r
                eval_rewards.append(er)
            agent.train()
            eval_mean = float(np.mean(eval_rewards))
            logger.log('eval_reward', eval_mean, ep)
            logger.log('eval_reward_std', float(np.std(eval_rewards)), ep)
            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                agent.save(os.path.join(save_dir, 'best_model.pt'))

        if ep % save_freq == 0:
            agent.save(os.path.join(save_dir, f'checkpoint_ep{ep}.pt'))

    total_time = time.time() - start_time
    print(f'\n  [{tag}] Done: {total_time/60:.1f}min | Best={best_eval_reward:.2f}')
    agent.save(os.path.join(save_dir, 'final_model.pt'))
    logger.save()
    return best_eval_reward


# ============================================================
#  Experiment A: Functional Diversity (per-expert evaluation)
# ============================================================

def evaluate_expert(agent: FEATAgent, env: MECEnvironment,
                    expert_k: int, num_episodes: int = 30) -> List[float]:
    """Force a specific expert to act alone, bypassing the steerer."""
    agent.eval()
    rewards = []
    for _ in range(num_episodes):
        _, tv, _ = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            tv_t = torch.FloatTensor(tv).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                a, _ = agent.meta_policy.sample_action(tv_t, expert_k,
                                                       deterministic=True)
            a = a.cpu().numpy()[0]
            _, tv, r, done, _ = env.step(a)
            ep_r += r
        rewards.append(ep_r)
    return rewards


def run_exp_a(agent: FEATAgent, K: int, variant: str,
              seed: int = 42) -> dict:
    """Run Experiment A: functional diversity measurement."""
    config = create_default_config()
    reward_matrix = {}  # expert_k -> {env_name -> mean_reward}

    for k in range(K):
        reward_matrix[k] = {}
        for env_name, overrides in TEST_ENVS.items():
            env_cfg = copy.deepcopy(config.env_config)
            for attr, val in overrides.items():
                setattr(env_cfg, attr, val)
            env_cfg.__post_init__()
            env = MECEnvironment(config=env_cfg, seed=seed + 500)
            rewards = evaluate_expert(agent, env, k, num_episodes=30)
            reward_matrix[k][env_name] = {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'raw': [float(r) for r in rewards],
            }

    env_names = list(TEST_ENVS.keys())
    unique_best = set()
    for env_name in env_names:
        best_k = max(range(K), key=lambda k: reward_matrix[k][env_name]['mean'])
        unique_best.add(best_k)

    cross_expert_stds = []
    for env_name in env_names:
        means = [reward_matrix[k][env_name]['mean'] for k in range(K)]
        cross_expert_stds.append(float(np.std(means)))

    spec_scores = []
    for k in range(K):
        means_k = [reward_matrix[k][e]['mean'] for e in env_names]
        spec_scores.append(max(means_k) - min(means_k))

    return {
        'variant': variant,
        'K': K,
        'reward_matrix': reward_matrix,
        'unique_best_experts': len(unique_best),
        'unique_best_set': list(unique_best),
        'cross_expert_std': float(np.mean(cross_expert_stds)),
        'specialisation_score': float(np.mean(spec_scores)),
    }


# ============================================================
#  Experiment C: Adaptation Capability
# ============================================================

class SteererOnlyAdapter:
    def __init__(self, agent: FEATAgent):
        self.agent = agent
        for p in agent.meta_policy.parameters():
            p.requires_grad = False
        for p in agent.q_network.parameters():
            p.requires_grad = False

    def select_action(self, es, tv, deterministic=False):
        return self.agent.select_action(es, tv, deterministic=deterministic)

    def update(self, batch):
        return self.agent.update_steerer_only(batch)


def evaluate_agent_simple(agent, env, num_episodes=30):
    agent.eval()
    rewards = []
    for _ in range(num_episodes):
        es, tv, _ = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            a, _ = agent.select_action(es, tv, deterministic=True)
            es, tv, r, done, _ = env.step(a)
            ep_r += r
        rewards.append(ep_r)
    return float(np.mean(rewards)), float(np.std(rewards))


def run_exp_c(model_path: str, K: int, variant: str,
              finetune_episodes: int = 200,
              seeds: List[int] = [42, 43, 44],
              device: str = 'cuda') -> dict:
    """Run Experiment C: steerer-only adaptation across environments."""
    config = create_default_config()
    config.net_config.num_meta_policies = K
    dims = config.get_state_dims()
    results = {}

    for env_name, overrides in ADAPT_ENVS.items():
        env_results = {'zero_shot': [], 'adapted': [], 'gain': []}

        for seed in seeds:
            set_seed(seed)
            env_cfg = copy.deepcopy(config.env_config)
            for attr, val in overrides.items():
                setattr(env_cfg, attr, val)
            env_cfg.__post_init__()
            env = MECEnvironment(config=env_cfg, seed=seed + 200)

            agent = FEATAgent(
                env_state_dim=dims['env_state_dim'],
                time_varying_state_dim=dims['time_varying_state_dim'],
                action_dim=dims['action_dim'],
                num_policies=K,
                device=device,
            )
            agent.load(model_path, load_optimizer=False)

            zs_mean, zs_std = evaluate_agent_simple(agent, env)

            adapter = SteererOnlyAdapter(agent)
            agent.train()
            buf = ReplayBuffer(capacity=30000)
            batch_size = 64
            warmup = 10

            for ep in range(finetune_episodes):
                es, tv, _ = env.reset()
                done = False
                while not done:
                    a, sk = adapter.select_action(es, tv, deterministic=False)
                    n_es, n_tv, r, done, info = env.step(a)
                    buf.push(es, tv, a, r, n_es, n_tv, done, sk)
                    es, tv = n_es, n_tv
                if len(buf) >= batch_size and ep >= warmup:
                    for _ in range(10):
                        batch = buf.sample(batch_size)
                        adapter.update(batch)

            final_mean, final_std = evaluate_agent_simple(agent, env)
            gain = final_mean - zs_mean

            env_results['zero_shot'].append(zs_mean)
            env_results['adapted'].append(final_mean)
            env_results['gain'].append(gain)

        results[env_name] = {
            'zero_shot_mean': float(np.mean(env_results['zero_shot'])),
            'zero_shot_std': float(np.std(env_results['zero_shot'])),
            'adapted_mean': float(np.mean(env_results['adapted'])),
            'adapted_std': float(np.std(env_results['adapted'])),
            'gain_mean': float(np.mean(env_results['gain'])),
            'gain_std': float(np.std(env_results['gain'])),
            'per_seed': env_results,
        }

    avg_gain = float(np.mean([r['gain_mean'] for r in results.values()]))
    return {'variant': variant, 'K': K, 'environments': results,
            'avg_gain': avg_gain}


# ============================================================
#  Plotting
# ============================================================

def plot_all(report: dict, output_dir: str):
    """Generate all K-sensitivity plots."""
    plots_dir = os.path.join(output_dir, 'plots')
    data_dir = os.path.join(plots_dir, PLOT_DATA_SUBDIR)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    k_values = sorted(report['k_values'])

    # --- 1. Training Convergence ---
    _plot_training(report, k_values, plots_dir, data_dir)
    # --- 2. Diversity Summary ---
    _plot_diversity(report, k_values, plots_dir, data_dir)
    # --- 3. Adaptation Gain ---
    _plot_adaptation(report, k_values, plots_dir, data_dir)
    # --- 4. Grand Summary ---
    _plot_summary(report, k_values, plots_dir, data_dir)


def _smooth(values, window=100):
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(np.mean(values[start:i + 1]))
    return out


def _plot_training(report, k_values, plots_dir, data_dir):
    """Training curves for each K."""
    n_k = len(k_values)
    fig, axes = plt.subplots(1, n_k, figsize=(5.5 * n_k, 4.5), sharey=True)
    if n_k == 1:
        axes = [axes]

    plot_data = {}
    for idx, K in enumerate(k_values):
        ax = axes[idx]
        k_key = f'K{K}'
        k_data = report.get('training', {}).get(k_key, {})

        for var_name in ['Full_PDM', 'No_PDM']:
            metrics_path = k_data.get(f'{var_name}_metrics')
            if not metrics_path or not os.path.exists(metrics_path):
                continue
            with open(metrics_path, 'r') as f:
                raw = json.load(f)
            metrics = raw.get('metrics', raw)
            if 'reward' not in metrics:
                continue
            episodes = [p[0] for p in metrics['reward']]
            values = [p[1] for p in metrics['reward']]
            smoothed = _smooth(values)

            color = '#2ecc71' if var_name == 'Full_PDM' else '#e74c3c'
            label = 'Full PDM' if var_name == 'Full_PDM' else 'No PDM'
            ls = '-' if var_name == 'Full_PDM' else '--'
            ax.plot(episodes, smoothed, label=label, color=color,
                    linestyle=ls, linewidth=1.8)

            plot_data[f'{k_key}_{var_name}'] = {
                'episodes': episodes, 'smoothed': smoothed}

        ax.set_xlabel('Episode', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Reward (smoothed)', fontsize=11)
        ax.set_title(f'K = {K}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, 'k_sensitivity_training.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    _save_json(plot_data, os.path.join(data_dir, 'training_curves.json'))
    print(f'  [OK] {path}')


def _plot_diversity(report, k_values, plots_dir, data_dir):
    """Diversity metrics comparison across K."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    exp_a = report.get('exp_a', {})

    # Unique Best Experts
    ax = axes[0]
    plot_data = {'unique_best': {}}
    x = np.arange(len(k_values))
    w = 0.35
    for i, var in enumerate(['Full_PDM', 'No_PDM']):
        vals = []
        for K in k_values:
            entry = exp_a.get(f'K{K}_{var}', {})
            vals.append(entry.get('unique_best_experts', 0))
        color = '#2ecc71' if var == 'Full_PDM' else '#e74c3c'
        label = 'Full PDM' if var == 'Full_PDM' else 'No PDM'
        offset = -w / 2 + i * w
        bars = ax.bar(x + offset, vals, w * 0.9, color=color, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    str(v), ha='center', fontsize=10, fontweight='bold')
        plot_data['unique_best'][var] = vals

    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in k_values], fontsize=11)
    ax.set_ylabel('Unique Best Experts', fontsize=11)
    ax.set_title('(a) Functional Specialisation', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    # Cross-Expert Std
    ax = axes[1]
    plot_data['cross_std'] = {}
    for i, var in enumerate(['Full_PDM', 'No_PDM']):
        vals = []
        for K in k_values:
            entry = exp_a.get(f'K{K}_{var}', {})
            vals.append(entry.get('cross_expert_std', 0))
        color = '#2ecc71' if var == 'Full_PDM' else '#e74c3c'
        label = 'Full PDM' if var == 'Full_PDM' else 'No PDM'
        offset = -w / 2 + i * w
        ax.bar(x + offset, vals, w * 0.9, color=color, label=label)
        plot_data['cross_std'][var] = vals

    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in k_values], fontsize=11)
    ax.set_ylabel('Cross-Expert Reward Std', fontsize=11)
    ax.set_title('(b) Expert Behavioural Difference', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    # Specialisation Score
    ax = axes[2]
    plot_data['spec_score'] = {}
    for i, var in enumerate(['Full_PDM', 'No_PDM']):
        vals = []
        for K in k_values:
            entry = exp_a.get(f'K{K}_{var}', {})
            vals.append(entry.get('specialisation_score', 0))
        color = '#2ecc71' if var == 'Full_PDM' else '#e74c3c'
        label = 'Full PDM' if var == 'Full_PDM' else 'No PDM'
        offset = -w / 2 + i * w
        ax.bar(x + offset, vals, w * 0.9, color=color, label=label)
        plot_data['spec_score'][var] = vals

    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in k_values], fontsize=11)
    ax.set_ylabel('Specialisation Score', fontsize=11)
    ax.set_title('(c) Expert Specialisation Depth', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    path = os.path.join(plots_dir, 'k_sensitivity_diversity.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    _save_json(plot_data, os.path.join(data_dir, 'diversity_data.json'))
    print(f'  [OK] {path}')


def _plot_adaptation(report, k_values, plots_dir, data_dir):
    """Adaptation gain comparison across K."""
    exp_c = report.get('exp_c', {})
    env_names = list(ADAPT_ENVS.keys())

    fig, axes = plt.subplots(1, len(env_names) + 1,
                             figsize=(4.5 * (len(env_names) + 1), 5))

    plot_data = {}
    x = np.arange(len(k_values))
    w = 0.35

    for env_idx, env_name in enumerate(env_names):
        ax = axes[env_idx]
        plot_data[env_name] = {}
        for i, var in enumerate(['Full_PDM', 'No_PDM']):
            gains, stds = [], []
            for K in k_values:
                entry = exp_c.get(f'K{K}_{var}', {})
                env_data = entry.get('environments', {}).get(env_name, {})
                gains.append(env_data.get('gain_mean', 0))
                stds.append(env_data.get('gain_std', 0))
            color = '#2ecc71' if var == 'Full_PDM' else '#e74c3c'
            label = 'Full PDM' if var == 'Full_PDM' else 'No PDM'
            offset = -w / 2 + i * w
            ax.bar(x + offset, gains, w * 0.9, yerr=stds, color=color,
                   label=label, capsize=3, alpha=0.85)
            plot_data[env_name][var] = {'gains': gains, 'stds': stds}

        ax.set_xticks(x)
        ax.set_xticklabels([f'K={k}' for k in k_values], fontsize=11)
        ax.set_ylabel('Adaptation Gain', fontsize=11)
        ax.set_title(env_name, fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    # Average across environments
    ax = axes[-1]
    plot_data['Average'] = {}
    for i, var in enumerate(['Full_PDM', 'No_PDM']):
        avg_gains, avg_stds = [], []
        for K in k_values:
            entry = exp_c.get(f'K{K}_{var}', {})
            all_gains = [entry.get('environments', {}).get(e, {}).get('gain_mean', 0)
                         for e in env_names]
            avg_gains.append(float(np.mean(all_gains)))
            all_stds = [entry.get('environments', {}).get(e, {}).get('gain_std', 0)
                        for e in env_names]
            avg_stds.append(float(np.mean(all_stds)))
        color = '#2ecc71' if var == 'Full_PDM' else '#e74c3c'
        label = 'Full PDM' if var == 'Full_PDM' else 'No PDM'
        offset = -w / 2 + i * w
        ax.bar(x + offset, avg_gains, w * 0.9, yerr=avg_stds, color=color,
               label=label, capsize=3, alpha=0.85)
        plot_data['Average'][var] = {'gains': avg_gains, 'stds': avg_stds}

    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in k_values], fontsize=11)
    ax.set_ylabel('Avg. Adaptation Gain', fontsize=11)
    ax.set_title('Average (All Envs)', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    path = os.path.join(plots_dir, 'k_sensitivity_adaptation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    _save_json(plot_data, os.path.join(data_dir, 'adaptation_data.json'))
    print(f'  [OK] {path}')


def _plot_summary(report, k_values, plots_dir, data_dir):
    """Grand summary: 2x2 grid showing key metrics vs K."""
    exp_a = report.get('exp_a', {})
    exp_c = report.get('exp_c', {})
    training = report.get('training', {})

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    plot_data = {}
    for var, color, marker in [('Full_PDM', '#2ecc71', 'o'),
                                ('No_PDM', '#e74c3c', 's')]:
        label = 'Full PDM' if var == 'Full_PDM' else 'No PDM'

        # (0,0) Unique Best Experts vs K
        ub = [exp_a.get(f'K{K}_{var}', {}).get('unique_best_experts', 0)
              for K in k_values]
        axes[0, 0].plot(k_values, ub, f'{marker}-', color=color, label=label,
                        linewidth=2, markersize=8)
        plot_data[f'{var}_unique_best'] = ub

        # (0,1) Avg Adaptation Gain vs K
        gains = [exp_c.get(f'K{K}_{var}', {}).get('avg_gain', 0)
                 for K in k_values]
        axes[0, 1].plot(k_values, gains, f'{marker}-', color=color,
                        label=label, linewidth=2, markersize=8)
        plot_data[f'{var}_avg_gain'] = gains

        # (1,0) Specialisation Score vs K
        spec = [exp_a.get(f'K{K}_{var}', {}).get('specialisation_score', 0)
                for K in k_values]
        axes[1, 0].plot(k_values, spec, f'{marker}-', color=color,
                        label=label, linewidth=2, markersize=8)
        plot_data[f'{var}_spec_score'] = spec

        # (1,1) Training Performance vs K
        tr_perf = [training.get(f'K{K}', {}).get(f'{var}_best_reward', 0)
                   for K in k_values]
        axes[1, 1].plot(k_values, tr_perf, f'{marker}-', color=color,
                        label=label, linewidth=2, markersize=8)
        plot_data[f'{var}_training_perf'] = tr_perf

    titles = [
        ('(a) Unique Best Experts vs K', 'Unique Best Experts'),
        ('(b) Avg Adaptation Gain vs K', 'Adaptation Gain'),
        ('(c) Specialisation Score vs K', 'Specialisation Score'),
        ('(d) Best Training Reward vs K', 'Best Eval Reward'),
    ]
    for (ax, (title, ylabel)) in zip(axes.flat, titles):
        ax.set_xlabel('Number of Experts (K)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(k_values)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.4)

    plt.tight_layout()
    path = os.path.join(plots_dir, 'k_sensitivity_summary.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    _save_json(plot_data, os.path.join(data_dir, 'summary_data.json'))
    print(f'  [OK] {path}')


# ============================================================
#  Main orchestration
# ============================================================

def find_k3_checkpoints(ablation_dir: str) -> dict:
    """Try to locate existing K=3 ablation checkpoints."""
    found = {}
    for var_name, dir_name in [('Full_PDM', 'FEAT_A+B'),
                                ('No_PDM', 'FEAT_vanilla')]:
        best = os.path.join(ablation_dir, 'checkpoints', dir_name,
                            'best_model.pt')
        if os.path.exists(best):
            found[var_name] = best
            log_dir = os.path.join(ablation_dir, 'logs', dir_name)
            metrics_file = os.path.join(log_dir, 'metrics.json')
            if os.path.exists(metrics_file):
                found[f'{var_name}_metrics'] = metrics_file
    return found


def parse_args():
    p = argparse.ArgumentParser(
        description='K-Sensitivity Experiment for FEAT')
    p.add_argument('--k_values', nargs='+', type=int, default=[2, 3, 5],
                   help='K values to test (default: 2 3 5)')
    p.add_argument('--num_episodes', type=int, default=5000,
                   help='Training episodes per variant')
    p.add_argument('--finetune_episodes', type=int, default=200,
                   help='Adaptation episodes for Exp C')
    p.add_argument('--num_seeds', type=int, default=3,
                   help='Number of seeds for Exp C')
    p.add_argument('--seed', type=int, default=42,
                   help='Base seed for training')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--k3_ablation_dir', type=str, default=None,
                   help='Existing K=3 ablation dir to reuse')
    p.add_argument('--skip_training', action='store_true',
                   help='Skip training, only run analysis')
    p.add_argument('--skip_exp_a', action='store_true')
    p.add_argument('--skip_exp_c', action='store_true')
    p.add_argument('--output_dir', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.output_dir = f'experiments/k_sensitivity_{ts}'

    os.makedirs(args.output_dir, exist_ok=True)

    k_values = sorted(args.k_values)
    seeds = list(range(args.seed, args.seed + args.num_seeds))

    if args.k3_ablation_dir is None:
        candidates = sorted(glob.glob('experiments/ablation_*'))
        for c in reversed(candidates):
            if os.path.isdir(os.path.join(c, 'checkpoints')) and \
               os.path.isdir(os.path.join(c, 'logs')):
                args.k3_ablation_dir = c
                break

    print(f'\n{"#"*60}')
    print(f'  K-Sensitivity Experiment')
    print(f'  K values: {k_values}')
    print(f'  Training episodes: {args.num_episodes}')
    print(f'  Adaptation episodes: {args.finetune_episodes}')
    print(f'  Seeds: {seeds}')
    print(f'  K3 ablation dir: {args.k3_ablation_dir}')
    print(f'  Output: {args.output_dir}')
    print(f'{"#"*60}\n')

    report = {
        'k_values': k_values,
        'num_episodes': args.num_episodes,
        'finetune_episodes': args.finetune_episodes,
        'seeds': seeds,
        'training': {},
        'exp_a': {},
        'exp_c': {},
    }

    # ---- Phase 1: Training ----
    for K in k_values:
        k_key = f'K{K}'
        k_dir = os.path.join(args.output_dir, k_key)
        report['training'][k_key] = {}

        # Try to reuse K=3 from existing ablation
        if K == 3 and args.k3_ablation_dir:
            k3_found = find_k3_checkpoints(args.k3_ablation_dir)
            if 'Full_PDM' in k3_found and 'No_PDM' in k3_found:
                print(f'[K=3] Reusing existing checkpoints from '
                      f'{args.k3_ablation_dir}')
                for var in ['Full_PDM', 'No_PDM']:
                    report['training'][k_key][f'{var}_model'] = k3_found[var]
                    if f'{var}_metrics' in k3_found:
                        report['training'][k_key][f'{var}_metrics'] = \
                            k3_found[f'{var}_metrics']
                    with open(k3_found[f'{var}_metrics'], 'r') as f:
                        raw = json.load(f)
                    metrics = raw.get('metrics', raw)
                    if 'eval_reward' in metrics and metrics['eval_reward']:
                        report['training'][k_key][f'{var}_best_reward'] = \
                            max(p[1] for p in metrics['eval_reward'])
                continue

        if args.skip_training:
            for var in ['Full_PDM', 'No_PDM']:
                model_path = os.path.join(k_dir, 'checkpoints', var,
                                          'best_model.pt')
                metrics_path = os.path.join(k_dir, 'logs', var,
                                            'metrics.json')
                if os.path.exists(model_path):
                    report['training'][k_key][f'{var}_model'] = model_path
                if os.path.exists(metrics_path):
                    report['training'][k_key][f'{var}_metrics'] = metrics_path
                    with open(metrics_path, 'r') as f:
                        raw = json.load(f)
                    metrics = raw.get('metrics', raw)
                    if 'eval_reward' in metrics and metrics['eval_reward']:
                        report['training'][k_key][f'{var}_best_reward'] = \
                            max(p[1] for p in metrics['eval_reward'])
            continue

        for var_name, pdm_params in PDM_VARIANTS.items():
            save_d = os.path.join(k_dir, 'checkpoints', var_name)
            log_d = os.path.join(k_dir, 'logs', var_name)

            best = train_variant(
                K=K, variant_name=var_name, pdm_params=pdm_params,
                num_episodes=args.num_episodes, seed=args.seed,
                device=args.device, save_dir=save_d, log_dir=log_d)

            report['training'][k_key][f'{var_name}_model'] = \
                os.path.join(save_d, 'best_model.pt')
            report['training'][k_key][f'{var_name}_metrics'] = \
                os.path.join(log_d, 'metrics.json')
            report['training'][k_key][f'{var_name}_best_reward'] = best

    # ---- Phase 2: Exp A (Functional Diversity) ----
    if not args.skip_exp_a:
        print(f'\n{"="*60}')
        print('  Phase 2: Experiment A — Functional Diversity')
        print(f'{"="*60}\n')

        config = create_default_config()
        dims = config.get_state_dims()

        for K in k_values:
            k_key = f'K{K}'
            for var_name in ['Full_PDM', 'No_PDM']:
                model_path = report['training'].get(k_key, {}).get(
                    f'{var_name}_model')
                if not model_path or not os.path.exists(model_path):
                    print(f'  [SKIP] {k_key} {var_name}: model not found')
                    continue

                print(f'  Running Exp A: K={K} {var_name}...')
                agent = FEATAgent(
                    env_state_dim=dims['env_state_dim'],
                    time_varying_state_dim=dims['time_varying_state_dim'],
                    action_dim=dims['action_dim'],
                    num_policies=K,
                    device=args.device,
                )
                agent.load(model_path, load_optimizer=False)

                result = run_exp_a(agent, K, var_name, seed=args.seed)
                report['exp_a'][f'{k_key}_{var_name}'] = result
                print(f'    Unique Best: {result["unique_best_experts"]} / {K}  |  '
                      f'Spec: {result["specialisation_score"]:.2f}')

    # ---- Phase 3: Exp C (Adaptation Capability) ----
    if not args.skip_exp_c:
        print(f'\n{"="*60}')
        print('  Phase 3: Experiment C — Adaptation Capability')
        print(f'{"="*60}\n')

        for K in k_values:
            k_key = f'K{K}'
            for var_name in ['Full_PDM', 'No_PDM']:
                model_path = report['training'].get(k_key, {}).get(
                    f'{var_name}_model')
                if not model_path or not os.path.exists(model_path):
                    print(f'  [SKIP] {k_key} {var_name}: model not found')
                    continue

                print(f'  Running Exp C: K={K} {var_name} '
                      f'({len(seeds)} seeds)...')
                result = run_exp_c(
                    model_path=model_path, K=K, variant=var_name,
                    finetune_episodes=args.finetune_episodes,
                    seeds=seeds, device=args.device)
                report['exp_c'][f'{k_key}_{var_name}'] = result
                print(f'    Avg Gain: {result["avg_gain"]:.3f}')

    # ---- Save & Plot ----
    report_path = os.path.join(args.output_dir, 'k_sensitivity_report.json')
    _save_json(report, report_path)
    print(f'\n  Report saved: {report_path}')

    print('\n  Generating plots...')
    plot_all(report, args.output_dir)

    # ---- Console Summary ----
    print(f'\n{"="*80}')
    print('  K-SENSITIVITY EXPERIMENT SUMMARY')
    print(f'{"="*80}')
    print(f'\n  {"K":>3} | {"Variant":<10} | {"Best Train":>11} | '
          f'{"Unique Best":>12} | {"Spec Score":>11} | {"Avg Gain":>9}')
    print(f'  {"-"*70}')
    for K in k_values:
        k_key = f'K{K}'
        for var in ['Full_PDM', 'No_PDM']:
            tr = report['training'].get(k_key, {}).get(
                f'{var}_best_reward', '--')
            ub = report['exp_a'].get(f'{k_key}_{var}', {}).get(
                'unique_best_experts', '--')
            sp = report['exp_a'].get(f'{k_key}_{var}', {}).get(
                'specialisation_score', '--')
            ag = report['exp_c'].get(f'{k_key}_{var}', {}).get(
                'avg_gain', '--')
            tr_s = f'{tr:.2f}' if isinstance(tr, (int, float)) else tr
            sp_s = f'{sp:.2f}' if isinstance(sp, (int, float)) else sp
            ag_s = f'{ag:.3f}' if isinstance(ag, (int, float)) else ag
            print(f'  {K:>3} | {var:<10} | {tr_s:>11} | '
                  f'{str(ub):>12} | {sp_s:>11} | {ag_s:>9}')
    print(f'{"="*80}')

    print(f'\nAll results saved to: {args.output_dir}')
    print(f'Plot data for re-plotting: {args.output_dir}/plots/{PLOT_DATA_SUBDIR}/')


if __name__ == '__main__':
    main()
