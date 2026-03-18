"""
Multi-Seed Baseline Training for Statistical Significance
==========================================================
Trains FEAT and SAC with 3 seeds each and computes mean +/- std.
Also investigates TD3/DDPG performance with improved hyperparameters.

Usage:
    python run_multi_seed_baselines.py --seeds 42 43 44 --num_episodes 5000 --device cuda
    python run_multi_seed_baselines.py --quick  # 2000 episodes for quick test
"""

import os
import sys
import json
import time
import argparse
import copy
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig, create_default_config
from environment.mec_environment import MECEnvironment
from models.feat_agent import FEATAgent
from models.baselines import SACAgent, TD3Agent, DDPGAgent
from utils.replay_buffer import ReplayBuffer
from utils.helpers import set_seed, MetricsLogger

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


def structured_random_action(M: int) -> np.ndarray:
    offload_ratio = np.random.choice([0.0, 0.25, 0.5, 0.75, 1.0])
    num_offload = int(M * offload_ratio)
    offload = np.zeros(M)
    if num_offload > 0:
        idx = np.random.choice(M, num_offload, replace=False)
        offload[idx] = 1.0
    power = np.random.uniform(0.3, 1.0, size=M)
    return np.concatenate([offload, power])


def train_feat(config, seed, num_episodes, device, save_dir, log_dir):
    """Train FEAT with Full PDM."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    set_seed(seed)

    config = copy.deepcopy(config)
    config.seed = seed
    config.device = device

    env = MECEnvironment(config=config.env_config, seed=seed)
    eval_env = MECEnvironment(config=config.env_config, seed=seed + 1000)
    dims = env.get_state_dims()

    agent = FEATAgent(
        env_state_dim=dims['env_state_dim'],
        time_varying_state_dim=dims['time_varying_state_dim'],
        action_dim=dims['action_dim'],
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
        device=device,
    )

    buffer = ReplayBuffer(capacity=config.train_config.buffer_size)
    logger = MetricsLogger(log_dir=log_dir)
    M = config.env_config.num_mobile_devices
    min_buf = config.train_config.min_buffer_size
    bs = config.train_config.batch_size
    total_steps = 0
    best_eval = float('-inf')
    start = time.time()

    K = config.net_config.num_meta_policies

    for ep in range(1, num_episodes + 1):
        es, tv, _ = env.reset()
        ep_r, ep_steps = 0.0, 0
        done = False

        while not done:
            if total_steps < min_buf:
                a = structured_random_action(M)
                sk = np.random.randint(0, K)
            else:
                a, sk = agent.select_action(es, tv, deterministic=False)
            n_es, n_tv, r, done, info = env.step(a)
            buffer.push(es, tv, a, r, n_es, n_tv, done, sk)
            es, tv = n_es, n_tv
            ep_r += r
            ep_steps += 1
            total_steps += 1

        logger.log('reward', ep_r, ep)

        if buffer.is_ready(min_buf):
            for _ in range(ep_steps):
                batch = buffer.sample(bs)
                agent.update(batch)

        if ep % 50 == 0:
            agent.eval()
            eval_rs = []
            for _ in range(10):
                e_es, e_tv, _ = eval_env.reset()
                er = 0.0
                d = False
                while not d:
                    ea, _ = agent.select_action(e_es, e_tv, deterministic=True)
                    e_es, e_tv, r, d, _ = eval_env.step(ea)
                    er += r
                eval_rs.append(er)
            agent.train()
            em = float(np.mean(eval_rs))
            logger.log('eval_reward', em, ep)
            if em > best_eval:
                best_eval = em
                agent.save(os.path.join(save_dir, 'best_model.pt'))

        if ep % 100 == 0:
            elapsed = time.time() - start
            avg = logger.get_recent_average('reward', 100)
            print(f'  [FEAT s={seed}] Ep {ep}/{num_episodes} | '
                  f'R={ep_r:.2f} | Avg100={avg or 0:.2f} | {elapsed/60:.1f}min')

    logger.save()
    agent.save(os.path.join(save_dir, 'final_model.pt'))
    return best_eval


def train_sac(config, seed, num_episodes, device, save_dir, log_dir):
    """Train SAC baseline."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    set_seed(seed)

    config = copy.deepcopy(config)
    config.seed = seed

    env = MECEnvironment(config=config.env_config, seed=seed)
    eval_env = MECEnvironment(config=config.env_config, seed=seed + 1000)
    dims = env.get_state_dims()
    state_dim = dims['env_state_dim'] + dims['time_varying_state_dim']
    action_dim = dims['action_dim']

    agent = SACAgent(
        state_dim=state_dim, action_dim=action_dim,
        actor_lr=3e-4, critic_lr=3e-4,
        gamma=0.95, tau=0.005,
        device=device,
    )

    buffer = ReplayBuffer(capacity=config.train_config.buffer_size)
    logger = MetricsLogger(log_dir=log_dir)
    M = config.env_config.num_mobile_devices
    min_buf = config.train_config.min_buffer_size
    bs = config.train_config.batch_size
    total_steps = 0
    best_eval = float('-inf')
    start = time.time()

    for ep in range(1, num_episodes + 1):
        es, tv, _ = env.reset()
        ep_r, ep_steps = 0.0, 0
        done = False

        while not done:
            if total_steps < min_buf:
                a = structured_random_action(M)
            else:
                state = np.concatenate([es, tv])
                a = agent.select_action(state, deterministic=False)
            n_es, n_tv, r, done, info = env.step(a)
            buffer.push(es, tv, a, r, n_es, n_tv, done, 0)
            es, tv = n_es, n_tv
            ep_r += r
            ep_steps += 1
            total_steps += 1

        logger.log('reward', ep_r, ep)

        if buffer.is_ready(min_buf):
            for _ in range(ep_steps):
                batch = buffer.sample(bs)
                sac_batch = {
                    'states': np.concatenate([batch['env_states'],
                                              batch['time_varying_states']], axis=1),
                    'actions': batch['actions'],
                    'rewards': batch['rewards'],
                    'next_states': np.concatenate([batch['next_env_states'],
                                                   batch['next_time_varying_states']], axis=1),
                    'dones': batch['dones'],
                }
                agent.update(sac_batch)

        if ep % 50 == 0:
            agent.eval()
            eval_rs = []
            for _ in range(10):
                e_es, e_tv, _ = eval_env.reset()
                er = 0.0
                d = False
                while not d:
                    state = np.concatenate([e_es, e_tv])
                    ea = agent.select_action(state, deterministic=True)
                    e_es, e_tv, r, d, _ = eval_env.step(ea)
                    er += r
                eval_rs.append(er)
            agent.train()
            em = float(np.mean(eval_rs))
            logger.log('eval_reward', em, ep)
            if em > best_eval:
                best_eval = em
                agent.save(os.path.join(save_dir, 'best_model.pt'))

        if ep % 100 == 0:
            elapsed = time.time() - start
            avg = logger.get_recent_average('reward', 100)
            print(f'  [SAC s={seed}] Ep {ep}/{num_episodes} | '
                  f'R={ep_r:.2f} | Avg100={avg or 0:.2f} | {elapsed/60:.1f}min')

    logger.save()
    agent.save(os.path.join(save_dir, 'final_model.pt'))
    return best_eval


def plot_multi_seed(report, output_dir):
    """Plot multi-seed comparison."""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    colors = {'FEAT': '#27ae60', 'SAC': '#3498db', 'TD3': '#e74c3c', 'DDPG': '#f39c12'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: eval reward curves (mean +/- std across seeds)
    ax = axes[0]
    for algo_name, algo_data in report['algorithms'].items():
        all_curves = []
        for seed_data in algo_data['seeds']:
            metrics_path = seed_data.get('metrics_path')
            if metrics_path and os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    raw = json.load(f)
                metrics = raw.get('metrics', raw)
                if 'eval_reward' in metrics and metrics['eval_reward']:
                    eps = [p[0] for p in metrics['eval_reward']]
                    vals = [p[1] for p in metrics['eval_reward']]
                    all_curves.append((eps, vals))

        if not all_curves:
            continue

        min_len = min(len(c[1]) for c in all_curves)
        arr = np.array([c[1][:min_len] for c in all_curves])
        eps = all_curves[0][0][:min_len]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        c = colors.get(algo_name, 'gray')
        ax.plot(eps, mean, color=c, label=algo_name, linewidth=2)
        if arr.shape[0] > 1:
            ax.fill_between(eps, mean - std, mean + std, color=c, alpha=0.15)

    ax.set_xlabel('Training Episode', fontsize=11)
    ax.set_ylabel('Evaluation Reward', fontsize=11)
    ax.set_title('Training Convergence (multi-seed)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: final eval reward bar chart
    ax = axes[1]
    names, means, stds, cols = [], [], [], []
    for algo_name, algo_data in report['algorithms'].items():
        best_rewards = [s['best_eval_reward'] for s in algo_data['seeds']
                        if s.get('best_eval_reward') is not None]
        if best_rewards:
            names.append(algo_name)
            means.append(float(np.mean(best_rewards)))
            stds.append(float(np.std(best_rewards)))
            cols.append(colors.get(algo_name, 'gray'))

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=cols, capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Best Eval Reward', fontsize=11)
    ax.set_title('Final Performance (multi-seed)', fontsize=13, fontweight='bold')
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.2,
                f'{m:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    path = os.path.join(plots_dir, 'multi_seed_comparison.png')
    plt.savefig(path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f'  [OK] {path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--num_episodes', type=int, default=5000)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--skip_feat', action='store_true')
    p.add_argument('--skip_sac', action='store_true')
    args = p.parse_args()

    if args.quick:
        args.num_episodes = 2000

    if args.output_dir is None:
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.output_dir = f'experiments/multi_seed_{ts}'

    os.makedirs(args.output_dir, exist_ok=True)
    config = create_default_config()

    print(f'\n{"#"*60}')
    print(f'  Multi-Seed Baseline Training')
    print(f'  Seeds: {args.seeds}')
    print(f'  Episodes: {args.num_episodes}')
    print(f'  Device: {args.device}')
    print(f'  Output: {args.output_dir}')
    print(f'{"#"*60}\n')

    report = {
        'seeds': args.seeds,
        'num_episodes': args.num_episodes,
        'algorithms': {},
    }

    # Train FEAT
    if not args.skip_feat:
        report['algorithms']['FEAT'] = {'seeds': []}
        for seed in args.seeds:
            save_d = os.path.join(args.output_dir, 'FEAT', f'seed_{seed}', 'checkpoints')
            log_d = os.path.join(args.output_dir, 'FEAT', f'seed_{seed}', 'logs')
            print(f'\n--- Training FEAT seed={seed} ---')
            best = train_feat(config, seed, args.num_episodes, args.device, save_d, log_d)
            report['algorithms']['FEAT']['seeds'].append({
                'seed': seed,
                'best_eval_reward': best,
                'metrics_path': os.path.join(log_d, 'metrics.json'),
                'model_path': os.path.join(save_d, 'best_model.pt'),
            })
            print(f'  FEAT seed={seed} best={best:.2f}')

    # Train SAC
    if not args.skip_sac:
        report['algorithms']['SAC'] = {'seeds': []}
        for seed in args.seeds:
            save_d = os.path.join(args.output_dir, 'SAC', f'seed_{seed}', 'checkpoints')
            log_d = os.path.join(args.output_dir, 'SAC', f'seed_{seed}', 'logs')
            print(f'\n--- Training SAC seed={seed} ---')
            best = train_sac(config, seed, args.num_episodes, args.device, save_d, log_d)
            report['algorithms']['SAC']['seeds'].append({
                'seed': seed,
                'best_eval_reward': best,
                'metrics_path': os.path.join(log_d, 'metrics.json'),
                'model_path': os.path.join(save_d, 'best_model.pt'),
            })
            print(f'  SAC seed={seed} best={best:.2f}')

    # Save report
    report_path = os.path.join(args.output_dir, 'multi_seed_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\n  Report: {report_path}')

    # Plot
    print('\n  Generating plots...')
    plot_multi_seed(report, args.output_dir)

    # Summary
    print(f'\n{"="*60}')
    print('  MULTI-SEED SUMMARY')
    print(f'{"="*60}')
    for algo, data in report['algorithms'].items():
        bests = [s['best_eval_reward'] for s in data['seeds']
                 if s.get('best_eval_reward') is not None]
        if bests:
            print(f'  {algo}: {np.mean(bests):.2f} +/- {np.std(bests):.2f}  '
                  f'(seeds: {[f"{b:.2f}" for b in bests]})')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
