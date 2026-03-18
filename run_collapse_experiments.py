"""
Policy Collapse Visualization and Analysis (v2)
=================================================
Provides compelling experimental evidence for the paper by running three
groups of experiments:

  Experiment A — Collapse Dynamics Visualization
      Train FEAT_vanilla (no PDM) and FEAT_A+B (full PDM) side by side,
      recording per-episode expert selection distributions. Produces:
        - Stacked-area plot of expert selection proportions over time
        - Selection entropy curve
        - Training reward comparison

  Experiment B — Expert Behavioral Divergence
      After training, feed identical state batches through all K experts.
      Metrics:
        - Pairwise L2 / cosine distance between expert actions
        - Offload decision agreement rate
        - Per-expert reward profile under varied environments
        - KL divergence between expert action distributions

  Experiment C — Collapse -> Adaptation Failure (Direct Evidence)
      For each variant (vanilla, A_only, B_only, A+B), use RGSA to
      fine-tune only the steerer + Q in a NEW environment. Collapsed
      variants cannot benefit from steerer re-routing because all
      experts produce nearly identical actions.

Usage:
    python run_collapse_experiments.py --all
    python run_collapse_experiments.py --exp A --num_episodes 3000
    python run_collapse_experiments.py --exp B --model_dir experiments/ablation_.../checkpoints
    python run_collapse_experiments.py --exp C --model_dir experiments/ablation_.../checkpoints
    python run_collapse_experiments.py --all --quick
"""

import os
import sys
import copy
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ExperimentConfig, create_default_config,
    EnvironmentConfig, NetworkConfig, TrainingConfig,
)
from environment.mec_environment import MECEnvironment
from models.feat_agent import FEATAgent
from models.networks import MetaPolicyNetwork
from utils.replay_buffer import ReplayBuffer
from utils.helpers import set_seed, soft_update

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
#  Variant configurations
# ============================================================

VARIANT_CONFIGS = {
    'FEAT_vanilla': {
        'steerer_entropy_coeff': 0.0,
        'forced_exploration_prob': 0.0,
        'forced_exploration_min': 0.0,
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
    'FEAT_A+B': {
        'steerer_entropy_coeff': 0.5,
        'forced_exploration_prob': 0.2,
        'forced_exploration_min': 0.02,
    },
}

VARIANT_COLORS = {
    'FEAT_vanilla': '#e74c3c',
    'FEAT_A+B':     '#2ecc71',
    'FEAT_A_only':  '#3498db',
    'FEAT_B_only':  '#e67e22',
}

EXPERT_COLORS = ['#e74c3c', '#2ecc71', '#3498db']


# ============================================================
#  Helpers
# ============================================================

def create_agent(config: ExperimentConfig, variant: str,
                 device: str = 'cuda') -> FEATAgent:
    dims = config.get_state_dims()
    v_cfg = VARIANT_CONFIGS[variant]
    return FEATAgent(
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
        steerer_entropy_coeff=v_cfg['steerer_entropy_coeff'],
        forced_exploration_prob=v_cfg['forced_exploration_prob'],
        forced_exploration_decay=config.net_config.forced_exploration_decay,
        forced_exploration_min=v_cfg['forced_exploration_min'],
        device=device,
    )


def collect_episode(agent, env, replay_buffer, config, total_steps,
                    deterministic=False):
    """Collect one episode and return stats."""
    env_state, tv_state, _ = env.reset()
    ep_reward = 0.0
    ep_steps = 0
    K = config.net_config.num_meta_policies
    M = config.env_config.num_mobile_devices
    policy_counts = {k: 0 for k in range(K)}

    done = False
    while not done:
        if total_steps < config.train_config.min_buffer_size:
            offload_ratio = np.random.choice([0.0, 0.25, 0.5, 0.75, 1.0])
            num_offload = int(M * offload_ratio)
            offload = np.zeros(M)
            if num_offload > 0:
                idx = np.random.choice(M, num_offload, replace=False)
                offload[idx] = 1.0
            power = np.random.uniform(0.3, 1.0, M)
            action = np.concatenate([offload, power])
            sel_k = np.random.randint(0, K)
        else:
            action, sel_k = agent.select_action(
                env_state, tv_state, deterministic=deterministic)

        n_env, n_tv, reward, done, info = env.step(action)
        replay_buffer.push(env_state, tv_state, action, reward,
                           n_env, n_tv, done, sel_k)
        policy_counts[sel_k] += 1
        ep_reward += reward
        ep_steps += 1
        total_steps += 1
        env_state, tv_state = n_env, n_tv

    return ep_reward, ep_steps, policy_counts, total_steps


def evaluate_agent_simple(agent, env, num_episodes=10):
    agent.eval()
    rewards = []
    for _ in range(num_episodes):
        es, tv, _ = env.reset()
        r_total = 0.0
        done = False
        while not done:
            a, _ = agent.select_action(es, tv, deterministic=True)
            es, tv, r, done, _ = env.step(a)
            r_total += r
        rewards.append(r_total)
    agent.train()
    return float(np.mean(rewards))


def _find_model(model_dir, variant):
    """Locate a model checkpoint for the given variant."""
    candidates = []
    if model_dir:
        for name in [f'{variant}/best_model.pt', f'{variant}_final.pt',
                     f'{variant}.pt', 'best_model.pt']:
            candidates.append(os.path.join(model_dir, name))

    if os.path.isdir('experiments'):
        for d in sorted(os.listdir('experiments')):
            full = os.path.join('experiments', d)
            if not os.path.isdir(full):
                continue
            candidates.append(
                os.path.join(full, 'checkpoints', variant, 'best_model.pt'))
            candidates.append(
                os.path.join(full, 'checkpoints', f'{variant}_final.pt'))

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ============================================================
#  Experiment A: Collapse Dynamics Visualization
# ============================================================

def experiment_a(num_episodes=3000, device='cuda', seed=42,
                 output_dir='experiments/collapse_dynamics'):
    """Train vanilla vs A+B, recording expert selection distributions."""
    print(f'\n{"="*60}')
    print('  Experiment A: Collapse Dynamics Visualization')
    print(f'  Episodes: {num_episodes}')
    print(f'{"="*60}')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    variants = ['FEAT_vanilla', 'FEAT_A+B']
    K = 3
    history = {v: {
        'rewards': [],
        'policy_dist': [],
        'q_dist_entropy': [],
        'eval_rewards': [],
        'eval_episodes': [],
        'steerer_entropy': [],
    } for v in variants}

    for variant in variants:
        print(f'\n--- Training {variant} ---')
        set_seed(seed)
        config = create_default_config()
        config.device = device
        config.train_config.num_episodes = num_episodes

        env = MECEnvironment(config=config.env_config, seed=seed)
        eval_env = MECEnvironment(config=config.env_config, seed=seed + 1000)
        buf = ReplayBuffer(capacity=config.train_config.buffer_size)
        agent = create_agent(config, variant, device)

        total_steps = 0
        t0 = time.time()

        for ep in range(1, num_episodes + 1):
            ep_rew, ep_steps, pcounts, total_steps = collect_episode(
                agent, env, buf, config, total_steps)
            history[variant]['rewards'].append(ep_rew)

            total = max(sum(pcounts.values()), 1)
            dist = [pcounts.get(k, 0) / total for k in range(K)]
            history[variant]['policy_dist'].append(dist)

            if buf.is_ready(config.train_config.min_buffer_size):
                metrics = None
                for _ in range(ep_steps):
                    batch = buf.sample(config.train_config.batch_size)
                    metrics = agent.update(batch)
                if metrics:
                    history[variant]['q_dist_entropy'].append(
                        metrics.get('q_dist_entropy', 0))
                else:
                    history[variant]['q_dist_entropy'].append(0)
            else:
                history[variant]['q_dist_entropy'].append(0)

            if ep % 50 == 0:
                eval_r = evaluate_agent_simple(agent, eval_env, 10)
                history[variant]['eval_rewards'].append(eval_r)
                history[variant]['eval_episodes'].append(ep)
                elapsed = time.time() - t0
                print(f'  Ep {ep:>5}/{num_episodes} | Rew={ep_rew:>8.2f} | '
                      f'Eval={eval_r:>8.2f} | '
                      f'Dist={[f"{d:.2f}" for d in dist]} | '
                      f'{elapsed/60:.1f}min')

        ckpt = os.path.join(output_dir, 'checkpoints', f'{variant}_final.pt')
        agent.save(ckpt)
        print(f'  Saved: {ckpt}')

    # ---- Plots ----
    _plot_collapse_dynamics(history, variants, K, output_dir)

    data = {}
    for v in variants:
        data[v] = {
            'rewards': [float(r) for r in history[v]['rewards']],
            'policy_dist': [[float(d) for d in row]
                            for row in history[v]['policy_dist']],
            'q_dist_entropy': [float(e) for e in history[v]['q_dist_entropy']],
            'eval_rewards': [float(r) for r in history[v]['eval_rewards']],
            'eval_episodes': history[v]['eval_episodes'],
        }
    with open(os.path.join(output_dir, 'collapse_dynamics.json'), 'w') as f:
        json.dump(data, f, indent=2)

    print(f'\nExperiment A done. -> {output_dir}')
    return history


def _plot_collapse_dynamics(history, variants, K, output_dir):
    """Main collapse dynamics figure (paper Figure 2 candidate)."""

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.3)

    for col, variant in enumerate(variants):
        dist_arr = np.array(history[variant]['policy_dist'])
        n_eps = len(dist_arr)
        window = max(50, n_eps // 40)

        # Smooth distributions
        smoothed = np.zeros_like(dist_arr)
        for k in range(K):
            smoothed[:, k] = np.convolve(
                dist_arr[:, k], np.ones(window) / window, mode='same')

        # Row 0: Stacked area — expert selection proportions
        ax = fig.add_subplot(gs[0, col])
        ax.stackplot(range(n_eps),
                     [smoothed[:, k] for k in range(K)],
                     labels=[f'Expert {k}' for k in range(K)],
                     colors=EXPERT_COLORS, alpha=0.75)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Training Episode')
        ax.set_ylabel('Selection Proportion')
        title_tag = '(No PDM)' if 'vanilla' in variant else '(With PDM)'
        ax.set_title(f'{variant} {title_tag}')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.2)

        # Row 1: Selection entropy
        ax = fig.add_subplot(gs[1, col])
        sel_entropy = []
        for d in smoothed:
            dc = np.clip(d, 1e-8, None)
            dc = dc / dc.sum()
            sel_entropy.append(-(dc * np.log(dc)).sum())
        ax.plot(range(n_eps), sel_entropy,
                color=VARIANT_COLORS[variant], linewidth=1.2)
        max_ent = np.log(K)
        ax.axhline(y=max_ent, color='gray', linestyle='--', alpha=0.7,
                   label=f'Max entropy (ln{K}={max_ent:.2f})')
        ax.set_ylim(0, max_ent * 1.15)
        ax.set_xlabel('Training Episode')
        ax.set_ylabel('Selection Entropy H(p)')
        ax.set_title(f'{variant} — Steerer Selection Entropy')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    # Row 2: Training reward comparison
    ax = fig.add_subplot(gs[2, :])
    window_r = 100
    for variant in variants:
        rews = history[variant]['rewards']
        if len(rews) > window_r:
            s = np.convolve(rews, np.ones(window_r) / window_r, mode='valid')
            ax.plot(range(len(s)), s, label=variant,
                    color=VARIANT_COLORS[variant], linewidth=1.5)
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Episode Reward (smoothed)')
    ax.set_title('Training Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, 'plots', 'collapse_dynamics.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # ---- Final distribution bar chart ----
    fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, variant in zip(axes, variants):
        dist_arr = np.array(history[variant]['policy_dist'])
        tail = min(200, len(dist_arr))
        final_dist = dist_arr[-tail:].mean(axis=0)
        bars = ax.bar(range(K), final_dist, color=EXPERT_COLORS)
        ax.set_xticks(range(K))
        ax.set_xticklabels([f'Expert {k}' for k in range(K)])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Avg Selection Proportion')
        ax.set_title(f'{variant}\n(last {tail} episodes)')
        for bar, val in zip(bars, final_dist):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'final_distribution.png'),
                dpi=200)
    plt.close()
    print('  Collapse dynamics plots saved.')


# ============================================================
#  Experiment B: Expert Behavioral Divergence
# ============================================================

def experiment_b(model_dir: str = None, device='cuda', seed=42,
                 output_dir='experiments/collapse_divergence'):
    """Measure pairwise expert divergence + per-expert env-specific rewards."""
    print(f'\n{"="*60}')
    print('  Experiment B: Expert Behavioral Divergence')
    print(f'{"="*60}')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    set_seed(seed)

    config = create_default_config()
    config.device = device
    dims = config.get_state_dims()
    K = config.net_config.num_meta_policies

    # Collect a fixed set of test states from the training distribution
    env = MECEnvironment(config=config.env_config, seed=seed)
    test_states = []
    for _ in range(200):
        es, tv, _ = env.reset()
        test_states.append((es, tv))
        done = False
        while not done:
            action = np.random.uniform(0, 1, dims['action_dim'])
            es, tv, _, done, _ = env.step(action)
            test_states.append((es, tv))
    test_states = test_states[:500]
    print(f'  {len(test_states)} test states collected')

    # Environment profiles for per-expert reward evaluation
    env_profiles = {
        'default':      {},
        'BW_5MHz':      {'bandwidth': 5e6},
        'ES_4GHz':      {'es_computing_capacity': 4e9},
        'Task_heavy':   {'task_size_mean': 1000e3, 'task_size_variance': 1500e3},
    }

    # Load models
    variants = ['FEAT_vanilla', 'FEAT_A+B', 'FEAT_A_only', 'FEAT_B_only']
    agents = {}
    for v in variants:
        path = _find_model(model_dir, v)
        if path is None:
            print(f'  [SKIP] {v}: model not found')
            continue
        agent = FEATAgent(
            env_state_dim=dims['env_state_dim'],
            time_varying_state_dim=dims['time_varying_state_dim'],
            action_dim=dims['action_dim'],
            num_policies=K, device=device)
        agent.load(path, load_optimizer=False)
        agent.eval()
        agents[v] = agent
        print(f'  Loaded {v}: {path}')

    if not agents:
        print('  No models. Run Experiment A first or supply --model_dir.')
        return

    results: Dict = {}

    for variant, agent in agents.items():
        print(f'\n  Analyzing {variant} ...')
        M = config.env_config.num_mobile_devices

        all_actions = {k: [] for k in range(K)}
        all_means   = {k: [] for k in range(K)}

        for es, tv in test_states:
            tv_t = torch.FloatTensor(tv).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                features = agent.meta_policy.get_features(tv_t)
                for k in range(K):
                    mean, log_std = agent.meta_policy.policy_heads[k](features)
                    action = torch.sigmoid(mean)
                    all_actions[k].append(action.cpu().numpy()[0])
                    all_means[k].append(mean.cpu().numpy()[0])

        # --- Pairwise L2 / cosine ---
        pw_l2, pw_cos = [], []
        for i in range(K):
            for j in range(i + 1, K):
                ai = np.array(all_actions[i])
                aj = np.array(all_actions[j])
                pw_l2.append(float(np.linalg.norm(ai - aj, axis=1).mean()))
                dot = (ai * aj).sum(axis=1)
                ni = np.linalg.norm(ai, axis=1)
                nj = np.linalg.norm(aj, axis=1)
                pw_cos.append(float((dot / (ni * nj + 1e-8)).mean()))

        # --- Offload-decision agreement ---
        agree = 0
        for idx in range(len(test_states)):
            decisions = [(all_actions[k][idx][:M] > 0.5) for k in range(K)]
            if all(np.array_equal(decisions[0], d) for d in decisions[1:]):
                agree += 1
        agree_rate = agree / len(test_states)

        # --- KL divergence between expert mean distributions ---
        kl_pairs = []
        for i in range(K):
            for j in range(i + 1, K):
                mi = np.array(all_means[i])
                mj = np.array(all_means[j])
                si = np.array(all_actions[i])
                sj = np.array(all_actions[j])
                std_i = si.std(axis=0).clip(1e-6)
                std_j = sj.std(axis=0).clip(1e-6)
                kl = (np.log(std_j / std_i)
                      + (std_i**2 + (mi.mean(0) - mj.mean(0))**2) / (2 * std_j**2)
                      - 0.5).mean()
                kl_pairs.append(float(kl))

        # --- Per-expert reward in varied environments ---
        expert_env_rewards = {}
        for env_name, overrides in env_profiles.items():
            ecfg = copy.deepcopy(config.env_config)
            for k_o, v_o in overrides.items():
                setattr(ecfg, k_o, v_o)
            ecfg.__post_init__()
            test_env = MECEnvironment(config=ecfg, seed=seed + 500)
            expert_env_rewards[env_name] = {}
            for k in range(K):
                rews = []
                for _ in range(30):
                    es, tv, _ = test_env.reset()
                    ep_r = 0.0
                    done = False
                    while not done:
                        tv_t = torch.FloatTensor(tv).unsqueeze(0).to(agent.device)
                        with torch.no_grad():
                            a, _ = agent.meta_policy.sample_action(tv_t, k,
                                                                    deterministic=True)
                        a = a.cpu().numpy()[0]
                        es, tv, r, done, _ = test_env.step(a)
                        ep_r += r
                    rews.append(ep_r)
                expert_env_rewards[env_name][k] = float(np.mean(rews))

        results[variant] = {
            'mean_pairwise_l2':  float(np.mean(pw_l2)),
            'max_pairwise_l2':   float(np.max(pw_l2)),
            'mean_pairwise_cos': float(np.mean(pw_cos)),
            'pairwise_l2': {f'E{i}-E{j}': l2
                            for (i, j), l2 in zip(
                                [(i, j) for i in range(K) for j in range(i+1, K)],
                                pw_l2)},
            'mean_kl_divergence': float(np.mean(kl_pairs)),
            'offload_agreement_rate': agree_rate,
            'expert_env_rewards': expert_env_rewards,
        }
        print(f'    L2={np.mean(pw_l2):.4f}  Cos={np.mean(pw_cos):.4f}  '
              f'KL={np.mean(kl_pairs):.4f}  Agree={agree_rate:.1%}')

    _plot_divergence(results, K, output_dir)

    with open(os.path.join(output_dir, 'divergence_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nExperiment B done. -> {output_dir}')
    return results


def _plot_divergence(results, K, output_dir):
    if not results:
        return
    variants = list(results.keys())
    colors = [VARIANT_COLORS.get(v, 'gray') for v in variants]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Pairwise L2
    ax = axes[0, 0]
    vals = [results[v]['mean_pairwise_l2'] for v in variants]
    bars = ax.bar(range(len(variants)), vals, color=colors)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.replace('FEAT_', '') for v in variants],
                       rotation=15, fontsize=9)
    ax.set_ylabel('Mean Pairwise L2')
    ax.set_title('Expert Action Divergence (higher = more diverse)')
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # (0,1) KL divergence
    ax = axes[0, 1]
    vals = [results[v]['mean_kl_divergence'] for v in variants]
    bars = ax.bar(range(len(variants)), vals, color=colors)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.replace('FEAT_', '') for v in variants],
                       rotation=15, fontsize=9)
    ax.set_ylabel('Mean KL Divergence')
    ax.set_title('Expert Distribution Divergence')
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # (1,0) Offload agreement (lower = more diverse)
    ax = axes[1, 0]
    vals = [results[v]['offload_agreement_rate'] for v in variants]
    bars = ax.bar(range(len(variants)), vals, color=colors)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.replace('FEAT_', '') for v in variants],
                       rotation=15, fontsize=9)
    ax.set_ylabel('Agreement Rate')
    ax.set_title('Offload Decision Agreement (lower = more diverse)')
    ax.set_ylim(0, 1)
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f'{val:.1%}', ha='center', fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # (1,1) Per-expert reward heatmap (for FEAT_A+B vs vanilla)
    ax = axes[1, 1]
    focus = ['FEAT_vanilla', 'FEAT_A+B']
    focus = [v for v in focus if v in results]
    if focus:
        env_names = list(results[focus[0]]['expert_env_rewards'].keys())
        n_env = len(env_names)
        x = np.arange(n_env)
        width = 0.12
        for vi, v in enumerate(focus):
            for k in range(K):
                rews = [results[v]['expert_env_rewards'][e][k] for e in env_names]
                offset = (vi * K + k - (len(focus) * K - 1) / 2) * width
                label = f'{v.replace("FEAT_","")}-E{k}' if vi == 0 or k == 0 else None
                ax.bar(x + offset, rews, width * 0.9,
                       color=EXPERT_COLORS[k],
                       alpha=0.9 if 'A+B' in v else 0.4,
                       edgecolor='black' if 'A+B' in v else 'none',
                       linewidth=0.5,
                       label=f'{v.replace("FEAT_","")}-E{k}')
        ax.set_xticks(x)
        ax.set_xticklabels(env_names, fontsize=8)
        ax.set_ylabel('Mean Episode Reward')
        ax.set_title('Per-Expert Reward Across Environments')
        ax.legend(fontsize=6, ncol=3)
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'divergence_comparison.png'),
                dpi=200)
    plt.close()
    print('  Divergence plots saved.')


# ============================================================
#  Experiment C: Collapse -> Adaptation Failure
# ============================================================

def experiment_c(model_dir: str = None, finetune_episodes: int = 200,
                 device='cuda', seed=42,
                 output_dir='experiments/collapse_adaptation'):
    """
    For each variant, use RGSA to attempt steerer-level adaptation in a
    new environment. Collapsed models should show negligible improvement
    because all experts produce nearly identical actions.
    """
    print(f'\n{"="*60}')
    print('  Experiment C: Collapse -> Adaptation Failure')
    print(f'{"="*60}')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    set_seed(seed)

    config = create_default_config()
    config.device = device
    dims = config.get_state_dims()
    K = config.net_config.num_meta_policies

    scenarios = {
        'BW_5MHz': {'bandwidth': 5e6},
        'ES_4GHz': {'es_computing_capacity': 4e9},
        'Combined_hard': {
            'bandwidth': 7e6,
            'task_size_mean': 1000e3,
            'es_computing_capacity': 6e9,
        },
    }

    variants = ['FEAT_vanilla', 'FEAT_A+B', 'FEAT_A_only', 'FEAT_B_only']
    results: Dict = {}

    for sc_name, overrides in scenarios.items():
        print(f'\n--- Scenario: {sc_name} ---')
        env_cfg = copy.deepcopy(config.env_config)
        for k, v in overrides.items():
            setattr(env_cfg, k, v)
        env_cfg.__post_init__()
        env = MECEnvironment(config=env_cfg, seed=seed + 300)

        results[sc_name] = {}

        for variant in variants:
            path = _find_model(model_dir, variant)
            if path is None:
                print(f'  [SKIP] {variant}: model not found')
                continue

            agent = FEATAgent(
                env_state_dim=dims['env_state_dim'],
                time_varying_state_dim=dims['time_varying_state_dim'],
                action_dim=dims['action_dim'],
                num_policies=K, device=device)
            agent.load(path, load_optimizer=False)

            # Zero-shot evaluation
            agent.eval()
            zs_rews = []
            for _ in range(30):
                es, tv, _ = env.reset()
                ep_r = 0.0
                done = False
                while not done:
                    a, _ = agent.select_action(es, tv, deterministic=True)
                    es, tv, r, done, _ = env.step(a)
                    ep_r += r
                zs_rews.append(ep_r)
            zs_mean = float(np.mean(zs_rews))

            # RGSA adaptation
            from run_improved_adaptation import RGSAAdapter, finetune_loop
            rgsa_agent = FEATAgent(
                env_state_dim=dims['env_state_dim'],
                time_varying_state_dim=dims['time_varying_state_dim'],
                action_dim=dims['action_dim'],
                num_policies=K, device=device)
            rgsa_agent.load(path, load_optimizer=False)
            adapter = RGSAAdapter(rgsa_agent,
                                  steerer_lr=1e-3, critic_lr=3e-4,
                                  reset_gumbel_temp=0.8,
                                  exploration_prob=0.15)
            ep_rews, eval_curve, final_eval = finetune_loop(
                adapter, env, num_episodes=finetune_episodes,
                eval_freq=20, eval_episodes=20, warmup_episodes=10)

            gain = final_eval['mean_reward'] - zs_mean
            results[sc_name][variant] = {
                'zero_shot_reward': zs_mean,
                'final_reward': final_eval['mean_reward'],
                'final_success_rate': final_eval['mean_success_rate'],
                'final_policy_dist': final_eval['policy_distribution'],
                'gain_over_zs': gain,
                'episode_rewards': [float(r) for r in ep_rews],
                'eval_curve': eval_curve,
            }
            print(f'  {variant:16s}  ZS={zs_mean:>7.2f} -> '
                  f'Final={final_eval["mean_reward"]:>7.2f}  '
                  f'(gain={gain:>+7.2f})  '
                  f'dist={final_eval["policy_distribution"]}')

    _plot_adaptation_comparison(results, output_dir)

    with open(os.path.join(output_dir, 'collapse_adaptation.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f'\nExperiment C done. -> {output_dir}')
    return results


def _plot_adaptation_comparison(results, output_dir):
    scenarios = list(results.keys())
    n_sc = len(scenarios)
    if n_sc == 0:
        return

    # Learning curves
    fig, axes = plt.subplots(1, n_sc, figsize=(6 * n_sc, 5), squeeze=False)
    for idx, sc_name in enumerate(scenarios):
        ax = axes[0, idx]
        for variant, vdata in results[sc_name].items():
            rews = vdata['episode_rewards']
            s = (np.convolve(rews, np.ones(15) / 15, mode='valid')
                 if len(rews) > 15 else np.array(rews))
            ax.plot(range(len(s)), s,
                    label=variant.replace('FEAT_', ''),
                    color=VARIANT_COLORS.get(variant, 'gray'),
                    linewidth=1.5)
            ax.axhline(y=vdata['zero_shot_reward'],
                       color=VARIANT_COLORS.get(variant, 'gray'),
                       linestyle=':', alpha=0.4)
        ax.set_xlabel('Fine-tune Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title(sc_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Adaptation: Collapsed vs. Diverse Experts (RGSA)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots',
                             'adaptation_comparison.png'), dpi=200)
    plt.close()

    # Bar chart: adaptation gain
    fig2, axes2 = plt.subplots(1, n_sc, figsize=(6 * n_sc, 4), squeeze=False)
    for idx, sc_name in enumerate(scenarios):
        ax = axes2[0, idx]
        sc_data = results[sc_name]
        vs = list(sc_data.keys())
        gains = [sc_data[v]['gain_over_zs'] for v in vs]
        cs = [VARIANT_COLORS.get(v, 'gray') for v in vs]
        bars = ax.bar(range(len(vs)), gains, color=cs)
        ax.set_xticks(range(len(vs)))
        ax.set_xticklabels([v.replace('FEAT_', '') for v in vs],
                           rotation=15, fontsize=9)
        ax.set_ylabel('Adaptation Gain over Zero-shot')
        ax.set_title(sc_name)
        ax.axhline(y=0, color='black', linewidth=0.8)
        for b, val in zip(bars, gains):
            y = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2,
                    y + (0.15 if y >= 0 else -0.4),
                    f'{val:+.2f}', ha='center', fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle('Adaptation Gain: Evidence of Collapse -> Failure',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots',
                             'adaptation_gain_bars.png'), dpi=200)
    plt.close()
    print('  Adaptation comparison plots saved.')


# ============================================================
#  CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='Policy Collapse Experiments')
    p.add_argument('--exp', type=str, choices=['A', 'B', 'C'], default=None)
    p.add_argument('--all', action='store_true')
    p.add_argument('--model_dir', type=str, default=None,
                   help='Directory containing variant checkpoints')
    p.add_argument('--num_episodes', type=int, default=3000)
    p.add_argument('--finetune_episodes', type=int, default=200)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--output_base', type=str, default='experiments')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if args.quick:
        args.num_episodes = min(args.num_episodes, 500)
        args.finetune_episodes = min(args.finetune_episodes, 50)

    if args.exp == 'A' or args.all:
        experiment_a(
            num_episodes=args.num_episodes,
            device=args.device, seed=args.seed,
            output_dir=os.path.join(args.output_base,
                                    f'collapse_dynamics_{ts}'))

    if args.exp == 'B' or args.all:
        experiment_b(
            model_dir=args.model_dir,
            device=args.device, seed=args.seed,
            output_dir=os.path.join(args.output_base,
                                    f'collapse_divergence_{ts}'))

    if args.exp == 'C' or args.all:
        experiment_c(
            model_dir=args.model_dir,
            finetune_episodes=args.finetune_episodes,
            device=args.device, seed=args.seed,
            output_dir=os.path.join(args.output_base,
                                    f'collapse_adaptation_{ts}'))
