"""
Improved Adaptation Experiment (v2)
====================================
Addresses fundamental flaws in the original steerer-only fine-tuning and
provides a corrected adaptation pipeline with multi-seed support and SAC
baseline comparison.

Root-cause diagnosis of the original negative adaptation gain:
  1. `update_steerer_only` calls `_update_steerer`, which derives supervisory
     labels (`max_q_indices`) from the SOURCE-env Q-network. In a new
     environment those Q-values are stale, so the steerer is trained towards
     the wrong expert.
  2. The Q-network is frozen during steerer-only fine-tuning, so the wrong
     supervisory signal never corrects itself.
  3. Post-training, Gumbel temperature (~0.2) and forced-exploration
     probability (~0.02) are near their minimums, leaving the steerer with
     almost no room to re-explore alternative experts.

Fix: **Reward-Guided Steerer Adaptation (RGSA)**
  - Freeze expert (meta-policy) weights entirely.
  - Co-update Q-network alongside the steerer so that the Q-based
    supervisory signal quickly reflects the new environment.
  - Reset steerer optimizer state and raise Gumbel temperature.
  - Temporarily raise forced-exploration probability.

Experiment matrix (per scenario):
  1. Zero-shot            — direct transfer, no fine-tuning
  2. Original steerer-only — flawed baseline (frozen Q)
  3. RGSA                 — our fix (co-update Q + steerer, freeze experts)
  4. Full fine-tune       — upper-bound reference (all params)
  5. SAC full fine-tune   — single-policy baseline comparison

Usage:
    python run_improved_adaptation.py ^
        --feat_model  experiments/baselines_.../checkpoints/FEAT/best_model.pt ^
        --sac_model   experiments/baselines_.../checkpoints/SAC/best_model.pt ^
        --scenarios BW_5MHz BW_7MHz ES_4GHz Combined_hard ^
        --finetune_episodes 200 --seeds 42 43 44
"""

import os
import sys
import copy
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EnvironmentConfig, ExperimentConfig, create_default_config
from environment.mec_environment import MECEnvironment
from models.feat_agent import FEATAgent
from models.baselines import SACAgent
from utils.replay_buffer import ReplayBuffer
from utils.helpers import set_seed, soft_update

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
#  Reward-Guided Steerer Adaptation (RGSA)
# ============================================================

class RGSAAdapter:
    """
    Reward-Guided Steerer Adaptation.

    Freezes expert (meta-policy) weights and co-updates Q-network +
    steerer in the new environment so that the supervisory Q-signal
    reflects actual new-environment dynamics.
    """

    def __init__(self, agent: FEATAgent, *,
                 steerer_lr: float = 1e-3,
                 critic_lr: float = 3e-4,
                 reset_gumbel_temp: float = 0.8,
                 exploration_prob: float = 0.15,
                 exploration_decay: float = 0.999,
                 exploration_min: float = 0.03,
                 temp_decay: float = 0.9995):
        self.agent = agent
        self._temp_decay = temp_decay

        # Freeze expert (meta-policy) weights
        for p in agent.meta_policy.parameters():
            p.requires_grad = False

        # Un-freeze Q-network and steerer
        for p in agent.q_network.parameters():
            p.requires_grad = True
        for p in agent.target_q_network.parameters():
            p.requires_grad = False
        for p in agent.steerer.parameters():
            p.requires_grad = True

        # Fresh optimizer state
        agent.steerer_optimizer = Adam(agent.steerer.parameters(), lr=steerer_lr)
        agent.critic_optimizer = Adam(agent.q_network.parameters(), lr=critic_lr)

        # Raise Gumbel temperature for re-exploration
        agent.steerer.temperature = reset_gumbel_temp

        # Raise forced-exploration probability
        agent.forced_exploration_prob = exploration_prob
        agent.forced_exploration_decay = exploration_decay
        agent.forced_exploration_min = exploration_min

    # ----- action selection (delegates to agent) -----

    def select_action(self, env_state, tv_state, deterministic=False):
        return self.agent.select_action(
            env_state, tv_state, deterministic=deterministic)

    # ----- joint Q + steerer update (experts frozen) -----

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        agent = self.agent

        env_states = torch.FloatTensor(batch['env_states']).to(agent.device)
        tv_states = torch.FloatTensor(batch['time_varying_states']).to(agent.device)
        actions = torch.FloatTensor(batch['actions']).to(agent.device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(agent.device)
        next_env = torch.FloatTensor(batch['next_env_states']).to(agent.device)
        next_tv = torch.FloatTensor(batch['next_time_varying_states']).to(agent.device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(agent.device)

        full_states = torch.cat([env_states, tv_states], dim=1)
        next_full = torch.cat([next_env, next_tv], dim=1)

        # ---- Update Q-network (CMQ target) ----
        with torch.no_grad():
            next_actions_list, next_logp_list = \
                agent.meta_policy.sample_all_actions(next_tv)
            target_qs = []
            for next_a, next_lp in zip(next_actions_list, next_logp_list):
                tq = agent.target_q_network.q_min(next_full, next_a)
                tq = tq - agent.alpha * next_lp.unsqueeze(1)
                target_qs.append(tq)
            stacked = torch.stack(target_qs, dim=-1)
            max_tq = stacked.max(dim=-1)[0]
            td_target = rewards + (1 - dones) * agent.gamma * max_tq

        q1, q2 = agent.q_network(full_states, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(agent.q_network.parameters(), 1.0)
        agent.critic_optimizer.step()

        soft_update(agent.target_q_network, agent.q_network, agent.tau)

        # ---- Update steerer (using refreshed Q signal) ----
        features = agent.meta_policy.get_features(tv_states)

        with torch.no_grad():
            actions_list, _ = agent.meta_policy.sample_all_actions(tv_states)
            q_vals = []
            for a in actions_list:
                q_vals.append(agent.q_network.q_min(full_states, a))
            q_stack = torch.stack(q_vals, dim=-1).squeeze(1)
            max_q_idx = q_stack.argmax(dim=-1)

        logits = agent.steerer.get_logits(env_states, features.detach())
        ce_loss = F.cross_entropy(logits, max_q_idx)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        steerer_loss = ce_loss - agent.steerer_entropy_coeff * entropy

        agent.steerer_optimizer.zero_grad()
        steerer_loss.backward()
        nn.utils.clip_grad_norm_(agent.steerer.parameters(), 1.0)
        agent.steerer_optimizer.step()

        # Decay temperature / exploration
        agent.steerer.update_temperature(self._temp_decay)
        agent.forced_exploration_prob = max(
            agent.forced_exploration_min,
            agent.forced_exploration_prob * agent.forced_exploration_decay)

        return {
            'critic_loss': critic_loss.item(),
            'steerer_loss': steerer_loss.item(),
            'steerer_entropy': entropy.item(),
            'gumbel_temp': agent.steerer.temperature,
            'explore_prob': agent.forced_exploration_prob,
        }


class OriginalSteererOnly:
    """Wrapper for original (flawed) steerer-only fine-tuning."""

    def __init__(self, agent: FEATAgent):
        self.agent = agent
        for p in agent.meta_policy.parameters():
            p.requires_grad = False
        for p in agent.q_network.parameters():
            p.requires_grad = False

    def select_action(self, env_state, tv_state, deterministic=False):
        return self.agent.select_action(
            env_state, tv_state, deterministic=deterministic)

    def update(self, batch):
        return self.agent.update_steerer_only(batch)


class FullFTAdapter:
    """Wrapper that fine-tunes all FEAT parameters."""

    def __init__(self, agent: FEATAgent):
        self.agent = agent
        agent.steerer.temperature = 0.8
        agent.forced_exploration_prob = 0.15

    def select_action(self, es, tv, deterministic=False):
        return self.agent.select_action(es, tv, deterministic=deterministic)

    def update(self, batch):
        return self.agent.update(batch)


class SACFTAdapter:
    """Wrapper that fine-tunes all SAC parameters.

    SACAgent.select_action returns (action,) not (action, sel_k), so we
    wrap it to return a dummy policy index for compatibility.
    """

    def __init__(self, agent: SACAgent):
        self.agent = agent

    def select_action(self, env_state, tv_state, deterministic=False):
        full_state = np.concatenate([env_state, tv_state])
        action = self.agent.select_action(full_state, deterministic=deterministic)
        return action, 0

    def update(self, batch):
        sac_batch = {
            'states': np.concatenate(
                [batch['env_states'], batch['time_varying_states']], axis=1),
            'actions': batch['actions'],
            'rewards': batch['rewards'],
            'next_states': np.concatenate(
                [batch['next_env_states'], batch['next_time_varying_states']],
                axis=1),
            'dones': batch['dones'],
        }
        return self.agent.update(sac_batch)


# ============================================================
#  Test scenarios
# ============================================================

def get_test_scenarios() -> Dict[str, Dict]:
    return {
        'BW_5MHz':  {'bandwidth': 5e6},
        'BW_7MHz':  {'bandwidth': 7e6},
        'BW_15MHz': {'bandwidth': 15e6},
        'ES_4GHz':  {'es_computing_capacity': 4e9},
        'ES_6GHz':  {'es_computing_capacity': 6e9},
        'Task_heavy': {
            'task_size_mean': 1000e3,
            'task_size_variance': 1500e3,
        },
        'Combined_hard': {
            'bandwidth': 7e6,
            'task_size_mean': 1000e3,
            'es_computing_capacity': 6e9,
        },
        'Combined_easy': {
            'bandwidth': 15e6,
            'task_size_mean': 500e3,
            'es_computing_capacity': 12e9,
        },
    }


# ============================================================
#  Evaluation helper
# ============================================================

def evaluate_agent(adapter, env, num_episodes=50, device='cuda'):
    """Evaluate any adapter that exposes select_action(es, tv, det)."""
    inner = getattr(adapter, 'agent', adapter)
    if hasattr(inner, 'eval'):
        inner.eval()

    rewards, successes, costs = [], [], []
    policy_usage: Dict[int, int] = {}

    for _ in range(num_episodes):
        env_state, tv_state, _ = env.reset()
        ep_rew = 0.0
        ep_suc, ep_cst = [], []
        done = False
        while not done:
            action, sel_k = adapter.select_action(
                env_state, tv_state, deterministic=True)
            env_state, tv_state, reward, done, info = env.step(action)
            ep_rew += reward
            ep_suc.extend(info['successes'])
            if 'cost' in info:
                ep_cst.append(info['cost'])
            policy_usage[sel_k] = policy_usage.get(sel_k, 0) + 1

        rewards.append(ep_rew)
        successes.append(np.mean(ep_suc))
        costs.append(np.mean(ep_cst) if ep_cst else 0.0)

    total = max(sum(policy_usage.values()), 1)
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_success_rate': float(np.mean(successes)),
        'mean_cost': float(np.mean(costs)),
        'policy_distribution': {
            str(k): round(v / total, 3)
            for k, v in sorted(policy_usage.items())
        },
    }


# ============================================================
#  Fine-tuning loop
# ============================================================

def finetune_loop(adapter, env, *,
                  num_episodes=200, batch_size=64,
                  eval_freq=20, eval_episodes=30,
                  warmup_episodes=10, updates_per_episode=10):
    """Run adaptation fine-tuning and return learning curve."""
    buf = ReplayBuffer(capacity=30000)
    episode_rewards: List[float] = []
    eval_curve: List[Dict] = []

    inner = getattr(adapter, 'agent', adapter)
    if hasattr(inner, 'train'):
        inner.train()

    for ep in range(num_episodes):
        env_state, tv_state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, sel_k = adapter.select_action(
                env_state, tv_state, deterministic=False)
            n_env, n_tv, reward, done, info = env.step(action)
            buf.push(env_state, tv_state, action, reward,
                     n_env, n_tv, done, sel_k)
            ep_reward += reward
            env_state, tv_state = n_env, n_tv

        episode_rewards.append(ep_reward)

        if len(buf) >= batch_size and ep >= warmup_episodes:
            for _ in range(updates_per_episode):
                batch = buf.sample(batch_size)
                adapter.update(batch)

        if (ep + 1) % eval_freq == 0:
            snap = evaluate_agent(adapter, env, eval_episodes)
            snap['episode'] = ep + 1
            eval_curve.append(snap)
            if hasattr(inner, 'train'):
                inner.train()

    final = evaluate_agent(adapter, env, eval_episodes)
    return episode_rewards, eval_curve, final


# ============================================================
#  Model loaders
# ============================================================

def load_feat_agent(model_path, device, config):
    dims = config.get_state_dims()
    agent = FEATAgent(
        env_state_dim=dims['env_state_dim'],
        time_varying_state_dim=dims['time_varying_state_dim'],
        action_dim=dims['action_dim'],
        num_policies=config.net_config.num_meta_policies,
        device=device,
    )
    agent.load(model_path, load_optimizer=False)
    return agent


def load_sac_agent(model_path, device, config):
    dims = config.get_state_dims()
    agent = SACAgent(
        state_dim=dims['env_state_dim'] + dims['time_varying_state_dim'],
        action_dim=dims['action_dim'],
        device=device,
    )
    agent.load(model_path)
    return agent


# ============================================================
#  Main experiment
# ============================================================

def run_experiment(feat_model: str, sac_model: Optional[str],
                   scenarios: List[str], finetune_episodes: int,
                   seeds: List[int], device: str, output_dir: str):

    config = create_default_config()
    config.device = device
    all_scenarios = get_test_scenarios()
    if not scenarios:
        scenarios = list(all_scenarios.keys())

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

    results: Dict = {'config': {
        'feat_model': feat_model,
        'sac_model': sac_model,
        'finetune_episodes': finetune_episodes,
        'seeds': seeds,
    }, 'scenarios': {}}

    methods = ['zero_shot', 'orig_steerer', 'rgsa', 'full_ft']
    if sac_model:
        methods.append('sac_ft')

    for sc_name in scenarios:
        if sc_name not in all_scenarios:
            print(f'[SKIP] Unknown scenario: {sc_name}')
            continue

        overrides = all_scenarios[sc_name]
        if 'num_mobile_devices' in overrides:
            print(f'[SKIP] {sc_name}: dimension-changing')
            continue

        print(f'\n{"="*70}')
        print(f'  Scenario: {sc_name}  |  Overrides: {overrides}')
        print(f'{"="*70}')

        sc_results: Dict[str, List] = {m: [] for m in methods}

        for seed in seeds:
            set_seed(seed)
            env_cfg = copy.deepcopy(config.env_config)
            for k, v in overrides.items():
                setattr(env_cfg, k, v)
            env_cfg.__post_init__()
            env = MECEnvironment(config=env_cfg, seed=seed + 200)

            # --- (1) Zero-shot ---
            agent_zs = load_feat_agent(feat_model, device, config)
            zs = evaluate_agent(agent_zs, env, 50)
            sc_results['zero_shot'].append(zs)

            # --- (2) Original steerer-only ---
            agent_orig = load_feat_agent(feat_model, device, config)
            orig_adapter = OriginalSteererOnly(agent_orig)
            orig_rew, orig_curve, orig_final = finetune_loop(
                orig_adapter, env, num_episodes=finetune_episodes)
            sc_results['orig_steerer'].append({
                'final': orig_final, 'curve': orig_curve,
                'rewards': [float(r) for r in orig_rew]})

            # --- (3) RGSA ---
            agent_rgsa = load_feat_agent(feat_model, device, config)
            rgsa_adapter = RGSAAdapter(agent_rgsa,
                                       steerer_lr=1e-3, critic_lr=3e-4,
                                       reset_gumbel_temp=0.8,
                                       exploration_prob=0.15)
            rgsa_rew, rgsa_curve, rgsa_final = finetune_loop(
                rgsa_adapter, env, num_episodes=finetune_episodes)
            sc_results['rgsa'].append({
                'final': rgsa_final, 'curve': rgsa_curve,
                'rewards': [float(r) for r in rgsa_rew]})

            # --- (4) Full fine-tune ---
            agent_full = load_feat_agent(feat_model, device, config)
            full_adapter = FullFTAdapter(agent_full)
            full_rew, full_curve, full_final = finetune_loop(
                full_adapter, env, num_episodes=finetune_episodes)
            sc_results['full_ft'].append({
                'final': full_final, 'curve': full_curve,
                'rewards': [float(r) for r in full_rew]})

            # --- (5) SAC full fine-tune ---
            if sac_model:
                sac_agent = load_sac_agent(sac_model, device, config)
                sac_adapter = SACFTAdapter(sac_agent)
                sac_rew, sac_curve, sac_final = finetune_loop(
                    sac_adapter, env, num_episodes=finetune_episodes)
                sc_results['sac_ft'].append({
                    'final': sac_final, 'curve': sac_curve,
                    'rewards': [float(r) for r in sac_rew]})

            print(f'  seed={seed}  ZS={zs["mean_reward"]:.2f}  '
                  f'Orig={orig_final["mean_reward"]:.2f}  '
                  f'RGSA={rgsa_final["mean_reward"]:.2f}  '
                  f'FullFT={full_final["mean_reward"]:.2f}', end='')
            if sac_model:
                print(f'  SAC_FT={sac_final["mean_reward"]:.2f}', end='')
            print()

        results['scenarios'][sc_name] = sc_results
        _plot_scenario(sc_name, sc_results, methods, output_dir)

    _save_results(results, output_dir)
    _print_summary(results, methods)
    print(f'\nResults saved to: {output_dir}')
    return results


# ============================================================
#  Visualization
# ============================================================

METHOD_STYLE = {
    'zero_shot':     {'color': '#95a5a6', 'label': 'Zero-shot'},
    'orig_steerer':  {'color': '#e74c3c', 'label': 'Orig Steerer-only'},
    'rgsa':          {'color': '#2ecc71', 'label': 'RGSA (Ours)'},
    'full_ft':       {'color': '#3498db', 'label': 'Full Fine-tune'},
    'sac_ft':        {'color': '#9b59b6', 'label': 'SAC Fine-tune'},
}


def _smooth(data, window=15):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')


def _plot_scenario(sc_name, sc_results, methods, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean training reward across seeds
    ax = axes[0]
    zs_mean = np.mean([r['mean_reward'] for r in sc_results['zero_shot']])
    ax.axhline(y=zs_mean, color='gray', linestyle='--', alpha=0.7,
               label=f'Zero-shot ({zs_mean:.2f})')

    for m in methods:
        if m == 'zero_shot':
            continue
        style = METHOD_STYLE[m]
        all_rews = [run['rewards'] for run in sc_results[m]]
        if not all_rews:
            continue
        min_len = min(len(r) for r in all_rews)
        arr = np.array([r[:min_len] for r in all_rews])
        mean_r = arr.mean(axis=0)
        smoothed = _smooth(mean_r)
        ax.plot(range(len(smoothed)), smoothed,
                label=style['label'], color=style['color'], linewidth=1.5)
        if arr.shape[0] > 1:
            std_r = arr.std(axis=0)
            s_std = _smooth(std_r)
            ml = min(len(smoothed), len(s_std))
            ax.fill_between(range(ml), smoothed[:ml] - s_std[:ml],
                           smoothed[:ml] + s_std[:ml],
                           color=style['color'], alpha=0.15)

    ax.set_xlabel('Fine-tune Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title(f'{sc_name} — Training Curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: final eval bar chart
    ax = axes[1]
    bar_data = []
    for m in methods:
        style = METHOD_STYLE[m]
        if m == 'zero_shot':
            vals = [r['mean_reward'] for r in sc_results[m]]
        else:
            vals = [run['final']['mean_reward'] for run in sc_results[m]]
        if vals:
            bar_data.append((style['label'], np.mean(vals), np.std(vals),
                            style['color']))

    x_pos = np.arange(len(bar_data))
    bars = ax.bar(x_pos, [d[1] for d in bar_data],
                  yerr=[d[2] for d in bar_data],
                  color=[d[3] for d in bar_data], capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d[0] for d in bar_data], rotation=20, fontsize=8)
    ax.set_ylabel('Final Eval Reward')
    ax.set_title(f'{sc_name} — Final Performance')
    for bar, (_, mean, std, _) in zip(bars, bar_data):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.1,
                f'{mean:.2f}', ha='center', fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots',
                             f'{sc_name}_adaptation.png'), dpi=150)
    plt.close()


def _save_results(results, output_dir):
    path = os.path.join(output_dir, 'improved_adaptation_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)


def _print_summary(results, methods):
    print(f'\n{"="*100}')
    header = f'{"Scenario":<20}'
    for m in methods:
        header += f' {METHOD_STYLE[m]["label"]:>18}'
    print(header)
    print(f'{"-"*100}')

    for sc_name, sc in results['scenarios'].items():
        row = f'{sc_name:<20}'
        for m in methods:
            if m == 'zero_shot':
                vals = [r['mean_reward'] for r in sc[m]]
            else:
                vals = [run['final']['mean_reward'] for run in sc[m]]
            if vals:
                row += f' {np.mean(vals):>8.2f}+/-{np.std(vals):<6.2f}'
            else:
                row += f' {"N/A":>16}'
        print(row)
    print(f'{"="*100}')


# ============================================================
#  CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='Improved Adaptation (RGSA)')
    p.add_argument('--feat_model', type=str, required=True,
                   help='Path to trained FEAT best_model.pt')
    p.add_argument('--sac_model', type=str, default=None,
                   help='Path to trained SAC best_model.pt (optional)')
    p.add_argument('--scenarios', nargs='*', default=[])
    p.add_argument('--finetune_episodes', type=int, default=200)
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--output_dir', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.output_dir is None:
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.output_dir = f'experiments/improved_adaptation_{ts}'
    run_experiment(
        feat_model=args.feat_model,
        sac_model=args.sac_model,
        scenarios=args.scenarios,
        finetune_episodes=args.finetune_episodes,
        seeds=args.seeds,
        device=args.device,
        output_dir=args.output_dir,
    )
