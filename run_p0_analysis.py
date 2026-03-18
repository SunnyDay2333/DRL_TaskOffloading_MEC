"""
###就是画收敛图的


P0 Analysis: Training Convergence Curves + Computational Overhead Table
========================================================================
Reads existing experiment data (no re-training needed) to produce:

1. Training reward convergence curves for 4 ablation variants
2. Evaluation reward curves during training
3. Computational overhead comparison table (parameter counts,
   adaptation episodes, final reward)

Usage:
    python run_p0_analysis.py
    python run_p0_analysis.py --ablation_dir experiments/ablation_2026-03-09_22-15-23
    python run_p0_analysis.py --adaptation_dir experiments/improved_adaptation_2026-03-16_19-53-13
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from config import create_default_config
from models.feat_agent import FEATAgent
from models.baselines import SACAgent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

VARIANT_STYLE = {
    'FEAT_A+B':     {'color': '#2ecc71', 'label': 'Full PDM (SDS+PLε)', 'ls': '-'},
    'FEAT_A_only':  {'color': '#3498db', 'label': 'SDS only',           'ls': '--'},
    'FEAT_B_only':  {'color': '#e67e22', 'label': 'PLε only',           'ls': '-.'},
    'FEAT_vanilla': {'color': '#e74c3c', 'label': 'No PDM (vanilla)',   'ls': ':'},
}

PLOT_DATA_SUBDIR = 'plot_data'


def _find_dir(pattern):
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _find_dir_with_subdir(pattern, required_subdir):
    """Find the latest directory matching pattern that contains required_subdir."""
    matches = sorted(glob.glob(pattern))
    for m in reversed(matches):
        if os.path.isdir(os.path.join(m, required_subdir)):
            return m
    return matches[-1] if matches else None


def _smooth(values, window=50):
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(np.mean(values[start:i + 1]))
    return out


def _save_plot_data(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serialisable = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            serialisable[k] = v.tolist()
        elif isinstance(v, list) and v and isinstance(v[0], np.floating):
            serialisable[k] = [float(x) for x in v]
        else:
            serialisable[k] = v
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)


# ============================================================
#  P0-1: Training Convergence Curves
# ============================================================

def plot_training_curves(ablation_dir: str, output_dir: str):
    """Load metrics.json from each ablation variant and plot curves."""
    log_base = os.path.join(ablation_dir, 'logs')
    if not os.path.isdir(log_base):
        print(f'[ERROR] Logs directory not found: {log_base}')
        return

    plots_dir = os.path.join(output_dir, 'plots')
    data_dir = os.path.join(plots_dir, PLOT_DATA_SUBDIR)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    all_data = {}
    for name in VARIANT_STYLE:
        fpath = os.path.join(log_base, name, 'metrics.json')
        if not os.path.exists(fpath):
            print(f'  [WARN] Missing {fpath}')
            continue
        with open(fpath, 'r') as f:
            raw = json.load(f)
        all_data[name] = raw.get('metrics', raw)

    if not all_data:
        print('[ERROR] No training data found.')
        return

    # --- Figure 1: Training Reward Curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_data_reward = {}
    ax = axes[0]
    for name, data in all_data.items():
        if 'reward' not in data:
            continue
        style = VARIANT_STYLE[name]
        episodes = [p[0] for p in data['reward']]
        values = [p[1] for p in data['reward']]
        smoothed = _smooth(values, window=100)
        ax.plot(episodes, smoothed, label=style['label'],
                color=style['color'], linestyle=style['ls'], linewidth=1.8)
        plot_data_reward[name] = {
            'episodes': episodes,
            'raw_reward': values,
            'smoothed_reward': smoothed,
        }

    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Episode Reward (smoothed)', fontsize=12)
    ax.set_title('(a) Training Reward Convergence', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)

    _save_plot_data(plot_data_reward,
                    os.path.join(data_dir, 'training_reward_data.json'))

    # --- Figure 2: Evaluation Reward Curves ---
    plot_data_eval = {}
    ax = axes[1]
    for name, data in all_data.items():
        if 'eval_reward' not in data:
            continue
        style = VARIANT_STYLE[name]
        episodes = [p[0] for p in data['eval_reward']]
        values = [p[1] for p in data['eval_reward']]
        ax.plot(episodes, values, 'o-', label=style['label'],
                color=style['color'], linestyle=style['ls'],
                linewidth=1.5, markersize=3)
        if 'eval_reward_std' in data:
            stds = [p[1] for p in data['eval_reward_std']]
            min_len = min(len(values), len(stds))
            lo = [values[i] - stds[i] for i in range(min_len)]
            hi = [values[i] + stds[i] for i in range(min_len)]
            ax.fill_between(episodes[:min_len], lo, hi,
                            color=style['color'], alpha=0.1)
        plot_data_eval[name] = {
            'episodes': episodes,
            'eval_reward': values,
        }

    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Evaluation Reward', fontsize=12)
    ax.set_title('(b) Evaluation Reward During Training', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)

    _save_plot_data(plot_data_eval,
                    os.path.join(data_dir, 'eval_reward_data.json'))

    plt.tight_layout()
    out_path = os.path.join(plots_dir, 'training_convergence.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  [OK] Training convergence plot saved: {out_path}')

    # --- Figure 3: Training Cost Curves ---
    has_cost = any('cost' in d for d in all_data.values())
    if has_cost:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_data_cost = {}
        for name, data in all_data.items():
            if 'cost' not in data:
                continue
            style = VARIANT_STYLE[name]
            episodes = [p[0] for p in data['cost']]
            values = [p[1] for p in data['cost']]
            smoothed = _smooth(values, window=100)
            ax.plot(episodes, smoothed, label=style['label'],
                    color=style['color'], linestyle=style['ls'], linewidth=1.8)
            plot_data_cost[name] = {
                'episodes': episodes,
                'raw_cost': values,
                'smoothed_cost': smoothed,
            }
        ax.set_xlabel('Training Episode', fontsize=12)
        ax.set_ylabel('Average Cost (smoothed)', fontsize=12)
        ax.set_title('Training Cost Convergence', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        cost_path = os.path.join(plots_dir, 'training_cost.png')
        plt.savefig(cost_path, dpi=200, bbox_inches='tight')
        plt.close()
        _save_plot_data(plot_data_cost,
                        os.path.join(data_dir, 'training_cost_data.json'))
        print(f'  [OK] Training cost plot saved: {cost_path}')

    # --- Figure 4: Final Training Performance Bar Chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_names, bar_vals, bar_colors = [], [], []
    plot_data_final = {}
    for name in VARIANT_STYLE:
        if name not in all_data:
            continue
        data = all_data[name]
        style = VARIANT_STYLE[name]
        if 'eval_reward' in data and data['eval_reward']:
            final_val = data['eval_reward'][-1][1]
        elif 'reward' in data:
            final_val = np.mean([p[1] for p in data['reward'][-100:]])
        else:
            continue
        bar_names.append(style['label'])
        bar_vals.append(final_val)
        bar_colors.append(style['color'])
        plot_data_final[name] = {'final_eval_reward': final_val}

    x = np.arange(len(bar_names))
    bars = ax.bar(x, bar_vals, color=bar_colors, width=0.55, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_names, fontsize=10)
    ax.set_ylabel('Final Evaluation Reward', fontsize=12)
    ax.set_title('Final Training Performance (5000 Episodes)', fontsize=13)
    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    bar_path = os.path.join(plots_dir, 'training_final_performance.png')
    plt.savefig(bar_path, dpi=200, bbox_inches='tight')
    plt.close()
    _save_plot_data(plot_data_final,
                    os.path.join(data_dir, 'final_performance_data.json'))
    print(f'  [OK] Final performance bar chart saved: {bar_path}')


# ============================================================
#  P0-2: Computational Overhead Table
# ============================================================

def _count_params(module):
    return sum(p.numel() for p in module.parameters())


def _count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def compute_overhead_table(adaptation_report_path: str, output_dir: str,
                           device: str = 'cpu'):
    """Build the computational overhead comparison table."""
    plots_dir = os.path.join(output_dir, 'plots')
    data_dir = os.path.join(plots_dir, PLOT_DATA_SUBDIR)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    config = create_default_config()
    dims = config.get_state_dims()

    feat = FEATAgent(
        env_state_dim=dims['env_state_dim'],
        time_varying_state_dim=dims['time_varying_state_dim'],
        action_dim=dims['action_dim'],
        num_policies=config.net_config.num_meta_policies,
        device=device,
    )
    sac = SACAgent(
        state_dim=dims['env_state_dim'] + dims['time_varying_state_dim'],
        action_dim=dims['action_dim'],
        device=device,
    )

    feat_meta = _count_params(feat.meta_policy)
    feat_steerer = _count_params(feat.steerer)
    feat_q = _count_params(feat.q_network)
    feat_total = feat_meta + feat_steerer + feat_q

    sac_actor = _count_params(sac.actor)
    sac_critic = _count_params(sac.critic)
    sac_total = sac_actor + sac_critic

    table_data = {
        'FEAT_components': {
            'meta_policy': feat_meta,
            'steerer': feat_steerer,
            'q_network': feat_q,
            'total': feat_total,
        },
        'SAC_components': {
            'actor': sac_actor,
            'critic': sac_critic,
            'total': sac_total,
        },
        'adaptation_methods': {},
    }

    # Load adaptation report for reward data
    adaptation_data = None
    if adaptation_report_path and os.path.exists(adaptation_report_path):
        with open(adaptation_report_path, 'r', encoding='utf-8') as f:
            adaptation_data = json.load(f)

    methods = [
        {
            'name': 'Steerer-only FT',
            'key': 'orig_steerer',
            'updated_params': feat_steerer,
            'total_params': feat_total,
            'frozen': feat_meta + feat_q,
        },
        {
            'name': 'FEAT Full FT',
            'key': 'full_ft',
            'updated_params': feat_total,
            'total_params': feat_total,
            'frozen': 0,
        },
        {
            'name': 'SAC Full FT',
            'key': 'sac_ft',
            'updated_params': sac_total,
            'total_params': sac_total,
            'frozen': 0,
        },
    ]

    for m in methods:
        entry = {
            'updated_params': m['updated_params'],
            'frozen_params': m['frozen'],
            'total_params': m['total_params'],
            'update_ratio': f"{m['updated_params'] / m['total_params'] * 100:.1f}%",
            'adaptation_episodes': 200,
        }

        if adaptation_data:
            rewards = []
            for sc_name, sc in adaptation_data.get('scenarios', {}).items():
                runs = sc.get(m['key'], [])
                for run in runs:
                    if isinstance(run, dict):
                        final = run.get('final', run)
                        if 'mean_reward' in final:
                            rewards.append(final['mean_reward'])
            if rewards:
                entry['mean_final_reward'] = float(np.mean(rewards))
                entry['std_final_reward'] = float(np.std(rewards))

        table_data['adaptation_methods'][m['name']] = entry

    # Also add zero-shot
    if adaptation_data:
        zs_rewards = []
        for sc_name, sc in adaptation_data.get('scenarios', {}).items():
            for r in sc.get('zero_shot', []):
                if 'mean_reward' in r:
                    zs_rewards.append(r['mean_reward'])
        if zs_rewards:
            table_data['adaptation_methods']['Zero-shot (no FT)'] = {
                'updated_params': 0,
                'frozen_params': feat_total,
                'total_params': feat_total,
                'update_ratio': '0.0%',
                'adaptation_episodes': 0,
                'mean_final_reward': float(np.mean(zs_rewards)),
                'std_final_reward': float(np.std(zs_rewards)),
            }

    _save_plot_data(table_data,
                    os.path.join(data_dir, 'overhead_table_data.json'))

    # --- Generate table figure ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    col_labels = ['Method', 'Updated\nParams', 'Frozen\nParams',
                  'Update\nRatio', 'Adapt.\nEpisodes', 'Final Reward\n(mean ± std)']

    rows = []
    row_colors = []
    color_map = {
        'Zero-shot (no FT)': '#f0f0f0',
        'Steerer-only FT': '#d5f5e3',
        'FEAT Full FT': '#d6eaf8',
        'SAC Full FT': '#fdebd0',
    }

    for method_name in ['Zero-shot (no FT)', 'Steerer-only FT',
                        'FEAT Full FT', 'SAC Full FT']:
        entry = table_data['adaptation_methods'].get(method_name)
        if not entry:
            continue
        rew_str = '--'
        if 'mean_final_reward' in entry:
            rew_str = f"{entry['mean_final_reward']:.2f} ± {entry.get('std_final_reward', 0):.2f}"
        rows.append([
            method_name,
            f"{entry['updated_params']:,}",
            f"{entry['frozen_params']:,}",
            entry['update_ratio'],
            str(entry['adaptation_episodes']),
            rew_str,
        ])
        row_colors.append(color_map.get(method_name, 'white'))

    table = ax.table(cellText=rows, colLabels=col_labels, loc='center',
                     cellLoc='center', colColours=['#34495e'] * len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_facecolor('#34495e')
        else:
            cell.set_facecolor(row_colors[row - 1])
        cell.set_edgecolor('#bdc3c7')

    ax.set_title('Computational Overhead Comparison for Environment Adaptation',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    table_path = os.path.join(plots_dir, 'computational_overhead_table.png')
    plt.savefig(table_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  [OK] Overhead table saved: {table_path}')

    # --- Print to console ---
    print(f'\n{"="*80}')
    print('  COMPUTATIONAL OVERHEAD COMPARISON')
    print(f'{"="*80}')
    print(f'\n  FEAT Architecture (K={config.net_config.num_meta_policies}):')
    print(f'    Meta-Policy (shared + {config.net_config.num_meta_policies} heads): '
          f'{feat_meta:>8,} params')
    print(f'    Steerer:                            {feat_steerer:>8,} params')
    print(f'    Q-Network (twin):                   {feat_q:>8,} params')
    print(f'    Total:                              {feat_total:>8,} params')
    print(f'\n  SAC Baseline:')
    print(f'    Actor:                              {sac_actor:>8,} params')
    print(f'    Critic:                             {sac_critic:>8,} params')
    print(f'    Total:                              {sac_total:>8,} params')
    print(f'\n  {"Method":<22} {"Updated":>10} {"Frozen":>10} {"Ratio":>8} '
          f'{"Episodes":>9} {"Reward":>18}')
    print(f'  {"-"*77}')
    for method_name in ['Zero-shot (no FT)', 'Steerer-only FT',
                        'FEAT Full FT', 'SAC Full FT']:
        entry = table_data['adaptation_methods'].get(method_name)
        if not entry:
            continue
        rew = '--'
        if 'mean_final_reward' in entry:
            rew = f"{entry['mean_final_reward']:.2f}±{entry.get('std_final_reward', 0):.2f}"
        print(f'  {method_name:<22} {entry["updated_params"]:>10,} '
              f'{entry["frozen_params"]:>10,} {entry["update_ratio"]:>8} '
              f'{entry["adaptation_episodes"]:>9} {rew:>18}')
    print(f'{"="*80}\n')

    return table_data


# ============================================================
#  Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='P0: Training curves + Overhead table')
    p.add_argument('--ablation_dir', type=str, default=None,
                   help='Ablation experiment directory')
    p.add_argument('--adaptation_dir', type=str, default=None,
                   help='Improved adaptation experiment directory')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--output_dir', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.ablation_dir is None:
        args.ablation_dir = _find_dir_with_subdir(
            'experiments/ablation_*', 'logs')
    if args.adaptation_dir is None:
        args.adaptation_dir = _find_dir('experiments/improved_adaptation_*')

    if args.output_dir is None:
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.output_dir = f'experiments/p0_analysis_{ts}'

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'\n{"#"*60}')
    print(f'  P0 Analysis')
    print(f'  Ablation dir:   {args.ablation_dir}')
    print(f'  Adaptation dir: {args.adaptation_dir}')
    print(f'  Output dir:     {args.output_dir}')
    print(f'{"#"*60}\n')

    # P0-1: Training Curves
    print('[P0-1] Generating training convergence curves...')
    if args.ablation_dir and os.path.isdir(args.ablation_dir):
        plot_training_curves(args.ablation_dir, args.output_dir)
    else:
        print(f'  [SKIP] Ablation directory not found: {args.ablation_dir}')

    # P0-2: Computational Overhead Table
    print('\n[P0-2] Computing overhead table...')
    adaptation_report = None
    if args.adaptation_dir:
        candidate = os.path.join(args.adaptation_dir,
                                 'improved_adaptation_report.json')
        if os.path.exists(candidate):
            adaptation_report = candidate
    compute_overhead_table(adaptation_report, args.output_dir,
                           device=args.device)

    print(f'\nAll P0 outputs saved to: {args.output_dir}')
