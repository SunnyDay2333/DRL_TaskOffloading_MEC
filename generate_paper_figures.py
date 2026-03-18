"""
Generate consolidated paper figures from existing experiment data.

Produces:
  1. RGSA adaptation summary bar chart (all scenarios, all methods)
  2. Expert functional specialization heatmap
  3. Collapse narrative reframing figure (specialized diversity vs random diversity)

Usage:
    python generate_paper_figures.py --device cpu
"""

import os
import sys
import json
import copy
import argparse
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'experiments/paper_figures'


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_latest(prefix):
    import glob
    candidates = sorted(glob.glob(f'experiments/{prefix}*'))
    for c in reversed(candidates):
        if os.path.isdir(c):
            return c
    return None


# ============================================================
#  Figure 1: RGSA Adaptation Summary
# ============================================================

def generate_rgsa_summary(report_path: str, output_dir: str):
    """Consolidated bar chart: all scenarios x all methods."""
    data = load_json(report_path)
    scenarios = data.get('scenarios', {})

    methods = ['zero_shot', 'orig_steerer', 'rgsa', 'full_ft', 'sac_ft']
    method_labels = {
        'zero_shot': 'Zero-shot',
        'orig_steerer': 'Orig. Steerer',
        'rgsa': 'RGSA (Ours)',
        'full_ft': 'FEAT Full FT',
        'sac_ft': 'SAC Full FT',
    }
    method_colors = {
        'zero_shot': '#95a5a6',
        'orig_steerer': '#e74c3c',
        'rgsa': '#27ae60',
        'full_ft': '#3498db',
        'sac_ft': '#8e44ad',
    }

    sc_names = list(scenarios.keys())
    n_sc = len(sc_names)
    n_methods = len(methods)

    means = {m: [] for m in methods}
    stds = {m: [] for m in methods}

    for sc_name in sc_names:
        sc = scenarios[sc_name]
        for m in methods:
            if m not in sc or not sc[m]:
                means[m].append(0)
                stds[m].append(0)
                continue
            if m == 'zero_shot':
                vals = [r['mean_reward'] for r in sc[m]]
            else:
                vals = [run['final']['mean_reward'] for run in sc[m]]
            means[m].append(float(np.mean(vals)))
            stds[m].append(float(np.std(vals)))

    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(n_sc)
    total_w = 0.75
    w = total_w / n_methods

    for i, m in enumerate(methods):
        offset = -total_w / 2 + w * (i + 0.5)
        bars = ax.bar(x + offset, means[m], w * 0.9,
                      yerr=stds[m], capsize=2,
                      color=method_colors[m],
                      label=method_labels[m],
                      alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    short_labels = []
    for s in sc_names:
        s_short = s.replace('Combined_', 'Comb.').replace('Task_', 'Task ')
        short_labels.append(s_short)
    ax.set_xticklabels(short_labels, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Mean Reward (after 200-episode adaptation)', fontsize=11)
    ax.set_title('Adaptation Performance Across Environment Scenarios', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=n_methods, loc='upper center',
              bbox_to_anchor=(0.5, -0.18), frameon=True)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'rgsa_adaptation_summary.png')
    plt.savefig(path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f'  [OK] {path}')

    table_path = os.path.join(output_dir, 'rgsa_summary_table.json')
    table = {}
    for sc_name in sc_names:
        table[sc_name] = {}
        for m in methods:
            idx = sc_names.index(sc_name)
            table[sc_name][method_labels[m]] = {
                'mean': round(means[m][idx], 2),
                'std': round(stds[m][idx], 2),
            }
    with open(table_path, 'w') as f:
        json.dump(table, f, indent=2)
    print(f'  [OK] {table_path}')

    return table


# ============================================================
#  Figure 2: Expert Functional Specialization Heatmap
# ============================================================

def generate_specialization_heatmap(divergence_path: str, output_dir: str):
    """Heatmap: per-expert, per-environment reward for each variant."""
    data = load_json(divergence_path)

    variants = ['FEAT_A+B', 'FEAT_vanilla']
    variant_titles = {'FEAT_A+B': 'Full PDM (Ours)', 'FEAT_vanilla': 'No PDM (Vanilla)'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    for idx, variant in enumerate(variants):
        ax = axes[idx]
        vdata = data.get(variant, {})
        expert_rewards = vdata.get('expert_env_rewards', {})
        if not expert_rewards:
            continue

        env_names = list(expert_rewards.keys())
        experts = sorted(expert_rewards[env_names[0]].keys(), key=int)
        n_experts = len(experts)
        n_envs = len(env_names)

        matrix = np.zeros((n_experts, n_envs))
        for j, env_name in enumerate(env_names):
            for i, ek in enumerate(experts):
                matrix[i, j] = expert_rewards[env_name][ek]

        pretty_envs = []
        for e in env_names:
            e = e.replace('default', 'Default').replace('BW_', 'BW ').replace('ES_', 'ES ').replace('Task_', 'Task ')
            pretty_envs.append(e)

        cmap = LinearSegmentedColormap.from_list('rg',
            ['#c0392c', '#f5f5f5', '#27ae60'], N=256)

        vmax = max(abs(matrix.max()), abs(matrix.min()))
        im = ax.imshow(matrix, cmap=cmap, aspect='auto',
                       vmin=-vmax, vmax=vmax)

        for i in range(n_experts):
            for j in range(n_envs):
                val = matrix[i, j]
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)

        for j in range(n_envs):
            best_i = np.argmax(matrix[:, j])
            ax.add_patch(plt.Rectangle((j - 0.5, best_i - 0.5), 1, 1,
                                        fill=False, edgecolor='gold',
                                        linewidth=2.5))

        ax.set_xticks(range(n_envs))
        ax.set_xticklabels(pretty_envs, fontsize=9, rotation=25, ha='right')
        ax.set_yticks(range(n_experts))
        ax.set_yticklabels([f'Expert {e}' for e in experts], fontsize=10)
        ax.set_title(variant_titles[variant], fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Mean Reward')

    fig.suptitle('Expert Functional Specialization (gold border = best expert per env)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'expert_specialization_heatmap.png')
    plt.savefig(path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f'  [OK] {path}')


# ============================================================
#  Figure 3: Collapse Narrative — Diversity Type Comparison
# ============================================================

def generate_collapse_narrative(divergence_path: str, params_path: str,
                                output_dir: str):
    """Combined figure showing the nuance: action diversity vs functional specialization."""
    div_data = load_json(divergence_path)
    param_data = load_json(params_path)

    variants = ['FEAT_A+B', 'FEAT_A_only', 'FEAT_B_only', 'FEAT_vanilla']
    short_names = {
        'FEAT_A+B': 'Full PDM',
        'FEAT_A_only': 'SDS Only',
        'FEAT_B_only': r'PL$\epsilon$ Only',
        'FEAT_vanilla': 'No PDM',
    }
    colors = {
        'FEAT_A+B': '#27ae60',
        'FEAT_A_only': '#3498db',
        'FEAT_B_only': '#f39c12',
        'FEAT_vanilla': '#e74c3c',
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # (a) Mean pairwise L2 (action space)
    ax = axes[0]
    vals = [div_data.get(v, {}).get('mean_pairwise_l2', 0) for v in variants]
    bars = ax.bar(range(len(variants)), vals,
                  color=[colors[v] for v in variants], alpha=0.85)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([short_names[v] for v in variants], fontsize=9, rotation=15)
    ax.set_ylabel('Mean Pairwise L2', fontsize=10)
    ax.set_title('(a) Action-Space Diversity', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=8)

    # (b) Mean KL divergence (action distribution)
    ax = axes[1]
    vals = [div_data.get(v, {}).get('mean_kl_divergence', 0) for v in variants]
    bars = ax.bar(range(len(variants)), vals,
                  color=[colors[v] for v in variants], alpha=0.85)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([short_names[v] for v in variants], fontsize=9, rotation=15)
    ax.set_ylabel('Mean KL Divergence', fontsize=10)
    ax.set_title('(b) Action Distribution Divergence', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=8)

    # (c) Pairwise expert cosine similarity (parameter space)
    ax = axes[2]
    vals = []
    for v in variants:
        pw = param_data.get(v, {}).get('pairwise_experts', {})
        cosines = [d['cosine'] for d in pw.values()]
        vals.append(float(np.mean(cosines)) if cosines else 0)
    bars = ax.bar(range(len(variants)), vals,
                  color=[colors[v] for v in variants], alpha=0.85)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([short_names[v] for v in variants], fontsize=9, rotation=15)
    ax.set_ylabel('Mean Cosine Similarity', fontsize=10)
    ax.set_title('(c) Parameter-Space Similarity', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', fontsize=8)

    # (d) Functional specialization: unique best experts + reward range
    ax = axes[3]
    for v in variants:
        expert_rewards = div_data.get(v, {}).get('expert_env_rewards', {})
        if not expert_rewards:
            continue
        envs = list(expert_rewards.keys())
        experts = sorted(expert_rewards[envs[0]].keys(), key=int)
        best_count = set()
        for env_name in envs:
            best_k = max(experts, key=lambda k: expert_rewards[env_name][k])
            best_count.add(best_k)

        all_means = []
        for k in experts:
            for env_name in envs:
                all_means.append(expert_rewards[env_name][k])
        reward_range = max(all_means) - min(all_means)

        ax.scatter(len(best_count), reward_range,
                   s=200, color=colors[v], edgecolors='black',
                   linewidth=1.5, zorder=5, label=short_names[v])
        ax.annotate(short_names[v], (len(best_count), reward_range),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=9)

    ax.set_xlabel('Unique Best Experts', fontsize=10)
    ax.set_ylabel('Reward Range (max - min)', fontsize=10)
    ax.set_title('(d) Functional Specialization', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'collapse_narrative_analysis.png')
    plt.savefig(path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f'  [OK] {path}')


# ============================================================
#  Figure 4: Training Convergence (publication quality)
# ============================================================

def generate_training_convergence(p0_dir: str, output_dir: str):
    """Publication-quality training convergence from p0 analysis data."""
    eval_path = os.path.join(p0_dir, 'plots', 'plot_data', 'eval_reward_data.json')
    if not os.path.exists(eval_path):
        print(f'  [SKIP] Training convergence: {eval_path} not found')
        return

    data = load_json(eval_path)

    variant_style = {
        'FEAT_A+B':    {'color': '#27ae60', 'ls': '-',  'label': 'Full PDM (A+B)'},
        'FEAT_A_only': {'color': '#3498db', 'ls': '--', 'label': 'SDS Only (A)'},
        'FEAT_B_only': {'color': '#f39c12', 'ls': '-.', 'label': r'PL$\epsilon$ Only (B)'},
        'FEAT_vanilla': {'color': '#e74c3c', 'ls': ':',  'label': 'No PDM (Vanilla)'},
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for variant, style in variant_style.items():
        vdata = data.get(variant, {})
        episodes = vdata.get('episodes', [])
        rewards = vdata.get('eval_reward', [])
        if not episodes or not rewards:
            continue
        ax.plot(episodes, rewards, color=style['color'],
                linestyle=style['ls'], label=style['label'],
                linewidth=2.0)

    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Evaluation Reward', fontsize=12)
    ax.set_title('Training Convergence: PDM Variants', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_convergence_pub.png')
    plt.savefig(path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f'  [OK] {path}')


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--improved_adaptation_dir', type=str, default=None)
    parser.add_argument('--divergence_path', type=str, default=None)
    parser.add_argument('--params_path', type=str, default=None)
    parser.add_argument('--p0_dir', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect paths
    if args.improved_adaptation_dir is None:
        args.improved_adaptation_dir = find_latest('improved_adaptation_')

    if args.divergence_path is None:
        d = find_latest('collapse_divergence_')
        if d:
            args.divergence_path = os.path.join(d, 'divergence_analysis.json')

    if args.params_path is None:
        d = find_latest('collapse_v2_params_')
        if d:
            args.params_path = os.path.join(d, 'param_degeneration.json')

    if args.p0_dir is None:
        args.p0_dir = find_latest('p0_analysis_')

    print(f'\n{"="*60}')
    print('  Generating Paper Figures')
    print(f'{"="*60}')
    print(f'  Output: {args.output_dir}')
    print(f'  Improved Adaptation: {args.improved_adaptation_dir}')
    print(f'  Divergence: {args.divergence_path}')
    print(f'  Params: {args.params_path}')
    print(f'  P0 Analysis: {args.p0_dir}')
    print()

    # 1. RGSA Adaptation Summary
    if args.improved_adaptation_dir:
        report_path = os.path.join(args.improved_adaptation_dir,
                                   'improved_adaptation_report.json')
        if os.path.exists(report_path):
            print('--- Figure 1: RGSA Adaptation Summary ---')
            generate_rgsa_summary(report_path, args.output_dir)
        else:
            print(f'  [SKIP] Report not found: {report_path}')
    else:
        print('  [SKIP] No improved_adaptation directory found')

    # 2. Expert Specialization Heatmap
    if args.divergence_path and os.path.exists(args.divergence_path):
        print('\n--- Figure 2: Expert Specialization Heatmap ---')
        generate_specialization_heatmap(args.divergence_path, args.output_dir)
    else:
        print('  [SKIP] Divergence analysis not found')

    # 3. Collapse Narrative
    if (args.divergence_path and os.path.exists(args.divergence_path) and
        args.params_path and os.path.exists(args.params_path)):
        print('\n--- Figure 3: Collapse Narrative Analysis ---')
        generate_collapse_narrative(args.divergence_path, args.params_path,
                                   args.output_dir)
    else:
        print('  [SKIP] Divergence/params data not found')

    # 4. Training Convergence (publication quality)
    if args.p0_dir:
        print('\n--- Figure 4: Training Convergence ---')
        generate_training_convergence(args.p0_dir, args.output_dir)
    else:
        print('  [SKIP] P0 analysis not found')

    print(f'\nAll figures saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
