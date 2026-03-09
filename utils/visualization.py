"""
可视化模块
===========
提供训练过程和评估结果的可视化功能

功能包括:
- 训练曲线绘制
- 策略分布可视化
- 信道状态可视化
- 性能对比图表
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
import os
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrainingVisualizer:
    """
    训练可视化器
    =============
    绘制训练过程中的各种指标
    """
    
    def __init__(self, log_dir: str = './logs'):
        """
        初始化可视化器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        self.metrics_path = os.path.join(log_dir, 'metrics.json')
        
    def load_metrics(self) -> Dict:
        """加载训练指标"""
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r') as f:
                return json.load(f)
        return {}
    
    def plot_training_curves(self, 
                            save_path: Optional[str] = None,
                            show: bool = True):
        """
        绘制训练曲线
        
        Args:
            save_path: 保存路径 (可选)
            show: 是否显示图表
        """
        metrics = self.load_metrics()
        
        if not metrics or 'metrics' not in metrics:
            print("没有找到训练指标数据")
            return
            
        data = metrics['metrics']
        
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # 1. 奖励曲线
        if 'reward' in data:
            ax1 = fig.add_subplot(gs[0, 0])
            steps, values = zip(*data['reward'])
            ax1.plot(steps, values, alpha=0.3, color='blue')
            # 添加平滑曲线
            window = min(50, len(values) // 5)
            if window > 1:
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                ax1.plot(steps[window-1:], smoothed, color='blue', linewidth=2)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('奖励')
            ax1.set_title('训练奖励')
            ax1.grid(True, alpha=0.3)
            
        # 2. Critic 损失
        if 'critic_loss' in data:
            ax2 = fig.add_subplot(gs[0, 1])
            steps, values = zip(*data['critic_loss'])
            ax2.plot(steps, values, alpha=0.5, color='red')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('损失')
            ax2.set_title('Critic 损失')
            ax2.grid(True, alpha=0.3)
            
        # 3. Actor 损失
        if 'actor_loss' in data:
            ax3 = fig.add_subplot(gs[0, 2])
            steps, values = zip(*data['actor_loss'])
            ax3.plot(steps, values, alpha=0.5, color='green')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('损失')
            ax3.set_title('Actor 损失')
            ax3.grid(True, alpha=0.3)
            
        # 4. Steerer 损失
        if 'steerer_loss' in data:
            ax4 = fig.add_subplot(gs[1, 0])
            steps, values = zip(*data['steerer_loss'])
            ax4.plot(steps, values, alpha=0.5, color='purple')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('损失')
            ax4.set_title('Steerer 损失')
            ax4.grid(True, alpha=0.3)
            
        # 5. 熵系数 Alpha
        if 'alpha' in data:
            ax5 = fig.add_subplot(gs[1, 1])
            steps, values = zip(*data['alpha'])
            ax5.plot(steps, values, color='orange')
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Alpha')
            ax5.set_title('SAC 熵系数')
            ax5.grid(True, alpha=0.3)
            
        # 6. 评估奖励
        if 'eval_reward' in data:
            ax6 = fig.add_subplot(gs[1, 2])
            steps, values = zip(*data['eval_reward'])
            ax6.plot(steps, values, marker='o', color='blue')
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('评估奖励')
            ax6.set_title('评估性能')
            ax6.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_evaluation_comparison(self,
                                   feat_results: Dict,
                                   baseline_results: List[Dict],
                                   save_path: Optional[str] = None,
                                   show: bool = True):
        """
        绘制 FEAT 与基线算法的对比图
        
        Args:
            feat_results: FEAT 评估结果
            baseline_results: 基线算法评估结果列表
            save_path: 保存路径
            show: 是否显示
        """
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        # 准备数据
        names = ['FEAT'] + [r['policy_name'] for r in baseline_results]
        rewards = [feat_results['mean_reward']] + [r['mean_reward'] for r in baseline_results]
        delays = [feat_results['mean_delay']*1000] + [r['mean_delay']*1000 for r in baseline_results]
        success_rates = [feat_results['mean_success_rate']*100] + [r['mean_success_rate']*100 for r in baseline_results]
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        
        # 1. 奖励对比
        bars = axes[0].bar(names, rewards, color=colors[:len(names)])
        axes[0].set_ylabel('平均奖励')
        axes[0].set_title('奖励对比')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, val in zip(bars, rewards):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 延迟对比
        bars = axes[1].bar(names, delays, color=colors[:len(names)])
        axes[1].set_ylabel('平均延迟 (ms)')
        axes[1].set_title('延迟对比')
        axes[1].tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars, delays):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 3. 成功率对比
        bars = axes[2].bar(names, success_rates, color=colors[:len(names)])
        axes[2].set_ylabel('成功率 (%)')
        axes[2].set_title('成功率对比')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].set_ylim(0, 105)
        
        for bar, val in zip(bars, success_rates):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"对比图已保存到: {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()


class ChannelVisualizer:
    """
    信道可视化器
    =============
    可视化 Jakes 信道模型的特性
    """
    
    @staticmethod
    def plot_channel_response(times: np.ndarray,
                             channel_gains_db: np.ndarray,
                             save_path: Optional[str] = None,
                             show: bool = True):
        """
        绘制信道响应
        
        Args:
            times: 时间数组
            channel_gains_db: 信道增益 (dB)
            save_path: 保存路径
            show: 是否显示
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 1. 时域响应
        axes[0].plot(times, channel_gains_db, linewidth=0.5)
        axes[0].set_xlabel('时间 (s)')
        axes[0].set_ylabel('信道增益 (dB)')
        axes[0].set_title('Jakes 信道时域响应')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 增益分布直方图
        axes[1].hist(channel_gains_db, bins=50, density=True, alpha=0.7, color='blue')
        axes[1].set_xlabel('信道增益 (dB)')
        axes[1].set_ylabel('概率密度')
        axes[1].set_title('信道增益分布')
        axes[1].grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_gain = np.mean(channel_gains_db)
        std_gain = np.std(channel_gains_db)
        axes[1].axvline(mean_gain, color='red', linestyle='--', 
                       label=f'均值: {mean_gain:.1f} dB')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
            
    @staticmethod
    def plot_doppler_spectrum(max_doppler: float,
                             num_samples: int = 1000,
                             save_path: Optional[str] = None,
                             show: bool = True):
        """
        绘制多普勒频谱
        
        Args:
            max_doppler: 最大多普勒频移
            num_samples: 采样数量
            save_path: 保存路径
            show: 是否显示
        """
        # 生成频率轴
        f = np.linspace(-max_doppler * 1.5, max_doppler * 1.5, num_samples)
        
        # Clarke/Jakes 频谱 (理论)
        S = np.zeros_like(f)
        mask = np.abs(f) < max_doppler
        S[mask] = 1 / (np.pi * max_doppler * np.sqrt(1 - (f[mask]/max_doppler)**2 + 1e-10))
        
        plt.figure(figsize=(10, 6))
        plt.plot(f, S, linewidth=2)
        plt.xlabel('频率 (Hz)')
        plt.ylabel('功率谱密度')
        plt.title(f'Jakes 多普勒功率谱 (f_d = {max_doppler:.1f} Hz)')
        plt.grid(True, alpha=0.3)
        plt.xlim(-max_doppler * 1.2, max_doppler * 1.2)
        
        # 标注最大多普勒频移
        plt.axvline(max_doppler, color='red', linestyle='--', alpha=0.7)
        plt.axvline(-max_doppler, color='red', linestyle='--', alpha=0.7)
        plt.text(max_doppler * 1.05, plt.ylim()[1] * 0.9, f'f_d', color='red')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()


class PolicyVisualizer:
    """
    策略可视化器
    =============
    可视化策略选择和动作分布
    """
    
    @staticmethod
    def plot_policy_distribution(policy_counts: Dict[int, int],
                                save_path: Optional[str] = None,
                                show: bool = True):
        """
        绘制策略使用分布
        
        Args:
            policy_counts: 策略使用次数字典
            save_path: 保存路径
            show: 是否显示
        """
        policies = list(policy_counts.keys())
        counts = list(policy_counts.values())
        total = sum(counts)
        percentages = [c/total*100 for c in counts]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 饼图
        colors = plt.cm.Set3(np.linspace(0, 1, len(policies)))
        axes[0].pie(counts, labels=[f'策略 {p}' for p in policies],
                   autopct='%1.1f%%', colors=colors)
        axes[0].set_title('策略使用分布')
        
        # 2. 柱状图
        bars = axes[1].bar([f'策略 {p}' for p in policies], counts, color=colors)
        axes[1].set_ylabel('使用次数')
        axes[1].set_title('策略使用频率')
        
        for bar, pct in zip(bars, percentages):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
            
    @staticmethod
    def plot_action_heatmap(offloading_decisions: np.ndarray,
                           power_allocations: np.ndarray,
                           num_devices: int,
                           save_path: Optional[str] = None,
                           show: bool = True):
        """
        绘制动作热力图
        
        Args:
            offloading_decisions: 卸载决策序列, shape: (time_steps, num_devices)
            power_allocations: 功率分配序列, shape: (time_steps, num_devices)
            num_devices: 设备数量
            save_path: 保存路径
            show: 是否显示
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 卸载决策热力图
        im1 = axes[0].imshow(offloading_decisions.T, aspect='auto', 
                            cmap='RdYlGn', vmin=0, vmax=1)
        axes[0].set_xlabel('时隙')
        axes[0].set_ylabel('设备 ID')
        axes[0].set_title('卸载决策 (绿=卸载, 红=本地)')
        plt.colorbar(im1, ax=axes[0])
        
        # 2. 功率分配热力图
        im2 = axes[1].imshow(power_allocations.T, aspect='auto',
                            cmap='YlOrRd', vmin=0, vmax=1)
        axes[1].set_xlabel('时隙')
        axes[1].set_ylabel('设备 ID')
        axes[1].set_title('功率分配 (归一化)')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()


class AdaptationVisualizer:
    """
    环境适应可视化器
    ==================
    可视化环境适应过程
    """
    
    @staticmethod
    def plot_adaptation_curve(finetune_rewards: List[float],
                             before_reward: float,
                             after_reward: float,
                             save_path: Optional[str] = None,
                             show: bool = True):
        """
        绘制适应曲线
        
        Args:
            finetune_rewards: 微调过程中的奖励
            before_reward: 微调前的平均奖励
            after_reward: 微调后的平均奖励
            save_path: 保存路径
            show: 是否显示
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        episodes = range(1, len(finetune_rewards) + 1)
        
        # 绘制微调曲线
        ax.plot(episodes, finetune_rewards, alpha=0.4, color='blue', label='Episode 奖励')
        
        # 添加平滑曲线
        window = min(20, len(finetune_rewards) // 5)
        if window > 1:
            smoothed = np.convolve(finetune_rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(finetune_rewards) + 1), smoothed, 
                   color='blue', linewidth=2, label='平滑奖励')
        
        # 添加参考线
        ax.axhline(before_reward, color='red', linestyle='--', 
                  label=f'微调前: {before_reward:.3f}')
        ax.axhline(after_reward, color='green', linestyle='--',
                  label=f'微调后: {after_reward:.3f}')
        
        ax.set_xlabel('微调 Episode')
        ax.set_ylabel('奖励')
        ax.set_title('环境适应过程 (仅微调 Steerer)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加提升信息
        improvement = after_reward - before_reward
        relative_improvement = improvement / abs(before_reward) * 100
        ax.text(0.02, 0.98, f'提升: {improvement:.3f} ({relative_improvement:.1f}%)',
               transform=ax.transAxes, va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    # 测试可视化功能
    print("=== 可视化模块测试 ===")
    
    # 测试多普勒频谱绘制
    print("\n绘制多普勒频谱...")
    ChannelVisualizer.plot_doppler_spectrum(
        max_doppler=24.0,  # 对应 3m/s 速度，2.4GHz 载波
        show=False
    )
    print("完成")
    
    # 测试策略分布
    print("\n绘制策略分布...")
    policy_counts = {0: 150, 1: 200, 2: 100}
    PolicyVisualizer.plot_policy_distribution(policy_counts, show=False)
    print("完成")
    
    # 测试适应曲线
    print("\n绘制适应曲线...")
    finetune_rewards = np.random.randn(200).cumsum() / 10 - 5
    AdaptationVisualizer.plot_adaptation_curve(
        finetune_rewards=list(finetune_rewards),
        before_reward=-8.0,
        after_reward=-3.0,
        show=False
    )
    print("完成")
    
    print("\n所有可视化测试完成!")
