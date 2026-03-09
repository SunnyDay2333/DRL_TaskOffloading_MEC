"""
FEAT 快速测试脚本
==================
本脚本用于快速验证整个系统是否正常工作

测试内容:
1. 环境模块测试
2. 网络模块测试
3. 训练流程测试 (少量 Episode)
4. 评估流程测试
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import time

from config import create_default_config, ExperimentConfig
from environment.mec_environment import MECEnvironment
from environment.channel_model import JakesChannelModel
from models.feat_agent import FEATAgent
from utils.replay_buffer import ReplayBuffer
from utils.helpers import set_seed


def test_channel_model():
    """测试 Jakes 信道模型"""
    print("\n" + "="*50)
    print("1. 测试 Jakes 信道模型")
    print("="*50)
    
    # 创建模型
    model = JakesChannelModel(
        num_devices=12,
        carrier_frequency=2.4e9,
        mobile_speed=3.0,
        num_sinusoids=20,
        seed=42
    )
    
    # 生成随机距离
    distances = np.random.uniform(20, 100, size=12)
    
    # 测试多个时隙
    print("测试时变信道...")
    for t in range(5):
        state = model.update_channel(distances, t * 0.5)
        mean_gain_db = 10 * np.log10(np.mean(state.channel_gain))
        print(f"  时隙 {t}: 平均信道增益 = {mean_gain_db:.1f} dB")
        
    # 测试 SINR 计算
    powers = np.ones(12) * 0.3
    noise_power = 1e-13
    sinr = model.compute_sinr(state.channel_gain, powers, noise_power)
    print(f"\nSINR 范围: [{10*np.log10(sinr.min()):.1f}, {10*np.log10(sinr.max()):.1f}] dB")
    
    print("✓ Jakes 信道模型测试通过")
    return True


def test_environment():
    """测试 MEC 环境"""
    print("\n" + "="*50)
    print("2. 测试 MEC 环境")
    print("="*50)
    
    # 创建环境
    env = MECEnvironment(seed=42)
    
    print(f"设备数量: {env.config.num_mobile_devices}")
    print(f"时隙数量: {env.config.num_time_slots}")
    print(f"带宽: {env.config.bandwidth / 1e6} MHz")
    
    # 重置环境
    env_state, time_varying_state, info = env.reset()
    
    print(f"\n状态维度:")
    print(f"  环境状态: {env_state.shape}")
    print(f"  时变状态: {time_varying_state.shape}")
    
    # 执行几步
    M = env.config.num_mobile_devices
    total_reward = 0
    
    print("\n执行 Episode:")
    for step in range(env.config.num_time_slots):
        # 随机动作
        actions = np.random.uniform(0, 1, size=M * 2)
        env_state, time_varying_state, reward, done, info = env.step(actions)
        total_reward += reward
        
    print(f"  Episode 奖励: {total_reward:.4f}")
    print(f"  成功任务数: {env.episode_stats['successful_tasks']}")
    print(f"  失败任务数: {env.episode_stats['failed_tasks']}")
    
    print("✓ MEC 环境测试通过")
    return True


def test_neural_networks():
    """测试神经网络"""
    print("\n" + "="*50)
    print("3. 测试神经网络")
    print("="*50)
    
    from models.networks import MetaPolicyNetwork, SteererNetwork, QNetwork
    
    # 参数
    batch_size = 32
    state_dim = 49
    env_state_dim = 5
    action_dim = 24
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建网络
    meta_policy = MetaPolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        num_policies=3
    ).to(device)
    
    steerer = SteererNetwork(
        env_state_dim=env_state_dim,
        feature_dim=256,
        num_policies=3
    ).to(device)
    
    q_net = QNetwork(
        state_dim=env_state_dim + state_dim,
        action_dim=action_dim
    ).to(device)
    
    # 测试前向传播
    state = torch.randn(batch_size, state_dim).to(device)
    env_state = torch.randn(batch_size, env_state_dim).to(device)
    action = torch.rand(batch_size, action_dim).to(device)
    
    # Meta-Policy
    means, log_stds, features = meta_policy(state)
    print(f"\nMeta-Policy:")
    print(f"  策略数量: {len(means)}")
    print(f"  均值形状: {means[0].shape}")
    print(f"  特征形状: {features.shape}")
    
    # Steerer
    weights, selected = steerer(env_state, features)
    print(f"\nSteerer:")
    print(f"  权重形状: {weights.shape}")
    print(f"  选择分布: {np.bincount(selected.cpu().numpy(), minlength=3)}")
    
    # Q-Network
    full_state = torch.cat([env_state, state], dim=1)
    q1, q2 = q_net(full_state, action)
    print(f"\nQ-Network:")
    print(f"  Q1 形状: {q1.shape}")
    print(f"  Q 值范围: [{min(q1.min(), q2.min()):.2f}, {max(q1.max(), q2.max()):.2f}]")
    
    print("\n✓ 神经网络测试通过")
    return True


def test_feat_agent():
    """测试 FEAT 智能体"""
    print("\n" + "="*50)
    print("4. 测试 FEAT 智能体")
    print("="*50)
    
    env_state_dim = 5
    time_varying_state_dim = 49
    action_dim = 24
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建智能体
    agent = FEATAgent(
        env_state_dim=env_state_dim,
        time_varying_state_dim=time_varying_state_dim,
        action_dim=action_dim,
        num_policies=3,
        device=device
    )
    
    print(f"Meta-Policy 参数: {sum(p.numel() for p in agent.meta_policy.parameters()):,}")
    print(f"Steerer 参数: {sum(p.numel() for p in agent.steerer.parameters()):,}")
    print(f"Q-Network 参数: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    
    # 测试动作选择
    env_state = np.random.randn(env_state_dim).astype(np.float32)
    time_varying_state = np.random.randn(time_varying_state_dim).astype(np.float32)
    
    action, selected_k = agent.select_action(env_state, time_varying_state)
    print(f"\n动作选择:")
    print(f"  动作形状: {action.shape}")
    print(f"  选择策略: {selected_k}")
    print(f"  动作范围: [{action.min():.3f}, {action.max():.3f}]")
    
    # 测试更新
    batch_size = 64
    batch = {
        'env_states': np.random.randn(batch_size, env_state_dim).astype(np.float32),
        'time_varying_states': np.random.randn(batch_size, time_varying_state_dim).astype(np.float32),
        'actions': np.random.rand(batch_size, action_dim).astype(np.float32),
        'rewards': np.random.randn(batch_size).astype(np.float32),
        'next_env_states': np.random.randn(batch_size, env_state_dim).astype(np.float32),
        'next_time_varying_states': np.random.randn(batch_size, time_varying_state_dim).astype(np.float32),
        'dones': np.zeros(batch_size).astype(np.float32),
        'selected_ks': np.random.randint(0, 3, batch_size)
    }
    
    metrics = agent.update(batch)
    print(f"\n更新指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
        
    print("\n✓ FEAT 智能体测试通过")
    return True


def test_training_loop():
    """测试完整训练流程 (少量 Episode)"""
    print("\n" + "="*50)
    print("5. 测试训练流程 (10 Episodes)")
    print("="*50)
    
    set_seed(42)
    
    # 创建配置
    config = create_default_config()
    
    # 创建环境
    env = MECEnvironment(config=config.env_config, seed=42)
    
    # 获取维度
    state_dims = env.get_state_dims()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建智能体
    agent = FEATAgent(
        env_state_dim=state_dims['env_state_dim'],
        time_varying_state_dim=state_dims['time_varying_state_dim'],
        action_dim=state_dims['action_dim'],
        num_policies=3,
        device=device
    )
    
    # 创建回放池
    buffer = ReplayBuffer(capacity=10000)
    
    # 训练循环
    start_time = time.time()
    total_steps = 0
    episode_rewards = []
    
    print("\n开始训练...")
    
    for episode in range(10):
        env_state, time_varying_state, _ = env.reset()
        episode_reward = 0
        
        while True:
            # 选择动作
            if total_steps < 100:
                action = np.random.uniform(0, 1, state_dims['action_dim'])
                selected_k = np.random.randint(0, 3)
            else:
                action, selected_k = agent.select_action(env_state, time_varying_state)
                
            # 执行动作
            next_env_state, next_time_varying_state, reward, done, _ = env.step(action)
            
            # 存储经验
            buffer.push(
                env_state, time_varying_state, action, reward,
                next_env_state, next_time_varying_state, done, selected_k
            )
            
            episode_reward += reward
            total_steps += 1
            
            # 更新
            if len(buffer) >= 64:
                batch = buffer.sample(64)
                agent.update(batch)
                
            if done:
                break
                
            env_state = next_env_state
            time_varying_state = next_time_varying_state
            
        episode_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: 奖励 = {episode_reward:.3f}")
        
    elapsed = time.time() - start_time
    
    print(f"\n训练完成:")
    print(f"  总步数: {total_steps}")
    print(f"  平均奖励: {np.mean(episode_rewards):.3f}")
    print(f"  用时: {elapsed:.1f}s")
    
    print("\n✓ 训练流程测试通过")
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("FEAT 算法快速测试")
    print("="*60)
    
    tests = [
        ("Jakes 信道模型", test_channel_model),
        ("MEC 环境", test_environment),
        ("神经网络", test_neural_networks),
        ("FEAT 智能体", test_feat_agent),
        ("训练流程", test_training_loop)
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} 测试失败: {e}")
            results.append((name, False))
            import traceback
            traceback.print_exc()
            
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {name}: {status}")
        
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过! 系统可以正常使用。")
        print("\n下一步:")
        print("  1. 运行训练: python train.py --exp_name my_exp")
        print("  2. 评估模型: python evaluate.py --model_path ./checkpoints/best_model.pt")
    else:
        print("\n⚠️ 存在失败的测试，请检查错误信息。")
        
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
