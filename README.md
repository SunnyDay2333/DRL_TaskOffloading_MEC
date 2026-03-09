# FEAT: Fast Environment-Adaptive Task Offloading Algorithm

基于 PyTorch 的 FEAT 边缘计算卸载算法完整实现

## 论文参考

> FEAT: Towards Fast Environment-Adaptive Task Offloading and Power Allocation in MEC

## 项目结构

```
FEAT3/
├── config.py                 # 配置参数管理
├── environment/
│   ├── __init__.py
│   ├── channel_model.py      # Jakes 时变信道模型
│   └── mec_environment.py    # MEC 仿真环境
├── models/
│   ├── __init__.py
│   ├── networks.py           # 神经网络架构
│   └── feat_agent.py         # FEAT 智能体
├── utils/
│   ├── __init__.py
│   ├── replay_buffer.py      # 经验回放池
│   └── helpers.py            # 工具函数
├── train.py                  # 训练脚本
├── evaluate.py               # 评估脚本
├── requirements.txt          # 依赖包
└── README.md                 # 说明文档
```

## 核心特性

### 1. 物理环境模型

- **Jakes 时变信道**: 使用正弦波叠加法 (SoS) 实现真实的多普勒衰落
- **大尺度衰落**: 路径损耗 + 对数正态阴影衰落
- **SINR 计算**: 考虑多用户干扰
- **队列模型**: 本地和边缘服务器的排队动态

### 2. FEAT 网络架构

- **Meta-Policies**: K=3 个共享底层的元策略网络
- **Steerer**: 策略选择器，使用 Gumbel-Softmax 保证可微分
- **Double Q-Network**: 双 Q 网络减少过估计

### 3. 训练策略 (论文 Table II 第5组)

- **Steerer 更新 (CMQ)**: 训练选择当前 Q 值最高的策略
- **Meta-Policy 更新 (HSD)**: 只更新被选择的策略
- **Q 网络更新 (CMQ)**: 选择最大 Q 值计算目标

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
# 默认训练
python train.py --exp_name my_experiment

# 自定义参数
python train.py \
    --exp_name feat_custom \
    --num_episodes 5000 \
    --batch_size 256 \
    --seed 42 \
    --device cuda
```

### 评估模型

```bash
# 标准评估
python evaluate.py --model_path ./checkpoints/best_model.pt

# 环境适应性测试
python evaluate.py \
    --model_path ./checkpoints/best_model.pt \
    --test_adaptation \
    --new_bandwidth 9e6 \
    --finetune_episodes 200

# 基线对比
python evaluate.py \
    --model_path ./checkpoints/best_model.pt \
    --compare_baselines
```

## 环境参数 (默认值)

| 参数 | 值 | 说明 |
|------|-----|------|
| M | 12 | 移动设备数量 |
| N | 10 | 每 Episode 时隙数 |
| δ | 0.5s | 时隙长度 |
| B | 10 MHz | 系统带宽 |
| f^ES | 9 GHz | 边缘服务器计算能力 |
| f^MD | 1 GHz | 移动设备计算能力 |
| p^max | 0.5 W | 最大发射功率 |
| σ² | -100 dBm | 噪声功率 |

## 任务参数

| 参数 | 值 | 说明 |
|------|-----|------|
| d | N(700, 1000) Kb | 任务大小分布 |
| c | [800, 900] cycles/bit | 计算密度 |
| t^max | [0.8, 0.9] s | 延迟容忍 |

## 代码扩展

### 添加新的策略网络

```python
from models.networks import PolicyHead

# 自定义策略头
class CustomPolicyHead(PolicyHead):
    def __init__(self, feature_dim, action_dim):
        super().__init__(feature_dim, action_dim)
        # 添加自定义层
```

### 修改环境配置

```python
from config import EnvironmentConfig

# 创建自定义配置
config = EnvironmentConfig()
config.bandwidth = 20e6  # 修改带宽
config.num_mobile_devices = 20  # 修改设备数量
```

### 环境适应性测试

```python
from evaluate import Evaluator

# 加载预训练模型
evaluator = Evaluator(model_path='./checkpoints/best_model.pt')

# 测试在新环境的适应能力
results = evaluator.test_environment_adaptation(
    new_bandwidth=9e6,      # 新带宽
    new_task_variance=1200e3,  # 新任务方差
    finetune_episodes=200   # 微调 Episode 数
)
```

## 实验结果说明

训练过程中会输出以下指标:

- **reward**: Episode 奖励 (越高越好)
- **critic_loss**: Q 网络损失
- **actor_loss**: 策略网络损失
- **steerer_loss**: Steerer 网络损失
- **alpha**: SAC 熵系数
- **steerer_temp**: Gumbel-Softmax 温度

评估指标:

- **mean_reward**: 平均奖励
- **mean_delay**: 平均延迟 (ms)
- **mean_energy**: 平均能耗 (mJ)
- **success_rate**: 任务成功率

## 关键算法说明

### CMQ (Current Max Q) 损失

```
用于 Steerer 和 Q 网络更新
目标: 选择当前 Q 值最高的策略
Loss_steerer = CE(steerer_output, one_hot(argmax_k Q(s,a_k)))
```

### HSD (Hard Selection with Detachment) 更新

```
用于 Meta-Policy 更新
只更新被选择的策略 k
其他 K-1 个策略的参数不更新
```

### Gumbel-Softmax

```
训练时: 使用 Gumbel-Softmax 保证梯度可导
推理时: 使用普通 Softmax 选择策略
```

## 常见问题

### Q: 训练不收敛怎么办?

1. 检查学习率是否合适
2. 增加经验回放池大小
3. 调整熵系数 alpha

### Q: 如何加速训练?

1. 使用 GPU (`--device cuda`)
2. 增加批量大小
3. 减少评估频率

### Q: 如何添加新的基线算法?

在 `evaluate.py` 的 `BaselineComparison` 类中添加新方法。

## 许可证

MIT License

## 致谢

本项目基于论文《FEAT: Towards Fast Environment-Adaptive Task Offloading and Power Allocation in MEC》实现。
