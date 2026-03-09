"""
MEC 环境类实现
================
本文件实现了完整的移动边缘计算 (MEC) 仿真环境，包括:
- 移动设备分布和任务生成
- 通信模型 (基于 Jakes 时变信道)
- 本地和边缘计算模型
- 排队模型
- 奖励计算

参考论文: FEAT: Towards Fast Environment-Adaptive Task Offloading 
         and Power Allocation in MEC
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EnvironmentConfig, create_default_config
from environment.channel_model import JakesChannelModel, ChannelState


@dataclass
class Task:
    """
    任务数据类
    ==========
    表示单个计算任务的属性
    """
    device_id: int                    # 任务所属的移动设备 ID
    size: float                       # 任务大小 (bits)
    computation_density: float        # 计算密度 (cycles/bit)
    delay_tolerance: float            # 最大延迟容忍 (秒)
    arrival_time: float               # 任务到达时间
    
    @property
    def total_cycles(self) -> float:
        """任务所需的总计算周期"""
        return self.size * self.computation_density


@dataclass
class DeviceState:
    """
    移动设备状态
    ============
    保存单个移动设备的当前状态
    """
    device_id: int
    distance: float                   # 到基站的距离 (米)
    current_task: Optional[Task]      # 当前任务
    local_queue: float                # 本地队列长度 (bits)
    battery_level: float              # 电池电量 (J)
    channel_gain: float               # 当前信道增益


@dataclass
class EdgeServerState:
    """
    边缘服务器状态
    ==============
    保存边缘服务器的当前状态
    """
    queue_length: float               # 当前队列长度 (bits)
    processing_capacity: float        # 每时隙处理能力 (bits)
    tasks_in_queue: List[Tuple[int, float]] = field(default_factory=list)  # (device_id, task_size)


@dataclass
class StepResult:
    """
    单步执行结果
    ============
    保存环境执行一步后的所有信息
    """
    delays: np.ndarray                # 每个设备的延迟
    energy_costs: np.ndarray          # 每个设备的能耗
    success_flags: np.ndarray         # 任务是否成功完成
    reward: float                     # 总奖励
    done: bool                        # Episode 是否结束
    info: Dict                        # 额外信息


class MECEnvironment:
    """
    MEC 仿真环境
    =============
    实现完整的移动边缘计算环境，支持:
    - 多设备场景
    - 时变 Jakes 信道
    - 本地/边缘计算选择
    - 队列动态更新
    
    遵循 OpenAI Gym 风格的接口设计
    
    Attributes:
        config: 环境配置
        channel_model: Jakes 信道模型
        devices: 设备状态列表
        edge_server: 边缘服务器状态
        current_step: 当前时隙索引
    """
    
    def __init__(self, 
                 config: Optional[EnvironmentConfig] = None,
                 seed: Optional[int] = None):
        """
        初始化 MEC 环境
        
        Args:
            config: 环境配置，如果为 None 则使用默认配置
            seed: 随机种子
        """
        # 加载配置
        if config is None:
            self.config = EnvironmentConfig()
        else:
            self.config = config
            
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None
            
        # 初始化信道模型
        self.channel_model = JakesChannelModel(
            num_devices=self.config.num_mobile_devices,
            carrier_frequency=self.config.carrier_frequency,
            mobile_speed=self.config.mobile_speed,
            num_sinusoids=self.config.num_sinusoids,
            path_loss_constant=self.config.path_loss_constant,
            path_loss_exponent=self.config.path_loss_exponent,
            shadow_fading_std=self.config.shadow_fading_std,
            time_slot_duration=self.config.time_slot_duration,
            seed=seed
        )
        
        # 初始化状态变量
        self.devices: List[DeviceState] = []
        self.edge_server: Optional[EdgeServerState] = None
        self.current_step: int = 0
        self.current_time: float = 0.0
        
        # 存储环境状态 (用于 Steerer)
        self.env_state = self._compute_env_state()
        
        # 初始化设备位置
        self.device_distances: Optional[np.ndarray] = None
        
        # 统计信息
        self.episode_stats = {
            'total_delay': 0.0,
            'total_energy': 0.0,
            'successful_tasks': 0,
            'failed_tasks': 0
        }
        
    def _compute_env_state(self) -> np.ndarray:
        """
        计算环境状态向量 (Steerer 输入)
        
        环境状态包含不可变的物理属性:
        - 任务分布参数 (均值, 方差)
        - 系统带宽
        - MD 计算能力
        - ES 计算能力
        
        Returns:
            环境状态向量, shape: (5,)
        """
        env_state = np.array([
            self.config.task_size_mean / 1e6,        # 归一化任务大小均值 (Mb)
            self.config.task_size_variance / 1e6,    # 归一化任务大小方差
            self.config.bandwidth / 1e6,             # 带宽 (MHz)
            self.config.md_computing_capacity / 1e9, # MD 计算能力 (GHz)
            self.config.es_computing_capacity / 1e9  # ES 计算能力 (GHz)
        ], dtype=np.float32)
        
        return env_state
    
    def _generate_device_positions(self) -> np.ndarray:
        """
        生成设备到基站的距离
        
        设备随机分布在距离 BS 20m-100m 的环形区域内
        
        Returns:
            距离数组, shape: (M,)
        """
        M = self.config.num_mobile_devices
        
        # 在环形区域内均匀分布
        # 使用平方根变换确保面积均匀
        r_min = self.config.min_distance
        r_max = self.config.max_distance
        
        # 均匀分布的半径 (考虑到面积与 r² 成正比)
        u = np.random.uniform(0, 1, size=M)
        distances = np.sqrt(u * (r_max**2 - r_min**2) + r_min**2)
        
        return distances
    
    def _generate_task(self, device_id: int) -> Task:
        """
        为指定设备生成新任务
        
        任务参数:
        - 大小: N(700, 1000) Kb
        - 计算密度: U[800, 900] cycles/bit
        - 延迟容忍: U[0.8, 0.9] s
        
        Args:
            device_id: 设备 ID
            
        Returns:
            Task: 生成的任务
        """
        # 任务大小 (高斯分布，限制为正值)
        task_size = np.random.normal(
            self.config.task_size_mean,
            np.sqrt(self.config.task_size_variance)
        )
        task_size = max(task_size, 100e3)  # 最小 100 Kb
        
        # 计算密度 (均匀分布)
        comp_density = np.random.uniform(
            self.config.computation_density_min,
            self.config.computation_density_max
        )
        
        # 延迟容忍 (均匀分布)
        delay_tol = np.random.uniform(
            self.config.delay_tolerance_min,
            self.config.delay_tolerance_max
        )
        
        return Task(
            device_id=device_id,
            size=task_size,
            computation_density=comp_density,
            delay_tolerance=delay_tol,
            arrival_time=self.current_time
        )
    
    def _init_devices(self):
        """
        初始化所有移动设备状态
        """
        self.devices = []
        
        for i in range(self.config.num_mobile_devices):
            device = DeviceState(
                device_id=i,
                distance=self.device_distances[i],
                current_task=self._generate_task(i),
                local_queue=0.0,
                battery_level=self.config.initial_battery,
                channel_gain=0.0  # 将在 reset 中更新
            )
            self.devices.append(device)
    
    def _init_edge_server(self):
        """
        初始化边缘服务器状态
        """
        # 每时隙处理能力 = 计算能力 * 时隙长度 / 平均计算密度
        avg_comp_density = (self.config.computation_density_min + 
                           self.config.computation_density_max) / 2
        processing_capacity = (self.config.es_computing_capacity * 
                              self.config.time_slot_duration / avg_comp_density)
        
        self.edge_server = EdgeServerState(
            queue_length=0.0,
            processing_capacity=processing_capacity,
            tasks_in_queue=[]
        )
    
    def reset(self, 
              new_config: Optional[EnvironmentConfig] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        重置环境到初始状态
        
        Args:
            new_config: 新的环境配置 (用于环境适应性测试)
            
        Returns:
            env_state: 环境状态向量 (Steerer 输入)
            time_varying_state: 时变状态向量 (Meta-Policy 输入)
            info: 额外信息
        """
        # 如果提供了新配置，则更新
        if new_config is not None:
            self.config = new_config
            self.env_state = self._compute_env_state()
            
        # 重置时间
        self.current_step = 0
        self.current_time = 0.0
        
        # 生成新的设备位置
        self.device_distances = self._generate_device_positions()
        
        # 重置信道模型
        self.channel_model.reset(reinit_phases=True)
        
        # 更新信道状态
        channel_state = self.channel_model.update_channel(self.device_distances)
        
        # 初始化设备和边缘服务器
        self._init_devices()
        self._init_edge_server()
        
        # 更新设备的信道增益
        for i, device in enumerate(self.devices):
            device.channel_gain = channel_state.channel_gain[i]
            
        # 重置统计信息
        self.episode_stats = {
            'total_delay': 0.0,
            'total_energy': 0.0,
            'successful_tasks': 0,
            'failed_tasks': 0
        }
        
        # 计算时变状态
        time_varying_state = self._compute_time_varying_state()
        
        info = {
            'device_distances': self.device_distances.copy(),
            'channel_gains': channel_state.channel_gain.copy(),
            'task_sizes': np.array([d.current_task.size for d in self.devices])
        }
        
        return self.env_state.copy(), time_varying_state, info
    
    def _compute_time_varying_state(self) -> np.ndarray:
        """
        计算时变状态向量 (Meta-Policy 输入)
        
        每个设备的状态:
        - 当前任务大小
        - 当前电池电量
        - 本地队列长度
        - 信道增益
        
        加上全局边缘队列长度
        
        Returns:
            时变状态向量
        """
        M = self.config.num_mobile_devices
        
        state = np.zeros(M * 4 + 1, dtype=np.float32)
        
        for i, device in enumerate(self.devices):
            # 归一化任务大小 (除以平均值)
            task_size_norm = device.current_task.size / self.config.task_size_mean
            
            # 归一化电池电量
            battery_norm = device.battery_level / self.config.initial_battery
            
            # 归一化本地队列 (除以平均任务大小)
            local_queue_norm = device.local_queue / self.config.task_size_mean
            
            # 归一化信道增益 (取对数并缩放)
            channel_gain_db = 10 * np.log10(max(device.channel_gain, 1e-30))
            channel_gain_norm = (channel_gain_db + 150) / 100  # 大致映射到 [0, 1]
            
            state[i * 4] = task_size_norm
            state[i * 4 + 1] = battery_norm
            state[i * 4 + 2] = local_queue_norm
            state[i * 4 + 3] = channel_gain_norm
            
        # 归一化边缘队列长度
        edge_queue_norm = self.edge_server.queue_length / (M * self.config.task_size_mean)
        state[-1] = edge_queue_norm
        
        return state
    
    def _compute_local_execution(self, 
                                 device: DeviceState,
                                 task: Task) -> Tuple[float, float]:
        """
        计算本地执行的延迟和能耗
        
        延迟: t^loc = (q_old + task_size) * c / f^MD
        能耗: e^loc = ξ * (f^MD)² * task_size * c
        
        Args:
            device: 设备状态
            task: 当前任务
            
        Returns:
            (delay, energy): 延迟 (秒) 和能耗 (J)
        """
        # 计算延迟
        total_bits = device.local_queue + task.size
        delay = (total_bits * task.computation_density) / self.config.md_computing_capacity
        
        # 计算能耗
        energy = (self.config.energy_coefficient * 
                 (self.config.md_computing_capacity ** 2) * 
                 task.size * task.computation_density)
        
        return delay, energy
    
    def _compute_edge_execution(self,
                               device: DeviceState,
                               task: Task,
                               transmit_power: float,
                               channel_gain: float,
                               num_offloading: int,
                               task_arrival_order: int) -> Tuple[float, float, float]:
        """
        计算边缘执行的延迟和能耗
        
        使用 OFDMA 模型：每个卸载设备分配独立子载波，无相互干扰
        
        总延迟 = 传输时间 + 排队等待时间 + 执行时间
        能耗 = 传输能耗
        
        Args:
            device: 设备状态
            task: 当前任务
            transmit_power: 发射功率 (W)
            channel_gain: 信道增益
            num_offloading: 本时隙卸载的设备数量
            task_arrival_order: 任务在本时隙的到达顺序
            
        Returns:
            (delay, energy, transmission_time): 延迟、能耗和传输时间
        """
        # OFDMA 模型：每个卸载设备分配独立子载波，无相互干扰
        num_offloading = max(num_offloading, 1)  # 至少1个
        
        # 每个设备分配的带宽 = 总带宽 / 卸载设备数
        allocated_bandwidth = self.config.bandwidth / num_offloading
        
        # 计算 SNR (无干扰，只有噪声)
        # SNR = g * p / σ²
        received_power = channel_gain * transmit_power
        snr = received_power / self.config.noise_power
        
        # 计算传输速率 (Shannon 公式)
        rate = allocated_bandwidth * np.log2(1 + max(snr, 1e-10))
        
        # 传输时间
        transmission_time = task.size / max(rate, 1e-10)
        
        # 排队等待时间
        # 等待时间取决于当前队列长度 + 本时隙内比该任务更早到达的任务
        wait_queue = self.edge_server.queue_length
        
        # 计算等待时间 (队列中的任务需要的处理时间)
        avg_comp_density = (self.config.computation_density_min + 
                           self.config.computation_density_max) / 2
        wait_time = (wait_queue * avg_comp_density) / self.config.es_computing_capacity
        
        # 执行时间
        execution_time = (task.size * task.computation_density) / self.config.es_computing_capacity
        
        # 总延迟
        total_delay = transmission_time + wait_time + execution_time
        
        # 传输能耗
        energy = transmit_power * transmission_time
        
        return total_delay, energy, transmission_time
    
    def _update_queues(self,
                      offloading_decisions: np.ndarray,
                      tasks: List[Task]):
        """
        更新本地和边缘队列
        
        队列动态:
        下一时刻队列 = max(0, 当前队列 + 新任务 - 本时隙处理能力)
        
        Args:
            offloading_decisions: 卸载决策 (0=本地, 1=边缘)
            tasks: 当前时隙的任务列表
        """
        # 更新边缘服务器队列
        new_edge_tasks = 0.0
        for i, decision in enumerate(offloading_decisions):
            if decision > 0.5:  # 卸载到边缘
                new_edge_tasks += tasks[i].size
                self.edge_server.tasks_in_queue.append((i, tasks[i].size))
                
        # 边缘队列更新
        self.edge_server.queue_length = max(
            0.0,
            self.edge_server.queue_length + new_edge_tasks - self.edge_server.processing_capacity
        )
        
        # 更新本地队列
        for i, device in enumerate(self.devices):
            if offloading_decisions[i] <= 0.5:  # 本地执行
                # 本地处理能力 (本时隙)
                local_capacity = (self.config.md_computing_capacity * 
                                 self.config.time_slot_duration /
                                 tasks[i].computation_density)
                
                # 更新队列
                device.local_queue = max(
                    0.0,
                    device.local_queue + tasks[i].size - local_capacity
                )
    
    def step(self, 
             actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        """
        执行一步环境交互
        
        Args:
            actions: 动作数组, shape: (M*2,)
                    - 前 M 个: 卸载决策 (0-1 连续值，>0.5 表示卸载)
                    - 后 M 个: 发射功率比例 (0-1，乘以 p_max 得到实际功率)
                    
        Returns:
            env_state: 环境状态
            time_varying_state: 时变状态
            reward: 奖励
            done: Episode 是否结束
            info: 额外信息
        """
        M = self.config.num_mobile_devices
        
        # 解析动作
        offloading_decisions = actions[:M]  # 卸载决策
        power_ratios = actions[M:]          # 功率比例
        
        # 将功率比例转换为实际功率
        transmit_powers = power_ratios * self.config.md_max_power
        
        # 获取当前信道增益
        channel_gains = np.array([d.channel_gain for d in self.devices])
        
        # 收集当前任务
        tasks = [d.current_task for d in self.devices]
        
        # 初始化结果数组
        delays = np.zeros(M)
        energies = np.zeros(M)
        successes = np.zeros(M, dtype=bool)
        
        # 计算卸载设备数量 (OFDMA 带宽分配)
        num_offloading = np.sum(offloading_decisions > 0.5)
        
        # 计算每个设备的延迟和能耗
        for i, device in enumerate(self.devices):
            task = device.current_task
            
            if offloading_decisions[i] > 0.5:
                # 边缘执行
                delay, energy, _ = self._compute_edge_execution(
                    device=device,
                    task=task,
                    transmit_power=transmit_powers[i],
                    channel_gain=channel_gains[i],
                    num_offloading=num_offloading,
                    task_arrival_order=i
                )
            else:
                # 本地执行
                delay, energy = self._compute_local_execution(device, task)
                
            delays[i] = delay
            energies[i] = energy
            
            # 检查是否满足延迟约束
            successes[i] = delay <= task.delay_tolerance
            
            # 更新电池电量
            device.battery_level = max(0.0, device.battery_level - energy)
            
        # 更新队列
        self._update_queues(offloading_decisions, tasks)
        
        # 更新时间
        self.current_step += 1
        self.current_time += self.config.time_slot_duration
        
        # 检查 Episode 是否结束
        done = self.current_step >= self.config.num_time_slots
        
        # 如果没有结束，生成新任务并更新信道
        if not done:
            # 更新信道
            channel_state = self.channel_model.update_channel(
                self.device_distances, 
                self.current_time
            )
            
            # 生成新任务并更新设备状态
            for i, device in enumerate(self.devices):
                device.current_task = self._generate_task(i)
                device.channel_gain = channel_state.channel_gain[i]
                
        # 获取更新后的电池电量 (用于约束检查)
        battery_levels = np.array([d.battery_level for d in self.devices])
        
        # 计算奖励 (严格按论文公式)
        reward = self._compute_reward(delays, energies, successes, tasks, battery_levels)
        
        # 更新统计信息
        self.episode_stats['total_delay'] += np.sum(delays)
        self.episode_stats['total_energy'] += np.sum(energies)
        self.episode_stats['successful_tasks'] += np.sum(successes)
        self.episode_stats['failed_tasks'] += np.sum(~successes)
        
        # 计算新的时变状态
        time_varying_state = self._compute_time_varying_state()
        
        # 计算 cost (用于监控，论文 Eq.12)
        avg_delay = np.mean(delays)
        avg_energy = np.mean(energies)
        cost = (self.config.delay_weight * avg_delay + 
               self.config.energy_weight * avg_energy)
        
        info = {
            'delays': delays,
            'energies': energies,
            'successes': successes,
            'offloading_decisions': offloading_decisions,
            'transmit_powers': transmit_powers,
            'channel_gains': channel_gains,
            'cost': cost,  # 当前时隙的 cost (论文 Eq.12)
            'avg_delay': avg_delay,
            'avg_energy': avg_energy,
            'battery_levels': battery_levels,
            'episode_stats': self.episode_stats.copy() if done else None
        }
        
        return self.env_state.copy(), time_varying_state, reward, done, info
    
    def _compute_local_cost(self, tasks: List[Task]) -> float:
        """
        计算所有任务本地执行时的 Cost (C_{i,local})
        
        用于 reward 归一化基准 (论文 reward 公式中的 C_{i,local}^{-1})
        
        Args:
            tasks: 当前时隙的任务列表
            
        Returns:
            全部本地执行的 cost
        """
        M = len(tasks)
        local_delays = np.zeros(M)
        local_energies = np.zeros(M)
        
        for i, device in enumerate(self.devices):
            task = tasks[i]
            delay, energy = self._compute_local_execution(device, task)
            local_delays[i] = delay
            local_energies[i] = energy
        
        # 平均延迟和平均能耗 (论文 Eq.10)
        avg_delay = np.mean(local_delays)
        avg_energy = np.mean(local_energies)
        
        # Cost = η * t_i + (1-η) * e_i (论文 Eq.12)
        cost_local = (self.config.delay_weight * avg_delay + 
                     self.config.energy_weight * avg_energy)
        
        return cost_local
    
    def _compute_reward(self,
                       delays: np.ndarray,
                       energies: np.ndarray,
                       successes: np.ndarray,
                       tasks: List[Task],
                       battery_levels: np.ndarray) -> float:
        """
        计算奖励 (严格按论文公式)
        
        论文 Reward 定义:
        r_i = 1(·) * α * (C_i^{-1} - C_{i,local}^{-1}) / C_{i,local}^{-1}
            + (1 - 1(·)) * Σ β_j * r_j^{penal}
        
        其中:
        - C_i = η * t_i + (1-η) * e_i (当前策略的 cost, 论文 Eq.12)
        - C_{i,local} 是所有任务本地执行的 cost
        - α 是归一化即时奖励的系数
        - 1(·) 是指示函数，所有约束满足时为1，否则为0
        - β_j 是每个约束的权重
        - r_j^{penal} 是与约束违反程度成比例的惩罚
        
        约束条件:
        - C1: a_i^m ∈ {0, 1} (二值卸载，由网络输出保证)
        - C2: 0 ≤ p_i^m ≤ pMD (功率范围，由动作空间保证)
        - C3: delay ≤ tmax (延迟约束)
        - C4: eb_i^m ≥ 0 (能量预算非负)
        
        Args:
            delays: 延迟数组
            energies: 能耗数组
            successes: 成功标志数组 (C3约束是否满足)
            tasks: 任务列表
            battery_levels: 各设备当前电池电量
            
        Returns:
            总奖励
        """
        M = len(tasks)
        
        # ==================== 计算当前策略的 Cost (C_i) ====================
        # 平均延迟 t_i = (1/M) * Σ delay_m (论文 Eq.10)
        avg_delay = np.mean(delays)
        
        # 平均能耗 e_i = (1/M) * Σ energy_m (论文 Eq.10)
        avg_energy = np.mean(energies)
        
        # Cost C_i = η * t_i + (1-η) * e_i (论文 Eq.12)
        cost_current = (self.config.delay_weight * avg_delay + 
                       self.config.energy_weight * avg_energy)
        
        # ==================== 计算本地执行的 Cost (C_{i,local}) ====================
        cost_local = self._compute_local_cost(tasks)
        
        # ==================== 检查约束条件 ====================
        # C3: 延迟约束 - delay ≤ tmax
        delay_tolerances = np.array([t.delay_tolerance for t in tasks])
        constraint_c3_satisfied = np.all(delays <= delay_tolerances)
        
        # C4: 能量预算约束 - eb_i^m ≥ 0
        constraint_c4_satisfied = np.all(battery_levels >= 0)
        
        # 所有约束是否满足
        all_constraints_satisfied = constraint_c3_satisfied and constraint_c4_satisfied
        
        # ==================== 计算 Reward ====================
        if all_constraints_satisfied:
            # 所有约束满足: r_i = α * (C_i^{-1} - C_{i,local}^{-1}) / C_{i,local}^{-1}
            # 即 r_i = α * (C_{i,local} / C_i - 1)
            # 这样当 C_i < C_{i,local} 时 reward > 0 (比全本地执行好)
            # 当 C_i > C_{i,local} 时 reward < 0 (比全本地执行差)
            
            # 防止除零
            cost_current = max(cost_current, 1e-10)
            cost_local = max(cost_local, 1e-10)
            
            # 归一化 reward (论文公式)
            reward = self.config.reward_coefficient * (cost_local / cost_current - 1.0)
            
        else:
            # 约束违反: r_i = Σ β_j * r_j^{penal}
            penalty = 0.0
            
            # C3 惩罚: 延迟超时惩罚 (与超时程度成比例)
            if not constraint_c3_satisfied:
                # 计算超时程度: (delay - tmax) / tmax
                delay_violations = np.maximum(0, delays - delay_tolerances) / delay_tolerances
                # 惩罚与平均超时程度成比例
                penalty += self.config.penalty_delay * np.mean(delay_violations)
            
            # C4 惩罚: 能量不足惩罚 (与能量不足程度成比例)
            if not constraint_c4_satisfied:
                # 计算能量不足程度: -battery / initial_battery (当 battery < 0 时)
                energy_violations = np.maximum(0, -battery_levels) / self.config.initial_battery
                # 惩罚与平均能量不足程度成比例
                penalty += self.config.penalty_energy * np.mean(energy_violations)
            
            # 惩罚为负数
            reward = -penalty
        
        return reward
    
    def get_state_dims(self) -> Dict[str, int]:
        """
        获取状态空间维度
        
        Returns:
            维度字典
        """
        M = self.config.num_mobile_devices
        return {
            'env_state_dim': 5,
            'time_varying_state_dim': M * 4 + 1,
            'action_dim': M * 2
        }
    
    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取动作空间边界
        
        Returns:
            (lower_bound, upper_bound): 动作空间的上下界
        """
        M = self.config.num_mobile_devices
        action_dim = M * 2
        
        lower_bound = np.zeros(action_dim)
        upper_bound = np.ones(action_dim)
        
        return lower_bound, upper_bound
    
    def update_config(self, **kwargs):
        """
        更新环境配置
        
        用于环境适应性测试，可以修改带宽、任务分布等参数
        
        Args:
            **kwargs: 要更新的配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"警告: 配置中不存在参数 '{key}'")
                
        # 重新计算派生参数
        self.config.__post_init__()
        
        # 更新环境状态
        self.env_state = self._compute_env_state()
        
    def render(self, mode: str = 'text'):
        """
        渲染环境状态
        
        Args:
            mode: 渲染模式 ('text' 或 'plot')
        """
        if mode == 'text':
            print(f"\n=== 时隙 {self.current_step} / {self.config.num_time_slots} ===")
            print(f"边缘队列长度: {self.edge_server.queue_length / 1e3:.2f} Kb")
            print(f"设备状态:")
            for i, device in enumerate(self.devices):
                print(f"  设备 {i}: 距离={device.distance:.1f}m, "
                      f"任务={device.current_task.size/1e3:.1f}Kb, "
                      f"电量={device.battery_level:.1f}J, "
                      f"信道={10*np.log10(max(device.channel_gain,1e-30)):.1f}dB")


if __name__ == "__main__":
    # 测试 MEC 环境
    print("=== MEC 环境测试 ===")
    
    # 创建环境
    env = MECEnvironment(seed=42)
    
    # 重置环境
    env_state, time_varying_state, info = env.reset()
    print(f"环境状态维度: {env_state.shape}")
    print(f"时变状态维度: {time_varying_state.shape}")
    print(f"环境状态: {env_state}")
    
    # 渲染初始状态
    env.render()
    
    # 执行几步
    M = env.config.num_mobile_devices
    for step in range(3):
        # 随机动作
        actions = np.random.uniform(0, 1, size=M * 2)
        
        env_state, time_varying_state, reward, done, info = env.step(actions)
        
        print(f"\n时隙 {step + 1}: 奖励 = {reward:.4f}, 完成 = {done}")
        print(f"  平均延迟: {np.mean(info['delays']):.4f}s")
        print(f"  平均能耗: {np.mean(info['energies']):.6f}J")
        print(f"  成功率: {np.mean(info['successes']):.2%}")
        
    # 测试配置更新
    print("\n=== 测试配置更新 (环境适应) ===")
    env.update_config(bandwidth=9e6)
    print(f"新带宽: {env.config.bandwidth / 1e6} MHz")
    print(f"新环境状态: {env.env_state}")
