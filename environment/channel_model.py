"""
Jakes 信道模型实现
==================
本文件实现了基于 Jakes 模型的时变无线信道，包括:
- 大尺度衰落 (路径损耗 + 阴影衰落)
- 小尺度衰落 (基于正弦波叠加法的瑞利衰落)

参考论文: FEAT: Towards Fast Environment-Adaptive Task Offloading 
         and Power Allocation in MEC
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChannelState:
    """
    信道状态数据类
    ==============
    保存当前信道的所有状态信息
    """
    large_scale_fading: np.ndarray   # 大尺度衰落 (线性值)
    small_scale_fading: np.ndarray   # 小尺度衰落复数值
    channel_gain: np.ndarray         # 总信道增益 g = β|h|²
    phase: np.ndarray                # 当前相位 (用于 Jakes 更新)


class JakesChannelModel:
    """
    Jakes 时变信道模型
    ==================
    使用正弦波叠加法 (Sum-of-Sinusoids, SoS) 模拟多普勒频移导致的快衰落
    
    信道模型:
    - 大尺度衰落: β = -148.1 - 37.6*log10(d_km) + z, z~N(0, 8²)
    - 小尺度衰落: 使用 Jakes 模型生成瑞利衰落
    - 总增益: g = β * |h|²
    
    Attributes:
        num_devices: 移动设备数量
        num_sinusoids: 正弦波数量 (决定模型精度)
        max_doppler: 最大多普勒频移
        sample_interval: 采样间隔
    """
    
    def __init__(self,
                 num_devices: int,
                 carrier_frequency: float = 2.4e9,
                 mobile_speed: float = 3.0,
                 num_sinusoids: int = 20,
                 path_loss_constant: float = -148.1,
                 path_loss_exponent: float = 37.6,
                 shadow_fading_std: float = 8.0,
                 time_slot_duration: float = 0.5,
                 seed: Optional[int] = None):
        """
        初始化 Jakes 信道模型
        
        Args:
            num_devices: 移动设备数量
            carrier_frequency: 载波频率 (Hz)
            mobile_speed: 移动设备速度 (m/s)
            num_sinusoids: 正弦波叠加数量
            path_loss_constant: 路径损耗常数 (dB)
            path_loss_exponent: 路径损耗指数
            shadow_fading_std: 阴影衰落标准差 (dB)
            time_slot_duration: 时隙长度 (s)
            seed: 随机种子
        """
        self.num_devices = num_devices
        self.carrier_frequency = carrier_frequency
        self.mobile_speed = mobile_speed
        self.num_sinusoids = num_sinusoids
        self.path_loss_constant = path_loss_constant
        self.path_loss_exponent = path_loss_exponent
        self.shadow_fading_std = shadow_fading_std
        self.time_slot_duration = time_slot_duration
        
        # 计算最大多普勒频移
        # f_d = v * f_c / c
        speed_of_light = 3e8
        self.max_doppler = (mobile_speed * carrier_frequency) / speed_of_light
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        
        # 初始化 Jakes 模型参数
        self._init_jakes_parameters()
        
        # 当前时间索引
        self.time_index = 0
        
        # 存储当前信道状态
        self.current_state: Optional[ChannelState] = None
        
    def _init_jakes_parameters(self):
        """
        初始化 Jakes 模型的正弦波参数
        
        使用改进的 Jakes 模型 (Modified Jakes Model)
        h(t) = (1/√N) * Σ exp(j*2π*f_n*t + j*θ_n)
        
        其中:
        - f_n = f_d * cos(2πn/N) 是第 n 个正弦波的多普勒频移
        - θ_n 是随机初始相位
        """
        N = self.num_sinusoids
        M = self.num_devices
        
        # 为每个设备生成独立的参数
        # 多普勒频率数组: f_n = f_d * cos(2π(n+0.5)/(N+1)) for n = 0,...,N-1
        # 使用均匀分布的到达角
        n_indices = np.arange(N)
        
        # 到达角度 (均匀分布在 [0, 2π])
        # 使用 α_n = 2π(n + 0.5) / N 保证对称性
        self.arrival_angles = 2 * np.pi * (n_indices + 0.5) / N  # shape: (N,)
        
        # 多普勒频率 f_n = f_d * cos(α_n)
        self.doppler_frequencies = self.max_doppler * np.cos(self.arrival_angles)  # shape: (N,)
        
        # 为每个设备生成随机初始相位 θ_n ~ U(0, 2π)
        # shape: (M, N) - 每个设备有 N 个正弦波，每个有独立的初始相位
        self.initial_phases = np.random.uniform(0, 2 * np.pi, size=(M, N))
        
        # 正弦波的权重 (归一化)
        self.amplitude = 1.0 / np.sqrt(N)
        
    def compute_large_scale_fading(self, distances: np.ndarray) -> np.ndarray:
        """
        计算大尺度衰落
        
        公式: β = -148.1 - 37.6*log10(d_km) + z
        其中 z ~ N(0, σ²), σ = 8 dB
        
        Args:
            distances: 设备到基站的距离 (米), shape: (M,)
            
        Returns:
            大尺度衰落 (线性值), shape: (M,)
        """
        # 将距离转换为 km
        distances_km = distances / 1000.0
        
        # 防止对数运算出错
        distances_km = np.maximum(distances_km, 1e-6)
        
        # 计算路径损耗 (dB)
        path_loss_db = self.path_loss_constant - self.path_loss_exponent * np.log10(distances_km)
        
        # 添加阴影衰落 (对数正态分布)
        shadow_fading_db = np.random.normal(0, self.shadow_fading_std, size=distances.shape)
        
        # 总大尺度衰落 (dB)
        large_scale_db = path_loss_db + shadow_fading_db
        
        # 转换为线性值
        large_scale_linear = 10 ** (large_scale_db / 10)
        
        return large_scale_linear
    
    def compute_small_scale_fading(self, time: float) -> np.ndarray:
        """
        使用 Jakes 正弦波叠加法计算小尺度衰落
        
        复数衰落系数:
        h(t) = (1/√N) * Σ exp(j*(2π*f_n*t + θ_n))
        
        Args:
            time: 当前时间 (秒)
            
        Returns:
            复数衰落系数, shape: (M,)
        """
        M = self.num_devices
        N = self.num_sinusoids
        
        # 计算每个正弦波的相位
        # phase[m,n] = 2π * f_n * t + θ_n[m]
        # self.doppler_frequencies: (N,)
        # self.initial_phases: (M, N)
        
        phase = (2 * np.pi * self.doppler_frequencies * time + self.initial_phases)  # (M, N)
        
        # 计算复数衰落系数 (正弦波叠加)
        # h = (1/√N) * Σ exp(j*phase)
        complex_fading = self.amplitude * np.sum(np.exp(1j * phase), axis=1)  # (M,)
        
        return complex_fading
    
    def update_channel(self, 
                       distances: np.ndarray,
                       time: Optional[float] = None) -> ChannelState:
        """
        更新信道状态
        
        Args:
            distances: 设备到基站的距离 (米), shape: (M,)
            time: 当前时间 (秒)，如果为 None 则使用内部时间索引
            
        Returns:
            ChannelState: 更新后的信道状态
        """
        if time is None:
            time = self.time_index * self.time_slot_duration
            
        # 计算大尺度衰落 (每个 Episode 开始时重新计算)
        if self.current_state is None or self.time_index == 0:
            large_scale = self.compute_large_scale_fading(distances)
        else:
            # 保持大尺度衰落不变 (假设设备位置在一个 Episode 内不变)
            large_scale = self.current_state.large_scale_fading
            
        # 计算小尺度衰落
        small_scale = self.compute_small_scale_fading(time)
        
        # 计算总信道增益 g = β * |h|²
        channel_gain = large_scale * np.abs(small_scale) ** 2
        
        # 更新时间索引
        self.time_index += 1
        
        # 保存当前状态
        self.current_state = ChannelState(
            large_scale_fading=large_scale,
            small_scale_fading=small_scale,
            channel_gain=channel_gain,
            phase=np.angle(small_scale)
        )
        
        return self.current_state
    
    def reset(self, 
              new_distances: Optional[np.ndarray] = None,
              reinit_phases: bool = True):
        """
        重置信道模型
        
        Args:
            new_distances: 新的设备距离 (可选)
            reinit_phases: 是否重新初始化随机相位
        """
        self.time_index = 0
        self.current_state = None
        
        if reinit_phases:
            # 重新生成随机初始相位
            self.initial_phases = np.random.uniform(
                0, 2 * np.pi, 
                size=(self.num_devices, self.num_sinusoids)
            )
    
    def get_current_channel_gain(self) -> np.ndarray:
        """
        获取当前信道增益
        
        Returns:
            信道增益数组, shape: (M,)
        """
        if self.current_state is None:
            raise ValueError("信道未初始化，请先调用 update_channel()")
        return self.current_state.channel_gain
    
    def compute_sinr(self,
                     channel_gains: np.ndarray,
                     transmit_powers: np.ndarray,
                     noise_power: float) -> np.ndarray:
        """
        计算信噪比 (SINR)
        
        公式: SINR_m = (g_m * p_m) / (Σ_{k≠m} g_k * p_k + σ²)
        
        Args:
            channel_gains: 信道增益, shape: (M,)
            transmit_powers: 发射功率, shape: (M,)
            noise_power: 噪声功率 (W)
            
        Returns:
            SINR 数组, shape: (M,)
        """
        M = len(channel_gains)
        
        # 计算接收信号功率
        received_power = channel_gains * transmit_powers  # (M,)
        
        # 计算干扰 + 噪声
        total_power = np.sum(received_power)
        interference_plus_noise = total_power - received_power + noise_power
        
        # 计算 SINR
        sinr = received_power / interference_plus_noise
        
        return sinr
    
    def compute_transmission_rate(self,
                                  sinr: np.ndarray,
                                  bandwidth: float) -> np.ndarray:
        """
        计算传输速率
        
        公式: R = B * log2(1 + SINR)
        
        Args:
            sinr: 信噪比, shape: (M,)
            bandwidth: 系统带宽 (Hz)
            
        Returns:
            传输速率 (bits/s), shape: (M,)
        """
        # 防止 log2(1 + 0) 的问题
        sinr = np.maximum(sinr, 1e-10)
        rate = bandwidth * np.log2(1 + sinr)
        return rate


class JakesChannelSimulator:
    """
    Jakes 信道仿真器
    =================
    用于生成和可视化 Jakes 信道的时变特性
    """
    
    def __init__(self, channel_model: JakesChannelModel):
        """
        初始化仿真器
        
        Args:
            channel_model: Jakes 信道模型实例
        """
        self.channel_model = channel_model
        
    def simulate_time_series(self,
                            distance: float,
                            duration: float,
                            sample_rate: float = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        仿真单个设备的时变信道
        
        Args:
            distance: 设备到基站的距离 (米)
            duration: 仿真时长 (秒)
            sample_rate: 采样率 (Hz)
            
        Returns:
            times: 时间点数组
            channel_gains: 对应的信道增益 (dB)
        """
        num_samples = int(duration * sample_rate)
        times = np.linspace(0, duration, num_samples)
        
        # 临时设置为单设备
        original_num_devices = self.channel_model.num_devices
        self.channel_model.num_devices = 1
        self.channel_model._init_jakes_parameters()
        
        # 计算大尺度衰落
        distances = np.array([distance])
        large_scale = self.channel_model.compute_large_scale_fading(distances)[0]
        
        # 计算时变小尺度衰落
        channel_gains_linear = []
        for t in times:
            small_scale = self.channel_model.compute_small_scale_fading(t)[0]
            gain = large_scale * np.abs(small_scale) ** 2
            channel_gains_linear.append(gain)
            
        channel_gains_linear = np.array(channel_gains_linear)
        
        # 转换为 dB
        channel_gains_db = 10 * np.log10(channel_gains_linear + 1e-30)
        
        # 恢复原始设置
        self.channel_model.num_devices = original_num_devices
        self.channel_model._init_jakes_parameters()
        
        return times, channel_gains_db


if __name__ == "__main__":
    # 测试 Jakes 信道模型
    print("=== Jakes 信道模型测试 ===")
    
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
    print(f"设备距离: {distances}")
    
    # 更新信道
    state = model.update_channel(distances)
    print(f"大尺度衰落 (dB): {10*np.log10(state.large_scale_fading)}")
    print(f"信道增益 (dB): {10*np.log10(state.channel_gain)}")
    
    # 测试 SINR 计算
    powers = np.ones(12) * 0.5  # 0.5W
    noise_power = 1e-13  # -100 dBm
    sinr = model.compute_sinr(state.channel_gain, powers, noise_power)
    print(f"SINR (dB): {10*np.log10(sinr)}")
    
    # 计算传输速率
    bandwidth = 10e6  # 10 MHz
    rate = model.compute_transmission_rate(sinr, bandwidth)
    print(f"传输速率 (Mbps): {rate / 1e6}")
