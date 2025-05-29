#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
同步压缩变换SST_CWT算法模块

该模块实现了基于连续小波变换(CWT)的同步压缩变换算法。
主要包含SST_Base基类和SST_CWT实现类。

作者: 课题组
日期: 2025年5月28日
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import scipy.integrate
from sklearn.preprocessing import scale

# 设置中文字体
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["simhei"]
plt.rcParams["font.size"] = 14
plt.rcParams["axes.unicode_minus"] = False


class SST_Base:
    """
    同步压缩变换基类
    """

    def __init__(self, signal, fs, transform_param=None, gamma=0.001):
        """
        初始化SST基类

        参数:
            signal: 输入信号
            fs: 采样频率
            transform_param: 变换参数
            gamma: 阈值参数
        """
        self.signal = signal
        self.fs = fs
        self.transform_param = transform_param
        self.gamma = gamma

        # 结果存储
        self.transform_result = None
        self.inst_freq = None
        self.sst_result = None

    def transform(self):
        """
        执行时频变换 - 抽象方法，子类必须实现
        """
        raise NotImplementedError("子类必须实现transform方法")

    def calc_inst_freq(self):
        """
        计算瞬时频率 - 抽象方法，子类必须实现
        """
        raise NotImplementedError("子类必须实现calc_inst_freq方法")

    def energy_reassign(self):
        """
        能量重排算法 - 抽象方法，子类必须实现
        """
        raise NotImplementedError("子类必须实现energy_reassign方法")
    
    def sst(self):
        """
        执行完整的SST变换
        """
        raise NotImplementedError("子类必须实现sst方法")

    def plot(self, type="sst", **kwargs):
        """
        绘制时频图
        """
        import matplotlib.pyplot as plt

        if type == "transform":
            if self.transform_result is None:
                result = self.transform(
                    self.signal, self.fs, self.transform_param
                )
            else:
                result = self.transform_result
            title = kwargs.get("title", "CWT时频图")
        elif type == "sst":
            if self.sst_result is None:
                result = self.sst()
            else:
                result = self.sst_result
            title = kwargs.get("title", "SST时频图")
        else:
            raise ValueError("type参数必须为'transform'或'sst'")
        kwargs.pop("title", None)  # 移除title参数以避免重复
        self._plot_tf_map(result["t_Axis"], result["f_Axis"],result["tf_map"], title=title, **kwargs)

    @staticmethod
    def _plot_tf_map(t, f, tf_map, **kwargs):
        """
        绘制时频图
        """
        plt.figure(figsize=kwargs.get("figsize", (8, 6)))
        mesh = plt.pcolormesh(
            t,
            f,
            np.abs(tf_map),
            cmap=kwargs.get("cmap", "jet"),
            shading=kwargs.get("shading", "auto"),
            vmin=kwargs.get("vmin", None),
            vmax=kwargs.get("vmax", None),
        )
        plt.colorbar(label=kwargs.get("colorbar_label", "幅值"))
        plt.xlabel(kwargs.get("xlabel", "时间/s"))
        plt.ylabel(kwargs.get("ylabel", "频率/Hz"))
        if "title" in kwargs:
            plt.title(kwargs["title"])
        if "xlim" in kwargs:
            plt.xlim(kwargs["xlim"])
        if "ylim" in kwargs:
            plt.ylim(kwargs["ylim"])
        plt.tight_layout()
        plt.show()


class SST_CWT(SST_Base):
    """
    基于CWT的同步压缩变换
    """

    def __init__(self, signal, fs, cwt_param=None, gamma=1e-3, isSmooth=False):
        """
        初始化CWT-SST对象，cwt参数通过cwt_param字典传入
        """
        # 默认CWT参数
        transform_param = {
            "wavelet": "cmor10-1",  # 复Morlet小波
            "scales": None,  # 默认尺度
            "scalesType": "log",  # 默认尺度范围
            "scalesNum": 64,  # 默认尺度数量
        }
        if cwt_param is not None:
            transform_param.update(cwt_param)
        super().__init__(signal, fs, transform_param, gamma)
        # 检查是否安装了Numba
        try:
            import numba
            self.has_numba = True
        except ImportError:
            self.has_numba = False
        self.isSmooth = isSmooth  # 是否对SST结果平滑处理

    def _check_numba(self):
        """检测 numba 是否可用"""
        try:
            from numba import jit
            return True
        except ImportError:
            return False

    @staticmethod
    def _get_cwt_scales(fs, scalesType, scalesNum):
        """
        根据尺度类型和数量生成CWT尺度
        """
        import numpy as np

        if scalesType == "log":
            log_fn=np.log10(fs/2)
            log_f_Axis=np.linspace(log_fn-np.log10(200), log_fn, scalesNum)# f= fn/200~fn
            f_Axis = np.power(10, log_f_Axis)  # 生成对数尺度
            scales = fs/ f_Axis[::-1]  # 计算增序尺度
            return scales
        elif scalesType == "linear":
            f_Axis = np.linspace(fs/2/1000, fs / 2, scalesNum)
            scales = fs / f_Axis[::-1]
            return scales
        else:
            raise ValueError("未知的尺度类型")

    def transform(self, data, fs, transform_param):
        """
        计算CWT
        """
        from pywt import cwt
        import numpy as np

        # --------- 1. 边界填充 ---------
        pad_len = len(data) // 10
        data_padded = np.pad(data, pad_width=pad_len, mode="edge")
        # --------- 2. 生成尺度 ---------
        scales = transform_param["scales"]
        scalesType = transform_param["scalesType"]
        scalesNum = transform_param["scalesNum"]
        if scales is None:
            scales = self._get_cwt_scales(fs,scalesType, scalesNum)
        wavelet = transform_param["wavelet"]  # 小波类型
        t_Axis = np.arange(len(data)) / fs  # 时间轴
        # --------- 3. 计算CWT ---------
        coeffs, f_Axis = cwt(
            data=data_padded,
            wavelet=wavelet,
            scales=scales,
            sampling_period=t_Axis[1],
            method="fft",
        )
        # --------- 4. 处理结果 ---------
        coeffs = coeffs[::-1, pad_len:-pad_len]  # # 反转系数顺序并去除填充部分
        f_Axis = f_Axis[::-1]  # 频率轴调整为增大
        # --------- 5. 归一化能量 ---------
        data_energy = np.sum(np.abs(data) ** 2)
        cwt_energy = np.sum(np.abs(coeffs) ** 2)
        coeffs = coeffs * np.sqrt(data_energy / cwt_energy)
        result = {"f_Axis": f_Axis, "t_Axis": t_Axis, "tf_map": coeffs,"scales": scales}
        return result

    def calc_inst_freq(self, method="gradRatio"):
        """
        基于CWT计算瞬时频率
        """
        if self.transform_result is None:
            self.transform_result = self.transform(
                self.signal, self.fs, self.transform_param
            )

        C_x = self.transform_result["tf_map"]  # CWT系数
        t = self.transform_result["t_Axis"]  # 时间轴

        if method == "gradRatio":  # 常用方法
            freq_remap = self.__calc_inst_freq_gradRatio(C_x, t)
        elif method == "phaseGrad":
            freq_remap = self.__calc_inst_freq_phaseGrad(C_x, t)
        else:
            raise ValueError("未知的瞬时频率计算方法")

        return np.asarray(freq_remap)

    def __calc_inst_freq_gradRatio(self, C_x, t):
        """
        梯度比率方法计算瞬时频率
        """
        # 计算时间方向梯度
        dC_x = np.gradient(C_x, t, axis=1)

        # 幅值过滤
        magnitude = np.abs(C_x)
        threshold = self.gamma * np.max(magnitude)
        mask = magnitude >= threshold

        # 计算瞬时频率
        ratio = np.zeros_like(C_x, dtype=complex)
        ratio[mask] = dC_x[mask] / C_x[mask]
        freq_remap = np.imag(ratio) / (2 * np.pi)
        freq_remap[~mask] = np.nan

        return freq_remap

    def __calc_inst_freq_phaseGrad(self, C_x, t):
        """
        相位梯度方法计算瞬时频率
        """
        # 相位解缠绕
        phase = np.unwrap(np.angle(C_x), axis=1)

        # 计算相位梯度
        dphase_dt = np.gradient(phase, t, axis=1)

        # 幅值过滤
        magnitude = np.abs(C_x)
        threshold = self.gamma * np.max(magnitude)
        mask = magnitude >= threshold

        # 仅保留有效区域的正频率
        valid_mask = mask & (dphase_dt > 0)
        freq_remap = np.full_like(C_x, np.nan, dtype=float)
        freq_remap[valid_mask] = dphase_dt[valid_mask] / (2 * np.pi)

        return freq_remap

    def energy_reassign(self):
        """
        CWT能量重排算法
        """
        if self.inst_freq is None:
            self.inst_freq = self.calc_inst_freq()

        # 获取有效的频率重映射点
        valid_mask = ~np.isnan(self.inst_freq)

        if not np.any(valid_mask):
            # 如果没有有效点，返回零矩阵
            sst_tf_map = np.zeros_like(self.transform_result["tf_map"], dtype=complex)
        else:
            # 提取二维数组中有效的频率和系数, 并一维化加快循环速度
            valid_freq_remap = self.inst_freq[valid_mask]
            valid_coeffs = self.transform_result["tf_map"][valid_mask]

            # 获取有效点的二维索引
            valid_indices = np.where(valid_mask)
            time_indices = valid_indices[1]  # 时间索引

            # 向量化计算最近的左侧频率索引
            freq_indices = np.searchsorted(
                self.transform_result["f_Axis"], valid_freq_remap, side="left"
            )

            # 处理不在有效频率轴范围内的点
            freq_indices = np.clip(
                freq_indices, 0, len(self.transform_result["f_Axis"]) - 1
            )

            # 左侧频率索引改为最近邻索引
            for i in range(len(freq_indices)):
                if freq_indices[i] > 0:
                    left_dist = abs(
                        valid_freq_remap[i]
                        - self.transform_result["f_Axis"][freq_indices[i] - 1]
                    )
                    right_dist = abs(
                        valid_freq_remap[i]
                        - self.transform_result["f_Axis"][freq_indices[i]]
                    )
                    if left_dist < right_dist:
                        freq_indices[i] -= 1

            # 初始化SST矩阵
            sst_tf_map = np.zeros_like(self.transform_result["tf_map"], dtype=complex)

            # 一维向量按目标索引向二维矩阵向量化累加操作
            np.add.at(sst_tf_map, (freq_indices, time_indices), valid_coeffs)

        return {
            "f_Axis": self.transform_result["f_Axis"],  # 保持与原始CWT一致的轴
            "t_Axis": self.transform_result["t_Axis"],
            "tf_map": sst_tf_map,
        }

    def energy_reassign_numba(self):
        """
        CWT能量重排算法 - Numba 加速版本
        """
        if self.inst_freq is None:
            self.inst_freq = self.calc_inst_freq()
        # 定义 Numba 加速的能量重排核心函数
        from numba import jit

        @jit(nopython=True)
        def __energy_reassign_numba_core(inst_freq, tf_map, f_axis):
            n_freq, n_time = tf_map.shape
            sst_tf_map = np.zeros_like(tf_map)
            for i in range(n_freq):
                for j in range(n_time):
                    freq_remap = inst_freq[i, j]
                    if not np.isnan(freq_remap):
                        # 找到最近的频率索引
                        freq_idx = 0
                        min_dist = abs(f_axis[0] - freq_remap)
                        for k in range(1, len(f_axis)):
                            dist = abs(f_axis[k] - freq_remap)
                            if dist < min_dist:
                                min_dist = dist
                                freq_idx = k
                        # 累加系数
                        sst_tf_map[freq_idx, j] += tf_map[i, j]
            return sst_tf_map

        # 使用 Numba 加速的核心函数
        sst_tf_map = __energy_reassign_numba_core(
            self.inst_freq,
            self.transform_result["tf_map"],
            self.transform_result["f_Axis"],
        )

        return {
            "f_Axis": self.transform_result["f_Axis"],
            "t_Axis": self.transform_result["t_Axis"],
            "tf_map": sst_tf_map,
        }

    def smooth_2D(self, data2D, smooth_param=None):
        """
        对2D数据进行平滑处理
        """
        # 默认参数
        default_param = {
            "method": "gaussian",
            "sigma": 3,
            "morph_shape": 3,
            "morph_type": "open",
        }
        if smooth_param is not None:
            default_param.update(smooth_param)
        method = default_param["method"]
        if method == "gaussian":
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(data2D, sigma=default_param["sigma"])
        elif method == "morph":
            from skimage.morphology import (
                footprint_rectangle,
                erosion,
                opening,
                closing,
            )

            selem = footprint_rectangle(
                (1, default_param["morph_shape"])
            )  # 创建长条结构元素
            abs_data = np.abs(data2D)
            if default_param["morph_type"] == "erosion":
                filtered = erosion(abs_data, selem)
            elif default_param["morph_type"] == "open":
                filtered = opening(abs_data, selem)
            elif default_param["morph_type"] == "close":
                filtered = closing(abs_data, selem)
            else:
                raise ValueError("morph_type must be 'open' or 'close'")
            return filtered * np.exp(1j * np.angle(data2D))
        else:
            raise ValueError("method must be 'gaussian' or 'morph'")

    def sst(self):
        """
        执行完整的SST变换
        """
        # 能量重排
        if self.sst_result is None:
            # 计算瞬时频率
            if self.inst_freq is None:
                # 执行时频变换
                if self.transform_result is None:
                    self.transform_result = self.transform(
                        self.signal, self.fs, self.transform_param
                    )
                self.inst_freq = self.calc_inst_freq()
            # 执行能量重排
            if self.has_numba:
                try:
                    self.sst_result = self.energy_reassign_numba()
                except Exception as e:  # numba加速失败, 使用普通方法
                    self.sst_result = self.energy_reassign()
            else:
                self.sst_result = self.energy_reassign()
        # 平滑处理
        if self.isSmooth:
            smoothed_tf_map = self.smooth_2D(
                self.sst_result["tf_map"],
                # 根据调频调幅信号模型, 进行线条形态学滤波
                smooth_param={
                    "method": "morph",
                    "morph_shape": self.sst_result["t_Axis"].size // 500,
                    "morph_type": "open",
                },
            )
            self.sst_result["tf_map"] = smoothed_tf_map
        return self.sst_result

    def reconstruct(self, fc=None, fb=None):
        """
        重构信号
        """
        if self.sst_result is None:
            raise ValueError("请先执行sst()方法获取SST结果")
        else:
            f_Axis= self.sst_result["f_Axis"]
            t_Axis = self.sst_result["t_Axis"]
            from ssqueezepy import issq_cwt
            scipy.integrate.trapz = np.trapz
            if fc is None or fb is None:
                recons = issq_cwt(
                    Tx=self.sst_result["tf_map"],
                    wavelet="morlet",
                )
            else:
                if fc.ndim==1 and fb.ndim==1:
                    fc=fc.reshape(1, -1)
                    fb=fb.reshape(1, -1)
                if fc.shape[1]==len(t_Axis) and fb.shape[1]==len(t_Axis):
                    # 根据理想频率范围查找对应索引范围
                    f_low,f_high=fc-fb,fc+fb
                    f_low_idx=np.searchsorted(f_Axis,f_low,side='left')
                    f_high_idx=np.searchsorted(f_Axis,f_high,side='right')
                    f_low_idx = np.clip(f_low_idx, 0, len(f_Axis) - 1)
                    f_high_idx = np.clip(f_high_idx, 0, len(f_Axis) - 1)
                    fc_idx= (f_low_idx+f_high_idx)//2
                    fb_idx= (f_high_idx-f_low_idx)//2
                    # 执行重构
                    recons = issq_cwt(
                        Tx=self.sst_result["tf_map"],
                        wavelet="morlet",
                        cc=fc_idx.T,
                        cw=fb_idx.T,
                    )
                else:
                    raise ValueError("fc和fb的时间轴长度必须与SST结果的时间轴长度相同")
        return recons


def test_sst_cwt():
    """
    测试SST_CWT算法的基本功能
    """
    print("测试SST_CWT算法...")
    
    # 生成测试信号
    fs = 1000
    t = np.linspace(0, 2, 2*fs)
    PI = np.pi
    
    # 调频调幅信号
    IF = (1 + 0.2*np.cos(2*PI*2*t)) * 100
    phase = np.cumsum(2*PI*IF/fs)
    A = (1 + 0.3*np.cos(2*PI*5*t))
    signal = A * np.cos(phase)
    
    # 初始化SST_CWT
    sst = SST_CWT(signal, fs, cwt_param={"scalesType":"linear","scalesNum":500}, gamma=0.1)
    
    # 执行SST变换
    result = sst.sst()
    
    print(f"变换结果维度: {result['tf_map'].shape}")
    print(f"时间轴长度: {len(result['t_Axis'])}")
    print(f"频率轴范围: {result['f_Axis'][0]:.2f} - {result['f_Axis'][-1]:.2f} Hz")
    print("SST_CWT算法测试完成！")
    
    return sst


if __name__ == "__main__":
    # 运行测试
    sst_example = test_sst_cwt()
    
    # 同时显示两幅图，便于对比
    import matplotlib.pyplot as plt
    result_cwt = sst_example.transform_result
    result_sst = sst_example.sst_result

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # CWT图
    mesh1 = axes[0].pcolormesh(
        result_cwt["t_Axis"], result_cwt["f_Axis"], np.abs(result_cwt["tf_map"]),
        cmap="jet", shading="auto", vmin=None, vmax=None
    )
    axes[0].set_title("CWT时频图测试")
    axes[0].set_xlabel("时间/s")
    axes[0].set_ylabel("频率/Hz")
    axes[0].set_ylim(0, 200)
    fig.colorbar(mesh1, ax=axes[0], label="幅值")
    # SST图
    mesh2 = axes[1].pcolormesh(
        result_sst["t_Axis"], result_sst["f_Axis"], np.abs(result_sst["tf_map"]),
        cmap="jet", shading="auto", vmin=None, vmax=None
    )
    axes[1].set_title("SST时频图测试")
    axes[1].set_xlabel("时间/s")
    axes[1].set_ylabel("频率/Hz")
    axes[1].set_ylim(0, 200)
    fig.colorbar(mesh2, ax=axes[1], label="幅值")
    plt.tight_layout()
    plt.show()
