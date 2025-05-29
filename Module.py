# PYTHON基础库
import math  # 数学计算库

# 数值计算库
import numpy as np
from numpy import random  # 随机数库

# 绘图库
from matplotlib import pyplot as plt
from matplotlib import font_manager
plt.rcParams["font.family"] = "sans-serif"  # 默认字体类型
plt.rcParams["font.sans-serif"] = ["simhei"]  # 默认字体设置为黑体
plt.rcParams["font.size"] = 16  # 设置全局字体大小
plt.rcParams["axes.unicode_minus"] = False  # 设置正常显示符号
zh_font = font_manager.FontProperties(family="SimHei")
en_font = font_manager.FontProperties(family="Times New Roman")

# 表格处理库
import pandas as pd

# 科学计算库
import scipy
from scipy import stats  # 统计库
from scipy import signal  # 信号处理库
from scipy import fft  # 傅里叶变换库

PI = np.pi
FLOAT_EPS = np.finfo(float).eps  # 浮点数精度
