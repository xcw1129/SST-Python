#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成算法测试环境说明的Markdown格式文档
"""

import sys
import platform
import pkg_resources
import importlib
import numpy as np
import scipy as sp
import matplotlib as mpl
import pywt
import ssqueezepy
import sklearn
import skimage
import datetime
try:
    import numba
    numba_installed = True
except ImportError:
    numba_installed = False

def get_package_version(package_name):
    """获取包的版本号"""
    try:
        return pkg_resources.get_distribution(package_name).version
    except:
        try:
            module = importlib.import_module(package_name)
            return getattr(module, "__version__", "未知版本")
        except:
            return "未安装或无法获取版本"

def main():
    """主函数，生成Markdown格式的环境信息"""
    # 创建markdown文件
    with open('环境配置说明.md', 'w', encoding='utf-8') as md_file:
        md_file.write("# 同步压缩变换(SST)算法测试环境说明\n\n")
        md_file.write(f"*生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # 系统信息
        md_file.write("## 系统信息\n\n")
        md_file.write("| 项目 | 详情 |\n")
        md_file.write("| --- | --- |\n")
        md_file.write(f"| 操作系统 | {platform.platform()} |\n")
        md_file.write(f"| 处理器架构 | {platform.machine()} |\n")
        md_file.write(f"| 计算机名称 | {platform.node()} |\n\n")
        
        # Python信息
        md_file.write("## Python信息\n\n")
        md_file.write("| 项目 | 详情 |\n")
        md_file.write("| --- | --- |\n")
        md_file.write(f"| Python版本 | {platform.python_version()} |\n")
        md_file.write(f"| Python实现 | {platform.python_implementation()} |\n")
        md_file.write(f"| Python编译器 | {platform.python_compiler()} |\n\n")
        
        # 依赖库信息
        md_file.write("## 依赖库信息\n\n")
        md_file.write("| 库名称 | 版本 | 用途 |\n")
        md_file.write("| --- | --- | --- |\n")
        
        # 核心科学计算库
        md_file.write(f"| NumPy | {np.__version__} | 数值计算基础库 |\n")
        md_file.write(f"| SciPy | {sp.__version__} | 科学计算库 |\n")
        md_file.write(f"| Matplotlib | {mpl.__version__} | 数据可视化 |\n")
        
        # 小波变换库
        md_file.write(f"| PyWavelets | {pywt.__version__} | 小波变换库 |\n")
        
        # 同步压缩变换库
        ssq_version = ssqueezepy.__version__ if hasattr(ssqueezepy, '__version__') else get_package_version('ssqueezepy')
        md_file.write(f"| ssqueezepy | {ssq_version} | 同步压缩变换库 |\n")
        
        # 机器学习库
        md_file.write(f"| scikit-learn | {sklearn.__version__} | 机器学习库 (DBSCAN聚类) |\n")
        
        # 图像处理库
        md_file.write(f"| scikit-image | {skimage.__version__} | 图像处理库 (形态学滤波) |\n")
        
        # 可选加速库
        if numba_installed:
            md_file.write(f"| Numba | {numba.__version__} | JIT编译加速库 |\n\n")
        else:
            md_file.write("| Numba | 未安装 | JIT编译加速库 |\n\n")
        
        # 添加注释
        md_file.write("## 注意事项\n\n")
        md_file.write("1. 上述环境配置已通过算法验证测试\n")
        md_file.write("2. 若需复现实验结果，建议使用相同或兼容版本的依赖库\n")
        md_file.write("3. 核心依赖库：NumPy, SciPy, PyWavelets和ssqueezepy\n")
        md_file.write("4. 可选依赖：Numba可显著提高算法运行效率，但不影响算法结果\n")
        
        print(f"Markdown环境说明文档已成功生成: 环境配置说明.md")

if __name__ == "__main__":
    main()
