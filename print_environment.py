#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
打印当前Python环境及所有已安装库的版本信息
用于记录算法测试环境
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
try:
    import numba
    numba_installed = True
except ImportError:
    numba_installed = False

def print_header(title):
    """打印格式化的标题"""
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)

def print_section(title):
    """打印格式化的分节标题"""
    print("\n" + "-" * 60)
    print(f"{title:^60}")
    print("-" * 60)

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
    """主函数，打印所有环境信息"""
    print_header("算法测试环境说明")
    
    # 系统信息
    print_section("系统信息")
    print(f"操作系统: {platform.platform()}")
    print(f"处理器架构: {platform.machine()}")
    print(f"计算机名称: {platform.node()}")
    
    # Python信息
    print_section("Python信息")
    print(f"Python版本: {platform.python_version()}")
    print(f"Python实现: {platform.python_implementation()}")
    print(f"Python编译器: {platform.python_compiler()}")
    
    # 核心科学计算库
    print_section("核心科学计算库")
    print(f"NumPy版本: {np.__version__}")
    print(f"SciPy版本: {sp.__version__}")
    print(f"Matplotlib版本: {mpl.__version__}")
    
    # 小波变换库
    print_section("小波变换库")
    print(f"PyWavelets版本: {pywt.__version__}")
    
    # 同步压缩变换库
    print_section("同步压缩变换库")
    print(f"ssqueezepy版本: {ssqueezepy.__version__ if hasattr(ssqueezepy, '__version__') else get_package_version('ssqueezepy')}")
    
    # 机器学习库
    print_section("机器学习库")
    print(f"scikit-learn版本: {sklearn.__version__}")
    
    # 图像处理库
    print_section("图像处理库")
    print(f"scikit-image版本: {skimage.__version__}")
    
    # 可选加速库
    print_section("可选加速库")
    if numba_installed:
        print(f"Numba版本: {numba.__version__}")
    else:
        print("Numba未安装")
      # 检查requirements.txt中列出但未在上面检查的包
    print_section("requirements.txt中的其他包")
    req_packages = []
    try:
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    req_packages.append(line)
        
        for package in req_packages:
            if package.lower() not in ['numpy', 'scipy', 'matplotlib', 'pywavelets', 
                                      'ssqueezepy', 'scikit-learn', 'scikit-image', 'numba']:
                print(f"{package}版本: {get_package_version(package)}")
    except Exception as e:
        print(f"读取requirements.txt时出错: {e}")

if __name__ == "__main__":
    main()
