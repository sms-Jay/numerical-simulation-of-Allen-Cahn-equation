# -*- coding: utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def load_data_from_file(filename):
    """从文件加载数据"""
    with open(filename, 'r') as f:
        # 读取元数据
        metadata = f.readline().split()
        time_steps = int(metadata[0])
        x_size = int(metadata[1])
        y_size = int(metadata[2])
        
        # 读取x坐标
        x_data = np.array([float(x) for x in f.readline().split()])/50-1.0
        
        # 读取y坐标
        y_data = np.array([float(y) for y in f.readline().split()])/50-1.0
        
        # 初始化u数组
        u_data = np.zeros((time_steps, x_size, y_size))
        
        # 读取u数据
        current_t = -1
        current_x = 0
        
        for line in f:
            line = line.strip()
            if line.startswith('t='):
                # 新时间步
                current_t = int(line[2:])
                current_x = 0
            elif line:
                # 数据行
                values = [float(v) for v in line.split()]
                u_data[current_t, current_x, :] = values
                current_x += 1
    
    return u_data, x_data, y_data

def create_contour_plots(u, x, y, time_indices=None,time=1):
    
    output_dir = 'plot_results/truncated_admm'
    os.makedirs(output_dir, exist_ok=True)
    
    """创建等高线图"""
    if time_indices is None:
        time_indices = range(min(5, u.shape[0]))  # 默认显示前5个时间步
    
    n_plots = len(time_indices)
    
    # 创建子图
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    # 创建网格
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # 确定颜色范围（所有图使用相同的颜色范围）
    vmin = np.min(u)
    vmax = np.max(u)
    
    # 创建每个时间步的等高线图
    for i, t in enumerate(time_indices):
        contour = axes[i].contourf(X, Y, u[t], 20, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].set_title(f't = {t}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        
        # 添加颜色条
        plt.colorbar(contour, ax=axes[i])
    
    plt.tight_layout()
    # plt.savefig(f"admm_t_{time}.png", dpi=1000)
    plt.savefig(os.path.join(output_dir, f"truncated_admm_t_{time}.png"), dpi=1000)
    

def main():
    """主函数"""
    # 加载数据
    data_file = "H:/undergraduate/scientific_research/allen_cahn_equation_simulation/serial/data_truncated_admm.txt"
    if not os.path.exists(data_file):
        print(f"数据文件 {data_file} 不存在")
        print("请先运行C++程序生成数据")
        return
    
    u, x, y = load_data_from_file(data_file)
    print(f"数据形状: u({u.shape}), x({x.shape}), y({y.shape})")
    
    # 选择要可视化的时间点
    for time in [0,10,20,30,40,50]:

        time_indices = [time]
        create_contour_plots(u, x, y, time_indices,time)

if __name__ == "__main__":
    main()