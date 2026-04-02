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
        L = 2.0
        # 读取x坐标
        x_data = np.array([float(x) for x in f.readline().split()])*L/x_size
        
        # 读取y坐标
        y_data = np.array([float(y) for y in f.readline().split()])*L/y_size
        
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

def create_contour_plots_combined(u, x, y, time_indices):
    """
    创建组合等高线图 - 多个时间步在一幅图中
    """
    output_dir = 'results/plot/admm'
    os.makedirs(output_dir, exist_ok=True)
    
    n_plots = len(time_indices)
    
    # 计算子图布局：2行3列（适合6幅图）
    n_rows = 2
    n_cols = 3
    
    # 创建大图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten()  # 展平以便索引
    
    # 创建网格
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # 确定颜色范围（所有图使用相同的颜色范围）
    vmin = np.min(u[time_indices])
    vmax = np.max(u[time_indices])
    
    # 创建每个时间步的等高线图
    for i, t in enumerate(time_indices):
        contour = axes[i].contourf(X, Y, u[t], 20, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'time step = {t}', fontsize=12)
        axes[i].set_xlabel('x', fontsize=10)
        axes[i].set_ylabel('y', fontsize=10)
        axes[i].set_aspect('equal')  # 保持纵横比
    
    # 隐藏多余的子图（如果时间点少于子图数量）
    for j in range(len(time_indices), len(axes)):
        axes[j].set_visible(False)
    
    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label('u value', fontsize=12)
    
    # 添加总标题
    fig.suptitle(f'The dynamical evolution of the phase variable computed by ADMM framework', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # 为颜色条和总标题留出空间
    plt.savefig(os.path.join(output_dir, "admm_evolution_0_to_50.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已生成组合图: admm_evolution_0_to_50.png")

def create_separate_plots(u, x, y, time_indices):
    """
    保留原功能：生成单张图（如果需要）
    """
    output_dir = 'results/plot/admm'
    os.makedirs(output_dir, exist_ok=True)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    for time in time_indices:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        contour = ax.contourf(X, Y, u[time], 20, cmap='viridis')
        ax.set_title(f't = {time}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(contour, ax=ax)
        # plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"admm_t_{time}.png"), dpi=300)
        # plt.close()
        print(f"已生成单张图: admm_t_{time}.png")

def main():
    """主函数"""
    # 加载数据
    data_file = "H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/data/data_ccfd_admm_adaptive.txt"
    if not os.path.exists(data_file):
        print(f"数据文件 {data_file} 不存在")
        print("请先运行C++程序生成数据")
        return
    
    u, x, y = load_data_from_file(data_file)
    print(f"数据形状: u({u.shape}), x({x.shape}), y({y.shape})")
    
    # 选择要可视化的时间点：0,10,20,30,40,50
    time_indices = [0, 30, 60, 90, 120, 139]
    
    # 生成组合图（6幅图一起展现）
    create_contour_plots_combined(u, x, y, time_indices)
    
    # 如果需要同时保留单张图，取消下面的注释
    # create_separate_plots(u, x, y, time_indices)

if __name__ == "__main__":
    main()