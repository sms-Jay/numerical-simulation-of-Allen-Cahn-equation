# -*- coding: utf8 -*-
import numpy as np
import plotly.graph_objects as go
import os

def load_3d_data(filename):
    """
    从三维数据文件加载数据，返回 (u, x, y, z)
    文件格式：
        第一行：time_steps Nx Ny Nz
        第二行：x坐标（Nx个值）
        第三行：y坐标（Ny个值）
        第四行：z坐标（Nz个值）
        之后对于每个时间步 t：
            一行 "t=<t>"
            然后 Nx * Ny 行数据，每行 Nz 个值，按先 y 后 x 的顺序排列
    """
    with open(filename, 'r') as f:
        # 元数据：time_steps Nx Ny Nz
        metadata = f.readline().split()
        time_steps = int(metadata[0])
        Nx = int(metadata[1])
        Ny = int(metadata[2])
        Nz = int(metadata[3])
        L = 2.0  # 物理域长度

        # 读取坐标（索引已缩放为物理坐标）
        x = np.array([float(v) for v in f.readline().split()]) * L / Nx
        y = np.array([float(v) for v in f.readline().split()]) * L / Ny
        z = np.array([float(v) for v in f.readline().split()]) * L / Nz

        # 初始化 u 数组
        u = np.zeros((time_steps, Nx, Ny, Nz))

        # 读取数据
        current_t = -1
        current_x = 0
        current_y = 0
        for line in f:
            line = line.strip()
            if line.startswith('t='):
                current_t = int(line[2:])
                current_x = 0
                current_y = 0
            elif line:
                values = [float(v) for v in line.split()]
                # 每行应包含 Nz 个值，对应固定 (x, y) 的所有 z
                u[current_t, current_x, current_y, :] = values
                current_y += 1
                if current_y == Ny:
                    current_y = 0
                    current_x += 1

    return u, x, y, z

def plot_volume_plotly(u, x, y, z, time_indices, opacity_scale=0.2, surface_count=20):
    """
    对每个时间步绘制体渲染图，保存为 HTML 文件。
    参数:
        u: 4D数组 (时间, Nx, Ny, Nz)
        x, y, z: 坐标数组
        time_indices: 要绘制的时间步列表
        opacity_scale: 透明度缩放因子（值越大越不透明）
        surface_count: 等值面数量，越多细节越丰富但渲染越慢
    """
    output_dir = 'results/plot/admm_3d_volume'
    os.makedirs(output_dir, exist_ok=True)

    for t in time_indices:
        values = u[t]  # shape (Nx, Ny, Nz)

        # 归一化数据到 [0,1] 以便颜色映射（可选，但有助于透明度调节）
        vmin, vmax = values.min(), values.max()
        if vmax - vmin > 1e-12:
            norm_values = (values - vmin) / (vmax - vmin)
        else:
            norm_values = values

        fig = go.Figure(data=go.Volume(
            x=x,
            y=y,
            z=z,
            value=values.flatten(),          # 原始数值用于颜色映射
            isomin=vmin,
            isomax=vmax,
            opacity=opacity_scale,            # 基础透明度
            surface_count=surface_count,      # 生成多少个等值面，越多越平滑
            colorscale='viridis',              # 颜色映射
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=True,                    # 显示颜色条
            # 可选的透明度映射：根据数值调整透明度，使低值更透明
            # opacityscale=[[0, 0], [0.5, 0.2], [1, 0.8]],
        ))

        fig.update_layout(
            title=f't = {t} 体渲染 (数据范围 [{vmin:.3f}, {vmax:.3f}])',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z',
                aspectmode='data'               # 保持真实比例
            ),
            width=900,
            height=900,
        )

        filepath = os.path.join(output_dir, f'volume_t_{t}.html')
        fig.write_html(filepath)
        print(f'已生成 t={t} 的体渲染 HTML 文件: {filepath}')

def main():
    # 请根据实际路径修改
    data_file = "H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/data/data_ccfd_admm_parallel_3d.txt"
    if not os.path.exists(data_file):
        print(f"数据文件 {data_file} 不存在，请先运行三维求解器。")
        return

    u, x, y, z = load_3d_data(data_file)
    print(f"数据形状: u {u.shape}, x {x.shape}, y {y.shape}, z {z.shape}")

    # 选择需要可视化的时间步（例如前 3 步，可根据需要修改）
    time_indices = list(range(min(3, u.shape[0])))
    plot_volume_plotly(u, x, y, z, time_indices, opacity_scale=0.2, surface_count=30)

if __name__ == "__main__":
    main()