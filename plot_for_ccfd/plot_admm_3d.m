%% 三维彩色等值面组合图 (2行3列)
% 读取指定时间步的数据，绘制等值面并用颜色表示数值，然后保存组合图
% 需修改 data_file 和 time_steps

clear; clc; close all;

%% 设置参数
data_file = 'H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/data/data_ccfd_admm_parallel_3d.txt';
time_steps = [0, 10, 20, 30, 40, 50];  % 要显示的时间步
isovalue = 0.5;  % 等值面值
output_dir = 'H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/plot/admm_3d_combined';
output_file = fullfile(output_dir, 'isosurface_ep=0.05.png');

% 创建输出目录
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% 预先加载所有时间步的数据，计算全局颜色范围
fprintf('正在加载数据并计算全局颜色范围...\n');
u_all = cell(length(time_steps), 1);
global_min = inf;
global_max = -inf;
for idx = 1:length(time_steps)
    t = time_steps(idx);
    [u, x, y, z] = load_single_time_step(data_file, t);
    u_all{idx} = u;
    global_min = min(global_min, min(u(:)));
    global_max = max(global_max, max(u(:)));
end
fprintf('全局数值范围: [%.6f, %.6f]\n', global_min, global_max);

%% 创建组合图
figure('Position', [100, 100, 1400, 900]);  % 调整窗口大小

for idx = 1:length(time_steps)
    t = time_steps(idx);
    u = u_all{idx};
    
    subplot(2, 3, idx);  % 2行3列
    
    % 提取等值面
    fv = isosurface(x, y, z, u, isovalue);
    
    if isempty(fv.vertices)
        warning('时间步 %d: 等值面值 %f 无数据，请调整 isovalue。', t, isovalue);
        axis off;
        continue;
    end
    
    % 计算每个顶点处的数值（通过插值）
    vert_vals = interp3(x, y, z, u, fv.vertices(:,1), fv.vertices(:,2), fv.vertices(:,3));
    
    % 绘制面，并用顶点数值定义颜色
    p = patch(fv);
    isonormals(x, y, z, u, p);  % 重新计算法向以获得平滑光照
    p.FaceColor = 'interp';      % 插值着色
    p.EdgeColor = 'none';
    p.FaceVertexCData = vert_vals;  % 每个顶点的颜色数据
    p.FaceAlpha = 1;              % 不透明
    
    % 设置统一颜色范围
    caxis([global_min, global_max]);
    colormap jet;  % 每个子图都设置一次，确保一致
    
    % 设置光照
    camlight; 
    lighting gouraud;
    
    % 坐标轴等比例并设置范围
    axis equal;
    xlim([min(x), max(x)]);
    disp(x);
    ylim([min(y), max(y)]);
    zlim([min(z), max(z)]);
    xlabel('x'); ylabel('y'); zlabel('z');
    
    title(sprintf('t = %d', t));
    view(45, 30);  % 统一视角
    grid on;
end

% 添加总颜色条（放在图右侧）
colormap jet;  % 确保颜色映射一致
c = colorbar('Position', [0.92 0.15 0.02 0.7]);
c.Label.String = 'u value';

% 添加总标题
sgtitle(sprintf('Dynamic evolution of the isosurface(isovalue = %.2f)', isovalue), 'FontSize', 14);

%% 保存图片
saveas(gcf, output_file);
fprintf('组合图已保存至: %s\n', output_file);

%% ------------------------------------------------------------------------
% 函数：加载单个时间步的数据（返回数组和坐标）
function [u, x, y, z] = load_single_time_step(filename, t_select)
    fid = fopen(filename, 'r');
    if fid == -1
        error('无法打开文件: %s', filename);
    end
    
    header = fgetl(fid);
    meta = sscanf(header, '%d');
    time_steps = meta(1);
    Nx = meta(2);
    Ny = meta(3);
    Nz = meta(4);
    L = 1.0;
    
    x_line = fgetl(fid);
    x = sscanf(x_line, '%f');
    y_line = fgetl(fid);
    y = sscanf(y_line, '%f');
    z_line = fgetl(fid);
    z = sscanf(z_line, '%f');
    
    found = false;
    for t = 0:time_steps-1
        tline = fgetl(fid);
        if ~ischar(tline)
            error('文件提前结束');
        end
        if t == t_select
            found = true;
            u = zeros(Nx, Ny, Nz);
            for i = 1:Nx
                for j = 1:Ny
                    dataline = fgetl(fid);
                    vals = sscanf(dataline, '%f');
                    if length(vals) ~= Nz
                        error('数据行长度错误');
                    end
                    u(i, j, :) = vals;
                end
            end
            break;
        else
            for i = 1:Nx*Ny
                fgetl(fid);
            end
        end
    end
    fclose(fid);
    
    if ~found
        error('未找到时间步 %d', t_select);
    end
end