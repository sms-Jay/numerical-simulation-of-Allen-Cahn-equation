%% 完全仿照 MATLAB 参考代码风格的三维等值面绘图（ADMM 相场数据，等值面 u=0.5）
% 生成 2×3 组合大图
clear; close all; clc;

%% 参数设置
data_file = 'H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/data/data_ccfd_admm_parallel_3d.txt';
output_dir = 'figures/3d';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 需要可视化的时间步索引（6 个，对应 2×3 布局）
time_steps_to_plot = [0, 10, 20, 30, 40, 50];
dt = 0.01;   % 时间步长（用于标题显示）

%% 读取三维数据
fprintf('正在读取三维数据文件：%s\n', data_file);
[u_all, x, y, z] = load_3d_data(data_file);
[time_steps, Nx, Ny, Nz] = size(u_all);
fprintf('数据形状: 时间步=%d, 网格=%d×%d×%d\n', time_steps, Nx, Ny, Nz);

N = Nx;
if Nx ~= Ny || Ny ~= Nz
    warning('网格不是立方体，将使用 Nx 构建网格，但绘图可能不正确');
end

%% 生成网格坐标
xi = ((0:(N+1)) - 0.5) / N;
[X, Y, Z] = meshgrid(xi, xi, xi);

%% 创建大图窗（2行3列）
figure('Position', [100, 100, 1500, 1000]);  % 屏幕显示大小（像素），可调整

% 全局颜色范围（所有子图统一）
global_clim = [-1, 9];

% 循环绘制子图
for idx = 1:length(time_steps_to_plot)
    t = time_steps_to_plot(idx) + 1;
    fprintf('正在绘制时间步 t = %d (u=0.5 等值面)...\n', t-1);
    
    u0 = squeeze(u_all(t, :, :, :));
    
    % 边界扩展
    u = u0;
    u(N+2, N+2, N+2) = 0;
    u(2:end-1, 2:end-1, 2:end-1) = u0;
    u(1, :, :)   = u(end-1, :, :);
    u(end, :, :) = u(2, :, :);
    u(:, 1, :)   = u(:, end-1, :);
    u(:, end, :) = u(:, 2, :);
    u(:, :, 1)   = u(:, :, end-1);
    u(:, :, end) = u(:, :, 2);
    
    % 插值函数：绘制 u=0.5 等值面
    F = @(xq, yq, zq) interp3(X, Y, Z, u, xq, yq, zq) - 0.5;
    interval = [0, 1, 0, 1, 0, 1];
    
    % 创建子图（按行优先：第一行 t=0,10,20；第二行 t=30,40,50）
    subplot(2, 3, idx);
    fimplicit3(F, interval, 'EdgeColor', 'none', 'FaceAlpha', 0.6);
    
    % 设置子图属性
    daspect([1,1,1]);
    axis([0,1,0,1,0,1]);
    xticks(0:0.2:1);
    yticks(0:0.2:1);
    zticks(0:0.2:1);
    
    % 光照和材质（每个子图独立设置）
    light("Style", "infinite", "Position", [-1, -1, 1]);
    lighting gouraud;
    material([0.2, 1, 0.5]);
    clim(global_clim);
    colormap bone;   % 全局颜色图，所有子图共享
    
    % 子图标题（数学字体）
    title(sprintf('$t = %.3f$', (t-1)*dt), 'Interpreter', 'latex', 'FontSize', 14);
    
    % 可选：隐藏坐标轴刻度标签（若希望更简洁，可保留）
    % 这里保持原样，显示刻度
end

% 添加总标题
%sgtitle('Dynamical evolution of the phase variable (ADMM, isosurface $u=0.5$)', ...
%    'Interpreter', 'latex', 'FontSize', 16, 'FontWeight', 'bold');

% 调整子图间距（避免过挤）
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperUnits', 'inches', 'PaperSize', [pos(3), pos(4)], ...
    'PaperPosition', [0, 0, pos(3), pos(4)]);

% 保存大图
outname = fullfile(output_dir, 'admm_3d_evolution_3_0.10');
saveas(gcf, [outname, '.fig']);
saveas(gcf, [outname, '.pdf']);
fprintf('已保存大图：%s.fig 和 %s.pdf\n', outname, outname);

close(gcf);
fprintf('所有图形生成完毕！\n');

%% ==================== 子函数：读取三维数据（保持不变） ====================
function [u_all, x, y, z] = load_3d_data(filename)
    fid = fopen(filename, 'r');
    if fid == -1
        error('无法打开文件: %s', filename);
    end
    
    metadata = fscanf(fid, '%d', 4);
    if length(metadata) < 4
        error('文件格式错误：无法读取元数据');
    end
    time_steps = metadata(1);
    Nx = metadata(2);
    Ny = metadata(3);
    Nz = metadata(4);
    L = 2.0;
    
    x_raw = fscanf(fid, '%f', Nx);
    x = x_raw * L / Nx;
    y_raw = fscanf(fid, '%f', Ny);
    y = y_raw * L / Ny;
    z_raw = fscanf(fid, '%f', Nz);
    z = z_raw * L / Nz;
    
    u_all = zeros(time_steps, Nx, Ny, Nz);
    
    for t = 1:time_steps
        tline = fgetl(fid);
        while isempty(strtrim(tline))
            tline = fgetl(fid);
        end
        for ix = 1:Nx
            for iy = 1:Ny
                line = fgetl(fid);
                while isempty(strtrim(line))
                    line = fgetl(fid);
                end
                values = sscanf(line, '%f');
                if length(values) ~= Nz
                    error('数据长度不匹配: t=%d, x=%d, y=%d', t-1, ix-1, iy-1);
                end
                u_all(t, ix, iy, :) = values;
            end
        end
    end
    
    fclose(fid);
end