%% 绘制 ADMM 求解 Allen-Cahn 方程的二维相变演化（纯 MATLAB 风格，2行3列大图）
clear; close all; clc;

dt = 0.1;
%% 参数设置
data_file = 'H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/data/data_ccfd_admm_parallel.txt';
output_dir = 'figures/2d';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 需要可视化的时间步索引（与 Python 一致）
time_indices = [0, 20, 40, 60, 80, 100];

%% 读取数据
fprintf('正在读取数据文件：%s\n', data_file);
[u, x, y] = load_admm_data(data_file);
fprintf('数据加载完成：u 大小 = %s, x 长度 = %d, y 长度 = %d\n', ...
    mat2str(size(u)), length(x), length(y));

%% 创建大图窗（2行3列）
figure('Position', [100, 100, 1200, 800]);  % 屏幕显示大小（像素），可根据需要调整

% 全局颜色范围（所有子图统一）
global_clim = [-1, 1];

% 循环绘制子图
for k = 1:length(time_indices)
    t_idx = time_indices(k);
    fprintf('正在绘制时间步 t = %d ...\n', t_idx);
    
    % 提取当前时间步的二维解
    sol = squeeze(u(t_idx+1, :, :));
    
    % 边界扩展（与原 MATLAB 代码完全一致）
    sol_ext = zeros(size(sol,1)+2, size(sol,2)+2);
    sol_ext(2:end-1, 2:end-1) = sol;
    sol_ext(end, :)   = sol_ext(2, :);
    sol_ext(1, :)     = sol_ext(end-1, :);
    sol_ext(:, end)   = sol_ext(:, 2);
    sol_ext(:, 1)     = sol_ext(:, end-1);
    
    % 生成网格坐标（范围 [-1.6,1.6] 对应 128 网格）
    n = size(sol, 1);
    [X, Y] = meshgrid(((-1:n)/n - 0.5) * 3.2, ((-1:n)/n - 0.5) * 3.2);
    
    % 创建子图（2行3列，按行优先）
    subplot(2, 3, k);
    h = pcolor(X, Y, sol_ext);
    set(h, 'LineStyle', 'none');
    shading interp;
    colormap bone;               % 全局 colormap，作用于整个大图
    clim(global_clim);
    
    % 坐标轴设置（每个子图独立）
    axis([-1.6, 1.6, -1.6, 1.6]);
    daspect([1, 1, 1]);
    box off;
    axis off;
    
    % 添加子图标题（时间信息）
    title(sprintf('$t = %.3f$', t_idx * dt), 'Interpreter', 'latex', 'FontSize', 12);
end

% 可选：添加总标题
% sgtitle('Dynamical evolution of the phase variable (ADMM)', ...
%    'Interpreter', 'latex', 'FontSize', 16, 'FontWeight', 'bold');

% 调整子图间距（可选）
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperUnits', 'inches', 'PaperSize', [pos(3), pos(4)], ...
    'PaperPosition', [0, 0, pos(3), pos(4)]);

% 保存大图（PDF 和 FIG）
outname = fullfile(output_dir, 'admm_evolution_2_5.0');
saveas(gcf, [outname, '.fig']);
saveas(gcf, [outname, '.pdf']);
fprintf('已保存大图：%s.fig 和 %s.pdf\n', outname, outname);

close(gcf);
fprintf('所有图形生成完毕！\n');

%% ==================== 子函数：读取自定义格式数据（保持不变） ====================
function [u, x_coords, y_coords] = load_admm_data(filename)
    fid = fopen(filename, 'r');
    if fid == -1
        error('无法打开文件：%s', filename);
    end
    
    % 第一行：元数据
    metadata = fgetl(fid);
    if ~ischar(metadata)
        error('文件为空或格式错误');
    end
    meta = sscanf(metadata, '%d');
    if length(meta) < 3
        error('元数据格式错误，应包含 time_steps, x_size, y_size');
    end
    time_steps = meta(1);
    x_size = meta(2);
    y_size = meta(3);
    L = 2.0;
    
    % 第二行：x 坐标
    x_line = fgetl(fid);
    x_raw = sscanf(x_line, '%f');
    if length(x_raw) ~= x_size
        error('x 坐标数量不匹配：应为 %d，实际 %d', x_size, length(x_raw));
    end
    x_coords = x_raw * L / x_size;
    
    % 第三行：y 坐标
    y_line = fgetl(fid);
    y_raw = sscanf(y_line, '%f');
    if length(y_raw) ~= y_size
        error('y 坐标数量不匹配：应为 %d，实际 %d', y_size, length(y_raw));
    end
    y_coords = y_raw * L / y_size;
    
    % 初始化 u
    u = zeros(time_steps, x_size, y_size);
    
    current_t = -1;
    current_x = 0;
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if isempty(line)
            continue;
        end
        if startsWith(line, 't=')
            tokens = split(line, '=');
            current_t = str2double(tokens{2}) + 1;
            if current_t > time_steps
                warning('时间步索引超出范围，忽略多余数据');
                break;
            end
            current_x = 0;
        else
            if current_t == -1
                error('在第一个时间标记之前出现数据行');
            end
            values = sscanf(line, '%f');
            if length(values) ~= y_size
                error('时间步 %d，行 %d 的数据长度不正确（应为 %d）', current_t-1, current_x+1, y_size);
            end
            u(current_t, current_x+1, :) = values;
            current_x = current_x + 1;
        end
    end
    
    fclose(fid);
    
    if any(u(:) == 0) && (current_t < time_steps)
        warning('数据可能不完整，仅读取了 %d 个时间步（共 %d）', current_t, time_steps);
        u = u(1:current_t, :, :);
    end
end