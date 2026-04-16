%% 纯 MATLAB 风格绘图（无虚线、无网格、无额外线）
clear; close all; clc;

dt = 0.001;
%% 创建输出目录
output_root = pwd;  % 当前工作目录
figures_dir = fullfile(output_root, 'figures/curve');
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end

%% 辅助读取函数（与 importdata 类似）
function data = read_data(filename, min_cols)
    data = [];
    if ~isfile(filename)
        fprintf('文件不存在: %s\n', filename);
        return;
    end
    tmp = importdata(filename);
    if isstruct(tmp)
        data = tmp.data;
        if isempty(data) && isfield(tmp, 'textdata')
            fprintf('警告: %s 包含文本头，可能无法正确读取\n', filename);
            return;
        end
    else
        data = tmp;
    end
    if isempty(data)
        fprintf('无有效数据: %s\n', filename);
        return;
    end
    if nargin > 1 && size(data,2) < min_cols
        fprintf('列数不足: %s (需要 %d 列)\n', filename, min_cols);
        data = [];
        return;
    end
    fprintf('读取成功: %s, 大小 %s\n', filename, mat2str(size(data)));
end

%% 1. 能量曲线
fprintf('\n=== 能量曲线 ===\n');
d = read_data('H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/data/energy_data_3d.txt', 1);
if ~isempty(d)
    if size(d,2) >= 2
        t = d(1:end,1); E = d(1:end,2);
    else
        t = (1:length(d))'; E = log (d(:,1));
    end
    t = dt * t;
    figure(1);
    plot(t, E, 'o', 'LineWidth', 1);
    xlabel('Time $t$', 'Interpreter', 'latex');
    ylabel('Energy', 'Interpreter', 'latex');
    set(gca, 'FontSize', 16);
    box on;
    saveas(gcf, 'figures/curve/energy_curve_3.fig');
    set(gcf, 'PaperSize', [12,9]);
    set(gcf, 'PaperPosition', [0, 0, 12, 9]);
    saveas(gcf, 'figures/curve/energy_curve_3.pdf');
    fprintf('✓ energy_curve\n');
end

%% 2. ADMM 残差收敛（对数纵轴）
fprintf('\n=== ADMM 残差收敛 ===\n');
d = read_data('H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/data/admm_residual_data_3d.txt', 2);
if ~isempty(d)
    iter = d(:,1); res = max(d(:,2), eps);
    figure(2);
    semilogy(iter, res, 'o', 'LineWidth', 1);
    xlabel('ADMM Iteration', 'Interpreter', 'latex');
    ylabel('Primal residual (log scale)', 'Interpreter', 'latex');
    set(gca, 'FontSize', 16);
    box on;
    set(gcf, 'PaperSize', [12, 10]);
    set(gcf, 'PaperPosition', [0, 0, 12, 10]);
    saveas(gcf, 'figures/curve/admm_convergence_3.fig');
    saveas(gcf, 'figures/curve/admm_convergence_3.pdf');
    fprintf('✓ admm_convergence\n');
end

%% 3. 上界违反： max(u)
fprintf('\n=== 上界违反 (1 - max(u)) ===\n');
d = read_data('H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/data/bounds_data_3d.txt', 3);
if ~isempty(d)
    t = d(:,1); 
    t = dt * t;
    max_u = d(:,3);
    y_vals = 1 - max_u;
    figure(3);
    semilogy(t, y_vals, 'o', 'LineWidth', 1);
    xlabel('Time $t$', 'Interpreter', 'latex');
    ylabel('$1-\max(u)$', 'Interpreter', 'latex');
    axis([-inf, inf, 1e-2, 1]);
    set(gca, 'FontSize', 16);
    box on;
    set(gcf, 'PaperSize', [12, 9]);
    set(gcf, 'PaperPosition', [0, 0, 12, 9]);
    saveas(gcf, 'figures/curve/bounds_upper_3.fig');
    saveas(gcf, 'figures/curve/bounds_upper_3.pdf');
    fprintf('✓ bounds_upper\n');
end

%% 4. 下界违反：min(u)
fprintf('\n=== 下界违反 (min(u)) ===\n');
d = read_data('H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/data/bounds_data_3d.txt', 3);
if ~isempty(d)
    t = d(:,1); 
    t = dt * t;
    min_u = d(:,2);
    y_vals = min_u;
    figure(4);
    semilogy(t, y_vals, 'o', 'LineWidth', 1);
    xlabel('Time $t$', 'Interpreter', 'latex');
    ylabel('$\min(u)$', 'Interpreter', 'latex');
    axis([-inf, inf, 1e-2, 1]);
    set(gca, 'FontSize', 16);
    box on;
    set(gcf, 'PaperSize', [12, 9]);
    set(gcf, 'PaperPosition', [0, 0, 12, 9]);
    saveas(gcf, 'figures/curve/bounds_lower_3.fig');
    saveas(gcf, 'figures/curve/bounds_lower_3.pdf');
    fprintf('✓ bounds_lower\n');
end

%% 5. ADMM 迭代次数
fprintf('\n=== ADMM 迭代次数 ===\n');
d = read_data('H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/data/admm_iterations_3d.txt', 1);
if ~isempty(d)
    if size(d,2) >= 2
        t = d(:,1); iters = d(:,2);
    else
        t = (1:length(d))'; iters = d(:,1);
    end
    t = dt * t;
    figure(5);
    plot(t, iters, 'o', 'LineWidth', 1);
    xlabel('Time $t$', 'Interpreter', 'latex');
    ylabel('Number of ADMM iterations', 'Interpreter', 'latex');
    set(gca, 'FontSize', 16);
    box on;
    set(gcf, 'PaperSize', [12, 9]);
    set(gcf, 'PaperPosition', [0, 0, 12, 9]);
    saveas(gcf, 'figures/curve/admm_iterations_3.fig');
    saveas(gcf, 'figures/curve/admm_iterations_3.pdf');
    fprintf('✓ admm_iterations\n');
end

fprintf('\n所有图形生成完毕，保存在 figures/ 目录\n');