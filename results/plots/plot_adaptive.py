
import numpy as np
import matplotlib.pyplot as plt
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['figure.dpi'] = 800
plt.rcParams['savefig.dpi'] = 800
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

plt.rcParams['axes.linewidth'] = 1.5  # 坐标轴线宽
plt.rcParams['grid.linewidth'] = 1.0  # 网格线宽
plt.rcParams['lines.linewidth'] = 2.0  # 线条宽度
plt.rcParams['patch.linewidth'] = 1.0  # 图形边框宽度
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.edgecolor'] = 'black'  # 图例边框颜色
plt.rcParams['legend.framealpha'] = 1.0  # 图例不透明

# Create output directory
os.makedirs('figures', exist_ok=True)

def safe_read_data(filename, expected_cols=None):
    """Safely read data file"""
    try:
        if not os.path.exists(filename):
            print(f'File not found: {filename}')
            return None
        
        data = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        values = list(map(float, line.split()))
                        if values:
                            data.append(values)
                    except:
                        continue
        
        if not data:
            print(f'No valid data in {filename}')
            return None
        
        data = np.array(data)
        print(f'Successfully read {filename}: shape = {data.shape}')
        return data
    except Exception as e:
        print(f'Error reading {filename}: {e}')
        return None

# Color palette
colors = {
    'energy': '#2E86AB',
    'residual': '#A23B72',
    'min': '#F18F01',
    'max': '#C73E1D',
    'bounds': '#6C757D',
    'admm': '#3B1F2B',
    'convergence': '#287D7D'  # 添加缺失的键
}

# 1. Energy decay curve
print('\n' + '='*50)
print('Generating energy decay curve...')
print('='*50)

try:
    energy_data = safe_read_data('energy_data_adaptive.txt')
    if energy_data is not None:
        if energy_data.shape[1] >= 2:
            time_steps = energy_data[:, 0]
            energy = energy_data[:, 1]
        else:
            time_steps = np.arange(len(energy_data))
            energy = energy_data[:, 0]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot with transparency
        ax.scatter(time_steps, energy, c=colors['energy'], s=60, alpha=0.7, 
                  edgecolors='white', linewidth=0.5, label='Energy')
        
        
        
        ax.set_xlabel('Time', fontweight='bold')
        ax.set_ylabel('Energy', fontweight='bold')
        ax.set_title('Energy', fontweight='bold', pad=15)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set scientific notation for y-axis if values are small
        if np.max(energy) < 1e-3 or np.min(energy) < 1e-3:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        plt.savefig('figures/energy_curve_adaptive.png')
        plt.close()
        print('✓ Energy decay curve saved')
        
    else:
        print('✗ No energy data available')
except Exception as e:
    print(f'✗ Error generating energy curve: {e}')

# 2. ADMM Convergence curve 
print('\n' + '='*50)
print('Generating ADMM convergence curve...')
print('='*50)

try:
    # Read ADMM residual data
    admm_residual_data = safe_read_data('admm_residual_data_adaptive.txt')
    
    if admm_residual_data is not None and admm_residual_data.shape[1] >= 2:
        iterations = admm_residual_data[:, 0]  # 迭代次数
        residuals = admm_residual_data[:, 1]    # 原始残差
        
        # Ensure positive residuals for log scale
        residuals = np.maximum(residuals, 1e-16)
        
        # 创建单个图 - 残差收敛（对数坐标）
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # 散点图
        ax.scatter(iterations, residuals, c=colors['convergence'], s=60, alpha=0.7,
                  edgecolors='white', linewidth=0.5, label='ADMM residual', marker='o')
        
        
        
        # 设置对数坐标
        ax.set_yscale('log')
        
        # 添加容差线
        tolerance = 1e-8
        ax.axhline(y=tolerance, color='gray', linestyle=':', linewidth=1.5,
                   alpha=0.7, label=f'Tolerance = {tolerance:.0e}')
        
        ax.set_xlabel('ADMM Iteration', fontweight='bold')
        ax.set_ylabel('Primal residual (log scale)', fontweight='bold')
        ax.set_title(f'Primal residual', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig('figures/admm_convergence_adaptive.png')
        plt.close()
        print('✓ ADMM convergence curve saved')
        
            
    else:
        print('✗ No ADMM residual data available or incorrect format')
        print('  Expected file: admm_residual_data.txt with columns: iteration residual')
        
except Exception as e:
    print(f'✗ Error generating ADMM convergence curve: {e}')
    import traceback
    traceback.print_exc()


# 3. Bounds preservation curve
print('\n' + '='*50)
print('Generating bounds preservation curve...')
print('='*50)

try:
    bounds_data = safe_read_data('bounds_data_adaptive.txt')
    if bounds_data is not None and bounds_data.shape[1] >= 3:
        time_steps = bounds_data[:, 0]
        min_vals = bounds_data[:, 1]
        max_vals = bounds_data[:, 2]
        
        fig, ax1 = plt.subplots(figsize=(9, 6))
        
        # Main plot: bounds
        ax1.scatter(time_steps, min_vals, c=colors['min'], s=50, alpha=0.7,
                   edgecolors='white', linewidth=0.5, label='Minimum value', marker='v')
        ax1.scatter(time_steps, max_vals, c=colors['max'], s=50, alpha=0.7,
                   edgecolors='white', linewidth=0.5, label='Maximum value', marker='^')
        
        # Add bounds lines
        ax1.axhline(y=0, color=colors['bounds'], linestyle='--', linewidth=1.5, 
                   alpha=0.7, label='Lower bound (0)')
        ax1.axhline(y=1, color=colors['bounds'], linestyle='--', linewidth=1.5, 
                   alpha=0.7, label='Upper bound (1)')
        
        # Shade feasible region
        ax1.fill_between(time_steps, 0, 1, alpha=0.1, color=colors['bounds'])
        
        ax1.set_xlabel('Time', fontweight='bold')
        ax1.set_ylabel('max(u) and min(u)', fontweight='bold')
        ax1.set_title('Bound preservation', fontweight='bold', pad=15)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        
        
        plt.tight_layout()
        plt.savefig('figures/bounds_preservation_adaptive.png')
        plt.close()
        print('✓ Bounds preservation curve saved')
        
    else:
        print('✗ No bounds data available')
except Exception as e:
    print(f'✗ Error generating bounds curve: {e}')

# 4. ADMM iterations
print('\n' + '='*50)
print('Generating ADMM iterations plot...')
print('='*50)

try:
    admm_data = safe_read_data('admm_iterations_adaptive.txt')
    if admm_data is not None:
        if admm_data.shape[1] >= 2:
            time_steps = admm_data[:, 0]
            iterations = admm_data[:, 1]
        else:
            time_steps = np.arange(len(admm_data))
            iterations = admm_data[:, 0]
        
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Scatter plot with different sizes based on iteration count
        scatter = ax.scatter(time_steps, iterations, c=colors['admm'], s=60, 
                           cmap='viridis', alpha=0.8, edgecolors='white', 
                           linewidth=0.5, vmin=0, vmax=np.max(iterations))
        
        
        # Add mean line
        mean_iter = np.mean(iterations)
        ax.axhline(y=mean_iter, color='red', linestyle='--', linewidth=1.5,
                  alpha=0.7, label=f'Mean: {mean_iter:.1f}')
        
        ax.set_xlabel('Time', fontweight='bold')
        ax.set_ylabel('Number of ADMM iterations', fontweight='bold')
        ax.set_title('Number of iterations', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Set integer y-ticks
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.savefig('figures/admm_iterations_adaptive.png')
        plt.close()
        print('✓ ADMM iterations plot saved')
        
    else:
        print('✗ No ADMM data available')
except Exception as e:
    print(f'✗ Error generating ADMM plot: {e}')
