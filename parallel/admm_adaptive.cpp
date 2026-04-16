// admm_plot.cpp – 时间自适应版本
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <omp.h>
#include <complex>
#include "fft_2d_solver.h"
#include <string>
#include <sstream>

using namespace std;

const double PI = 3.1415926535897932384626;
const double EPS = 1e-24;

// 初始条件函数（保持不变）
double ini_1(double x, double y){
    return (1+sin(2*PI*x)*sin(2*PI*y))/3.0;
}
double ini_2(double x, double y){
    if(abs(x-1)<=0.35 && abs(y-1)<=0.35) return 1e-5;
    else return 1-1e-5;
}
double ini_3(double x, double y){
    return 0.20 + 0.60 * ((double)rand() / (RAND_MAX + 1.0));
}
double ini_4(double x, double y){ return 0.5; }

double F(double t, double theta){
    t = max(EPS, min(1.0 - EPS, t));
    return t*log(t) + (1-t)*log(1-t) + theta*t*(1-t);
}

double safe_log_ratio(double u) {
    u = max(EPS, min(1.0 - EPS, u));
    return log(u / (1 - u));
}

// StaggeredGrid 类（与原始代码完全相同，省略以保持简洁，实际使用时需完整保留）
class StaggeredGrid {
private:
    int Nx, Ny;
    double dx, dy, dx2, dy2, dxdy;
    
public:
    StaggeredGrid(int grid_x, int grid_y, double Lx, double Ly) 
        : Nx(grid_x), Ny(grid_y) {
        dx = Lx / Nx;
        dy = Ly / Ny;
        dx2 = dx * dx;
        dy2 = dy * dy;
        dxdy = dx * dy;
    }
    
    // index function
    int idx_center(int i, int j) const { return i * Ny + j; }
    int idx_grad_x(int i, int j) const { return i * Ny + j; }      // i: 0..Nx
    int idx_grad_y(int i, int j) const { return i * (Ny+1) + j; }  // j: 0..Ny
    
    // Gradient
    void compute_gradient(const vector<double>& u_center,
                         vector<double>& grad_x,
                         vector<double>& grad_y) const {
        // clear
        fill(grad_x.begin(), grad_x.end(), 0.0);
        fill(grad_y.begin(), grad_y.end(), 0.0);
        
        // Gradient x
        #pragma omp parallel for collapse(2)
        for (int i = 0; i <= Nx; i++) {  
            for (int j = 0; j < Ny; j++) {
                int idx_gx = idx_grad_x(i, j);

                int i_right = i % Nx;              
                int i_left = (i - 1 + Nx) % Nx;   
            
                int idx_c_right = idx_center(i_right, j);
                int idx_c_left = idx_center(i_left, j);
            
                grad_x[idx_gx] = (u_center[idx_c_right] - u_center[idx_c_left]) / dx;
            }
        }
    
        // Gradient y
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j <= Ny; j++) {  
                int idx_gy = idx_grad_y(i, j);

                int j_up = j % Ny;              
                int j_down = (j - 1 + Ny) % Ny; 
            
                int idx_c_up = idx_center(i, j_up);
                int idx_c_down = idx_center(i, j_down);
            
                grad_y[idx_gy] = (u_center[idx_c_up] - u_center[idx_c_down]) / dy;
            }
        }
    }
    
    // Divergence
    vector<double> compute_divergence(const vector<double>& grad_x,
                                     const vector<double>& grad_y) const {
        vector<double> div(Nx * Ny, 0.0);
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                int idx_c = idx_center(i, j);
                
                // D_x u

                int idx_gx_right = idx_grad_x(i+1, j);   
                int idx_gx_left = idx_grad_x(i, j);      
                double div_x = (grad_x[idx_gx_right] - grad_x[idx_gx_left]) / dx;
                
                // D_y u

                int idx_gy_up = idx_grad_y(i, j+1);     
                int idx_gy_down = idx_grad_y(i, j);      
                double div_y = (grad_y[idx_gy_up] - grad_y[idx_gy_down]) / dy;
                
                div[idx_c] = div_x + div_y;
            }
        }
        
        return div;
    }
    
    // Laplace
    vector<double> laplacian_staggered(const vector<double>& u_center) const {
        vector<double> grad_x((Nx+1) * Ny, 0.0);
        vector<double> grad_y(Nx * (Ny+1), 0.0);
        
        compute_gradient(u_center, grad_x, grad_y);
        return compute_divergence(grad_x, grad_y);
    }
    
    // Energy
    double compute_energy_staggered(const vector<double>& u_center,
                                   double ep2, double theta) const {
        double E = 0.0;

        vector<double> grad_x((Nx+1) * Ny, 0.0);
        vector<double> grad_y(Nx * (Ny+1), 0.0);
        compute_gradient(u_center, grad_x, grad_y);
        
        #pragma omp parallel for reduction(+:E)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                int idx_g = idx_grad_x(i, j);
                E += 0.5 * ep2 * grad_x[idx_g] * grad_x[idx_g] * dxdy;
            }
        }
        
        #pragma omp parallel for reduction(+:E)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                int idx_g = idx_grad_y(i, j);
                E += 0.5 * ep2 * grad_y[idx_g] * grad_y[idx_g] * dxdy;
            }
        }
        
        #pragma omp parallel for collapse(2) reduction(+:E)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                int idx = idx_center(i, j);
                E += dxdy * F(u_center[idx], theta);
            }
        }
        
        return E;
    }
    
    int get_Nx() const { return Nx; }
    int get_Ny() const { return Ny; }
    double get_dx() const { return dx; }
    double get_dy() const { return dy; }
    double get_dx2() const { return dx2; }
    double get_dxdy() const { return dxdy; }
    int get_N() const { return Nx * Ny; }
};

// Allen-Cahn 方程 ADMM 求解器（自适应时间步长版）
class allen_cahn_equation_admm_staggered {
private:
    // 时间参数
    double T_total;          // 总模拟时间
    double dt;                // 当前时间步长
    double dt_min, dt_max;    // 步长限制
    double theta = 4.0;
    
    double ep, ep2;
    StaggeredGrid grid;
    int Nx, Ny, N;
    double dx, dy, dx2, dxdy;
    
    // 动态存储
    vector<double> time_points;           // 每个时间步的实际时间
    vector<vector<double>> u;              // 解序列 u[step][idx]
    vector<vector<vector<double>>> U;      // 用于输出的三维数组（可选）
    vector<double> Energy;
    vector<double> residual_history;
    vector<double> min_value_history;
    vector<double> max_value_history;
    vector<int> admm_iterations;           // 每个时间步的 ADMM 迭代次数
    vector<double> admm_primal_residual_history;
    vector<int> admm_iteration_numbers;
    bool first_step_admm_saved;
    
    FFT2DSolver fft_solver;

public:
    // 构造函数：传入总时间、初始步长、网格参数
    allen_cahn_equation_admm_staggered(double total_time, double init_dt,
                                       int grid_x, int grid_y, double epsilon)
        : T_total(total_time), dt(init_dt), ep(epsilon),
          grid(grid_x, grid_y, 2.0, 2.0),
          fft_solver(grid_x, grid_y, 2.0/grid_x, 2.0/grid_y) {
        
        Nx = grid.get_Nx();
        Ny = grid.get_Ny();
        N = grid.get_N();
        dx = grid.get_dx();
        dy = grid.get_dy();
        dx2 = grid.get_dx2();
        dxdy = grid.get_dxdy();
        ep2 = ep * ep;
        
        // 步长限制（可根据需要调整）
        dt_min = init_dt / 100.0;
        dt_max = T_total / 100.0;
        
        // 初始化第一个时间步
        time_points.push_back(0.0);
        u.push_back(vector<double>(N, 0.0));
        // 设置初始条件
        for(int i = 0; i < Nx; i++){
            for(int j = 0; j < Ny; j++){
                double x = (i + 0.5) * dx;
                double y = (j + 0.5) * dy;
                int idx = grid.idx_center(i, j);
                u[0][idx] = ini_2(x, y);
            }
        }
        
        first_step_admm_saved = false;
    }
    
    vector<double> solve_with_fft_matrix_free(const vector<double>& b, double rho) {
        double alpha = 1.0 / dt + rho;
        double beta = ep2;
        return fft_solver.solve_linear_system(b, alpha, beta);
    }
    // 原辅助函数（vec_dot, residual, energy, mat_vec_product, rhs, newton, safe_bisection, f）保持不变
    double vec_dot(const vector<double>& a, const vector<double>& b){
        double result = 0.0;
        #pragma omp parallel for reduction(+:result) schedule(static, 256)
        for(int i = 0; i < a.size(); i++){
            result += a[i] * b[i];
        }
        return result;
    }
    

    double residual(const vector<double>& u_n, const vector<double>& u_new){
        double r_norm = 0.0;
        
        auto lap = grid.laplacian_staggered(u_new);
        
        #pragma omp parallel for reduction(+:r_norm) schedule(static)
        for(int idx = 0; idx < N; idx++){
            double r = (u_new[idx] - u_n[idx]) / dt - ep2 * lap[idx] + 
                       safe_log_ratio(u_new[idx]) + theta * (1.0 - 2.0 * u_n[idx]);
            r_norm += r * r;
        }
        r_norm = sqrt(r_norm);
        return r_norm;
    }
    
    double energy(const vector<double>& U_vec){
        return grid.compute_energy_staggered(U_vec, ep2, theta);
    }
    

    vector<double> mat_vec_product(const vector<double>& U_vec, double rho){
        vector<double> AU(N, 0.0);
        
        auto lap = grid.laplacian_staggered(U_vec);
        
        #pragma omp parallel for schedule(static)
        for(int idx = 0; idx < N; idx++){
            AU[idx] = (1.0 / dt + rho) * U_vec[idx] - ep2 * lap[idx];
        }
        
        return AU;
    }
    
    vector<double> rhs(const vector<double>& Un, const vector<double>& Y, 
                      double rho, const vector<double>& U2){
        vector<double> b(N, 0.0);
        
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < N; i++){
            b[i] = Un[i] / dt - Y[i] + rho * U2[i];
        }
        return b;
    }
    
    vector<double> conjugate_gradient(const vector<double>& b, 
                                     const vector<double>& x0, 
                                     double tol, double rho, int max_iter){
        vector<double> x = x0;
        vector<double> r = b;
        vector<double> p = r;
        
        auto Ax0 = mat_vec_product(x0, rho);
        
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < N; i++){
            r[i] = b[i] - Ax0[i];
            p[i] = r[i];
        }
        
        double b_norm = sqrt(vec_dot(b, b));
        if (b_norm < 1e-10) b_norm = 1.0;
        double rold = vec_dot(r, r);
        
        for (int k = 1; k <= max_iter; k++){
            auto Ap = mat_vec_product(p, rho);
            double pAp = vec_dot(p, Ap);
            
            if (fabs(pAp) < 1e-14) {
                cout << "CG: pAp too small, early stop at " << k << endl;
                break;
            }
            
            double alpha = rold / pAp;
            
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; i++){
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }
            
            double rnew = vec_dot(r, r);
            double r_norm = sqrt(rnew);
            
            if(r_norm / b_norm < tol) {
                break;
            }
            
            double beta = rnew / rold;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; i++){
                p[i] = r[i] + beta * p[i];
            }
            rold = rnew;
            
            if(k == max_iter) {
                cout << "CG not converge, final residual = " << r_norm / b_norm << endl;
            }
        }
        return x;
    }
    
    double newton(double u2_old, double u1, double un, double y, double rho){
        double u = 0.5;  
        double fu, dfu;
        
        if (f(u, u1, un, y, rho) > 0)     u = EPS;
        else if (f(u, u1, un, y, rho) < 0)     u = 1.0 - EPS;
        else return u;

        for(int iter = 1; iter <= 50; iter++){
            fu = safe_log_ratio(u) + theta * (1 - 2 * un) - y - rho * (u1 - u);
            dfu = 1.0 / (u * (1 - u)) + rho;
            double u_new = u - fu / dfu;
            
            if(fabs(f(u_new, u1, un, y, rho)) < 1e-10){
                return u_new;
            }
            u = u_new;
            
            if(iter == 50) {
                u = safe_bisection(u1, un, y, rho);
                break;
            }
        }
        return u;
    }
    
    double safe_bisection(double u1, double un, double y, double rho) {
        double a = EPS, b = 1.0 - EPS;
        double fa = f(a, u1, un, y, rho);
        double fb = f(b, u1, un, y, rho);
        
        if (fa * fb > 0) {
            return fabs(fa) < fabs(fb) ? a : b;
        }
        
        for (int iter = 0; iter < 50; iter++) {
            double mid = (a + b) / 2;
            double fmid = f(mid, u1, un, y, rho);
            
            if (fabs(fmid) < 1e-10 || (b - a) < 1e-10) {
                return mid;
            }
            
            if (fa * fmid < 0) {
                b = mid;
                fb = fmid;
            } else {
                a = mid;
                fa = fmid;
            }
        }
        return (a + b) / 2;
    }
    
    double f(double u2, double u1, double un, double y, double rho){
        return safe_log_ratio(u2) + theta * (1 - 2 * un) - y - rho * (u1 - u2);
    }
    
    vector<double> admm(vector<double>& Un, int max_iter, double tolerance, int time_step){
        double rho = 10.0;
        double tau = 1.5;
        double mu = 10.0;
        double gamma = 2.0;

        auto U_1 = Un;
        auto U_2 = Un;
        auto U_2_new = Un;
        vector<double> Y(N, 0.0);
        
        double prev_energy = energy(Un);
        int iter_count = 0;
        
        if (time_step == 0) {
            admm_primal_residual_history.clear();
            admm_iteration_numbers.clear();
            first_step_admm_saved = true;
            cout << "\n=== Recording ADMM iteration history for first time step ===" << endl;
        }

        for(int k = 1; k <= max_iter; k++){
            iter_count = k;
            
            auto b = rhs(Un, Y, rho, U_2);
            U_1 = solve_with_fft_matrix_free(b, rho);  
            // U_1 = conjugate_gradient(b, U_1, 1e-8, rho, 1000);  
       
            double r = 0.0;
            double s = 0.0;
            
            #pragma omp parallel for reduction(+:r, s) schedule(static)
            for(int i = 0; i < N; i++){
                U_2_new[i] = newton(U_2[i], U_1[i], Un[i], Y[i], rho);
                Y[i] = Y[i] + tau * rho * (U_1[i] - U_2_new[i]);
                
                r += (U_1[i] - U_2_new[i]) * (U_1[i] - U_2_new[i]);
                s += (U_2[i] - U_2_new[i]) * (U_2[i] - U_2_new[i]);
            }
            
            U_2 = U_2_new;
            r = sqrt(r);
            s = sqrt(s);
            
            // 保存第一个时间步的迭代历史（原始残差r）
            if (time_step == 0) {
                admm_primal_residual_history.push_back(r);
                admm_iteration_numbers.push_back(k);
                
            }

            if(r > mu * s)    rho = gamma * rho;
            else if(s > mu * r)    rho = rho / gamma;
            else rho = rho;
            
            if(max(r, s) < tolerance) {
                Un = U_2;
                cout << "ADMM converge at " << k << " iterations." << endl;
                break;
            }
            
            if(k == max_iter){
                Un = U_2;
                cout << "ADMM not converge." << endl;
            }
        }
        
        double new_energy = energy(Un);
        if (new_energy > prev_energy + 1e-8) {
            cout << "WARNING: Energy increased from " << prev_energy 
                 << " to " << new_energy << endl;
        }
        
        // 保存ADMM迭代次数
        admm_iterations[time_step] = iter_count;
        
        return Un;
    }

    // 核心函数：时间自适应求解
    void solve(){
        ofstream history_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/history/history_ccfd_admm_adaptive.txt", ios::app);
        history_file << "Begin: T_total = " << T_total << ", initial dt = " << dt
                     << ", Nx = " << Nx << ", Ny = " << Ny << ", epsilon = " << ep << endl;

        int num_threads = omp_get_max_threads();
        cout << "Using " << num_threads << " threads, staggered grid, adaptive time stepping" << endl;

        // 打开输出文件（绘图数据）
        ofstream energy_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/plots/energy_data_adaptive.txt");
        ofstream residual_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/plots/residual_data_adaptive.txt");
        ofstream bounds_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/plots/bounds_data_adaptive.txt");
        ofstream admm_iter_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/plots/admm_iterations_adaptive.txt");
        
        energy_file << "# time energy" << endl;
        residual_file << "# time residual" << endl;
        bounds_file << "# time min_value max_value" << endl;
        admm_iter_file << "# time admm_iterations" << endl;

        int step = 0;          // 当前已完成的步数索引
        double t = 0.0;        // 当前时间
        double current_dt = dt;
        admm_iterations.push_back(0); 

        // 输出初始状态
        double E0 = energy(u[0]);
        Energy.push_back(E0);
        double min_val0, max_val0;
        compute_bounds(u[0], min_val0, max_val0);
        energy_file << t << " " << E0 << endl;
        bounds_file << t << " " << min_val0 << " " << max_val0 << endl;
        history_file << "t = " << t << ", energy = " << E0 << endl;

        while (t < T_total - 1e-12) {
            cout << "Step " << step << ", t = " << t << ", dt = " << dt << endl;
            if (admm_iterations.size() <= step) {
                admm_iterations.resize(step + 1, 0);
            }

            vector<double> u_old = u[step];
            vector<double> u_new = u_old;  // 初始猜测
            int attempt = 0;
            const int max_attempts = 20;
            bool accepted = false;

            while (!accepted && attempt < max_attempts) {
                this->dt = current_dt;
                cout << dt << endl;
                vector<double> u_old_copy = u_old;
                u_new = admm(u_old_copy, 1000, 1e-8, step);

                double E_old = Energy[step];
                double E_new = energy(u_new);

                // 检查能量是否上升（超过容忍值）

                if (E_new > E_old + 1e-12) {
                    // 能量上升，拒绝步长，减半重试
                    current_dt = max(current_dt * 0.5, dt_min);
                    attempt++;
                    cout << "  Energy increased (ΔE = " << E_new - E_old << "), retrying with dt = " << current_dt << endl;
                    continue;
                }

                // 能量下降正常，接受该步
                accepted = true;
                
                // 计算能量下降率（相对值）
                double rel_drop = (E_old - E_new) / max(abs(E_old), 1e-12);
                // 根据下降率调整下一步步长
                const double target_low = 0.1;   // 下降太慢的阈值
                const double target_high = 0.1;   // 下降太快的阈值
                const double inc_factor = 2.0;    // 增大因子
                const double dec_factor = 0.5;    // 减小因子

                if (rel_drop < target_low) {
                    // 下降太慢，增大步长
                    current_dt = min(current_dt * inc_factor, dt_max);
                } else if (rel_drop > target_high) {
                    // 下降太快，减小步长
                    current_dt = max(current_dt * dec_factor, dt_min);
                }
                // 否则保持不变

                // 更新时间和存储

                double used_dt = current_dt;  // 保存本次实际步长
                t += used_dt;
                time_points.push_back(t);
                u.push_back(u_new);
                Energy.push_back(E_new);
                compute_bounds(u_new, min_val0, max_val0);  // 复用变量
                min_value_history.push_back(min_val0);
                max_value_history.push_back(max_val0);

                // 计算残差（可选）
                double r_norm = residual(u_old, u_new);
                residual_history.push_back(r_norm);

                // 记录 ADMM 迭代次数（需从 admm 内部获取，目前 admm 未返回迭代次数，需修改 admm 返回迭代次数，或通过成员变量记录）
                // 假设 admm_iterations 在 admm 函数内已更新，我们直接使用 admm_iterations[step]（但注意 step 是旧索引）
                // 简单起见，我们在 admm 中通过 time_step 参数保存了迭代次数，但需要对应存储。这里我们假设 admm_iterations 的索引与 step 对应。
                // 在 admm 函数中，有 admm_iterations[time_step] = iter_count; 而 time_step 是传入的 step。
                // 所以直接使用 admm_iterations[step] 即可。
                int iter_used = admm_iterations[step];  // 注意：admm 函数中 time_step 是旧步索引，它保存到 admm_iterations[time_step]。
                // 但 admm_iterations 是在 admm 内部赋值的，这里 step 是调用时的旧步索引，所以可以。
                
                // 写入文件（以实际时间 t 为横坐标）
                energy_file << t << " " << E_new << endl;
                bounds_file << t << " " << min_val0 << " " << max_val0 << endl;
                residual_file << t << " " << r_norm << endl;
                admm_iter_file << t << " " << iter_used << endl;

                history_file << "t = " << t << ", energy = " << E_new << ", residual = " << r_norm
                             << ", ADMM iterations = " << iter_used << ", dt = " << used_dt << endl;

                step++;  // 步数增加
            }

            if (!accepted) {
                cerr << "Failed to advance after " << max_attempts << " attempts. Stopping." << endl;
                break;
            }
        }

        history_file << "Final time reached: t = " << t << ", total steps = " << step << endl;
        history_file.close();
        energy_file.close();
        residual_file.close();
        bounds_file.close();
        admm_iter_file.close();

        // 保存 ADMM 迭代残差历史（可选）
        save_admm_residual_history();
    }

    // 辅助函数：计算最值
    void compute_bounds(const vector<double>& u_vec, double& min_val, double& max_val) {
        min_val = *min_element(u_vec.begin(), u_vec.end());
        max_val = *max_element(u_vec.begin(), u_vec.end());
    }

    void save_admm_residual_history() {
        if (!first_step_admm_saved || admm_iteration_numbers.empty()) return;
        string filename = "H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/plots/admm_residual_data_adaptive.txt";
        ofstream history_file(filename);
        if (!history_file.is_open()) return;
        history_file << "# ADMM iteration residual history for first time step" << endl;
        history_file << "# iteration primal_residual" << endl;
        for (size_t i = 0; i < admm_iteration_numbers.size(); i++) {
            history_file << admm_iteration_numbers[i] << " "
                         << scientific << setprecision(8)
                         << admm_primal_residual_history[i] << endl;
        }
        history_file.close();
    }

    vector<vector<double>> vec_to_arr(const vector<double>& a){
        vector<vector<double>> A(Nx, vector<double>(Ny, 0.0));
        #pragma omp parallel for collapse(2) schedule(static)
        for(int i = 0; i < Nx; i++){
            for(int j = 0; j < Ny; j++){
                int idx = grid.idx_center(i, j);
                A[i][j] = a[idx];
            }
        }
        return A;
    }

    const vector<vector<double>>& getu() const{ return u; }

    vector<vector<vector<double>>>& getU(){
        U.resize(u.size(), vector<vector<double>>(Nx, vector<double>(Ny, 0.0)));
        #pragma omp parallel for schedule(static)
        for (size_t n = 0; n < u.size(); n++) {
            U[n] = vec_to_arr(u[n]);
        }
        return U;
    }
};

// 数据保存函数（适配动态步数）
void saveDataToFile(const vector<vector<vector<double>>>& u, const string& filename) {
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cerr << "Cannot Open File: " << filename << endl;
        return;
    }
    outFile << fixed << setprecision(6);
    int timeSteps = u.size();
    int xSize = u[0].size();
    int ySize = u[0][0].size();
    outFile << timeSteps << " " << xSize << " " << ySize << endl;
    // 输出坐标（网格中心）
    double L = 2.0;
    for (int x = 0; x < xSize; ++x) {
        outFile << (x + 0.5) * L / xSize;
        if (x < xSize - 1) outFile << " ";
    }
    outFile << endl;
    for (int y = 0; y < ySize; ++y) {
        outFile << (y + 0.5) * L / ySize;
        if (y < ySize - 1) outFile << " ";
    }
    outFile << endl;
    for (int t = 0; t < timeSteps; ++t) {
        outFile << "t=" << t << endl;
        for (int x = 0; x < xSize; ++x) {
            for (int y = 0; y < ySize; ++y) {
                outFile << u[t][x][y];
                if (y < ySize - 1) outFile << " ";
            }
            outFile << endl;
        }
    }
    outFile.close();
    cout << "Data Saved to: " << filename << endl;
}

int main(){
    int desired_threads = 16;
    omp_set_num_threads(desired_threads);
    
    double T_total = 1e12;      // 总模拟时间
    double init_dt = 1;      // 初始步长
    int Nx = 256;
    int Ny = 256;
    double ep = 0.05;
    
    allen_cahn_equation_admm_staggered allen_cahn_u(T_total, init_dt, Nx, Ny, ep);

    clock_t start = clock();
    allen_cahn_u.solve();
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    ofstream history_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/history/history_ccfd_admm_adaptive.txt", ios::app);
    history_file << "cpu_time_used = " << cpu_time_used << " seconds." << endl << endl;
    history_file.close();

    auto U = allen_cahn_u.getU();
    saveDataToFile(U, "H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/data/data_ccfd_admm_adaptive.txt");
    
    cout << "admm_staggered adaptive CPU time: " << cpu_time_used << " s." << endl;
    
    return 0;
}