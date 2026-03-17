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

double ini_1(double x, double y){
    return (1+sin(2*PI*x)*sin(2*PI*y))/3.0;
}

double ini_2(double x, double y){
    if(abs(x)<=0.35 && abs(y)<=0.35) return 1e-5;
    else return 1-1e-5;
}

double ini_3(double x, double y){
    return 0.01 + 0.98 * ((double)rand() / (RAND_MAX + 1.0));
}

double ini_4(double x, double y){
    return 0.5;
}

double F(double t, double theta){
    t = max(EPS, min(1.0 - EPS, t));
    return t*log(t) + (1-t)*log(1-t) + theta*t*(1-t);
}

double safe_log_ratio(double u) {
    u = max(EPS, min(1.0 - EPS, u));
    return log(u / (1 - u));
}

// StaggerGrid

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


class allen_cahn_equation_admm_staggered {
private:
    int Nt;
    double dt;
    double ep;
    double ep2;
    double theta = 4.0;
    
    StaggeredGrid grid;
    int Nx, Ny, N;
    double dx, dy, dx2, dxdy;
    
    vector<vector<double>> u;  
    vector<vector<vector<double>>> U;  
    vector<double> Energy;
    
    // 新增：保存绘图数据
    vector<double> residual_history;
    vector<double> min_value_history;
    vector<double> max_value_history;
    vector<int> admm_iterations;
    vector<double> admm_primal_residual_history;
    vector<int> admm_iteration_numbers;
    bool first_step_admm_saved;
    
    FFT2DSolver fft_solver;
    
public:
    allen_cahn_equation_admm_staggered(double time, int time_steps, 
                                      int grid_x, int grid_y, double epsilon)
        : dt(time), Nt(time_steps), ep(epsilon), 
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
        
        u.resize(Nt + 1, vector<double>(N, 0.0));
        U.resize(Nt + 1, vector<vector<double>>(Nx, vector<double>(Ny, 0.0)));
        Energy.resize(Nt + 1, 0.0);
        
        // 初始化历史记录数组
        residual_history.resize(Nt + 1, 0.0);
        min_value_history.resize(Nt + 1, 0.0);
        max_value_history.resize(Nt + 1, 0.0);
        admm_iterations.resize(Nt + 1, 0);

        first_step_admm_saved = false;
        
        // #pragma omp parallel for collapse(2) schedule(static)
        for(int i = 0; i < Nx; i++){
            for(int j = 0; j < Ny; j++){
                double x = (i + 0.5) * dx;
                double y = (j + 0.5) * dy;
                int idx = grid.idx_center(i, j);
                u[0][idx] = ini_1(x, y);  
            }
        }
    }
    
    
    vector<double> solve_with_fft_matrix_free(const vector<double>& b, double rho) {
        double alpha = 1.0 / dt + rho;
        double beta = ep2;
        return fft_solver.solve_linear_system(b, alpha, beta);
    }
    
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
    
    // 新增：计算并保存保界性信息
    void compute_bounds(const vector<double>& u_vec, double& min_val, double& max_val) {
        min_val = *min_element(u_vec.begin(), u_vec.end());
        max_val = *max_element(u_vec.begin(), u_vec.end());
    }

    // 新增：保存ADMM迭代残差历史到文件
    void save_admm_residual_history() {
        if (!first_step_admm_saved || admm_iteration_numbers.empty()) {
            cout << "No ADMM residual history data to save." << endl;
            return;
        }
        
        string filename = "H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/plots/admm_residual_data.txt";
        ofstream history_file(filename);
        
        if (!history_file.is_open()) {
            cerr << "Cannot open file: " << filename << endl;
            return;
        }
        
        history_file << "# ADMM iteration residual history for first time step" << endl;
        history_file << "# iteration primal_residual" << endl;
        
        for (size_t i = 0; i < admm_iteration_numbers.size(); i++) {
            history_file << admm_iteration_numbers[i] << " "
                        << scientific << setprecision(8)
                        << admm_primal_residual_history[i] << endl;
        }
        
        history_file.close();
        cout << "\nADMM residual history saved to: " << filename << endl;
        cout << "Total iterations recorded: " << admm_iteration_numbers.size() << endl;
    }

    
    // 修改后的 solve 函数
    void solve(){
        ofstream history_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/history/history_ccfd_admm_parallel.txt", ios::app);
        history_file << "Begin: dt = " << dt << ", Nx = " << Nx 
                     << ", Ny = " << Ny << ", Nt = " << Nt 
                     << ", epsilon = " << ep << "." << endl;

        int num_threads = omp_get_max_threads();
        cout << "Using " << num_threads << " threads, staggered grid" << endl;
        
        // 打开专门用于绘图的数据文件
        ofstream energy_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/plots/energy_data.txt");
        ofstream residual_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/plots/residual_data.txt");
        ofstream bounds_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/plots/bounds_data.txt");
        ofstream admm_iter_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/plots/admm_iterations.txt");
        
        energy_file << "# time_step energy" << endl;
        residual_file << "# time_step residual" << endl;
        bounds_file << "# time_step min_value max_value" << endl;
        admm_iter_file << "# time_step admm_iterations" << endl;
        
        for(int n = 0; n < Nt; n++){
            auto Un = u[n];
            Energy[n] = energy(Un);
            
            // 计算并保存保界性数据
            double min_val, max_val;
            compute_bounds(Un, min_val, max_val);
            min_value_history[n] = min_val;
            max_value_history[n] = max_val;
            
            // 写入绘图数据
            energy_file << n << " " << Energy[n] << endl;
            bounds_file << n << " " << min_val << " " << max_val << endl;
            
            history_file << "time step " << n << "/" << Nt 
                         << ", energy = " << Energy[n];
            
            u[n+1] = admm(Un, 1000, 1e-8, n);
            
            double r_norm = residual(u[n], u[n+1]);
            residual_history[n] = r_norm;
            residual_file << n << " " << r_norm << endl;
            admm_iter_file << n << " " << admm_iterations[n] << endl;
            
            history_file << ", residual: " << r_norm << ", ADMM iterations: " << admm_iterations[n] << endl;
            
            double next_energy = energy(u[n+1]);
            if (next_energy > Energy[n] + 1e-8) {
                cout << "WARNING: Energy increased at time step " << n 
                     << " from " << Energy[n] << " to " << next_energy << endl;
            }
        }
        
        // 最后一个时间步的数据
        Energy[Nt] = energy(u[Nt]);
        double min_val, max_val;
        compute_bounds(u[Nt], min_val, max_val);
        
        energy_file << Nt << " " << Energy[Nt] << endl;
        bounds_file << Nt << " " << min_val << " " << max_val << endl;
        
        history_file << "time step " << Nt << "/" << Nt 
                     << ", energy = " << Energy[Nt] << endl;
        
        history_file.close();
        energy_file.close();
        residual_file.close();
        bounds_file.close();
        admm_iter_file.close();
        
       // 保存ADMM迭代残差历史
        save_admm_residual_history();
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
        #pragma omp parallel for schedule(static)
        for(int n = 0; n <= Nt; n++){
            U[n] = vec_to_arr(u[n]);
        }
        return U;
    }
};


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
    
    for (int x = 0; x < xSize; ++x) {
        outFile << static_cast<double>(x + 0.5);
        if (x < xSize - 1) outFile << " ";
    }
    outFile << endl;
    
    for (int y = 0; y < ySize; ++y) {
        outFile << static_cast<double>(y + 0.5);
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
    
    double dt = 1e-2;  
    int Nx = 256;
    int Ny = 256;
    int Nt = 100;      
    double ep = 0.05;
    
    allen_cahn_equation_admm_staggered allen_cahn_u(dt, Nt, Nx, Ny, ep);

    clock_t start = clock();
    allen_cahn_u.solve();
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    ofstream history_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/history/history_ccfd_admm_parallel.txt", ios::app);
    history_file << "cpu_time_used = " << cpu_time_used << " seconds." << endl << endl;
    history_file.close();

    auto U = allen_cahn_u.getU();
    saveDataToFile(U, "H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/data/data_ccfd_admm_parallel.txt");
    
    cout << "admm_staggered CPU time: " << cpu_time_used << " s." << endl;
    cout << "\nTo generate plots, run:" << endl;
    cout << "  cd H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/plots" << endl;
    cout << "  python plot_results.py" << endl;
    cout << "\nMake sure you have numpy and matplotlib installed:" << endl;
    cout << "  pip install numpy matplotlib" << endl;
    
    return 0;
}