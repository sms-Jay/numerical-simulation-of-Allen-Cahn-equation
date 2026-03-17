// ccfd_admm_3d.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <omp.h>
#include <complex>
#include "fft_3d_solver.h"
using namespace std;

const double PI = 3.1415926535897932384626;
const double EPS = 1e-24;

// 三维初始条件
double ini_1(double x, double y, double z) {
    return (1 + sin(2*PI*x) * sin(2*PI*y) * sin(2*PI*z)) / 3.0;
}

double ini_2(double x, double y, double z) {
    // 中心立方体区域
    if (fabs(x-1) <= 0.35 && fabs(y-1) <= 0.35 && fabs(z-1) <= 0.35)
        return 1e-5;
    else
        return 1 - 1e-5;
}

double ini_3(double x, double y, double z) {
    return 0.01 + 0.98 * ((double)rand() / (RAND_MAX + 1.0));
}

double ini_4(double x, double y, double z) {
    return 0.5;
}

double F(double t, double theta) {
    t = max(EPS, min(1.0 - EPS, t));
    return t * log(t) + (1 - t) * log(1 - t) + theta * t * (1 - t);
}

double safe_log_ratio(double u) {
    u = max(EPS, min(1.0 - EPS, u));
    return log(u / (1 - u));
}

// 三维交错网格
class StaggeredGrid3D {
private:
    int Nx, Ny, Nz;
    double dx, dy, dz, dx2, dy2, dz2, dV; // dV = dx*dy*dz

public:
    StaggeredGrid3D(int grid_x, int grid_y, int grid_z, double Lx, double Ly, double Lz)
        : Nx(grid_x), Ny(grid_y), Nz(grid_z) {
        dx = Lx / Nx;
        dy = Ly / Ny;
        dz = Lz / Nz;
        dx2 = dx * dx;
        dy2 = dy * dy;
        dz2 = dz * dz;
        dV = dx * dy * dz;
    }

    // 索引：中心点 (i,j,k) -> i*Ny*Nz + j*Nz + k
    int idx_center(int i, int j, int k) const { return (i * Ny + j) * Nz + k; }

    // x方向梯度点位于 (i+0.5, j, k)，i从0..Nx，索引 idx_gx = i*Ny*Nz + j*Nz + k
    int idx_grad_x(int i, int j, int k) const { return (i * Ny + j) * Nz + k; }

    // y方向梯度点位于 (i, j+0.5, k)，j从0..Ny，索引 idx_gy = i*(Ny+1)*Nz + j*Nz + k
    int idx_grad_y(int i, int j, int k) const { return i * (Ny+1) * Nz + j * Nz + k; }

    // z方向梯度点位于 (i, j, k+0.5)，k从0..Nz，索引 idx_gz = i*Ny*(Nz+1) + j*(Nz+1) + k
    int idx_grad_z(int i, int j, int k) const { return (i * Ny + j) * (Nz+1) + k; }

    void compute_gradient(const vector<double>& u_center,
                          vector<double>& grad_x,
                          vector<double>& grad_y,
                          vector<double>& grad_z) const {
        fill(grad_x.begin(), grad_x.end(), 0.0);
        fill(grad_y.begin(), grad_y.end(), 0.0);
        fill(grad_z.begin(), grad_z.end(), 0.0);

        // x方向梯度
        #pragma omp parallel for collapse(3)
        for (int i = 0; i <= Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    int idx_g = idx_grad_x(i, j, k);
                    int i_right = i % Nx;
                    int i_left = (i - 1 + Nx) % Nx;
                    int idx_c_right = idx_center(i_right, j, k);
                    int idx_c_left = idx_center(i_left, j, k);
                    grad_x[idx_g] = (u_center[idx_c_right] - u_center[idx_c_left]) / dx;
                }
            }
        }

        // y方向梯度
        #pragma omp parallel for collapse(3)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j <= Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    int idx_g = idx_grad_y(i, j, k);
                    int j_up = j % Ny;
                    int j_down = (j - 1 + Ny) % Ny;
                    int idx_c_up = idx_center(i, j_up, k);
                    int idx_c_down = idx_center(i, j_down, k);
                    grad_y[idx_g] = (u_center[idx_c_up] - u_center[idx_c_down]) / dy;
                }
            }
        }

        // z方向梯度
        #pragma omp parallel for collapse(3)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k <= Nz; k++) {
                    int idx_g = idx_grad_z(i, j, k);
                    int k_up = k % Nz;
                    int k_down = (k - 1 + Nz) % Nz;
                    int idx_c_up = idx_center(i, j, k_up);
                    int idx_c_down = idx_center(i, j, k_down);
                    grad_z[idx_g] = (u_center[idx_c_up] - u_center[idx_c_down]) / dz;
                }
            }
        }
    }

    vector<double> compute_divergence(const vector<double>& grad_x,
                                      const vector<double>& grad_y,
                                      const vector<double>& grad_z) const {
        vector<double> div(Nx * Ny * Nz, 0.0);

        #pragma omp parallel for collapse(3)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    int idx_c = idx_center(i, j, k);

                    int idx_gx_right = idx_grad_x(i+1, j, k);
                    int idx_gx_left  = idx_grad_x(i,   j, k);
                    double div_x = (grad_x[idx_gx_right] - grad_x[idx_gx_left]) / dx;

                    int idx_gy_up   = idx_grad_y(i, j+1, k);
                    int idx_gy_down = idx_grad_y(i, j,   k);
                    double div_y = (grad_y[idx_gy_up] - grad_y[idx_gy_down]) / dy;

                    int idx_gz_up   = idx_grad_z(i, j, k+1);
                    int idx_gz_down = idx_grad_z(i, j, k);
                    double div_z = (grad_z[idx_gz_up] - grad_z[idx_gz_down]) / dz;

                    div[idx_c] = div_x + div_y + div_z;
                }
            }
        }
        return div;
    }

    vector<double> laplacian_staggered(const vector<double>& u_center) const {
        vector<double> grad_x((Nx+1) * Ny * Nz, 0.0);
        vector<double> grad_y(Nx * (Ny+1) * Nz, 0.0);
        vector<double> grad_z(Nx * Ny * (Nz+1), 0.0);

        compute_gradient(u_center, grad_x, grad_y, grad_z);
        return compute_divergence(grad_x, grad_y, grad_z);
    }

    double compute_energy_staggered(const vector<double>& u_center,
                                    double ep2, double theta) const {
        double E = 0.0;

        vector<double> grad_x((Nx+1) * Ny * Nz, 0.0);
        vector<double> grad_y(Nx * (Ny+1) * Nz, 0.0);
        vector<double> grad_z(Nx * Ny * (Nz+1), 0.0);
        compute_gradient(u_center, grad_x, grad_y, grad_z);

        #pragma omp parallel for reduction(+:E) collapse(3)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    int idx_gx = idx_grad_x(i, j, k);
                    E += 0.5 * ep2 * grad_x[idx_gx] * grad_x[idx_gx] * dV;
                }
            }
        }

        #pragma omp parallel for reduction(+:E) collapse(3)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    int idx_gy = idx_grad_y(i, j, k);
                    E += 0.5 * ep2 * grad_y[idx_gy] * grad_y[idx_gy] * dV;
                }
            }
        }

        #pragma omp parallel for reduction(+:E) collapse(3)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    int idx_gy = idx_grad_z(i, j, k);
                    E += 0.5 * ep2 * grad_y[idx_gy] * grad_y[idx_gy] * dV;
                }
            }
        }

        #pragma omp parallel for reduction(+:E) collapse(3)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    int idx = idx_center(i, j, k);
                    E += dV * F(u_center[idx], theta);
                }
            }
        }

        return E;
    }

    int get_Nx() const { return Nx; }
    int get_Ny() const { return Ny; }
    int get_Nz() const { return Nz; }
    double get_dx() const { return dx; }
    double get_dy() const { return dy; }
    double get_dz() const { return dz; }
    double get_dV() const { return dV; }
    int get_N() const { return Nx * Ny * Nz; }
};

// 三维 Allen-Cahn ADMM 求解器
class allen_cahn_equation_admm_staggered_3d {
private:
    int Nt;
    double dt;
    double ep;
    double ep2;
    double theta = 4.0;

    StaggeredGrid3D grid;
    int Nx, Ny, Nz, N;
    double dx, dy, dz, dV;

    vector<vector<double>> u;          // u[n][idx]
    vector<vector<vector<vector<double>>>> U; // U[n][i][j][k]
    vector<double> Energy;

    FFT3DSolver fft_solver;

public:
    allen_cahn_equation_admm_staggered_3d(double time, int time_steps,
                                          int grid_x, int grid_y, int grid_z, double epsilon)
        : dt(time), Nt(time_steps), ep(epsilon),
          grid(grid_x, grid_y, grid_z, 1.0, 1.0, 1.0),
          fft_solver(grid_x, grid_y, grid_z, 1.0/grid_x, 1.0/grid_y, 1.0/grid_z) {

        Nx = grid.get_Nx();
        Ny = grid.get_Ny();
        Nz = grid.get_Nz();
        N = grid.get_N();
        dx = grid.get_dx();
        dy = grid.get_dy();
        dz = grid.get_dz();
        dV = grid.get_dV();
        ep2 = ep * ep;

        u.resize(Nt + 1, vector<double>(N, 0.0));
        U.resize(Nt + 1, vector<vector<vector<double>>>(Nx,
                vector<vector<double>>(Ny, vector<double>(Nz, 0.0))));
        Energy.resize(Nt + 1, 0.0);

        // 初始化使用 ini_3 (随机)
        
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    double x = (i + 0.5) * dx;
                    double y = (j + 0.5) * dy;
                    double z = (k + 0.5) * dz;
                    int idx = grid.idx_center(i, j, k);
                    u[0][idx] = ini_3(x, y, z);
                }
            }
        }
    }

    vector<double> solve_with_fft_matrix_free(const vector<double>& b, double rho) {
        double alpha = 1.0 / dt + rho;
        double beta = ep2;
        return fft_solver.solve_linear_system(b, alpha, beta);
    }

    double vec_dot(const vector<double>& a, const vector<double>& b) {
        double result = 0.0;
        #pragma omp parallel for reduction(+:result) schedule(static, 256)
        for (int i = 0; i < (int)a.size(); i++) result += a[i] * b[i];
        return result;
    }

    double residual(const vector<double>& u_n, const vector<double>& u_new) {
        double r_norm = 0.0;
        auto lap = grid.laplacian_staggered(u_new);
        #pragma omp parallel for reduction(+:r_norm) schedule(static)
        for (int idx = 0; idx < N; idx++) {
            double r = (u_new[idx] - u_n[idx]) / dt - ep2 * lap[idx] +
                       safe_log_ratio(u_new[idx]) + theta * (1.0 - 2.0 * u_n[idx]);
            r_norm += r * r;
        }
        return sqrt(r_norm);
    }

    double energy(const vector<double>& U_vec) {
        return grid.compute_energy_staggered(U_vec, ep2, theta);
    }

    vector<double> mat_vec_product(const vector<double>& U_vec, double rho) {
        vector<double> AU(N, 0.0);
        auto lap = grid.laplacian_staggered(U_vec);
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < N; idx++) {
            AU[idx] = (1.0 / dt + rho) * U_vec[idx] - ep2 * lap[idx];
        }
        return AU;
    }

    vector<double> rhs(const vector<double>& Un, const vector<double>& Y,
                       double rho, const vector<double>& U2) {
        vector<double> b(N, 0.0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) b[i] = Un[i] / dt - Y[i] + rho * U2[i];
        return b;
    }

    double newton(double u2_old, double u1, double un, double y, double rho) {
        double u = 0.5;
        if (f(u, u1, un, y, rho) > 0) u = EPS;
        else if (f(u, u1, un, y, rho) < 0) u = 1.0 - EPS;
        else return u;

        for (int iter = 1; iter <= 50; iter++) {
            double fu = safe_log_ratio(u) + theta * (1 - 2 * un) - y - rho * (u1 - u);
            double dfu = 1.0 / (u * (1 - u)) + rho;
            double u_new = u - fu / dfu;
            if (fabs(f(u_new, u1, un, y, rho)) < 1e-10) return u_new;
            u = u_new;
        }
        // 若 Newton 不收敛，改用二分法
        return safe_bisection(u1, un, y, rho);
    }

    double safe_bisection(double u1, double un, double y, double rho) {
        double a = EPS, b = 1.0 - EPS;
        double fa = f(a, u1, un, y, rho);
        double fb = f(b, u1, un, y, rho);
        if (fa * fb > 0) return fabs(fa) < fabs(fb) ? a : b;
        for (int iter = 0; iter < 50; iter++) {
            double mid = (a + b) / 2;
            double fmid = f(mid, u1, un, y, rho);
            if (fabs(fmid) < 1e-10 || (b - a) < 1e-10) return mid;
            if (fa * fmid < 0) { b = mid; fb = fmid; }
            else { a = mid; fa = fmid; }
        }
        return (a + b) / 2;
    }

    double f(double u2, double u1, double un, double y, double rho) {
        return safe_log_ratio(u2) + theta * (1 - 2 * un) - y - rho * (u1 - u2);
    }

    vector<double> admm(vector<double>& Un, int max_iter, double tolerance) {
        double rho = 10.0;
        double tau = 1.5;
        double mu = 10.0;
        double gamma = 2.0;

        auto U_1 = Un;
        auto U_2 = Un;
        auto U_2_new = Un;
        vector<double> Y(N, 0.0);

        double prev_energy = energy(Un);

        for (int k = 1; k <= max_iter; k++) {
            auto b = rhs(Un, Y, rho, U_2);
            U_1 = solve_with_fft_matrix_free(b, rho);

            double r = 0.0, s = 0.0;
            #pragma omp parallel for reduction(+:r, s) schedule(static)
            for (int i = 0; i < N; i++) {
                U_2_new[i] = newton(U_2[i], U_1[i], Un[i], Y[i], rho);
                Y[i] = Y[i] + tau * rho * (U_1[i] - U_2_new[i]);
                r += (U_1[i] - U_2_new[i]) * (U_1[i] - U_2_new[i]);
                s += (U_2[i] - U_2_new[i]) * (U_2[i] - U_2_new[i]);
            }
            U_2 = U_2_new;
            r = sqrt(r);
            s = sqrt(s);

            if (r > mu * s)      rho = gamma * rho;
            else if (s > mu * r) rho = rho / gamma;

            if (max(r, s) < tolerance) {
                Un = U_2;
                cout << "ADMM converge at " << k << " iterations." << endl;
                break;
            }
            if (k == max_iter) {
                Un = U_2;
                cout << "ADMM not converge." << endl;
            }
        }

        double new_energy = energy(Un);
        if (new_energy > prev_energy + 1e-8) {
            cout << "WARNING: Energy increased from " << prev_energy
                 << " to " << new_energy << endl;
        }
        return Un;
    }

    void solve() {
        ofstream history_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/history/history_ccfd_admm_parallel_3d.txt", ios::app);
        history_file << "Begin: dt = " << dt << ", Nx = " << Nx
                     << ", Ny = " << Ny << ", Nz = " << Nz << ", Nt = " << Nt
                     << ", epsilon = " << ep << "." << endl;

        int num_threads = omp_get_max_threads();
        cout << "Using " << num_threads << " threads, staggered grid 3D" << endl;

        for (int n = 0; n < Nt; n++) {
            auto Un = u[n];
            Energy[n] = energy(Un);
            history_file << "time step " << n << "/" << Nt
                         << ", energy = " << Energy[n];

            u[n+1] = admm(Un, 1000, 1e-6);

            double r_norm = residual(u[n], u[n+1]);
            history_file << ", residual: " << r_norm << endl;

            double next_energy = energy(u[n+1]);
            if (next_energy > Energy[n] + 1e-8) {
                cout << "WARNING: Energy increased at time step " << n
                     << " from " << Energy[n] << " to " << next_energy << endl;
            }
        }
        Energy[Nt] = energy(u[Nt]);
        history_file << "time step " << Nt << "/" << Nt
                     << ", energy = " << Energy[Nt] << endl;
        history_file.close();
    }

    vector<vector<vector<double>>> vec_to_arr(const vector<double>& a) {
        vector<vector<vector<double>>> A(Nx,
            vector<vector<double>>(Ny, vector<double>(Nz, 0.0)));
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    int idx = grid.idx_center(i, j, k);
                    A[i][j][k] = a[idx];
                }
            }
        }
        return A;
    }

    const vector<vector<double>>& getu() const { return u; }

    vector<vector<vector<vector<double>>>>& getU() {
        #pragma omp parallel for schedule(static)
        for (int n = 0; n <= Nt; n++) {
            U[n] = vec_to_arr(u[n]);
        }
        return U;
    }
};

void saveDataToFile(const vector<vector<vector<vector<double>>>>& u, const string& filename) {
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cerr << "Cannot Open File: " << filename << endl;
        return;
    }
    outFile << fixed << setprecision(6);

    int timeSteps = u.size();
    int xSize = u[0].size();
    int ySize = u[0][0].size();
    int zSize = u[0][0][0].size();

    outFile << timeSteps << " " << xSize << " " << ySize << " " << zSize << endl;

    // 输出 x 坐标 (网格中心)
    for (int x = 0; x < xSize; ++x) {
        outFile << (x + 0.5) * (1.0 / xSize);
        if (x < xSize - 1) outFile << " ";
    }
    outFile << endl;

    // 输出 y 坐标
    for (int y = 0; y < ySize; ++y) {
        outFile << (y + 0.5) * (1.0 / ySize);
        if (y < ySize - 1) outFile << " ";
    }
    outFile << endl;

    // 输出 z 坐标
    for (int z = 0; z < zSize; ++z) {
        outFile << (z + 0.5) * (1.0 / zSize);
        if (z < zSize - 1) outFile << " ";
    }
    outFile << endl;

    for (int t = 0; t < timeSteps; ++t) {
        outFile << "t=" << t << endl;
        for (int x = 0; x < xSize; ++x) {
            for (int y = 0; y < ySize; ++y) {
                for (int z = 0; z < zSize; ++z) {
                    outFile << u[t][x][y][z];
                    if (z < zSize - 1) outFile << " ";
                }
                outFile << endl;
            }
        }
    }
    outFile.close();
    cout << "Data Saved to: " << filename << endl;
}

int main() {
    int desired_threads = 16;
    omp_set_num_threads(desired_threads);

    double dt = 0.01;          // 时间步长
    int Nx = 64;               // 网格点数（建议为2的幂）
    int Ny = 64;
    int Nz = 64;
    int Nt = 50;               // 时间步数
    double ep = 0.05;

    allen_cahn_equation_admm_staggered_3d allen_cahn_u(dt, Nt, Nx, Ny, Nz, ep);

    clock_t start = clock();
    allen_cahn_u.solve();
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    ofstream history_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/history/history_ccfd_admm_parallel_3d.txt", ios::app);
    history_file << "cpu_time_used = " << cpu_time_used << " seconds." << endl << endl;
    history_file.close();

    auto U = allen_cahn_u.getU();
    saveDataToFile(U, "H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/data/data_ccfd_admm_parallel_3d.txt");

    cout << "admm_staggered 3D CPU time: " << cpu_time_used << " s." << endl;
    return 0;
}