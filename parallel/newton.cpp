#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <omp.h>
#include <complex>

using namespace std;

const double PI = 3.1415926535897932384626;
const double EPS = 1e-24;

double ini_1(double x, double y){
    return (1+sin(2*PI*x)*sin(2*PI*y))/3.0;
}

double ini_2(double x, double y){
    if(abs(x-1)<=0.35 && abs(y-1)<=0.35) return 1e-5;
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

double safe_log_gradient(double u) {
    u = max(EPS, min(1.0 - EPS, u));
    return 1.0/(u*(1-u));
}

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

class allen_cahn_equation_newton_staggered {
private:
    int Nt;
    double dt;
    double ep;
    double ep2;
    double theta = 3.0;
    
    StaggeredGrid grid;
    int Nx, Ny, N;
    double dx, dy, dx2, dxdy;
    
    vector<vector<double>> u;  
    vector<vector<vector<double>>> U;  
    vector<double> Energy;

public:
    allen_cahn_equation_newton_staggered(double time, int time_steps, 
                                        int grid_x, int grid_y, double epsilon)
        : dt(time), Nt(time_steps), ep(epsilon), 
          grid(grid_x, grid_y, 2.0, 2.0) {
        
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
        
        // #pragma omp parallel for collapse(2) schedule(static)
        for(int i = 0; i < Nx; i++){
            for(int j = 0; j < Ny; j++){
                double x = (i + 0.5) * dx;
                double y = (j + 0.5) * dy;
                int idx = grid.idx_center(i, j);
                u[0][idx] = ini_3(x, y);
            }
        }
    }

    double vec_dot(const vector<double>& a, const vector<double>& b){
        double result = 0.0;
        #pragma omp parallel for reduction(+:result) schedule(static)
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

    vector<double> rhs(const vector<double>& u_n){
        vector<double> b(N, 0.0);
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < N; i++){
            b[i] = (1.0/dt) * u_n[i] + theta * (2 * u_n[i] - 1);
        }
        return b;
    }
    
    vector<double> rhs_tilde(const vector<double>& u_k, const vector<double>& b){
        vector<double> b_tilde(N, 0.0);
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < N; i++){
            b_tilde[i] += safe_log_gradient(u_k[i]) * u_k[i];
            b_tilde[i] -= safe_log_ratio(u_k[i]);
            b_tilde[i] += b[i];
        }
        return b_tilde;
    }
    
    vector<double> mat_vec_product(const vector<double>& v, const vector<double>& u_k){
        vector<double> Av(N, 0.0);
        
        // Laplace part
        auto lap = grid.laplacian_staggered(v);
        
        #pragma omp parallel for schedule(static)
        for(int idx = 0; idx < N; idx++){
            Av[idx] = (1.0/dt) * v[idx] - ep2 * lap[idx];
            Av[idx] += safe_log_gradient(u_k[idx]) * v[idx];
        }
        
        return Av;
    }

    vector<double> conjugate_gradient(const vector<double>& u_k, const vector<double>& b, 
                                     double tol, int max_iter){
        vector<double> x = u_k;
        vector<double> r = b;
        vector<double> p = r;
        
        auto Ax0 = mat_vec_product(x, u_k);
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < N; i++){
            r[i] = b[i] - Ax0[i];
            p[i] = r[i];
        }
        
        double b_norm = sqrt(vec_dot(b, b));
        if (b_norm < 1e-10) b_norm = 1.0;
        double rold = vec_dot(r, r);
        
        for (int k = 1; k <= max_iter; k++){
            auto Ap = mat_vec_product(p, u_k);
            double pAp = vec_dot(p, Ap);
            
            if (fabs(pAp) < 1e-30) {
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
        }
        return x;
    }

    vector<double> newton(const vector<double>& u_n, double tol, int max_iter){
        auto b = rhs(u_n);
        auto u_k = u_n;
        
        for(int k = 1; k <= max_iter; k++){
            auto b_tilde = rhs_tilde(u_k, b);
            auto u_new = conjugate_gradient(u_k, b_tilde, 1e-8, 10000);
            
            double r_norm = 0.0;
            #pragma omp parallel for reduction(+:r_norm) schedule(static)
            for(int i = 0; i < N; i++){
                // u_new[i] = max(EPS, min(1.0 - EPS, u_new[i]));
                r_norm += (u_new[i] - u_k[i]) * (u_new[i] - u_k[i]);
            }
            r_norm = sqrt(r_norm);
            
            if(r_norm < tol){
                cout << "Newton converged at " << k << " iterations." << endl;
                return u_new;
            } 

            u_k = u_new;
        }
        
        cout << "Newton not converged after " << max_iter << " iterations." << endl;
        return u_k;
    }

    void solve(){
        ofstream history_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/history/history_ccfd_newton_parallel.txt", ios::app);
        
        history_file << "Begin: dt = " << dt << ", Nx = " << Nx 
                     << ", Ny = " << Ny << ", Nt = " << Nt 
                     << ", epsilon = " << ep << "." << endl;

        int num_threads = omp_get_max_threads();
        cout << "Using " << num_threads << " threads for Newton method (staggered grid)" << endl;
        
        for(int n = 0; n < Nt; n++){
            auto u_n = u[n];
            Energy[n] = energy(u_n);
            
            history_file << "time step " << n << "/" << Nt 
                         << ", energy = " << Energy[n];

            u[n+1] = newton(u_n, 1e-6, 1e4);
            
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
    
    double dt = 1;  
    int Nx = 1024;
    int Ny = 1024;
    int Nt = 50;      
    double ep = 0.05;
    
    allen_cahn_equation_newton_staggered allen_cahn_u(dt, Nt, Nx, Ny, ep);
    
    clock_t start = clock();
    allen_cahn_u.solve();
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    ofstream history_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/history/history_ccfd_newton_parallel.txt", ios::app);
    history_file << "cpu_time_used = " << cpu_time_used << " seconds." << endl << endl;
    history_file.close();

    auto U = allen_cahn_u.getU();
    saveDataToFile(U, "H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/data/data_ccfd_newton_parallel.txt");
    
    cout << "newton_staggered CPU time: " << cpu_time_used << " s." << endl;
    
    return 0;
}