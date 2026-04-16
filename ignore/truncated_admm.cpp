#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <omp.h>

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

double F(double t, double theta){
    t = max(EPS, min(1.0 - EPS, t));
    return t*log(t) + (1-t)*log(1-t) + theta*t*(1-t);
}

double safe_log_ratio(double u) {
    u = max(EPS, min(1.0 - EPS, u));
    return log(u / (1 - u));
}

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

class allen_cahn_equation_admm {
private:
    int Nt;
    double dt;
    int Nx;
    double dx;
    double dx2;
    int Ny;
    double dy;
    double dy2;
    double dxdy;
    int N;
    double ep;
    double ep2;
    double theta = 4.0;
    vector<vector<double>> u;
    vector<vector<vector<double>>> U;
    vector<double> Energy;

public:
    allen_cahn_equation_admm(double time, int time_steps, int grid_x, int grid_y, double epsilon)
        : dt(time), Nt(time_steps), Nx(grid_x), Ny(grid_y), ep(epsilon) {
        
        dx = 2.0 / Nx;
        dx2 = dx * dx;
        dy = 2.0 / Ny;
        dy2 = dy * dy;
        dxdy = dx * dy;
        ep2 = ep * ep;
        N = Nx * Ny;
        
        u.resize(Nt + 1, vector<double>(N, 0.0));
        U.resize(Nt + 1, vector<vector<double>>(Nx, vector<double>(Ny, 0.0)));
        Energy.resize(Nt + 1, 0.0);
        
        #pragma omp parallel for collapse(2) schedule(static)
        for(int i = 0; i < Nx; i++){
            for(int j = 0; j < Ny; j++){
                double x = (i + 0.5)* dx;
                double y = (j + 0.5)* dy;
                int idx = i * Ny + j;
                u[0][idx] = ini_3(x, y);  
            }
        }
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
        #pragma omp parallel for collapse(2) reduction(+:r_norm) schedule(static)
        for(int i = 0; i < Nx; i++){
            for(int j = 0; j < Ny; j++){
                int ip = (i + 1) % Nx;
                int im = (i - 1 + Nx) % Nx;
                int jp = (j + 1) % Ny;
                int jm = (j - 1 + Ny) % Ny;
                int idx = i * Ny + j;
                int idx_ip = ip * Ny + j;
                int idx_im = im * Ny + j;
                int idx_jp = i * Ny + jp;
                int idx_jm = i * Ny + jm;

                double laplace = (u_new[idx_ip] + u_new[idx_jp] - 4.0 * u_new[idx] + u_new[idx_im] + u_new[idx_jm]) / dx2;

                double r = (u_new[idx] - u_n[idx]) / dt - ep2 * laplace + safe_log_ratio(u_new[idx]) + theta * (1.0 - 2.0 * u_n[idx]);
                r_norm += r * r;
            }
        }
        r_norm = sqrt(r_norm);
        return r_norm;
    }
    double energy(const vector<double>& U_vec){
        double E = 0.0;
      
        #pragma omp parallel for collapse(2) reduction(+:E) schedule(static)
        for(int i = 0; i < Nx; i++){
            for(int j = 0; j < Ny; j++){
                int ip = (i + 1) % Nx;
                int im = (i - 1 + Nx) % Nx;
                int jp = (j + 1) % Ny;
                int jm = (j - 1 + Ny) % Ny;
                int idx = i * Ny + j;
                int idx_ip = ip * Ny + j;
                int idx_im = im * Ny + j;
                int idx_jp = i * Ny + jp;
                int idx_jm = i * Ny + jm;
                
                double gx = (U_vec[idx_ip] - U_vec[idx_im]) / (2 * dx);
                double gy = (U_vec[idx_jp] - U_vec[idx_jm]) / (2 * dy);
                E += dxdy * (0.5 * ep2 * (gx * gx + gy * gy) + F(U_vec[idx], theta));
            }
        }
        return E;
    }

    vector<double> mat_vec_product(const vector<double>& U_vec, double rho){
        vector<double> AU(N, 0.0);

        #pragma omp parallel for collapse(2) schedule(static)
        for(int i = 0; i < Nx; i++){
            for(int j = 0; j < Ny; j++){
                int idx = i * Ny + j;
                AU[idx] = (1.0 / dt + rho) * U_vec[idx];
                
                int ip = (i + 1) % Nx;
                int im = (i - 1 + Nx) % Nx;
                int jp = (j + 1) % Ny;
                int jm = (j - 1 + Ny) % Ny;
                int idx_ip = ip * Ny + j;
                int idx_im = im * Ny + j;
                int idx_jp = i * Ny + jp;
                int idx_jm = i * Ny + jm;
                
                AU[idx] -= ep2 * ((U_vec[idx_ip] - 2 * U_vec[idx] + U_vec[idx_im]) / dx2 
                                + (U_vec[idx_jp] - 2 * U_vec[idx] + U_vec[idx_jm]) / dy2);
            }
        }
        return AU;
    }
    
    vector<double> rhs(const vector<double>& Un, const vector<double>& Y, double rho, const vector<double>& U2){
        vector<double> b(N, 0.0);
        
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < N; i++){
            b[i] = Un[i] / dt - Y[i] + rho * U2[i];
        }
        return b;
    }

    vector<double> conjugate_gradient(const vector<double>& b, const vector<double>& x0, double tol, double rho, int max_iter){
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
        double u = u2_old;  
        double fu, dfu;
        
        for(int iter = 1; iter <= 50; iter++){
            u = max(EPS, min(1.0 - EPS, u));
            
            fu = safe_log_ratio(u) + theta * (1 - 2 * un) - y - rho * (u1 - u);
            dfu = 1.0 / (u * (1 - u)) + rho;
            
            if (fabs(dfu) < 1e-14) {
                u = max(EPS, min(1.0 - EPS, u + 0.001 * (0.5 - u)));
                continue;
            }
            
            double u_new = u - fu / dfu;
            
            double max_change = 0.1;
            if (fabs(u_new - u) > max_change) {
                u_new = u + copysign(max_change, u_new - u);
            }
            
            u_new = max(EPS, min(1.0 - EPS, u_new));
            
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
        u2 = max(EPS, min(1.0 - EPS, u2));
        return safe_log_ratio(u2) + theta * (1 - 2 * un) - y - rho * (u1 - u2);
    }

    vector<double> admm(vector<double>& Un, int max_iter, double tolerance){
        double rho = 10.0;
        double tau = 1.8;
        
        double mu = 10.0;
        double gamma = 2.0;

        auto U_1 = Un;
        auto U_2 = Un;
        auto U_2_new = Un;
        vector<double> Y(N, 0.0);
        
        double prev_energy = energy(Un);
        
        for(int k = 1; k <= max_iter; k++){
            // CG to solve U1 
            auto b = rhs(Un, Y, rho, U_2);
            U_1 = conjugate_gradient(b, U_1, 1e-8, rho, 1000);
            
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; i++) {
                U_1[i] = max(EPS, min(1.0 - EPS, U_1[i]));
            }
            
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
            cout << "WARNING: Energy increased from " << prev_energy << " to " << new_energy << endl;
        }
        
        return Un;
    }
    
    void solve(){
        ofstream history_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/history/history_truncated_admm_parallel.txt", ios::app);
        
        history_file << "Begin: dt = " << dt << ", Nx = Ny = " << Nx << ", Nt = " << Nt << ", epsilon = " << ep << "." << endl;

        int num_threads = omp_get_max_threads();
        cout << "Using " << num_threads << " threads" << endl;
        
        for(int n = 0; n < Nt; n++){
            auto Un = u[n];
            Energy[n] = energy(Un);
            
            history_file << "time step " << n << "/" << Nt << ", energy = " << Energy[n];
            
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
        history_file << "time step " << Nt << "/" << Nt << ", energy = " << Energy[Nt]<<endl;
        history_file.close();
    }
    
    vector<vector<double>> vec_to_arr(const vector<double>& a){
        vector<vector<double>> A(Nx, vector<double>(Ny, 0.0));
        #pragma omp parallel for collapse(2) schedule(static)
        for(int i = 0; i < Nx; i++){
            for(int j = 0; j < Ny; j++){
                int idx = i * Ny + j;
                A[i][j] = a[idx];
            }
        }
        return A;
    }
    
    const vector<vector<double>>& getu() const{
        return u;
    }
    
    vector<vector<vector<double>>>& getU(){
        #pragma omp parallel for schedule(static)
        for(int n = 0; n <= Nt; n++){
            U[n] = vec_to_arr(u[n]);
        }
        return U;
    }
};

int main(){

    int desired_threads = 16;  
    omp_set_num_threads(desired_threads);
    
    double dt = 1e10;  
    int Nx = 1024;
    int Ny = 1024;
    int Nt = 1;      
    double ep = 0.05;
    
    allen_cahn_equation_admm allen_cahn_u(dt, Nt, Nx, Ny, ep);

    clock_t start = clock();
    allen_cahn_u.solve();
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    ofstream history_file("H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/history/history_truncated_admm_parallel.txt", ios::app);
    history_file << "cpu_time_used = " << cpu_time_used << " seconds." << endl << endl;
    history_file.close();

    auto U = allen_cahn_u.getU();
    saveDataToFile(U, "H:/undergraduate/scientific_research/allen_cahn_equation_simulation/results/data/data_truncated_admm.txt");
    
    cout << "truncated_admm_parallel CPU time: " << cpu_time_used << " s." << endl;
    
    
    return 0;
}