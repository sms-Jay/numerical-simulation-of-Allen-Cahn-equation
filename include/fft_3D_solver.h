// fft_3d_solver.h
#ifndef FFT_3D_SOLVER_H
#define FFT_3D_SOLVER_H

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <omp.h>

class FFT3DSolver {
private:
    int Nx, Ny, Nz, N;
    double dx, dy, dz, dx2, dy2, dz2;
    std::vector<std::complex<double>> cached_laplace_eigenvalues;
    bool eigenvalues_cached;

    const double PI = 3.1415926535897932384626;

public:
    FFT3DSolver(int grid_x, int grid_y, int grid_z, double delta_x, double delta_y, double delta_z)
        : Nx(grid_x), Ny(grid_y), Nz(grid_z), dx(delta_x), dy(delta_y), dz(delta_z) {
        N = Nx * Ny * Nz;
        dx2 = dx * dx;
        dy2 = dy * dy;
        dz2 = dz * dz;
        eigenvalues_cached = false;
    }

    void precompute_laplace_eigenvalues() {
        if (eigenvalues_cached) return;

        cached_laplace_eigenvalues.resize(N);

        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    int idx = (i * Ny + j) * Nz + k;

                    int kx = (i <= Nx/2) ? i : i - Nx;
                    int ky = (j <= Ny/2) ? j : j - Ny;
                    int kz = (k <= Nz/2) ? k : k - Nz;

                    double eigen = -2.0 * (
                        (1.0 - std::cos(2.0 * PI * kx / Nx)) / dx2 +
                        (1.0 - std::cos(2.0 * PI * ky / Ny)) / dy2 +
                        (1.0 - std::cos(2.0 * PI * kz / Nz)) / dz2
                    );
                    cached_laplace_eigenvalues[idx] = std::complex<double>(eigen, 0.0);
                }
            }
        }

        eigenvalues_cached = true;
    }

    const std::vector<std::complex<double>>& get_laplace_eigenvalues() {
        precompute_laplace_eigenvalues();
        return cached_laplace_eigenvalues;
    }

    void fft_1d(std::vector<std::complex<double>>& x, bool inverse) const {
        int n = x.size();
        if (n <= 1) return;

        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j >= bit; bit >>= 1) j -= bit;
            j += bit;
            if (i < j) std::swap(x[i], x[j]);
        }

        for (int len = 2; len <= n; len <<= 1) {
            double angle = 2 * PI / len * (inverse ? 1 : -1);
            std::complex<double> wlen(std::cos(angle), std::sin(angle));
            for (int i = 0; i < n; i += len) {
                std::complex<double> w(1);
                for (int j = 0; j < len/2; j++) {
                    std::complex<double> u = x[i + j];
                    std::complex<double> v = w * x[i + j + len/2];
                    x[i + j] = u + v;
                    x[i + j + len/2] = u - v;
                    w *= wlen;
                }
            }
        }

        if (inverse) {
            for (int i = 0; i < n; i++) x[i] /= n;
        }
    }

    std::vector<std::complex<double>> fft_3d(const std::vector<std::complex<double>>& input, bool inverse) const {
        std::vector<std::complex<double>> output(N);
        std::vector<std::complex<double>> temp(N);

        // FFT along x (i dimension)
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                std::vector<std::complex<double>> vec(Nx);
                for (int i = 0; i < Nx; i++) {
                    int idx = (i * Ny + j) * Nz + k;
                    vec[i] = input[idx];
                }
                fft_1d(vec, inverse);
                for (int i = 0; i < Nx; i++) {
                    int idx = (i * Ny + j) * Nz + k;
                    temp[idx] = vec[i];
                }
            }
        }

        // FFT along y (j dimension)
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < Nx; i++) {
            for (int k = 0; k < Nz; k++) {
                std::vector<std::complex<double>> vec(Ny);
                for (int j = 0; j < Ny; j++) {
                    int idx = (i * Ny + j) * Nz + k;
                    vec[j] = temp[idx];
                }
                fft_1d(vec, inverse);
                for (int j = 0; j < Ny; j++) {
                    int idx = (i * Ny + j) * Nz + k;
                    output[idx] = vec[j];
                }
            }
        }

        // FFT along z (k dimension)
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                std::vector<std::complex<double>> vec(Nz);
                for (int k = 0; k < Nz; k++) {
                    int idx = (i * Ny + j) * Nz + k;
                    vec[k] = output[idx];
                }
                fft_1d(vec, inverse);
                for (int k = 0; k < Nz; k++) {
                    int idx = (i * Ny + j) * Nz + k;
                    output[idx] = vec[k];
                }
            }
        }

        return output;
    }

    std::vector<std::complex<double>> fft_3d_real(const std::vector<double>& input, bool inverse) const {
        std::vector<std::complex<double>> complex_input(N);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) complex_input[i] = std::complex<double>(input[i], 0.0);
        return fft_3d(complex_input, inverse);
    }

    std::vector<double> solve_linear_system(const std::vector<double>& b, double alpha, double beta = 1.0) {
        precompute_laplace_eigenvalues();

        auto b_freq = fft_3d_real(b, false);
        std::vector<std::complex<double>> x_freq(N);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            double system_eigenvalue = alpha - beta * cached_laplace_eigenvalues[i].real();
            x_freq[i] = (std::abs(system_eigenvalue) > 1e-12) ? b_freq[i] / system_eigenvalue : 0.0;
        }

        auto x_complex = fft_3d(x_freq, true);
        std::vector<double> x(N);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) x[i] = x_complex[i].real();

        return x;
    }

    int get_total_size() const { return N; }
    int get_nx() const { return Nx; }
    int get_ny() const { return Ny; }
    int get_nz() const { return Nz; }

    void clear_cache() {
        cached_laplace_eigenvalues.clear();
        eigenvalues_cached = false;
    }
};

#endif // FFT_3D_SOLVER_H