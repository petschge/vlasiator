#pragma once

// Minimal 3-D array (row-major, shape Nx × Ny × Nz) indexed as A(i,j k)  where
// i is the x-index (0..Nx-1), j is the y-index (0..Ny-1) and k is the z index
// (0..Ny-1).
struct Array3D {
    int Nx, Ny, Nz;
    std::vector<double> data;

    Array3D() : Nx(0), Ny(0), Nz(0) {}
    Array3D(int nx, int ny, int nz, double val = 0.0)
        : Nx(nx), Ny(ny), Nz(nz), data(nx*ny*nz, val) {}

    double& operator()(int i, int j, int k)       { return data[i*Ny*Nz + j*Nz + k]; }
    double  operator()(int i, int j, int k) const { return data[i*Ny*Nz + j*Nz + k]; }

    void fill(double v) { std::fill(data.begin(), data.end(), v); }

    double max_abs() const {
        double m = 0.0;
        for (double v : data) m = std::max(m, std::abs(v));
        return m;
    }
};


struct ComplexArray3D {
    int Nx, Ny, Nz;
    std::vector<fftw_complex> data;

    ComplexArray3D() : Nx(0), Ny(0), Nz(0) {}

    ComplexArray3D(int nx, int ny, int nz)
        : Nx(nx), Ny(ny), Nz(nz), data(nx * ny * nz) {}

    fftw_complex& operator()(int i, int j, int k) {
        return data[i * Ny * Nz + j * Nz + k];
    }

    const fftw_complex& operator()(int i, int j, int k) const {
        return data[i * Ny * Nz + j * Nz + k];
    }

    void fill(double re, double im = 0.0) {
        for (auto &z : data) {
            z[0] = re;
            z[1] = im;
        }
    }

    fftw_complex* ptr() { return data.data(); }
    const fftw_complex* ptr() const { return data.data(); }
};
