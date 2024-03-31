#include <iostream>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <chrono>

#include "common.h"
#include "happly.h"


void fill_cube(std::vector<particle_t>& parts, physics_t& begin, physics_t& end, physics_t& velocity) {
    double x_len = abs(begin.x - end.x);
    idx_t num_x = floor(x_len / particle_radius);
    double x_stride = x_len / (num_x + 1);

    double y_len = abs(begin.y - end.y);
    idx_t num_y = floor(y_len / particle_radius);
    double y_stride = y_len / (num_y + 1);

    double z_len = abs(begin.z - end.z);
    idx_t num_z = floor(z_len / particle_radius);
    double z_stride = z_len / (num_z + 1);

    idx_t num_parts = num_x * num_y * num_z;
    parts.reserve(parts.size() + num_parts);

    physics_t acc = {0,0,0};
    for (idx_t x_idx = 0; x_idx < num_x; ++x_idx) {
        for (idx_t y_idx = 0; y_idx < num_y; ++y_idx) {
            for (idx_t z_idx = 0; z_idx < num_z; ++z_idx) {
                physics_t pos = {x_idx * x_stride, y_idx * y_stride, z_idx * z_stride};
                parts.emplace_back(pos, velocity, acc);
            }
        }
    }
}


// return number of particles
void init_particles(std::vector<particle_t>& parts) {
    physics_t begin = {tank_size, tank_size, tank_size};
    double tmp = tank_size * 3 / 4;
    physics_t end = {tmp, tmp, tmp};
    physics_t vel = {0,0,0};
    fill_cube(parts, begin, end, vel);
}


void write_ply(const std::vector<particle_t>& parts, const std::string& filename) {

}


int main(int argc, char** argv) {

    std::vector<particle_t> parts;
    init_particles(parts);


    idx_t num_parts = parts.size();

    particle_t* parts_gpu;
    cudaMalloc((void**)&parts_gpu, num_parts * sizeof(particle_t));
    cudaMemcpy(parts_gpu, parts.data(), num_parts * sizeof(particle_t), cudaMemcpyHostToDevice);

    std::string file_prefix = "./PLY/SPH_FLUID_";
    idx_t ply_idx = 0;

    // begin time counting
    auto start_time = std::chrono::steady_clock::now();

    init_simul(parts_gpu, num_parts);

    for (idx_t step = 0; step < num_steps; ++ step) {
        simul_one_step(parts_gpu, num_parts);
        cudaDeviceSynchronize();

        if (step % check_steps == 0) {
            cudaMemcpy(parts.data(), parts_gpu, num_parts * sizeof(particle_t), cudaMemcpyDeviceToHost);
            std::string filename = file_prefix + std::to_string(ply_idx++);
            write_ply(parts, filename);
        }
    }

    clear_simul();
    cudaDeviceSynchronize();

    auto end_time = std::chrono::steady_clock::now();
    // end time counting

    std::chrono::duration<double> diff_time = end_time - start_time;
    double seconds = diff_time.count();

    std::cout << "Simulation TIme = " << seconds << " seconds for " << num_parts << " particles.\n";
    cudaFree(parts_gpu);

    return 0;
}