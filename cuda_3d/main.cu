#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

#include "common.h"
#include "happly.h"


void fill_cube(std::vector<particle_t>& parts, const Vector3d& lower_corner, const Vector3d& cube_size) {
    int x_num = floor(cube_size.x / (2 * particle_radius));
    int y_num = floor(cube_size.y / (2 * particle_radius));
    int z_num = floor(cube_size.z / (2 * particle_radius));

    parts.reserve(parts.size() + x_num * y_num * z_num);
    for (int i = 0; i < x_num; ++i) {
        for (int j = 0; j < y_num; ++j) {
            for (int k = 0; k < z_num; ++k) {
                Vector3d pos = {lower_corner.x + 2 * particle_radius * i + particle_radius,
                                lower_corner.y + 2 * particle_radius * j + particle_radius,
                                lower_corner.z + 2 * particle_radius * k + particle_radius};
                Vector3d velocity = {0, 0, 0};
                Vector3d acceleration = {0, 0, 0};
                parts.emplace_back(pos, velocity, acceleration);
            }
        }
    }
}


void init_particles(std::vector<particle_t>& parts) {
    fill_cube(parts, {0, 40, 0}, {30, 30, 30});
}


void write_ply(const std::vector<particle_t>& parts, const std::string& filename) {

}


int main(int argc, char** argv) {

    std::vector<particle_t> parts;
    init_particles(parts);

    auto num_parts = static_cast<idx_t>(parts.size());

    particle_t* parts_gpu;
    cudaMalloc(&parts_gpu, num_parts * sizeof(particle_t));
    cudaMemcpy(parts_gpu, parts.data(), num_parts * sizeof(particle_t), cudaMemcpyHostToDevice);

    std::string file_prefix = "./PLY/SPH_FLUID_";
    idx_t ply_idx = 0;

    auto start_time = std::chrono::steady_clock::now();

    init_simul(parts_gpu, num_parts);

    for (idx_t step = 0; step < num_steps; ++step) {
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

    std::chrono::duration<double> diff_time = end_time - start_time;
    double seconds = diff_time.count();

    std::cout << "Simulation TIme = " << seconds << " seconds for " << num_parts << " particles.\n";
    cudaFree(parts_gpu);

    return 0;
}