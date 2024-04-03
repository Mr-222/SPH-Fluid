#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

#include "common.h"
#include "happly.h"


void fill_cube(std::vector<particle_t>& parts, const Vector3f& lower_corner, const Vector3f& cube_size) {
    int x_num = floor(cube_size.x / (2 * particle_radius));
    int y_num = floor(cube_size.y / (2 * particle_radius));
    int z_num = floor(cube_size.z / (2 * particle_radius));

    parts.reserve(parts.size() + x_num * y_num * z_num);
    for (int i = 0; i < x_num; ++i) {
        for (int j = 0; j < y_num; ++j) {
            for (int k = 0; k < z_num; ++k) {
                Vector3f pos = {lower_corner.x + 2 * particle_radius * static_cast<float>(i) + particle_radius,
                                lower_corner.y + 2 * particle_radius * static_cast<float>(j) + particle_radius,
                                lower_corner.z + 2 * particle_radius * static_cast<float>(k) + particle_radius};
                Vector3f velocity = {0, 0, 0};
                Vector3f acceleration = {0, 0, 0};
                parts.emplace_back(pos, velocity, acceleration, 1000.0, 0.0, true);
            }
        }
    }
}

void init_particles(std::vector<particle_t>& parts) {
    fill_cube(parts, {0, 2, 0}, {10, 10, 10});
}

void save_point_cloud_data(const std::vector<particle_t>& parts, const std::string& path) {
    happly::PLYData plyOut;
    std::vector<std::array<double, 3>> points;

    for (const auto& part : parts)
        points.push_back({part.pos.x, part.pos.y, part.pos.z});
    plyOut.addVertexPositions(points);

    plyOut.write(path, happly::DataFormat::ASCII);
}

int main(int argc, char** argv) {

    std::vector<particle_t> parts;
    init_particles(parts);

    auto num_parts = static_cast<idx_t>(parts.size());

    particle_t* parts_gpu;
    cudaMalloc(&parts_gpu, num_parts * sizeof(particle_t));
    cudaMemcpy(parts_gpu, parts.data(), num_parts * sizeof(particle_t), cudaMemcpyHostToDevice);

    std::string file_prefix = "../point_cloud_data/";

    auto start_time = std::chrono::steady_clock::now();

    init_simul(parts_gpu, num_parts);

    for (idx_t step = 0; step < num_steps; ++step) {
        simul_one_step(parts_gpu, num_parts);
        cudaMemcpy(parts.data(), parts_gpu, num_parts * sizeof(particle_t), cudaMemcpyDeviceToHost);
        save_point_cloud_data(parts, file_prefix + std::to_string(step) + ".ply");
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