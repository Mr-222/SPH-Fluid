#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

#include "common.h"
#include "happly.h"


void add_boundaries(std::vector<particle_t>& parts) {
    // Add two layers in each 6 faces of the tank
    int num_per_axis = ceil(tank_size / (2 * particle_radius));
    parts.reserve(num_per_axis * num_per_axis * 12);

    enum class Axis {
        X_AXIS = 0,
        Y_AXIS = 1,
        Z_AXIS = 2
    };

    auto layer_boundary_particles = [&parts](int x_num, int y_num, int z_num, Axis opposite_axis) {
        for (int i = 0; i < x_num; ++i) {
            for (int j = 0; j < y_num; ++j) {
                for (int k = 0; k < z_num; ++k) {
                    Vector3f pos = {2 * particle_radius * static_cast<float>(i) + particle_radius,
                                    2 * particle_radius * static_cast<float>(j) + particle_radius,
                                    2 * particle_radius * static_cast<float>(k) + particle_radius};
                    parts.push_back({ pos, {0, 0, 0}, {0, 0, 0}, density_0, 0.0, false });

                    pos[(int)opposite_axis] = tank_size - pos[(int)opposite_axis];
                    parts.push_back({ pos, {0, 0, 0}, {0, 0, 0}, density_0, 0.0, false });
                }
            }
        }
    };

    // bottom and top face
    layer_boundary_particles(num_per_axis, num_per_axis, 2, Axis::Z_AXIS);

    // front and back face
    layer_boundary_particles(num_per_axis, 2, num_per_axis, Axis::Y_AXIS);

    // left and right face
    layer_boundary_particles(2, num_per_axis, num_per_axis, Axis::X_AXIS);
}


void fill_cube(std::vector<particle_t>& parts, const Vector3f& lower_corner, const Vector3f& cube_size, const Vector3f& velocity) {
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
                Vector3f acceleration = {0, 0, 0};
                parts.emplace_back(pos, velocity, acceleration, density_0, 0.0, true);
            }
        }
    }
}

void init_particles(std::vector<particle_t>& parts) {
    add_boundaries(parts);
    fill_cube(parts, {2, 2, 3}, {6, 6, 6}, {0, 0, 0});
}

void save_point_cloud_data(const std::vector<particle_t>& parts, const std::string& path) {
    happly::PLYData plyOut;
    std::vector<std::array<double, 3>> points;
    points.reserve(parts.size());

    for (const auto& part : parts) {
        if (!part.is_fluid)
            continue;
        points.push_back({part.pos.x, part.pos.y, part.pos.z});
    }
    plyOut.addVertexPositions(points);

    plyOut.write(path, happly::DataFormat::ASCII);
}

int main() {

    std::vector<particle_t> parts;
    init_particles(parts);

    auto num_parts = static_cast<idx_t>(parts.size());

    particle_t* parts_gpu;
    cudaMalloc(&parts_gpu, num_parts * sizeof(particle_t));
    cudaMemcpy(parts_gpu, parts.data(), num_parts * sizeof(particle_t), cudaMemcpyHostToDevice);

    particle_t* parts_gpu_sorted;
    cudaMalloc(&parts_gpu_sorted, num_parts * sizeof(particle_t));

    std::string file_prefix = "../point_cloud_data/";

    auto start_time = std::chrono::steady_clock::now();

    init_simul(num_parts);

    int frame_number = 0;
    for (idx_t step = 0; step < num_steps; ++step) {
        simul_one_step(parts_gpu, num_parts, parts_gpu_sorted);

        if (write_to_file && (step % check_steps == 0 || step == num_steps - 1)) {
            cudaMemcpy(parts.data(), parts_gpu, num_parts * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save_point_cloud_data(parts, file_prefix + std::to_string(frame_number) + ".ply");
            frame_number++;
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