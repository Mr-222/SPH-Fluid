#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <cassert>

#include "common.h"
#include "happly.h"
#include "llrq.h"


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

void save_point_cloud_data(particle_t* parts, idx_t num_parts, const std::string& path) {
    happly::PLYData plyOut;
    std::vector<std::array<double, 3>> points;
    points.reserve(num_parts);

    for (idx_t i = 0; i < num_parts; ++i) {
        particle_t& part = parts[i];
        if (!part.is_fluid)
            continue;
        points.push_back({part.pos.x, part.pos.y, part.pos.z});
    }
    plyOut.addVertexPositions(points);

    plyOut.write(path, happly::DataFormat::ASCII);
}


LLRQ<std::pair<idx_t, particle_t*>> buff_to_read(gpu_buffer_num);
LLRQ<particle_t*> buff_to_write(gpu_buffer_num);

std::string file_prefix = "../point_cloud_data/";

volatile bool gpu_running = true;

void save_thread_fct(idx_t num_parts) {
    particle_t* buffer = new particle_t[num_parts];
    std::pair<idx_t, particle_t*> to_read_pair;

    while (true) {
        if (buff_to_read.pop(to_read_pair)) {
            idx_t frame_number = to_read_pair.first;
            particle_t* parts_to_save = to_read_pair.second;
            cudaMemcpy(buffer, parts_to_save, num_parts * sizeof(particle_t), cudaMemcpyDeviceToHost);
            buff_to_write.push(parts_to_save);

            save_point_cloud_data(buffer, num_parts, file_prefix + std::to_string(frame_number) + ".ply");
        }
        else if (!gpu_running){
            break;
        }
    }
    delete[] buffer;
}

int main() {

    std::vector<particle_t> parts;
    init_particles(parts);

    auto num_parts = static_cast<idx_t>(parts.size());
    idx_t frame_number = 0;
    particle_t* last_output_buffer, *output_buffer;

    particle_t* gpu_buffer;
    cudaMalloc((void**)&gpu_buffer, num_parts * sizeof(particle_t));
    cudaMemcpy(gpu_buffer, parts.data(), num_parts * sizeof(particle_t), cudaMemcpyHostToDevice);
    last_output_buffer = gpu_buffer;
    bool success = buff_to_read.push(std::make_pair(frame_number++, gpu_buffer));
    assert((success == true));
    for (idx_t i = 1; i < gpu_buffer_num; ++i) {
        cudaMalloc((void**)&gpu_buffer, num_parts * sizeof(particle_t));
        bool success = buff_to_write.push(gpu_buffer);
        assert((success == true));
    }

    auto start_time = std::chrono::steady_clock::now();

    std::thread thread_arr[save_thread_num];
    for (idx_t i = 0; i < save_thread_num; ++i) {
        thread_arr[i] = std::thread(save_thread_fct, num_parts);
    }

    init_simul(num_parts);

    success = buff_to_write.pop(output_buffer);
    assert((success == true));
    for (idx_t step = 0; step < num_steps; ++step) {
        
        assert((output_buffer != last_output_buffer));
        simul_one_step(last_output_buffer, num_parts, output_buffer);

        last_output_buffer = output_buffer;
        while (!buff_to_write.pop(output_buffer));

        if (write_to_file && (step % check_steps == 0 || step == num_steps - 1)) {
            while (!buff_to_read.push(std::make_pair(frame_number++, last_output_buffer)));
        }
        else {
            while (!buff_to_write.push(last_output_buffer));
        }
    }

    clear_simul();
    cudaDeviceSynchronize();

    gpu_running = false;

    for (idx_t i = 0; i < save_thread_num; ++i) {
        std::thread* thread_ptr = &thread_arr[i];
        if (thread_ptr->joinable()) {
            thread_ptr->join();
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_time = end_time - start_time;
    double seconds = diff_time.count();
    std::cout << "Simulation TIme = " << seconds << " seconds for " << num_parts << " particles.\n";

    for (idx_t i = 1; i < gpu_buffer_num; ++i) {
        particle_t* to_delete;
        bool success = buff_to_write.pop(to_delete);
        assert((success == true));
        cudaFree(to_delete);
    }
    cudaFree(output_buffer);

    return 0;
}