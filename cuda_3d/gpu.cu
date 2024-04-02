#include <cub/cub.cuh>

#include "common.h"
#include "sph.cuh"

constexpr idx_t cuda_threads = 256;
idx_t cuda_blks;

idx_t grid_dim;
idx_t num_bins;

idx_t* parts_bin_idx;
idx_t* parts_sorted;
idx_t* bins_parts_cnt;
idx_t* bins_begin;
idx_t* bins_curr_pos;

void* temp_mem;
size_t temp_mem_size;


__device__ idx_t get_part_bin_idx(particle_t& part, double support_radius, idx_t grid_dim) {
    idx_t grid_x = floor(part.pos.x / support_radius);
    idx_t grid_y = floor(part.pos.y / support_radius);
    idx_t grid_z = floor(part.pos.z / support_radius);
    return (grid_x * grid_dim + grid_y) * grid_dim + grid_z;
}

__global__ void get_bins_parts_cnt(particle_t* parts, idx_t num_parts, idx_t* parts_bin_idx,
                                   idx_t* bins_parts_cnt, double support_radius, idx_t grid_dim) {

    idx_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    idx_t part_bin_idx = get_part_bin_idx(parts[tid], support_radius, grid_dim);
    parts_bin_idx[tid] = part_bin_idx;
    atomicAdd(&bins_parts_cnt[part_bin_idx], 1);
}

__global__ void get_parts_sorted(particle_t* parts, idx_t num_parts, idx_t* parts_bin_idx,
                                 idx_t* parts_sorted, idx_t* bins_curr_pos) {

    idx_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    idx_t part_bin_idx = parts_bin_idx[tid];
    idx_t pos = atomicAdd(&bins_curr_pos[part_bin_idx], 1);
    parts_sorted[pos] = tid;
}

void sort_particles(particle_t* parts, idx_t num_parts) {
    cudaMemset(bins_parts_cnt, 0, num_bins * sizeof(idx_t));

    get_bins_parts_cnt<<<cuda_blks, cuda_threads>>>(parts, num_parts, parts_bin_idx,
                                                    bins_parts_cnt, support_radius, grid_dim);

    cub::DeviceScan::ExclusiveSum(temp_mem, temp_mem_size, bins_parts_cnt, bins_begin, num_bins);
    cudaMemcpy(bins_curr_pos, bins_begin, num_bins * sizeof(idx_t), cudaMemcpyDeviceToDevice);
    
    get_parts_sorted<<<cuda_blks, cuda_threads>>>(parts, num_parts, parts_bin_idx, parts_sorted, bins_curr_pos);
}

__global__ void update_densities(particle_t* parts, idx_t num_parts, double h, idx_t* parts_sorted, idx_t* bins_begin, idx_t grid_dim) {

    idx_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    idx_t part_idx = parts_sorted[tid];
    particle_t& part = parts[part_idx];
    part.density = 0;

    idx_t bin_grid_x = floor(part.pos.x / h);
    idx_t bin_grid_y = floor(part.pos.y / h);
    idx_t bin_grid_z = floor(part.pos.z / h);

    for (idx_t i = -1; i <= 1; ++i) {
        for (idx_t j = -1; j <= 1; ++j) {
            for (idx_t k = -1; k <= 1; ++k) {
                idx_t apply_bin_grid_x = bin_grid_x + i;
                idx_t apply_bin_grid_y = bin_grid_y + j;
                idx_t apply_bin_grid_z = bin_grid_z + k;
                if (apply_bin_grid_x < 0 || apply_bin_grid_x >= grid_dim ||
                    apply_bin_grid_y < 0 || apply_bin_grid_y >= grid_dim ||
                    apply_bin_grid_z < 0 || apply_bin_grid_z >= grid_dim) {
                    continue;
                }

                idx_t apply_bin_idx = (apply_bin_grid_x * grid_dim + apply_bin_grid_y) * grid_dim + apply_bin_grid_z;
                idx_t bin_begin = bins_begin[apply_bin_idx];
                idx_t bin_end = bins_begin[apply_bin_idx + 1];
                for (idx_t l = bin_begin; l < bin_end; ++l) {
                    idx_t neighbor_idx = parts_sorted[l];
                    particle_t& neighbor = parts[neighbor_idx];
                    if (part == neighbor) continue;

                    Vector3d r = {part.pos.x - neighbor.pos.x, part.pos.y - neighbor.pos.y, part.pos.z - neighbor.pos.z};
                    double r_norm = norm(r);
                    if (r_norm < h) {
                        part.density += particle_mass * cubic_kernel(r_norm, h);
                    }
                }
            }
        }
    }
}

__global__ void update_pressures(particle_t* parts, idx_t num_parts) {
    idx_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    particle_t& part = parts[tid];
    part.density = max(part.density, density_0); // Handle free surface
    part.pressure = k1 * (pow(part.density / density_0, k2) - 1.0);
}

__device__ void apply_gravity(particle_t& particle) {
    particle.a.z += gravity;
}

__device__ void apply_pressure(particle_t& particle, particle_t& neighbor,Vector3d& r) {
    Vector3d kernel_derivative = cubic_kernel_derivative(r, support_radius);

    // TODO: Handle boundary particles

    double partial_a = density_0 * particle_volume * (particle.pressure / (particle.density * particle.density) + neighbor.pressure / (neighbor.density * neighbor.density));

    particle.a += kernel_derivative * partial_a;
}

__device__ void apply_viscosity(particle_t& particle, particle_t& neighbor, Vector3d& r) {
    Vector3d v_difference = {particle.v.x - neighbor.v.x, particle.v.y - neighbor.v.y, particle.v.z - neighbor.v.z};
    
    double v_dot_x = dot(v_difference, r);
    double denominator = normSquared(r) + 0.01 * support_radius * support_radius;
    
    Vector3d kernel_derivative = cubic_kernel_derivative(r, support_radius);
    double partial_a = 2 * (dim + 2) * viscosity * (particle_mass / neighbor.density) * v_dot_x / denominator;

    particle.a += kernel_derivative * partial_a;
}

__device__ void apply_mutual_force(particle_t& particle, particle_t& neighbor) {
    Vector3d r = {particle.pos.x - neighbor.pos.x, particle.pos.y - neighbor.pos.y, particle.pos.z - neighbor.pos.z};
    apply_pressure(particle, neighbor,r);
    apply_viscosity(particle, neighbor,r);
}

__device__ void apply_bin_force(particle_t& part, idx_t bin_idx, particle_t* parts,
                                idx_t* parts_sorted, idx_t* bins_begin) {

    idx_t bin_begin = bins_begin[bin_idx];
    idx_t bin_end = bins_begin[bin_idx + 1];
    for (idx_t i = bin_begin; i < bin_end; ++i) {
        idx_t part_idx = parts_sorted[i];
        if (part != parts[part_idx])
            apply_mutual_force(part, parts[part_idx]);
    }
}

__global__ void compute_forces(particle_t* parts, idx_t num_parts, double support_radius,
                               idx_t* parts_sorted, idx_t* bins_begin, idx_t grid_dim) {

    idx_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    idx_t part_idx = parts_sorted[tid];
    particle_t& part = parts[part_idx];
    part.a.x = part.a.y = part.a.z = 0;
    idx_t bin_grid_x = floor(part.pos.x / support_radius);
    idx_t bin_grid_y = floor(part.pos.y / support_radius);
    idx_t bin_grid_z = floor(part.pos.z / support_radius);

    apply_gravity(part);

    for (idx_t i = -1; i <= 1; ++i) {
        for (idx_t j = -1; j <= 1; ++j) {
            for (idx_t k = -1; k <= 1; ++k) {
                idx_t apply_bin_grid_x = bin_grid_x + i;
                idx_t apply_bin_grid_y = bin_grid_y + j;
                idx_t apply_bin_grid_z = bin_grid_z + k;
                if (apply_bin_grid_x < 0 || apply_bin_grid_x >= grid_dim ||
                apply_bin_grid_y < 0 || apply_bin_grid_y >= grid_dim ||
                apply_bin_grid_z < 0 || apply_bin_grid_z >= grid_dim) {
                    continue;
                }
                int apply_bin_idx = (apply_bin_grid_x * grid_dim + apply_bin_grid_y) * grid_dim + apply_bin_grid_z;
                apply_bin_force(part, apply_bin_idx, parts, parts_sorted, bins_begin);
            }
        }
    }
}

__global__ void move_particles(particle_t* parts, idx_t num_parts, double size, idx_t* parts_sorted, double dt) {

    idx_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    idx_t part_idx = parts_sorted[tid];
    particle_t& part = parts[part_idx];

    part.v.x += part.a.x * dt;
    part.v.y += part.a.y * dt;
    part.v.z += part.a.z * dt;

    part.pos.x += part.v.x * dt;
    part.pos.y += part.v.y * dt;
    part.pos.z += part.v.z * dt;

    while (part.pos.x < 0 || part.pos.x > size) {
        part.pos.x = part.pos.x < 0 ? -part.pos.x : 2 * size - part.pos.x;
        part.v.x = -part.v.x;
    }
    while (part.pos.y < 0 || part.pos.y > size) {
        part.pos.y = part.pos.y < 0 ? -part.pos.y : 2 * size - part.pos.y;
        part.v.y = -part.v.y;
    }
    while (part.pos.z < 0 || part.pos.z > size) {
        part.pos.z = part.pos.z < 0 ? -part.pos.z : 2 * size - part.pos.z;
        part.v.z = -part.v.z;
    }

    part.a.x = part.a.y = part.a.z = 0;
}

void init_simul(particle_t* parts, idx_t num_parts) {
    cuda_blks = (num_parts + cuda_threads - 1) / cuda_threads;

    grid_dim = ceil(tank_size / support_radius);
    num_bins = grid_dim * grid_dim * grid_dim;

    idx_t num_parts_bytes = num_parts * sizeof(idx_t);
    idx_t num_bins_bytes = num_bins * sizeof(idx_t);

    cudaMalloc(&parts_bin_idx, num_parts_bytes);
    cudaMalloc(&parts_sorted, num_parts_bytes);
    cudaMalloc(&bins_parts_cnt, num_bins_bytes);
    cudaMalloc(&bins_begin, num_bins_bytes + sizeof(idx_t));
    cudaMalloc(&bins_curr_pos, num_bins_bytes);

    cudaMemcpy(&bins_begin[num_bins], &num_parts, sizeof(idx_t), cudaMemcpyHostToDevice);

    cub::DeviceScan::ExclusiveSum(nullptr, temp_mem_size, bins_parts_cnt, bins_begin, num_bins);
    temp_mem = nullptr;
    cudaMalloc((void**)&temp_mem, temp_mem_size);
}

void simul_one_step(particle_t* parts, idx_t num_parts) {
    // See https://drive.google.com/file/d/1j5Lu3G80BgsSRyEjEBc4y5klSip1eil6/view for details about the following code
    sort_particles(parts, num_parts);

    update_densities<<<cuda_blks, cuda_threads>>>(parts, num_parts, support_radius, parts_sorted, bins_begin, grid_dim);

    update_pressures<<<cuda_blks, cuda_threads>>>(parts, num_parts);

    compute_forces<<<cuda_blks, cuda_threads>>>(parts, num_parts, support_radius, parts_sorted, bins_begin, grid_dim);

    move_particles<<<cuda_blks, cuda_threads>>>(parts, num_parts, tank_size, parts_sorted, delta_time);
}

void clear_simul() {
    cudaFree(parts_bin_idx);
    cudaFree(parts_sorted);
    cudaFree(bins_parts_cnt);
    cudaFree(bins_begin);
    cudaFree(bins_curr_pos);
    cudaFree(temp_mem);
}