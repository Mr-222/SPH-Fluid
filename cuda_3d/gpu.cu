#include <cub/cub.cuh>

#include "common.h"
#include "sph.cuh"

constexpr idx_t cuda_threads = 256;
idx_t cuda_blks;

idx_t grid_dim;
idx_t num_bins;

idx_t* parts_bin_idx;
idx_t* bins_parts_cnt;
idx_t* bins_begin;
idx_t* bins_curr_pos;

void* temp_mem;
size_t temp_mem_size;


__device__ idx_t get_part_bin_idx(particle_t& part, float support_radius, idx_t grid_dim) {
    idx_t grid_x = floor(part.pos.x / support_radius);
    idx_t grid_y = floor(part.pos.y / support_radius);
    idx_t grid_z = floor(part.pos.z / support_radius);
    return (grid_x * grid_dim + grid_y) * grid_dim + grid_z;
}

__global__ void get_bins_parts_cnt(particle_t* parts, idx_t num_parts, idx_t* parts_bin_idx,
                                   idx_t* bins_parts_cnt, float support_radius, idx_t grid_dim) {

    idx_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    idx_t part_bin_idx = get_part_bin_idx(parts[tid], support_radius, grid_dim);
    parts_bin_idx[tid] = part_bin_idx;
    atomicAdd(&bins_parts_cnt[part_bin_idx], 1);
}

__global__ void get_parts_sorted(idx_t num_parts, idx_t* parts_bin_idx, particle_t* parts,
                                 particle_t* parts_sorted, idx_t* bins_curr_pos) {

    idx_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    idx_t part_bin_idx = parts_bin_idx[tid];
    idx_t pos = atomicAdd(&bins_curr_pos[part_bin_idx], 1);
    parts_sorted[pos] = parts[tid];
}

void sort_particles(particle_t* parts, idx_t num_parts, particle_t* parts_sorted) {
    cudaMemset(bins_parts_cnt, 0, num_bins * sizeof(idx_t));

    get_bins_parts_cnt<<<cuda_blks, cuda_threads>>>(parts, num_parts, parts_bin_idx,
                                                    bins_parts_cnt, support_radius, grid_dim);
    cudaDeviceSynchronize();

    cub::DeviceScan::ExclusiveSum(temp_mem, temp_mem_size, bins_parts_cnt, bins_begin, num_bins);
    cudaMemcpy(bins_curr_pos, bins_begin, num_bins * sizeof(idx_t), cudaMemcpyDeviceToDevice);
    
    get_parts_sorted<<<cuda_blks, cuda_threads>>>(num_parts, parts_bin_idx, parts, parts_sorted, bins_curr_pos);
    cudaDeviceSynchronize();
}

__global__ void update_densities(particle_t* parts, idx_t num_parts, float h, idx_t* bins_begin, idx_t grid_dim) {

    idx_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    particle_t& part = parts[tid];

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
                    apply_bin_grid_z < 0 || apply_bin_grid_z >= grid_dim) continue;

                idx_t apply_bin_idx = (apply_bin_grid_x * grid_dim + apply_bin_grid_y) * grid_dim + apply_bin_grid_z;
                idx_t bin_begin = bins_begin[apply_bin_idx];
                idx_t bin_end = bins_begin[apply_bin_idx + 1];
                for (idx_t l = bin_begin; l < bin_end; ++l) {
                    particle_t& neighbor = parts[l];
                    if (part == neighbor) continue;

                    Vector3f r = {part.pos.x - neighbor.pos.x, part.pos.y - neighbor.pos.y, part.pos.z - neighbor.pos.z};
                    float r_norm = norm(r);
                    if (r_norm < h) {
                        part.density += particle_mass * cubic_kernel(r_norm, h);
                    }
                }
            }
        }
    }
    part.density = max(part.density, density_0); // Handle free surface
}

// https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf Chapter 4.4
__global__ void update_pressures(particle_t* parts, idx_t num_parts) {
    idx_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    particle_t& part = parts[tid];
    if (!part.is_fluid) return;

    part.pressure = k1 * (pow(part.density / density_0, k2) - 1.0);
}

__device__ void inline apply_gravity(particle_t& particle) {
    particle.a.z += gravity;
}

// https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf Chapter 5.1
__device__ void apply_pressure(particle_t& particle, particle_t& neighbor, Vector3f& r) {
    Vector3f kernel_derivative = cubic_kernel_derivative(r, support_radius);

    float partial_a = 0.0f;
    if (neighbor.is_fluid)
        partial_a = -particle_mass * (particle.pressure / (particle.density * particle.density) + neighbor.pressure / (neighbor.density * neighbor.density));
    else
        partial_a = -particle_mass * (particle.pressure / (particle.density * particle.density) + particle.pressure / (density_0 * density_0));

    particle.a += (kernel_derivative * partial_a);
}

// Finite difference approximation of the Laplacian of the velocity field
// From https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf Chapter 6.2
// See also https://github.com/taichiCourse01/taichiCourse01/blob/main/material/10_fluid_lagrangian.pdf
__device__ void apply_viscosity(particle_t& particle, particle_t& neighbor, Vector3f& r) {
    Vector3f v_difference = particle.v - neighbor.v;
    
    float v_dot_x = dot(v_difference, r);
    float denominator = normSquared(r) + 0.01 * support_radius * support_radius;
    
    Vector3f kernel_derivative = cubic_kernel_derivative(r, support_radius);
    float partial_a = 2 * (dim + 2) * viscosity * (particle_mass / neighbor.density) * v_dot_x / denominator;

    particle.a += (kernel_derivative * partial_a);
}

__device__ void apply_mutual_force(particle_t& particle, particle_t& neighbor) {
    Vector3f r = particle.pos - neighbor.pos;
    apply_pressure(particle, neighbor,r);
    apply_viscosity(particle, neighbor,r);
}

__device__ void apply_bin_force(particle_t& part, idx_t bin_idx, particle_t* parts, idx_t* bins_begin) {

    idx_t bin_begin = bins_begin[bin_idx];
    idx_t bin_end = bins_begin[bin_idx + 1];
    for (idx_t i = bin_begin; i < bin_end; ++i) {
        particle_t& neighbor_part = parts[i];
        if (part != neighbor_part)
            apply_mutual_force(part, neighbor_part);
    }
}

__global__ void compute_forces(particle_t* parts, idx_t num_parts, float support_radius,
                               idx_t* bins_begin, idx_t grid_dim) {

    idx_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    particle_t& part = parts[tid];
    if (!part.is_fluid) return;

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
                    apply_bin_grid_z < 0 || apply_bin_grid_z >= grid_dim) continue;

                int apply_bin_idx = (apply_bin_grid_x * grid_dim + apply_bin_grid_y) * grid_dim + apply_bin_grid_z;
                apply_bin_force(part, apply_bin_idx, parts, bins_begin);
            }
        }
    }
}

__global__ void move_particles(particle_t* parts, idx_t num_parts, float tank_size, float dt) {

    idx_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

    particle_t& part = parts[tid];
    if (!part.is_fluid) return;

    // Explicit Euler integration
    part.v += part.a * dt;
    part.pos += part.v * dt;

    // Enforce boundaries, particles and boundary should not overlap
    part.pos.x = max(support_radius + particle_radius, min(tank_size - support_radius - particle_radius, part.pos.x));
    part.pos.y = max(support_radius + particle_radius, min(tank_size - support_radius - particle_radius, part.pos.y));
    part.pos.z = max(support_radius + particle_radius, min(tank_size - support_radius - particle_radius, part.pos.z));

    part.a.x = part.a.y = part.a.z = 0;
}

void init_simul(idx_t num_parts) {
    cuda_blks = (num_parts + cuda_threads - 1) / cuda_threads;

    grid_dim = ceil(tank_size / support_radius);
    num_bins = grid_dim * grid_dim * grid_dim;

    idx_t num_parts_bytes = num_parts * sizeof(idx_t);
    idx_t num_bins_bytes = num_bins * sizeof(idx_t);

    cudaMalloc(&parts_bin_idx, num_parts_bytes);
    cudaMalloc(&bins_parts_cnt, num_bins_bytes);
    cudaMalloc(&bins_begin, num_bins_bytes + sizeof(idx_t));
    cudaMalloc(&bins_curr_pos, num_bins_bytes);

    cudaMemcpy(&bins_begin[num_bins], &num_parts, sizeof(idx_t), cudaMemcpyHostToDevice);

    cub::DeviceScan::ExclusiveSum(nullptr, temp_mem_size, bins_parts_cnt, bins_begin, num_bins);
    temp_mem = nullptr;
    cudaMalloc((void**)&temp_mem, temp_mem_size);
}

void simul_one_step(particle_t* parts, idx_t num_parts, particle_t* parts_sorted) {
    // See https://on-demand.gputechconf.com/gtc/2014/presentations/S4117-fast-fixed-radius-nearest-neighbor-gpu.pdf for details
    sort_particles(parts, num_parts, parts_sorted);
    cudaDeviceSynchronize();

    update_densities<<<cuda_blks, cuda_threads>>>(parts_sorted, num_parts, support_radius, bins_begin, grid_dim);
    cudaDeviceSynchronize();

    update_pressures<<<cuda_blks, cuda_threads>>>(parts_sorted, num_parts);
    cudaDeviceSynchronize();

    compute_forces<<<cuda_blks, cuda_threads>>>(parts_sorted, num_parts, support_radius, bins_begin, grid_dim);
    cudaDeviceSynchronize();

    move_particles<<<cuda_blks, cuda_threads>>>(parts_sorted, num_parts, tank_size, delta_time);
    cudaDeviceSynchronize();
}

void clear_simul() {
    cudaFree(parts_bin_idx);
    cudaFree(bins_parts_cnt);
    cudaFree(bins_begin);
    cudaFree(bins_curr_pos);
    cudaFree(temp_mem);
}