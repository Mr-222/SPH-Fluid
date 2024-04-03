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

__global__ void get_parts_sorted(idx_t num_parts, idx_t* parts_bin_idx,
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
    cudaDeviceSynchronize();

    cub::DeviceScan::ExclusiveSum(temp_mem, temp_mem_size, bins_parts_cnt, bins_begin, num_bins);
    cudaMemcpy(bins_curr_pos, bins_begin, num_bins * sizeof(idx_t), cudaMemcpyDeviceToDevice);
    
    get_parts_sorted<<<cuda_blks, cuda_threads>>>(num_parts, parts_bin_idx, parts_sorted, bins_curr_pos);
    cudaDeviceSynchronize();
}

__global__ void update_densities(particle_t* parts, idx_t num_parts, float h, idx_t* parts_sorted, idx_t* bins_begin, idx_t grid_dim) {

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
                    apply_bin_grid_z < 0 || apply_bin_grid_z >= grid_dim) continue;

                idx_t apply_bin_idx = (apply_bin_grid_x * grid_dim + apply_bin_grid_y) * grid_dim + apply_bin_grid_z;
                idx_t bin_begin = bins_begin[apply_bin_idx];
                idx_t bin_end = bins_begin[apply_bin_idx + 1];
                for (idx_t l = bin_begin; l < bin_end; ++l) {
                    idx_t neighbor_idx = parts_sorted[l];
                    particle_t& neighbor = parts[neighbor_idx];
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

__global__ void update_pressures(particle_t* parts, idx_t num_parts) {
    idx_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    particle_t& part = parts[tid];
    part.pressure = k1 * (pow(part.density / density_0, k2) - 1.0);
}

__device__ void inline apply_gravity(particle_t& particle) {
    particle.a.z += gravity;
}

__device__ void apply_pressure(particle_t& particle, particle_t& neighbor, Vector3f& r) {
    Vector3f kernel_derivative = cubic_kernel_derivative(r, support_radius);

    // TODO: Handle boundary particles

    float partial_a = -particle_mass * (particle.pressure / (particle.density * particle.density) + neighbor.pressure / (neighbor.density * neighbor.density));

    particle.a += (kernel_derivative * partial_a);
}

__device__ void apply_viscosity(particle_t& particle, particle_t& neighbor, Vector3f& r) {
    Vector3f v_difference = {particle.v.x - neighbor.v.x, particle.v.y - neighbor.v.y, particle.v.z - neighbor.v.z};
    
    float v_dot_x = dot(v_difference, r);
    float denominator = normSquared(r) + 0.01 * support_radius * support_radius;
    
    Vector3f kernel_derivative = cubic_kernel_derivative(r, support_radius);
    float partial_a = 2 * (dim + 2) * viscosity * (particle_mass / neighbor.density) * v_dot_x / denominator;

    particle.a += (kernel_derivative * partial_a);
}

__device__ void apply_mutual_force(particle_t& particle, particle_t& neighbor) {
    Vector3f r = {particle.pos.x - neighbor.pos.x, particle.pos.y - neighbor.pos.y, particle.pos.z - neighbor.pos.z};
    apply_pressure(particle, neighbor,r);
    apply_viscosity(particle, neighbor,r);
}

__device__ void apply_bin_force(particle_t& part, idx_t bin_idx, particle_t* parts,
                                idx_t* parts_sorted, idx_t* bins_begin) {

    idx_t bin_begin = bins_begin[bin_idx];
    idx_t bin_end = bins_begin[bin_idx + 1];
    for (idx_t i = bin_begin; i < bin_end; ++i) {
        idx_t part_idx = parts_sorted[i];
        particle_t& neighbor_part = parts[part_idx];
        if (part != neighbor_part)
            apply_mutual_force(part, neighbor_part);
    }
}

__global__ void compute_forces(particle_t* parts, idx_t num_parts, float support_radius,
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
                    apply_bin_grid_z < 0 || apply_bin_grid_z >= grid_dim) continue;

                int apply_bin_idx = (apply_bin_grid_x * grid_dim + apply_bin_grid_y) * grid_dim + apply_bin_grid_z;
                apply_bin_force(part, apply_bin_idx, parts, parts_sorted, bins_begin);
            }
        }
    }
}

__device__ void simulate_collision(particle_t& part, const Vector3f& normal, float distance) {
    // Collision factor, assume roughly (1-c_f)*velocity loss after collision
    float c_f = 0.3;
    part.pos += normal * distance;
    float v_dot_n = dot(part.v, normal);
    part.v -= normal * (1 + c_f) * v_dot_n;
}

__global__ void move_particles(particle_t* parts, idx_t num_parts, float size, idx_t* parts_sorted, float dt) {

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

    if (part.pos.x < 0)
        simulate_collision(part, {1, 0, 0}, -part.pos.x);
    else if (part.pos.x > size)
        simulate_collision(part, {-1, 0, 0}, part.pos.x - size);

    if (part.pos.y < 0)
        simulate_collision(part, {0, 1, 0}, -part.pos.y);
    else if (part.pos.y > size)
        simulate_collision(part, {0, -1, 0}, part.pos.y - size);

    if (part.pos.z < 0)
        simulate_collision(part, {0, 0, 1}, -part.pos.z);
    else if (part.pos.z > size)
        simulate_collision(part, {0, 0, -1}, part.pos.z - size);

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
    // See https://drive.google.com/file/d/1j5Lu3G80BgsSRyEjEBc4y5klSip1eil6/view for details
    sort_particles(parts, num_parts);

    update_densities<<<cuda_blks, cuda_threads>>>(parts, num_parts, support_radius, parts_sorted, bins_begin, grid_dim);
    cudaDeviceSynchronize();

    update_pressures<<<cuda_blks, cuda_threads>>>(parts, num_parts);
    cudaDeviceSynchronize();

    compute_forces<<<cuda_blks, cuda_threads>>>(parts, num_parts, support_radius, parts_sorted, bins_begin, grid_dim);
    cudaDeviceSynchronize();

    move_particles<<<cuda_blks, cuda_threads>>>(parts, num_parts, tank_size, parts_sorted, delta_time);
    cudaDeviceSynchronize();
}

void clear_simul() {
    cudaFree(parts_bin_idx);
    cudaFree(parts_sorted);
    cudaFree(bins_parts_cnt);
    cudaFree(bins_begin);
    cudaFree(bins_curr_pos);
    cudaFree(temp_mem);
}