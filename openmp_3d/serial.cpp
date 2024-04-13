
#include "common.h"
#include "sph.h"
#include <vector>
#include <algorithm>
#include <numeric>

idx_t grid_dim;
idx_t num_bins;

std::vector<idx_t> parts_bin_idx;
std::vector<idx_t> bins_parts_cnt;
std::vector<idx_t> bins_begin;
std::vector<idx_t> bins_curr_pos;


idx_t get_part_bin_idx(const particle_t& part, float support_radius, idx_t grid_dim) {
    idx_t grid_x = floor(part.pos.x / support_radius);
    idx_t grid_y = floor(part.pos.y / support_radius);
    idx_t grid_z = floor(part.pos.z / support_radius);
    return (grid_x * grid_dim + grid_y) * grid_dim + grid_z;
}

void sort_particles(std::vector<particle_t>& parts, std::vector<particle_t>& parts_sorted) {
    std::fill(bins_parts_cnt.begin(), bins_parts_cnt.end(), 0);

    for (idx_t i = 0; i < parts.size(); ++i) {
        idx_t part_bin_idx = get_part_bin_idx(parts[i], support_radius, grid_dim);
        parts_bin_idx[i] = part_bin_idx;
        bins_parts_cnt[part_bin_idx]++;
    }

    bins_begin[0] = 0;
    std::partial_sum(bins_parts_cnt.begin(), bins_parts_cnt.end(), bins_begin.begin() + 1);
    bins_curr_pos = bins_begin;

    for (idx_t i = 0; i < parts.size(); ++i) {
        idx_t part_bin_idx = parts_bin_idx[i];
        idx_t pos = bins_curr_pos[part_bin_idx]++;
        parts_sorted[pos] = parts[i];
    }
}

void update_densities(std::vector<particle_t>& parts, float h, std::vector<idx_t>& bins_begin, idx_t grid_dim) {
    for (auto& part : parts) {
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
        part.density = std::max(part.density, density_0); // Handle free surface
    }
}

void update_pressures(std::vector<particle_t>& parts) {
    for (auto& part : parts) {
        if (!part.is_fluid) continue;
        part.pressure = k1 * (std::pow(part.density / density_0, k2) - 1.0);
    }
}


void inline apply_gravity(particle_t& particle) {
    particle.a.z += gravity;
}

void apply_pressure(particle_t& particle, particle_t& neighbor, Vector3f& r) {
    Vector3f kernel_derivative = cubic_kernel_derivative(r, support_radius);

    float partial_a = 0.0f;
    if (neighbor.is_fluid)
        partial_a = -particle_mass * (particle.pressure / (particle.density * particle.density) + neighbor.pressure / (neighbor.density * neighbor.density));
    else
        partial_a = -particle_mass * (particle.pressure / (particle.density * particle.density) + particle.pressure / (density_0 * density_0));

    particle.a += kernel_derivative * partial_a;
}

void apply_viscosity(particle_t& particle, particle_t& neighbor, Vector3f& r) {
    Vector3f v_difference = particle.v - neighbor.v;
    
    float v_dot_r = dot(v_difference, r);
    float denominator = normSquared(r) + 0.01f * support_radius * support_radius;
    
    Vector3f kernel_derivative = cubic_kernel_derivative(r, support_radius);
    float partial_a = 2 * (dim + 2) * viscosity * (particle_mass / neighbor.density) * v_dot_r / denominator;

    particle.a += kernel_derivative * partial_a;
}

void apply_mutual_force(particle_t& particle, particle_t& neighbor) {
    Vector3f r = particle.pos - neighbor.pos;
    apply_pressure(particle, neighbor, r);
    apply_viscosity(particle, neighbor, r);
}



void apply_bin_force(particle_t& part, idx_t bin_idx, std::vector<particle_t>& parts, std::vector<idx_t>& bins_begin) {

    idx_t bin_begin = bins_begin[bin_idx];
    idx_t bin_end = bins_begin[bin_idx + 1];
    for (idx_t i = bin_begin; i < bin_end; ++i) {
        particle_t& neighbor_part = parts[i];
        if (part != neighbor_part)
            apply_mutual_force(part, neighbor_part);
    }
}

void compute_forces(std::vector<particle_t>& parts, std::vector<idx_t>& bins_begin, idx_t grid_dim) {
    for (auto& part : parts) {
        if (!part.is_fluid) continue;

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
}

void move_particles(std::vector<particle_t>& parts, float tank_size, float dt) {
    for (auto& part : parts) {
        if (!part.is_fluid) continue;

        // Explicit Euler integration to update velocity and position
        part.v.x += part.a.x * dt;
        part.v.y += part.a.y * dt;
        part.v.z += part.a.z * dt;

        part.pos.x += part.v.x * dt;
        part.pos.y += part.v.y * dt;
        part.pos.z += part.v.z * dt;

        // Enforce boundary conditions
        part.pos.x = std::max(2.0f * particle_radius, std::min(tank_size - 2.0f * particle_radius, part.pos.x));
        part.pos.y = std::max(2.0f * particle_radius, std::min(tank_size - 2.0f * particle_radius, part.pos.y));
        part.pos.z = std::max(2.0f * particle_radius, std::min(tank_size - 2.0f * particle_radius, part.pos.z));

        // Reset accelerations for the next time step
        part.a.x = 0;
        part.a.y = 0;
        part.a.z = 0;
    }
}


void init_simul(idx_t num_parts) {
    grid_dim = ceil(tank_size / support_radius);
    num_bins = grid_dim * grid_dim * grid_dim;

    parts_bin_idx.resize(num_parts);
    bins_parts_cnt.resize(num_bins, 0);
    bins_begin.resize(num_bins + 1);
    bins_curr_pos.resize(num_bins);
}

void simul_one_step(std::vector<particle_t>& parts, std::vector<particle_t>& parts_sorted, idx_t num_parts) {
    sort_particles(parts, parts_sorted);
    update_densities(parts_sorted, support_radius, bins_begin, grid_dim);
    update_pressures(parts_sorted);
    compute_forces(parts_sorted, bins_begin, grid_dim);
    move_particles(parts_sorted,tank_size,delta_time);
}

void clear_simul() {
   
}