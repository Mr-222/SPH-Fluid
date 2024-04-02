#pragma once

#include <cstdint>
using idx_t = int;

constexpr idx_t num_steps = 1000;
constexpr idx_t dim = 3;
constexpr idx_t check_steps = 10;
constexpr double delta_time = 2e-4;

constexpr double gravity = 10.0;
constexpr double stiffness = 50.0;
constexpr double viscosity = 0.05;
constexpr double density_0 = 1000.0;

constexpr double tank_size = 100.0;
constexpr double particle_radius = tank_size / 1000;
constexpr double support_radius = 4 * particle_radius;
constexpr double particle_volume = (4 * 3.14 * particle_radius * particle_radius * particle_radius) / 3;
constexpr double particle_mass = particle_volume * density_0;

struct Vector3d {
    double x;
    double y;
    double z;
};

struct particle_t {
    Vector3d pos;
    Vector3d v;
    Vector3d a;
    double density;
    double pressure;

    particle_t(const Vector3d& my_pos, const Vector3d& my_v, const Vector3d& my_a, double my_density, double my_pressure)
    : pos(my_pos), v(my_v), a(my_a), density(my_density), pressure(my_pressure) {}
};

void init_simul(particle_t* parts, idx_t num_parts);
void simul_one_step(particle_t* parts, idx_t num_parts);
void clear_simul();