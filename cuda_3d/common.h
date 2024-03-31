#pragma once

#include <cstdint>
using idx_t = int;

constexpr idx_t num_steps = 1000;
constexpr idx_t check_steps = 10;
constexpr double delta_time = 0.01;

constexpr double gravity = 10.0;
constexpr double stiffness = 50.0;
constexpr double viscosity = 0.05;
constexpr double density_0 = 1000.0;

constexpr double tank_size = 100.0;
constexpr double particle_radius = tank_size / 1000;
constexpr double support_radius = 4 * particle_radius;
constexpr double particle_volume = 4 * particle_radius * particle_radius * particle_radius;
constexpr double particle_mass = particle_volume * density_0;

struct physics_t {
    double x;
    double y;
    double z;
};

struct particle_t {
    physics_t pos;
    physics_t v;
    physics_t a;

    particle_t(const physics_t& my_pos, const physics_t& my_v, const physics_t& my_a):
    pos(my_pos), v(my_v), a(my_a) {}
};

void init_simul(particle_t* parts, idx_t num_parts);
void simul_one_step(particle_t* parts, idx_t num_parts);
void clear_simul();