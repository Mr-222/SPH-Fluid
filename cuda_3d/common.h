#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
using idx_t = int;

constexpr idx_t num_steps = 1000;
constexpr idx_t dim = 3;
constexpr idx_t check_steps = 10;
constexpr double PI = 3.14159265358979323846;
constexpr double delta_time = 2e-4;

constexpr double gravity = -9.8;
constexpr double k1 = 50.0; // stiffness constant1
constexpr double k2 = 7.0; // stiffness constant2
constexpr double viscosity = 0.05;
constexpr double density_0 = 1000.0;

constexpr double tank_size = 100.0;
constexpr double particle_radius = tank_size / 1000.0;
constexpr double support_radius = 4.0 * particle_radius;
constexpr double particle_volume = (4.0 * 3.14 * particle_radius * particle_radius * particle_radius) / 3.0;
constexpr double particle_mass = particle_volume * density_0;

struct Vector3d {
    double x;
    double y;
    double z;

    Vector3d(const Vector3d& other) = default;

    __device__ void operator+=(const Vector3d& other) {
        x += other.x;
        y += other.y;
        z += other.z;
    }

    __device__ void operator-=(const Vector3d& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
    }
};

__device__ inline double dot(const Vector3d& a, const Vector3d& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline double normSquared(const Vector3d& r) {
    return r.x * r.x + r.y * r.y + r.z * r.z;
}

__device__ inline double norm(const Vector3d& r) {
    return sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
}

__device__ inline Vector3d operator*(const Vector3d& a, double b) {
    return {a.x * b, a.y * b, a.z * b};
}

struct particle_t {
    Vector3d pos;
    double density;
    Vector3d v;
    double pressure;
    Vector3d a;
    bool is_fluid;

    __device__ bool operator== (const particle_t& other) const {
        return pos.x == other.pos.x && pos.y == other.pos.y && pos.z == other.pos.z;
    }

    __device__ bool operator!= (const particle_t& other) const {
        return !(*this == other);
    }

    particle_t(const Vector3d& my_pos, const Vector3d& my_v, const Vector3d& my_a, double my_density, double my_pressure, bool im_fluid)
    : pos(my_pos), v(my_v), a(my_a), density(my_density), pressure(my_pressure), is_fluid(im_fluid) {}
};

void init_simul(particle_t* parts, idx_t num_parts);
void simul_one_step(particle_t* parts, idx_t num_parts);
void clear_simul();

#endif //COMMON_H