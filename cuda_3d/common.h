#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
using idx_t = int;

constexpr int num_steps = 100000;
constexpr int check_steps = 500;
constexpr idx_t dim = 3;
constexpr float PI = 3.14159265358979323846;
constexpr float delta_time = 2e-4;

constexpr float gravity = -9.8;
constexpr float k1 = 50.0; // stiffness constant1
constexpr float k2 = 7.0; // stiffness constant2
constexpr float viscosity = 0.05;
constexpr float density_0 = 1000.0;

constexpr float tank_size = 10.0;
constexpr float particle_radius = 0.05;
constexpr float support_radius = 4.0 * particle_radius;
constexpr float particle_volume = (4.0 * PI * particle_radius * particle_radius * particle_radius) / 3.0;
constexpr float particle_mass = particle_volume * density_0;

struct Vector3f {
    float x;
    float y;
    float z;

    //Vector3f(const Vector3f& other) = default;

    __device__ void operator+=(const Vector3f& other) {
        x += other.x;
        y += other.y;
        z += other.z;
    }

    __device__ void operator-=(const Vector3f& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
    }
};

__device__ inline float dot(const Vector3f& a, const Vector3f& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float normSquared(const Vector3f& r) {
    return r.x * r.x + r.y * r.y + r.z * r.z;
}

__device__ inline float norm(const Vector3f& r) {
    return sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
}

__device__ inline Vector3f operator*(const Vector3f& a, float b) {
    return {a.x * b, a.y * b, a.z * b};
}

struct particle_t {
    Vector3f pos;
    float density;
    Vector3f v;
    float pressure;
    Vector3f a;
    bool is_fluid;

    particle_t(const Vector3f& my_pos, const Vector3f& my_v, const Vector3f& my_a, float my_density, float my_pressure, bool im_fluid)
    : pos(my_pos), v(my_v), a(my_a), density(my_density), pressure(my_pressure), is_fluid(im_fluid) {}

    __device__ bool operator== (const particle_t& other) const {
        return pos.x == other.pos.x && pos.y == other.pos.y && pos.z == other.pos.z;
    }

    __device__ bool operator!= (const particle_t& other) const {
        return !(*this == other);
    }
};

void init_simul(particle_t* parts, idx_t num_parts);
void simul_one_step(particle_t* parts, idx_t num_parts);
void clear_simul();

#endif //COMMON_H