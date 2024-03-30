import taichi as ti
import numpy as np


@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = -9.80  # Gravity
        self.viscosity = 0.05  # viscosity
        self.density_0 = 1000.0  # reference density
        self.mass = self.ps.m_V * self.density_0
        self.dt = 1e-3  # time step

    # Kernel function from https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf Chapter 2.2
    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = self.ps.support_radius
        # value of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k /= h ** self.ps.dim
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        h = self.ps.support_radius
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k = 6. * k / h ** self.ps.dim
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    # Finite difference approximation of the Laplacian of the velocity field
    # From https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf Chapter 6.2
    # See also https://github.com/taichiCourse01/taichiCourse01/blob/main/material/10_fluid_lagrangian.pdf
    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (self.ps.v[p_i] -
                self.ps.v[p_j]).dot(r)
        res = (2 * (self.ps.dim + 2) * self.viscosity * (self.mass / (self.ps.density[p_j])) * v_xy /
               (r.norm() ** 2 + 0.01 * self.ps.support_radius ** 2) * self.cubic_kernel_derivative(r))
        return res

    # https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf Chapter 5.1
    @ti.func
    def pressure_force(self, p_i, p_j, r):
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        # Compute the pressure force contribution, Symmetric Formula
        if self.ps.material[p_j] == self.ps.material_boundary:
            res = -self.density_0 * self.ps.m_V * (self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
                                                   + self.ps.pressure[p_i] / self.density_0 ** 2) * self.cubic_kernel_derivative(r)
        else:
            res = -self.density_0 * self.ps.m_V * (self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
                                                   + self.ps.pressure[p_j] / self.ps.density[
                                                       p_j] ** 2) * self.cubic_kernel_derivative(r)
        return res

    def substep(self):
        pass

    def step(self):
        self.ps.initialize_particle_system()
        self.substep()
