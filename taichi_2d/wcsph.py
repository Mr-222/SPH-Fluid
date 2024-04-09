import taichi as ti
from sph_base import SPHBase


class WCSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        # Pressure state function parameters(WCSPH), they are stiffness constants
        # See https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf Chapter 4.4
        self.exponent = 7.0
        self.stiffness = 50.0

        self.d_velocity = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.d_velocity)

    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            # if self.ps.material[p_i] != self.ps.material_fluid:
            #     continue

            x_i = self.ps.x[p_i]
            self.ps.density[p_i] = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                self.ps.density[p_i] += self.ps.m_V * self.cubic_kernel((x_i - x_j).norm())
            self.ps.density[p_i] *= self.density_0

    # https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf Chapter 4.4
    @ti.kernel
    def compute_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)  # Handle free surface, Chapter 4.1
                self.ps.pressure[p_i] = self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0)

        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                # Compute Pressure force contribution
                d_v += self.pressure_force(p_i, p_j, x_i-x_j)
            self.d_velocity[p_i] += d_v

    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            x_i = self.ps.x[p_i]
            # Add body force
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            d_v[1] = self.g
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                d_v += self.viscosity_force(p_i, p_j, x_i - x_j)
            self.d_velocity[p_i] = d_v

    @ti.kernel
    def advect(self):
        # Symplectic Euler
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.v[p_i] += self.dt * self.d_velocity[p_i]
                self.ps.x[p_i] += self.dt * self.ps.v[p_i]

                # Enforce boundary
                self.ps.x[p_i][0] = ti.max(self.ps.support_radius, self.ps.x[p_i][0])
                self.ps.x[p_i][0] = ti.min(self.ps.bound[0] - self.ps.support_radius, self.ps.x[p_i][0])
                self.ps.x[p_i][1] = ti.max(self.ps.support_radius, self.ps.x[p_i][1])
                self.ps.x[p_i][1] = ti.min(self.ps.bound[1] - self.ps.support_radius, self.ps.x[p_i][1])

    def substep(self):
        self.compute_densities()
        self.compute_non_pressure_forces()
        self.compute_pressure_forces()
        self.advect()
