import taichi as ti
import numpy as np
from functools import reduce


@ti.data_oriented
class ParticleSystem:
    def __init__(self, res):
        self.res = res
        self.dim = len(res)
        assert self.dim > 1
        self.screen_to_world_ratio = 50
        self.bound = np.array(res) / self.screen_to_world_ratio
        # Material
        self.material_boundary = 0
        self.material_fluid = 1

        self.particle_radius = 0.05  # particle radius
        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V = 0.8 * self.particle_diameter ** self.dim  # Area of a particle, V_cube / V_sphere approximately equal to 0.8
        self.particle_max_num = 2 ** 15
        self.particle_max_num_per_cell = 100
        self.particle_max_num_neighbor = 100
        self.particle_num = ti.field(int, shape=())

        # Grid related properties
        self.grid_size = self.support_radius
        self.grid_num = np.ceil(np.array(res) / self.grid_size).astype(int)
        self.grid_particles_num = ti.field(int)
        self.grid_particles = ti.field(int)
        self.padding = self.grid_size

        # Particle related properties
        self.x = ti.Vector.field(self.dim, dtype=float)
        self.v = ti.Vector.field(self.dim, dtype=float)
        self.density = ti.field(dtype=float)
        self.pressure = ti.field(dtype=float)
        self.material = ti.field(dtype=int)
        self.color = ti.field(dtype=int)
        self.particle_neighbors = ti.field(int)
        self.particle_neighbors_num = ti.field(int)

        self.particles_node = ti.root.dense(ti.i, self.particle_max_num)
        self.particles_node.place(self.x, self.v, self.density, self.pressure, self.material,
                                  self.color, self.particle_neighbors_num)
        self.particle_node = self.particles_node.dense(ti.j, self.particle_max_num_neighbor)
        self.particle_node.place(self.particle_neighbors)

        index = ti.ij if self.dim == 2 else ti.ijk
        grid_node = ti.root.dense(index, self.grid_num)
        grid_node.place(self.grid_particles_num)

        cell_index = ti.k if self.dim == 2 else ti.l
        cell_node = grid_node.dense(cell_index, self.particle_max_num_per_cell)
        cell_node.place(self.grid_particles)

    @ti.func
    def add_particle(self, p, x, v, density, pressure, material, color):
        self.x[p] = x
        self.v[p] = v
        self.density[p] = density
        self.pressure[p] = pressure
        self.material[p] = material
        self.color[p] = color

    @ti.kernel
    def add_particles(self, new_particles_num: int,
                      new_particles_positions: ti.ext_arr(),
                      new_particles_velocity: ti.ext_arr(),
                      new_particle_density: ti.ext_arr(),
                      new_particle_pressure: ti.ext_arr(),
                      new_particles_material: ti.ext_arr(),
                      new_particles_color: ti.ext_arr()):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(p, x, v,
                              new_particle_density[p - self.particle_num[None]],
                              new_particle_pressure[p - self.particle_num[None]],
                              new_particles_material[p - self.particle_num[None]],
                              new_particles_color[p - self.particle_num[None]])
        self.particle_num[None] += new_particles_num

    def add_boundary_particles(self):
        width = self.bound[0]
        height = self.bound[1]
        positions = []

        # lower boundary
        for x_pos in np.arange(0, width, self.particle_radius):
            for y_pos in np.arange(0, self.support_radius, self.particle_radius):
                positions.append([x_pos, y_pos])

        # upper boundary
        for x_pos in np.arange(0, width, self.particle_radius):
            for y_pos in np.arange(height - self.support_radius, height, self.particle_radius):
                positions.append([x_pos, y_pos])

        # left boundary
        for x_pos in np.arange(0, self.support_radius, self.particle_radius):
            for y_pos in np.arange(0, height, self.particle_radius):
                positions.append([x_pos, y_pos])

        # right boundary
        for x_pos in np.arange(width - self.support_radius, width, self.particle_radius):
            for y_pos in np.arange(0, height, self.particle_radius):
                positions.append([x_pos, y_pos])

        positions = np.array(positions)
        velocities = np.zeros_like(positions)
        densities = np.array([1000.0 for _ in range(positions.shape[0])])
        pressures = np.array([0.0 for _ in range(positions.shape[0])])
        materials = np.array([self.material_boundary for _ in range(positions.shape[0])])
        colors = np.array([0xa52a2a for _ in range(positions.shape[0])])

        print("boundary particles positions shape ", positions.shape)

        self.add_particles(positions.shape[0], positions, velocities, densities, pressures, materials, colors)

    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)

    @ti.func
    def is_valid_cell(self, cell):
        # Check whether the cell is in the grid
        flag = True
        for d in ti.static(range(self.dim)):
            flag = flag and (0 <= cell[d] < self.grid_num[d])
        return flag

    @ti.kernel
    def allocate_particles_to_grid(self):
        for p in range(self.particle_num[None]):
            cell = self.pos_to_index(self.x[p])
            if not self.is_valid_cell(cell):
                continue  # Skip particles outside the grid

            offset = ti.atomic_add(self.grid_particles_num[cell], 1)
            if offset >= self.particle_max_num_per_cell:
                print("Offset: ", offset)
                print("Max number per cell: ", self.particle_max_num_per_cell)
            assert offset < self.particle_max_num_per_cell
            self.grid_particles[cell, offset] = p

    @ti.kernel
    def search_neighbors(self):
        for p_i in range(self.particle_num[None]):
            center_cell = self.pos_to_index(self.x[p_i])
            cnt = 0
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
                assert cnt < self.particle_max_num_neighbor

                cell = center_cell + offset
                if not self.is_valid_cell(cell):
                    continue

                for j in range(self.grid_particles_num[cell]):
                    p_j = self.grid_particles[cell, j]
                    distance = (self.x[p_i] - self.x[p_j]).norm()
                    if p_i != p_j and distance < self.support_radius:
                        self.particle_neighbors[p_i, cnt] = p_j
                        cnt += 1
            self.particle_neighbors_num[p_i] = cnt

    def initialize_particle_system(self):
        self.grid_particles_num.fill(0)
        self.particle_neighbors.fill(-1)
        self.allocate_particles_to_grid()
        self.search_neighbors()

    @ti.kernel
    def copy_to_numpy_nd(self, np_arr: ti.ext_arr(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            for j in ti.static(range(self.dim)):
                np_arr[i, j] = src_arr[i][j]

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.ext_arr(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]

    def dump(self):
        np_x = np.ndarray((self.particle_num[None], self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(np_x, self.x)

        np_v = np.ndarray((self.particle_num[None], self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(np_v, self.v)

        np_material = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_to_numpy(np_material, self.material)

        np_color = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_to_numpy(np_color, self.color)

        return {
            'position': np_x,
            'velocity': np_v,
            'material': np_material,
            'color': np_color
        }

    def add_cube(self,
                 lower_corner,
                 cube_size,
                 material,
                 color=0xFFFFFF,
                 density=None,
                 pressure=None,
                 velocity=None):

        num_dim = []
        for i in range(self.dim):
            num_dim.append(np.arange(lower_corner[i], lower_corner[i] + cube_size[i], self.particle_radius))
        num_new_particles = reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])
        assert self.particle_num[
                   None] + num_new_particles <= self.particle_max_num
        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        print("new position shape ", new_positions.shape)
        if velocity is None:
            velocity = np.full_like(new_positions, 0)
        else:
            velocity = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material = np.full_like(np.zeros(num_new_particles), material)
        color = np.full_like(np.zeros(num_new_particles), color)
        density = np.full_like(np.zeros(num_new_particles), density if density is not None else 1000.)
        pressure = np.full_like(np.zeros(num_new_particles), pressure if pressure is not None else 0.)
        self.add_particles(num_new_particles, new_positions, velocity, density, pressure, material, color)
