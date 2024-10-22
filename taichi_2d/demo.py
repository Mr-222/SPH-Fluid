import taichi as ti
from particle_system import ParticleSystem
from wcsph import WCSPHSolver

ti.init(arch=ti.cuda, device_memory_GB=3)

if __name__ == "__main__":
    ps = ParticleSystem((512, 512))

    ps.add_cube(lower_corner=[6, 2],
                cube_size=[3.0, 5.0],
                velocity=[-5.0, -10.0],
                density=1000.0,
                color=0x956333,
                material=1)

    ps.add_cube(lower_corner=[3, 1],
                cube_size=[2.0, 6.0],
                velocity=[0.0, -20.0],
                density=1000.0,
                color=0x956333,
                material=1)

    ps.add_boundary_particles()

    print("Number of particles: ", ps.particle_num[None])

    wcsph_solver = WCSPHSolver(ps)
    gui = ti.GUI(background_color=0xFFFFFF)
    while gui.running:
        for i in range(5):
            wcsph_solver.step()
        particle_info = ps.dump()
        fluid_positions = particle_info['position'][particle_info['material'] == ps.material_fluid]
        gui.circles(fluid_positions * ps.screen_to_world_ratio / 512,
                    radius=ps.particle_radius * ps.screen_to_world_ratio / 1.2,
                    color=0x72bcd4)
        gui.show()
