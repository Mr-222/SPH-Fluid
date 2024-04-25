# SPH-Fluid

In our final project for CSE6230: High Performance Parallel Computing - Tools and Applications, we implemented a Smoothed Particle Hydrodynamics (SPH) fluid simulation leveraging both CUDA and Taichi technologies. Our project features the Weakly Compressible SPH (WCSPH) model, a rapid fixed-radius neighbor search optimized for GPU performance, and a lockless ring buffer that facilitates concurrent operations between the CPU and GPU.



### Result

<img src="./output.gif" title="" alt="" data-align="center">

### References

[Authors | Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids (physics-simulation.org)](https://sph-tutorial.physics-simulation.org/)

[S4117-fast-fixed-radius-nearest-neighbor-gpu.pdf (gputechconf.com)](https://on-demand.gputechconf.com/gtc/2014/presentations/S4117-fast-fixed-radius-nearest-neighbor-gpu.pdf)

[taichiCourse01 (TaichiCourse) (github.com)](https://github.com/taichiCourse01)

[11. Ring Library — Data Plane Development Kit 24.03.0 documentation (dpdk.org)](https://doc.dpdk.org/guides/prog_guide/ring_lib.html?highlight=rte_ring)

[A Parallel SPH Implementation on Multi‐Core CPUs - Ihmsen - 2011 - Computer Graphics Forum - Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2010.01832.x)
