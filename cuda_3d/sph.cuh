//
// Created by 82162 on 4/2/2024.
//

#ifndef SPH_FLUID_SPH_CUH
#define SPH_FLUID_SPH_CUH

#include "common.h"

__device__ double cubic_kernel(double r_norm, double h) {
    double k = 8.0 / PI;
    k = k / (h * h * h);
    double q = r_norm / h;

    if (q <= 1.0) {
        if (q <= 0.5) {
            double q2 = q * q;
            double q3 = q2 * q;
            return k * (6.0 * q3 - 6.0 * q2 + 1.0);
        }
        else
            return k * (2.0 * pow(1.0 - q, 3));
    }
}

__device__ Vector3d cubic_kernel_derivative(Vector3d& r, double h) {
    // Constants
    double k = 8.0 / PI;
    k = 6.0 * k / (h * h * h);

    // Calculate norm of r
    double r_norm = norm(r);
    double q = r_norm / h;

    // Initialize the result vector
    Vector3d res = {0.0, 0.0, 0.0};

    if (r_norm > 1e-5 && q <= 1.0) {
        Vector3d grad_q = {r.x / (r_norm * h), r.y / (r_norm * h), r.z / (r_norm * h)};

        if (q <= 0.5) {
            res.x = k * q * (3.0 * q - 2.0) * grad_q.x;
            res.y = k * q * (3.0 * q - 2.0) * grad_q.y;
            res.z = k * q * (3.0 * q - 2.0) * grad_q.z;
        } else {
            double factor = 1.0 - q;
            res.x = k * (-factor * factor) * grad_q.x;
            res.y = k * (-factor * factor) * grad_q.y;
            res.z = k * (-factor * factor) * grad_q.z;
        }
    }
    return res;
}


#endif //SPH_FLUID_SPH_CUH
