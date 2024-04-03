#ifndef SPH_FLUID_SPH_CUH
#define SPH_FLUID_SPH_CUH

#include "common.h"

__device__ float cubic_kernel(float r_norm, float h) {
    float k = 8.0f / PI;
    k = k / (h * h * h);
    float q = r_norm / h;

    if (q <= 1.0) {
        if (q <= 0.5) {
            float q2 = q * q;
            float q3 = q2 * q;
            return k * (6.0 * q3 - 6.0 * q2 + 1.0);
        }
        else
            return k * (2.0 * pow(1.0 - q, 3));
    }

    return 0.0;
}

__device__ Vector3f cubic_kernel_derivative(Vector3f& r, float h) {
    // Constants
    float k = 8.0 / PI;
    k = 6.0f * k / (h * h * h);

    // Calculate norm of r
    float r_norm = norm(r);
    float q = r_norm / h;

    // Initialize the result vector
    Vector3f res = {0.0, 0.0, 0.0};

    if (r_norm > 1e-5 && q <= 1.0) {
        Vector3f grad_q = {r.x / (r_norm * h), r.y / (r_norm * h), r.z / (r_norm * h)};

        if (q <= 0.5) {
            res.x = k * q * (3.0 * q - 2.0) * grad_q.x;
            res.y = k * q * (3.0 * q - 2.0) * grad_q.y;
            res.z = k * q * (3.0 * q - 2.0) * grad_q.z;
        } else {
            float factor = 1.0 - q;
            res.x = k * (-factor * factor) * grad_q.x;
            res.y = k * (-factor * factor) * grad_q.y;
            res.z = k * (-factor * factor) * grad_q.z;
        }
    }
    return res;
}


#endif //SPH_FLUID_SPH_CUH
