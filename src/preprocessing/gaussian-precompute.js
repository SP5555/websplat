'use strict';

import { quat, mat3 } from 'gl-matrix';

export function GaussianPrecompute(raw) {
    // positions, scales, rotations, opacities are Float32Arrays
    const { vertexCount, positions, scales, rotations, colors, opacities } = raw;

    const covariances = new Float32Array(vertexCount * 6);

    const R = mat3.create(); // temp rotation matrix
    for (let i = 0; i < vertexCount; i++) {

        // unpack scale
        const sx = scales[i*3 + 0];
        const sy = scales[i*3 + 1];
        const sz = scales[i*3 + 2];

        // convert quaternion to rotation matrix
        const q = quat.fromValues(
            rotations[i*4 + 0],
            rotations[i*4 + 1],
            rotations[i*4 + 2],
            rotations[i*4 + 3]
        );

        quat.normalize(q, q);
        mat3.fromQuat(R, q);

        // build covariance = R * diag(s^2) * R^T
        const sxx = sx*sx, syy = sy*sy, szz = sz*sz;

        const r00 = R[0], r01 = R[1], r02 = R[2];
        const r10 = R[3], r11 = R[4], r12 = R[5];
        const r20 = R[6], r21 = R[7], r22 = R[8];

        // Covariance matrix layout:
        // [ cxx cxy cxz ]
        // [ cxy cyy cyz ]
        // [ cxz cyz czz ]
        // note there is only 6 unique entries

        const cxx = r00*r00*sxx + r01*r01*syy + r02*r02*szz;
        const cxy = r00*r10*sxx + r01*r11*syy + r02*r12*szz;
        const cxz = r00*r20*sxx + r01*r21*syy + r02*r22*szz;

        const cyy = r10*r10*sxx + r11*r11*syy + r12*r12*szz;
        const cyz = r10*r20*sxx + r11*r21*syy + r12*r22*szz;

        const czz = r20*r20*sxx + r21*r21*syy + r22*r22*szz;

        const base = i*6;
        covariances[base + 0] = cxx;
        covariances[base + 1] = cxy;
        covariances[base + 2] = cxz;
        covariances[base + 3] = cyy;
        covariances[base + 4] = cyz;
        covariances[base + 5] = czz;

        // ===== FLIP Y AND Z AXES =====
        // Reflect * C * Reflect^T where Reflect = diag(1, -1, -1)
        // [ cxx  -cxy  -cxz ]
        // [ -cxy  cyy   cyz ]
        // [ -cxz  cyz   czz ]

        // flip positions
        positions[i*3 + 1] *= -1; // y
        positions[i*3 + 2] *= -1; // z
        // flip covariance
        covariances[base + 1] *= -1; // cxy
        covariances[base + 2] *= -1; // cxz
        // cyz, cxx, cyy, czz stay the same
    }

    return {
        vertexCount,
        positions,
        covariances,
        colors,
        opacities
    };
}