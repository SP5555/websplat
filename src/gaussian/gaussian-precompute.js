'use strict';

import { quat, mat3 } from 'gl-matrix';

export function GaussianPrecompute(raw) {
    // positions, scales, rotations, opacities are Float32Arrays
    const { vertexCount, positions, scales, rotations, colors, opacities } = raw;

    const covariances = new Float32Array(vertexCount * 6);

    const R = mat3.create();
    const S = mat3.create();
    const cov = mat3.create();

    for (let i = 0; i < vertexCount; i++) {

        // unpack scale
        const sx = scales[i*3 + 0];
        const sy = scales[i*3 + 1];
        const sz = scales[i*3 + 2];

        // convert quaternion to rotation matrix
        const q = quat.fromValues(
            rotations[i*4 + 0], // x
            rotations[i*4 + 1], // y
            rotations[i*4 + 2], // z
            rotations[i*4 + 3]  // w
        );
        quat.normalize(q, q);
        mat3.fromQuat(R, q);

        // build covariance = R * S * S^T * R^T
        S[0] = sx; // x
        S[4] = sy; // y
        S[8] = sz; // z
        // R * S
        mat3.multiply(cov, R, S);
        // R * S * S^T * R^T
        mat3.multiply(cov, cov, mat3.transpose(mat3.create(), cov));
        
        // Covariance matrix layout:
        // [ cxx cxy cxz ]
        // [ cxy cyy cyz ]
        // [ cxz cyz czz ]
        // note there is only 6 unique entries
        const base = i * 6;
        covariances[base + 0] = cov[0]; // cxx
        covariances[base + 1] = cov[1]; // cxy
        covariances[base + 2] = cov[2]; // cxz
        covariances[base + 3] = cov[4]; // cyy
        covariances[base + 4] = cov[5]; // cyz
        covariances[base + 5] = cov[8]; // czz

        // flip y and z axes
        positions[i*3 + 1] *= -1;
        positions[i*3 + 2] *= -1;

        covariances[base + 1] *= -1; // cxy
        covariances[base + 2] *= -1; // cxz
    }

    return {
        vertexCount,
        positions,
        covariances,
        colors,
        opacities
    };
}